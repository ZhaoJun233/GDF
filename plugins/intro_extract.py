import asyncio
import json
import math
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import Any

from loguru import logger

from core.bot import Bot, flag_config
from core.protocol.events import Event, GroupAtMessage
from plugins import user_map

DEFAULT_CONFIDENCE = 0.8
INTRO_COOLDOWN = 15.0
INTRO_QUOTA_DEFAULT = 3
INTRO_QUOTA_HARD_CAP = 10
INTRO_QUOTA_WINDOW = 60.0
INTRO_CLOCK_LIMIT = 1000
INTRO_SCAN_LEN = 120
INTRO_SEND_LEN = 200
INTRO_MAX_TOKENS = 80
CLAUSE_FINDER = re.compile(r"[^?？。！!，,\n]+[?？。！!，,\n]?")
INTRO_QUESTIONS = (
    "我是谁", "我叫什么", "我叫啥", "叫我什么", "我的名字是什么",
    "who am i", "what is my name", "what's my name",
)
INTRO_HINTS = (
    "我叫", "叫我", "我是", "名字是", "名字叫", "称呼我", "喊我",
    "备注", "本名", "昵称", "外号", "my name", "call me",
)

DEFAULT_INTRO_PROMPT = (
    "你是群聊身份识别器。请判断这条消息中【发言人本人】是否在主动告知自己的名字、"
    "昵称或希望别人对他的称呼。\n"
    "判定规则：\n"
    "1. 只有发言人明确介绍自己（如“我叫小明”“以后叫我阿梓”“我是老王”）时，is_self_intro 才为 true。\n"
    "2. 若是在谈论他人（如“A叫小明”“你知道小明吗”）、开玩笑占便宜（如“我是你爹”“我是秦始皇”）、"
    "描述状态或身份属性（如“我是学生”“我是新来的”“我叫外卖了”），is_self_intro 必须为 false。\n"
    "3. 只输出一行合法 JSON，禁止输出 Markdown 代码块或任何解释：\n"
    '{"is_self_intro": false, "claimed_name": null, "confidence": 0.0}'
)

CompleteFn = Callable[[Bot, list[dict[str, Any]], int], Awaitable[str]]


def _state(bot: Bot) -> dict[str, Any]:
    state = getattr(bot, "_intro_extract_state", None)
    if not isinstance(state, dict):
        state = {}
        setattr(bot, "_intro_extract_state", state)
    state.setdefault("tasks", set())
    state.setdefault("in_flight", set())
    state.setdefault("clock", {})
    state.setdefault("conv_quota", {})
    state.setdefault("cleanup_added", False)
    return state


def might_be_intro(prompt: str) -> bool:
    head = prompt.strip()[:INTRO_SCAN_LEN].lower()
    if len(head) < 2:
        return False
    for question in INTRO_QUESTIONS:
        head = head.replace(question, " ")
    for clause in CLAUSE_FINDER.findall(head):
        part = clause.strip()
        if not part or part.endswith(("?", "？", "吗")):
            continue
        if any(hint in part for hint in INTRO_HINTS):
            return True
    return False


def _parse_finite_float(raw: Any, default: float) -> float:
    if isinstance(raw, bool):
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    return val if math.isfinite(val) else default


def _parse_int(raw: Any, default: int) -> int:
    if isinstance(raw, bool):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError, OverflowError):
        return default


def _confidence_threshold(bot: Bot) -> float:
    val = _parse_finite_float(
        bot.config.get("llm_name_confidence_threshold", DEFAULT_CONFIDENCE),
        DEFAULT_CONFIDENCE,
    )
    return max(0.0, min(1.0, val))


def parse_intro_result(raw_text: str) -> tuple[bool, str | None, float]:
    text = raw_text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return (False, None, 0.0)
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return (False, None, 0.0)
    if not isinstance(data, dict):
        return (False, None, 0.0)
    raw_name = data.get("claimed_name")
    claimed = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    confidence = _parse_finite_float(data.get("confidence", 0.0), 0.0)
    return (data.get("is_self_intro") is True, claimed, max(0.0, min(1.0, confidence)))


def _record_clock(clock: dict[str, float], clock_key: str, now: float) -> None:
    if len(clock) >= INTRO_CLOCK_LIMIT:
        for key in [k for k, ts in clock.items() if now - ts >= INTRO_COOLDOWN]:
            clock.pop(key, None)
        while len(clock) >= INTRO_CLOCK_LIMIT:
            clock.pop(next(iter(clock)))
    clock[clock_key] = now


def _consume_conv_quota(
    state: dict[str, Any], conv_id: str, now: float, bot: Bot
) -> bool:
    limit = _parse_int(bot.config.get("llm_max_concurrency"), INTRO_QUOTA_DEFAULT)
    if limit <= 0:
        limit = INTRO_QUOTA_HARD_CAP
    window = _parse_finite_float(
        bot.config.get("llm_concurrency_window"), INTRO_QUOTA_WINDOW
    )
    if window <= 0:
        window = INTRO_QUOTA_WINDOW
    quotas: dict[str, list[float]] = state["conv_quota"]
    stamps = [ts for ts in quotas.get(conv_id, []) if now - ts < window]
    if len(stamps) >= limit:
        quotas[conv_id] = stamps
        return False
    if len(quotas) >= INTRO_CLOCK_LIMIT and conv_id not in quotas:
        for key in [
            k for k, items in quotas.items() if not items or now - items[-1] >= window
        ]:
            quotas.pop(key, None)
        while len(quotas) >= INTRO_CLOCK_LIMIT:
            quotas.pop(next(iter(quotas)))
    stamps.append(now)
    quotas[conv_id] = stamps
    return True


def _keyword_blocked(event: Event, bot: Bot, prompt: str, wake_words: Sequence[str]) -> bool:
    if not isinstance(event, GroupAtMessage):
        return False
    mode = str(bot.config.get("llm_reply_mode") or "smart").strip().lower()
    if mode != "keyword":
        return False
    return not any(w in prompt for w in wake_words if isinstance(w, str) and w)


def spawn_extract(
    event: Event,
    bot: Bot,
    prompt: str,
    conv_id: str,
    user_id: str,
    complete_fn: CompleteFn,
    wake_words: Sequence[str] = (),
) -> asyncio.Task[None] | None:
    auto_enabled = (
        flag_config(bot.config, "llm_auto_name_enabled")
        if "llm_auto_name_enabled" in bot.config
        else True
    )
    if not auto_enabled or not user_id or user_id == "unknown":
        return None
    if _keyword_blocked(event, bot, prompt, wake_words):
        return None
    if not might_be_intro(prompt):
        return None
    state = _state(bot)
    clock_key = f"{conv_id}:{user_id}"
    now = time.monotonic()
    clock: dict[str, float] = state["clock"]
    if clock_key in state["in_flight"]:
        return None
    if clock_key in clock and now - clock[clock_key] < INTRO_COOLDOWN:
        return None
    if not _consume_conv_quota(state, conv_id, now, bot):
        return None
    _record_clock(clock, clock_key, now)
    if not state["cleanup_added"]:
        state["cleanup_added"] = True
        bot.on_shutdown(lambda: _cancel_task_set(state["tasks"]))
    state["in_flight"].add(clock_key)
    task = asyncio.create_task(
        _extract_and_save(
            event, bot, prompt, conv_id, user_id, complete_fn, clock_key, wake_words
        )
    )
    state["tasks"].add(task)

    def _on_done(done_task: asyncio.Task[None]) -> None:
        state["tasks"].discard(done_task)
        state["in_flight"].discard(clock_key)

    task.add_done_callback(_on_done)
    return task


async def _extract_and_save(
    event: Event,
    bot: Bot,
    prompt: str,
    conv_id: str,
    user_id: str,
    complete_fn: CompleteFn,
    clock_key: str,
    wake_words: Sequence[str] = (),
) -> None:
    state = _state(bot)
    try:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": DEFAULT_INTRO_PROMPT},
            {"role": "user", "content": prompt.strip()[:INTRO_SEND_LEN]},
        ]
        raw = await complete_fn(bot, messages, INTRO_MAX_TOKENS)
        is_intro, claimed, confidence = parse_intro_result(raw)
        threshold = _confidence_threshold(bot)
        if not is_intro or not claimed or confidence < threshold:
            return
        await user_map.set_alias(
            bot,
            user_id,
            claimed,
            group_openid=user_map.extract_group_id(event),
            wake_words=wake_words,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.opt(exception=True).error("自称识别失败 conv={}", conv_id)
    finally:
        state["in_flight"].discard(clock_key)


async def _cancel_task_set(tasks: set[asyncio.Task[None]]) -> None:
    for task in list(tasks):
        task.cancel()
    for task in list(tasks):
        with suppress(asyncio.CancelledError):
            await task
    tasks.clear()
