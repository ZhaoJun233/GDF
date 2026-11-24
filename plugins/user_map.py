import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger

from core.bot import Bot
from core.protocol.events import C2CMessage, Event, GroupAtMessage

PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_MAP_PATH = "data/users_map.json"
DEFAULT_CONFIDENCE = 0.8
DEFAULT_COOLDOWN = 15.0
INTRO_CLOCK_LIMIT = 1000
INTRO_MIN_LEN = 2
INTRO_MAX_LEN = 80
INTRO_MAX_TOKENS = 80
ALIAS_MAX_LEN = 16
DEFAULT_WAKE_KEYWORDS = ("诺喵莉", "喵莉")

INTRO_HINTS = (
    "我叫",
    "叫我",
    "我是",
    "名字是",
    "名字叫",
    "称呼我",
    "喊我",
    "备注",
    "本名",
    "昵称",
    "外号",
    "my name",
    "call me",
)
BLOCKED_ALIASES = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "群主",
    "管理员",
    "机器人",
    "助手",
    "系统",
    "system",
    "assistant",
    "user",
    "admin",
    "null",
    "none",
    "unknown",
    "爸爸",
    "爹",
    "爷爷",
    "妈妈",
    "主人",
    "群友",
    "路人",
    "新人",
}
FORBIDDEN_CHARS = set("[]\r\n\t:：")

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

_MAP_LOCK = asyncio.Lock()
_CACHE: dict[str, dict[str, Any]] | None = None
_CACHE_KEY: tuple[str, int, int] | None = None
_INTRO_TASKS: set[asyncio.Task[None]] = set()
_INTRO_CLOCK: dict[str, float] = {}


def map_path(bot: Bot) -> Path:
    raw = str(bot.config.get("llm_users_map_path") or DEFAULT_MAP_PATH)
    path = Path(raw)
    return path if path.is_absolute() else PLUGIN_DIR.parent / path


def empty_map() -> dict[str, dict[str, Any]]:
    return {"groups": {}, "users": {}}


def _clone_map(mapping: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "groups": {k: dict(v) for k, v in mapping.get("groups", {}).items()},
        "users": dict(mapping.get("users", {})),
    }


def normalize_map(raw: Any) -> dict[str, dict[str, Any]]:
    result = empty_map()
    if not isinstance(raw, dict):
        return result
    raw_groups = raw.get("groups")
    if isinstance(raw_groups, dict):
        for gid, members in raw_groups.items():
            _merge_group_entries(result["groups"], gid, members)
    raw_users = raw.get("users")
    if isinstance(raw_users, dict):
        for uid, alias in raw_users.items():
            clean_uid = str(uid or "").strip()
            if isinstance(alias, str) and clean_uid and alias.strip():
                result["users"][clean_uid] = alias.strip()
    for key, value in raw.items():
        if key in ("groups", "users") or not isinstance(value, dict):
            continue
        _merge_group_entries(result["groups"], key, value)
    return result


def _merge_group_entries(
    groups: dict[str, dict[str, str]], group_id: Any, members: Any
) -> None:
    gid = str(group_id or "").strip()
    if not gid or not isinstance(members, dict):
        return
    target = groups.setdefault(gid, {})
    for uid, alias in members.items():
        clean_uid = str(uid or "").strip()
        if isinstance(alias, str) and clean_uid and alias.strip():
            target.setdefault(clean_uid, alias.strip())
    if not target:
        groups.pop(gid, None)


def _stat_file(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


async def load_map(bot: Bot) -> dict[str, dict[str, Any]]:
    global _CACHE, _CACHE_KEY
    path = map_path(bot)
    path_key = str(path)
    stat = await asyncio.to_thread(_stat_file, path)
    if stat is None:
        empty = empty_map()
        _CACHE = empty
        _CACHE_KEY = (path_key, -1, -1)
        return _clone_map(empty)
    key = (path_key, stat[0], stat[1])
    if _CACHE is not None and _CACHE_KEY == key:
        return _clone_map(_CACHE)
    try:
        raw_text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        data = json.loads(raw_text)
    except (OSError, json.JSONDecodeError):
        logger.opt(exception=True).error("用户映射文件损坏, 本次按空表处理")
        return empty_map()
    normalized = normalize_map(data)
    _CACHE = normalized
    _CACHE_KEY = key
    return _clone_map(normalized)


def _wake_keywords(bot: Bot) -> tuple[str, ...]:
    raw = bot.config.get("llm_wake_keywords")
    if isinstance(raw, list):
        items = tuple(str(item).strip() for item in raw if str(item).strip())
        if items:
            return items
    return DEFAULT_WAKE_KEYWORDS


def valid_claimed_name(name: str, wake_keywords: Sequence[str] = ()) -> bool:
    cleaned = name.strip()
    if not (1 <= len(cleaned) <= ALIAS_MAX_LEN):
        return False
    if any(ch in FORBIDDEN_CHARS for ch in cleaned):
        return False
    lowered = cleaned.lower()
    if lowered in BLOCKED_ALIASES:
        return False
    blocked_wake = {str(w).strip().lower() for w in wake_keywords if str(w).strip()}
    if lowered in blocked_wake:
        return False
    return any(ch.isalnum() for ch in cleaned)


def _read_unlocked(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return empty_map()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.opt(exception=True).error("读取用户映射文件失败, 将重建映射结构")
        return empty_map()
    return normalize_map(data)


def _write_unlocked(path: Path, mapping: dict[str, dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(mapping, ensure_ascii=False, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


async def set_alias(
    bot: Bot, user_id: str, alias: str, group_openid: str = ""
) -> bool:
    global _CACHE, _CACHE_KEY
    uid = user_id.strip()
    cleaned = alias.strip()
    gid = group_openid.strip()
    if not uid or uid == "unknown":
        return False
    if not valid_claimed_name(cleaned, _wake_keywords(bot)):
        return False
    path = map_path(bot)
    async with _MAP_LOCK:
        current = await asyncio.to_thread(_read_unlocked, path)
        if gid:
            current["groups"].setdefault(gid, {})[uid] = cleaned
        else:
            current["users"][uid] = cleaned
        stat = await asyncio.to_thread(_write_unlocked, path, current)
        _CACHE = current
        _CACHE_KEY = (str(path), stat[0], stat[1])
    logger.info(
        "用户备注已更新 group={} user={} alias={}",
        gid or "c2c",
        uid[-6:],
        cleaned,
    )
    return True


def extract_user_id(event: Event) -> str:
    raw_author = event.raw.get("author") if isinstance(event.raw, dict) else None
    if isinstance(raw_author, dict):
        if isinstance(event, GroupAtMessage):
            candidate = raw_author.get("member_openid") or raw_author.get("id")
        elif isinstance(event, C2CMessage):
            candidate = raw_author.get("user_openid") or raw_author.get("id")
        else:
            candidate = (
                raw_author.get("id")
                or raw_author.get("member_openid")
                or raw_author.get("user_openid")
            )
        if candidate:
            return str(candidate).strip()
    author = getattr(event, "author", None)
    author_id = str(getattr(author, "id", "") or "").strip()
    return author_id or "unknown"


def extract_group_id(event: Event) -> str:
    if isinstance(event, GroupAtMessage):
        return event.group_openid.strip()
    return ""


def extract_member_nick(event: Event) -> str | None:
    if isinstance(event, GroupAtMessage) and event.member.nick:
        nick = event.member.nick.strip()
        return nick or None
    return None


def fallback_label(user_id: str) -> str:
    uid = user_id.strip()
    if not uid or uid == "unknown":
        return "未知用户"
    suffix = uid[-6:] if len(uid) >= 6 else uid
    return f"用户_{suffix}"


def custom_name(
    mapping: dict[str, Any], user_id: str, group_openid: str = ""
) -> str | None:
    uid = user_id.strip()
    gid = group_openid.strip()
    if not uid:
        return None
    if gid:
        group_map = (mapping.get("groups") or {}).get(gid)
        if isinstance(group_map, dict):
            alias = group_map.get(uid)
            if isinstance(alias, str) and alias.strip():
                return alias.strip()
    user_map_dict = mapping.get("users")
    if isinstance(user_map_dict, dict):
        alias = user_map_dict.get(uid)
        if isinstance(alias, str) and alias.strip():
            return alias.strip()
    return None


def resolve_name(
    mapping: dict[str, Any],
    user_id: str,
    group_openid: str = "",
    member_nick: str | None = None,
) -> str:
    uid = user_id.strip()
    gid = group_openid.strip()
    if gid:
        group_map = (mapping.get("groups") or {}).get(gid)
        if isinstance(group_map, dict):
            alias = group_map.get(uid)
            if isinstance(alias, str) and alias.strip():
                return alias.strip()
        if member_nick and member_nick.strip():
            return member_nick.strip()
        user_map_dict = mapping.get("users")
        if isinstance(user_map_dict, dict):
            alias = user_map_dict.get(uid)
            if isinstance(alias, str) and alias.strip():
                return alias.strip()
        return fallback_label(uid)
    user_map_dict = mapping.get("users")
    if isinstance(user_map_dict, dict):
        alias = user_map_dict.get(uid)
        if isinstance(alias, str) and alias.strip():
            return alias.strip()
    if member_nick and member_nick.strip():
        return member_nick.strip()
    return fallback_label(uid)


def might_be_intro(prompt: str, max_len: int = INTRO_MAX_LEN) -> bool:
    text = prompt.strip()
    if not (INTRO_MIN_LEN <= len(text) <= max_len):
        return False
    lowered = text.lower()
    return any(hint in lowered for hint in INTRO_HINTS)


def parse_intro_result(raw_text: str) -> tuple[bool, str | None, float]:
    text = raw_text.strip()
    if not text:
        return (False, None, 0.0)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return (False, None, 0.0)
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return (False, None, 0.0)
    if not isinstance(data, dict):
        return (False, None, 0.0)
    is_intro = data.get("is_self_intro") is True
    raw_name = data.get("claimed_name")
    claimed = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return (is_intro, claimed, confidence)


def spawn_extract(
    event: Event,
    bot: Bot,
    prompt: str,
    conv_id: str,
    user_id: str,
    complete_fn: CompleteFn,
) -> asyncio.Task[None] | None:
    if not _bool_config(bot.config, "llm_auto_name_enabled", default=True):
        return None
    if not user_id or user_id == "unknown":
        return None
    if not might_be_intro(prompt):
        return None
    cooldown = _float_config(bot, "llm_intro_cooldown", DEFAULT_COOLDOWN)
    now = time.monotonic()
    clock_key = f"{conv_id}:{user_id}"
    if cooldown > 0 and now - _INTRO_CLOCK.get(clock_key, 0.0) < cooldown:
        return None
    if len(_INTRO_CLOCK) >= INTRO_CLOCK_LIMIT:
        cutoff = max(cooldown, 0.0)
        expired = [k for k, ts in _INTRO_CLOCK.items() if now - ts >= cutoff]
        for k in expired:
            _INTRO_CLOCK.pop(k, None)
        while len(_INTRO_CLOCK) >= INTRO_CLOCK_LIMIT:
            _INTRO_CLOCK.pop(next(iter(_INTRO_CLOCK)))
    _INTRO_CLOCK[clock_key] = now
    if cancel_tasks not in bot.cleanups:
        bot.on_shutdown(cancel_tasks)
    task = asyncio.create_task(
        _extract_and_save(event, bot, prompt, conv_id, user_id, complete_fn)
    )
    _INTRO_TASKS.add(task)
    task.add_done_callback(_INTRO_TASKS.discard)
    return task


async def _extract_and_save(
    event: Event,
    bot: Bot,
    prompt: str,
    conv_id: str,
    user_id: str,
    complete_fn: CompleteFn,
) -> None:
    try:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": DEFAULT_INTRO_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = await complete_fn(bot, messages, INTRO_MAX_TOKENS)
        is_intro, claimed, confidence = parse_intro_result(raw)
        threshold = _float_config(
            bot, "llm_name_confidence_threshold", DEFAULT_CONFIDENCE
        )
        if not is_intro or not claimed or confidence < threshold:
            return
        await set_alias(bot, user_id, claimed, group_openid=extract_group_id(event))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.opt(exception=True).error("自称识别失败 conv={}", conv_id)


async def cancel_tasks() -> None:
    for task in list(_INTRO_TASKS):
        task.cancel()
    for task in list(_INTRO_TASKS):
        with suppress(asyncio.CancelledError):
            await task
    _INTRO_TASKS.clear()


def _bool_config(config: dict[str, Any], key: str, default: bool = False) -> bool:
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _float_config(bot: Bot, key: str, default: float) -> float:
    try:
        return float(bot.config.get(key, default))
    except (TypeError, ValueError):
        return default
