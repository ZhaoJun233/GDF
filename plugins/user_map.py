import asyncio
import json
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger

from core.bot import Bot, flag_config
from core.protocol.events import C2CMessage, Event, GroupAtMessage

PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_MAP_PATH = "data/users_map.json"
DEFAULT_CONFIDENCE = 0.8
INTRO_COOLDOWN = 15.0
INTRO_CLOCK_LIMIT = 1000
INTRO_SCAN_LEN = 120
INTRO_SEND_LEN = 200
INTRO_MAX_TOKENS = 80
ALIAS_MAX_LEN = 16
DEFAULT_WAKE_KEYWORDS = ("诺喵莉", "喵莉")
DISALLOWED_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Zl", "Zp"}
FORBIDDEN_CHARS = set("[]【】［］|:：\r\n\t")
CLAUSE_FINDER = re.compile(r"[^?？。！!，,\n]+[?？。！!，,\n]?")
INTRO_QUESTIONS = (
    "我是谁", "我叫什么", "我叫啥", "叫我什么", "我的名字是什么",
    "who am i", "what is my name", "what's my name",
)
INTRO_HINTS = (
    "我叫", "叫我", "我是", "名字是", "名字叫", "称呼我", "喊我",
    "备注", "本名", "昵称", "外号", "my name", "call me",
)
BLOCKED_ALIASES = {
    "我", "你", "他", "她", "它", "群主", "管理员", "机器人", "助手", "系统",
    "system", "assistant", "user", "admin", "null", "none", "unknown",
    "爸爸", "爹", "爷爷", "妈妈", "主人", "群友", "路人", "新人",
}

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
    state = getattr(bot, "_user_map_state", None)
    if not isinstance(state, dict):
        state = {
            "lock": None,
            "lock_loop": None,
            "cache_key": None,
            "cache_data": None,
            "corrupt_key": None,
            "tasks": set(),
            "in_flight": set(),
            "clock": {},
            "cleanup_added": False,
        }
        setattr(bot, "_user_map_state", state)
    return state


def _get_lock(state: dict[str, Any]) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = state.get("lock")
    if lock is None or state.get("lock_loop") is not loop:
        lock = asyncio.Lock()
        state["lock"] = lock
        state["lock_loop"] = loop
    return lock


def wake_keywords(bot: Bot) -> tuple[str, ...]:
    raw = bot.config.get("llm_wake_keywords")
    if isinstance(raw, list):
        items = tuple(str(item).strip() for item in raw if str(item).strip())
        if items:
            return items
    return DEFAULT_WAKE_KEYWORDS


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


def _fold_name(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip()).casefold()


def valid_claimed_name(name: str, wake_words: Sequence[str] = ()) -> bool:
    cleaned = name.strip()
    if not (1 <= len(cleaned) <= ALIAS_MAX_LEN):
        return False
    if any(
        ch in FORBIDDEN_CHARS or unicodedata.category(ch) in DISALLOWED_CATEGORIES
        for ch in cleaned
    ):
        return False
    norm = unicodedata.normalize("NFKC", cleaned)
    if any(ch in FORBIDDEN_CHARS for ch in norm):
        return False
    folded = norm.casefold()
    if folded.startswith("用户_") or folded == "未知用户" or folded in BLOCKED_ALIASES:
        return False
    if folded in {_fold_name(str(w)) for w in wake_words if str(w).strip()}:
        return False
    return any(ch.isalnum() for ch in norm)


def _alias_taken(group_map: dict[str, str], uid: str, alias: str) -> bool:
    target = _fold_name(alias)
    return any(
        other_uid != uid and _fold_name(other_name) == target
        for other_uid, other_name in group_map.items()
    )


def normalize_map(raw: Any, wake_words: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    result = empty_map()
    if not isinstance(raw, dict):
        return result
    raw_groups = raw.get("groups")
    if isinstance(raw_groups, dict):
        for gid, members in raw_groups.items():
            clean_gid = str(gid or "").strip()
            if not clean_gid or not isinstance(members, dict):
                continue
            target: dict[str, str] = {}
            for uid, alias in members.items():
                clean_uid = str(uid or "").strip()
                if (
                    isinstance(alias, str)
                    and clean_uid
                    and valid_claimed_name(alias, wake_words)
                    and not _alias_taken(target, clean_uid, alias)
                ):
                    target[clean_uid] = alias.strip()
            if target:
                result["groups"][clean_gid] = target
    raw_users = raw.get("users")
    if isinstance(raw_users, dict):
        for uid, alias in raw_users.items():
            clean_uid = str(uid or "").strip()
            if isinstance(alias, str) and clean_uid and valid_claimed_name(alias, wake_words):
                result["users"][clean_uid] = alias.strip()
    return result


def _stat_file(path: Path) -> tuple[str, tuple[int, int]]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return ("missing", (-1, -1))
    except OSError:
        return ("error", (-2, -2))
    return ("ok", (st.st_mtime_ns, st.st_size))


def _read_disk(
    path: Path, wake_words: Sequence[str]
) -> tuple[str, dict[str, dict[str, Any]], tuple[int, int], BaseException | None]:
    status, stat = _stat_file(path)
    if status == "missing":
        return ("missing", empty_map(), stat, None)
    if status != "ok":
        return ("corrupt", empty_map(), stat, OSError(f"stat failed: {path}"))
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
        data = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ("corrupt", empty_map(), stat, exc)
    if not isinstance(data, dict):
        return ("corrupt", empty_map(), stat, ValueError("users_map root must be object"))
    return ("ok", normalize_map(data, wake_words), stat, None)


def _write_unlocked(path: Path, mapping: dict[str, dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(mapping, ensure_ascii=False, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


def _set_empty_cache(
    state: dict[str, Any], path_key: str, wakes: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    empty = empty_map()
    state["cache_data"] = empty
    state["cache_key"] = (path_key, -1, -1, wakes)
    state["corrupt_key"] = None
    return _clone_map(empty)


async def load_map(bot: Bot) -> dict[str, dict[str, Any]]:
    state = _state(bot)
    path = map_path(bot)
    path_key = str(path)
    wakes = wake_keywords(bot)
    async with _get_lock(state):
        st_status, stat = await asyncio.to_thread(_stat_file, path)
        if st_status == "missing":
            return _set_empty_cache(state, path_key, wakes)
        key = (path_key, stat[0], stat[1], wakes)
        if st_status == "ok" and state["cache_data"] is not None and state["cache_key"] == key:
            return _clone_map(state["cache_data"])
        disk_status, data, disk_stat, exc = await asyncio.to_thread(_read_disk, path, wakes)
        if disk_status == "missing":
            return _set_empty_cache(state, path_key, wakes)
        disk_key = (path_key, disk_stat[0], disk_stat[1])
        if disk_status == "corrupt":
            empty = empty_map()
            state["cache_data"] = empty
            state["cache_key"] = (
                (path_key, disk_stat[0], disk_stat[1], wakes)
                if disk_stat != (-2, -2)
                else None
            )
            if state["corrupt_key"] != disk_key:
                state["corrupt_key"] = disk_key
                logger.opt(exception=exc).error("用户映射文件损坏, 本次按空表处理 path={}", path)
            return _clone_map(empty)
        state["corrupt_key"] = None
        state["cache_data"] = data
        state["cache_key"] = (path_key, disk_stat[0], disk_stat[1], wakes)
        return _clone_map(data)


async def set_alias(
    bot: Bot, user_id: str, alias: str, group_openid: str = ""
) -> bool:
    uid = user_id.strip()
    cleaned = alias.strip()
    gid = group_openid.strip()
    wakes = wake_keywords(bot)
    if not uid or uid == "unknown" or not valid_claimed_name(cleaned, wakes):
        return False
    state = _state(bot)
    path = map_path(bot)
    path_key = str(path)
    async with _get_lock(state):
        disk_status, current, disk_stat, exc = await asyncio.to_thread(_read_disk, path, wakes)
        disk_key = (path_key, disk_stat[0], disk_stat[1])
        if disk_status == "corrupt":
            if state["corrupt_key"] != disk_key:
                state["corrupt_key"] = disk_key
                logger.opt(exception=exc).error("用户映射文件损坏, 拒绝覆写写入 path={}", path)
            return False
        state["corrupt_key"] = None
        if gid:
            group_map = current["groups"].setdefault(gid, {})
            if _alias_taken(group_map, uid, cleaned):
                if not group_map:
                    current["groups"].pop(gid, None)
                return False
            existing = group_map.get(uid)
            group_map[uid] = cleaned
        else:
            existing = current["users"].get(uid)
            current["users"][uid] = cleaned
        if existing == cleaned and disk_status == "ok":
            state["cache_data"] = current
            state["cache_key"] = (path_key, disk_stat[0], disk_stat[1], wakes)
            return True
        new_stat = await asyncio.to_thread(_write_unlocked, path, current)
        state["cache_data"] = current
        state["cache_key"] = (path_key, new_stat[0], new_stat[1], wakes)
    logger.info("用户备注已更新 group={} user={} alias={}", gid or "c2c", uid[-6:], cleaned)
    return True


def extract_user_id(event: Event) -> str:
    if isinstance(event, GroupAtMessage) and isinstance(event.raw, dict):
        raw_author = event.raw.get("author")
        if isinstance(raw_author, dict) and raw_author.get("member_openid"):
            return str(raw_author["member_openid"]).strip()
    author = getattr(event, "author", None)
    author_id = str(getattr(author, "id", "") or "").strip()
    return author_id or "unknown"


def extract_group_id(event: Event) -> str:
    return event.group_openid.strip() if isinstance(event, GroupAtMessage) else ""


def extract_member_nick(event: Event, bot: Bot | None = None) -> str | None:
    if isinstance(event, GroupAtMessage) and event.member.nick:
        nick = event.member.nick.strip()
        wakes = wake_keywords(bot) if bot is not None else DEFAULT_WAKE_KEYWORDS
        if valid_claimed_name(nick, wakes):
            return nick
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
    source = ((mapping.get("groups") or {}).get(gid) if gid else mapping.get("users"))
    if isinstance(source, dict):
        alias = source.get(uid)
        if isinstance(alias, str) and alias.strip():
            return alias.strip()
    return None


def resolve_name(
    mapping: dict[str, Any],
    user_id: str,
    group_openid: str = "",
    member_nick: str | None = None,
    wake_words: Sequence[str] = (),
) -> str:
    uid = user_id.strip()
    gid = group_openid.strip()
    alias = custom_name(mapping, uid, gid)
    if alias:
        return alias
    if member_nick and valid_claimed_name(member_nick, wake_words):
        group_map = ((mapping.get("groups") or {}).get(gid) or {}) if gid else {}
        if not (isinstance(group_map, dict) and _alias_taken(group_map, uid, member_nick)):
            return member_nick.strip()
    return fallback_label(uid)


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
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return (data.get("is_self_intro") is True, claimed, max(0.0, min(1.0, confidence)))


def _record_clock(state: dict[str, Any], clock_key: str) -> None:
    clock = state["clock"]
    now = time.monotonic()
    if len(clock) >= INTRO_CLOCK_LIMIT:
        for key in [k for k, ts in clock.items() if now - ts >= INTRO_COOLDOWN]:
            clock.pop(key, None)
        while len(clock) >= INTRO_CLOCK_LIMIT:
            clock.pop(next(iter(clock)))
    clock[clock_key] = now


def spawn_extract(
    event: Event,
    bot: Bot,
    prompt: str,
    conv_id: str,
    user_id: str,
    complete_fn: CompleteFn,
) -> asyncio.Task[None] | None:
    auto_enabled = (
        flag_config(bot.config, "llm_auto_name_enabled")
        if "llm_auto_name_enabled" in bot.config
        else True
    )
    if not auto_enabled or not user_id or user_id == "unknown" or not might_be_intro(prompt):
        return None
    state = _state(bot)
    clock_key = f"{conv_id}:{user_id}"
    clock = state["clock"]
    if clock_key in state["in_flight"]:
        return None
    if clock_key in clock and time.monotonic() - clock[clock_key] < INTRO_COOLDOWN:
        return None
    if not state["cleanup_added"]:
        state["cleanup_added"] = True
        bot.on_shutdown(lambda: _cancel_task_set(state["tasks"]))
    state["in_flight"].add(clock_key)
    task = asyncio.create_task(
        _extract_and_save(event, bot, prompt, conv_id, user_id, complete_fn, clock_key)
    )
    state["tasks"].add(task)
    task.add_done_callback(state["tasks"].discard)
    return task


async def _extract_and_save(
    event: Event,
    bot: Bot,
    prompt: str,
    conv_id: str,
    user_id: str,
    complete_fn: CompleteFn,
    clock_key: str,
) -> None:
    state = _state(bot)
    try:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": DEFAULT_INTRO_PROMPT},
            {"role": "user", "content": prompt.strip()[:INTRO_SEND_LEN]},
        ]
        raw = await complete_fn(bot, messages, INTRO_MAX_TOKENS)
        is_intro, claimed, confidence = parse_intro_result(raw)
        try:
            threshold = float(bot.config.get("llm_name_confidence_threshold", DEFAULT_CONFIDENCE))
        except (TypeError, ValueError):
            threshold = DEFAULT_CONFIDENCE
        if not is_intro or not claimed or confidence < threshold:
            return
        if await set_alias(bot, user_id, claimed, group_openid=extract_group_id(event)):
            _record_clock(state, clock_key)
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
