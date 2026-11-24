import asyncio
import json
import os
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from core.bot import Bot
from core.protocol.events import Event, GroupAtMessage

PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_MAP_PATH = "data/users_map.json"
ALIAS_MAX_LEN = 16
ALLOWED_ROOT_KEYS = {"groups", "users"}
DISALLOWED_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Zl", "Zp"}
FORBIDDEN_CHARS = set("[]【】［］|:：\r\n\t")
BLOCKED_ALIASES = {
    "我", "你", "他", "她", "它", "群主", "管理员", "机器人", "助手", "系统",
    "system", "assistant", "user", "admin", "null", "none", "unknown",
    "爸爸", "爹", "爷爷", "妈妈", "主人", "群友", "路人", "新人",
}


def _state(bot: Bot) -> dict[str, Any]:
    state = getattr(bot, "_user_map_state", None)
    if not isinstance(state, dict):
        state = {}
        setattr(bot, "_user_map_state", state)
    state.setdefault("lock", None)
    state.setdefault("lock_loop", None)
    state.setdefault("cache_key", None)
    state.setdefault("cache_data", None)
    state.setdefault("corrupt_key", None)
    return state


def _get_lock(state: dict[str, Any]) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = state.get("lock")
    if lock is None or state.get("lock_loop") is not loop:
        lock = asyncio.Lock()
        state["lock"] = lock
        state["lock_loop"] = loop
    return lock


def map_path(bot: Bot) -> Path:
    raw = str(bot.config.get("llm_users_map_path") or DEFAULT_MAP_PATH)
    path = Path(raw)
    return path if path.is_absolute() else PLUGIN_DIR.parent / path


def empty_map() -> dict[str, dict[str, Any]]:
    return {"groups": {}, "users": {}}


def _clone_map(mapping: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = mapping.get("groups")
    users = mapping.get("users")
    return {
        "groups": {
            k: dict(v) for k, v in groups.items() if isinstance(v, dict)
        } if isinstance(groups, dict) else {},
        "users": dict(users) if isinstance(users, dict) else {},
    }


def _fold_name(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip()).casefold()


def valid_claimed_name(name: str, wake_words: Sequence[str] = ()) -> bool:
    if not isinstance(name, str):
        return False
    cleaned = name.strip()
    if not (1 <= len(cleaned) <= ALIAS_MAX_LEN):
        return False
    if any(
        ch in FORBIDDEN_CHARS or unicodedata.category(ch) in DISALLOWED_CATEGORIES
        for ch in cleaned
    ):
        return False
    norm = unicodedata.normalize("NFKC", cleaned).strip()
    if not (1 <= len(norm) <= ALIAS_MAX_LEN):
        return False
    if any(
        ch in FORBIDDEN_CHARS or unicodedata.category(ch) in DISALLOWED_CATEGORIES
        for ch in norm
    ):
        return False
    folded = norm.casefold()
    if folded.startswith("用户_") or folded == "未知用户" or folded in BLOCKED_ALIASES:
        return False
    wake_folded = {
        _fold_name(w) for w in wake_words if isinstance(w, str) and w.strip()
    }
    if folded in wake_folded:
        return False
    return any(ch.isalnum() for ch in norm)


def _alias_taken(group_map: dict[str, Any], uid: str, alias: str) -> bool:
    if not isinstance(group_map, dict):
        return False
    target = _fold_name(alias)
    return any(
        isinstance(other_uid, str)
        and other_uid != uid
        and isinstance(other_name, str)
        and _fold_name(other_name) == target
        for other_uid, other_name in group_map.items()
    )


def _validate_map_data(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("users_map root must be object")
    unknown_keys = {str(key) for key in raw.keys()} - ALLOWED_ROOT_KEYS
    if unknown_keys:
        raise ValueError(f"users_map has unknown top-level keys: {sorted(unknown_keys)}")
    result = empty_map()
    raw_groups = raw.get("groups", {})
    if not isinstance(raw_groups, dict):
        raise ValueError("users_map.groups must be object")
    for gid, members in raw_groups.items():
        if not isinstance(gid, str) or not gid.strip():
            raise ValueError("users_map group id must be non-empty string")
        clean_gid = gid.strip()
        if clean_gid in result["groups"]:
            raise ValueError(f"duplicate group id after strip: {clean_gid}")
        if not isinstance(members, dict):
            raise ValueError(f"users_map.groups[{clean_gid}] must be object")
        target: dict[str, str] = {}
        for uid, alias in members.items():
            if not isinstance(uid, str) or not uid.strip() or uid.strip() == "unknown":
                raise ValueError(f"invalid member uid in group {clean_gid}")
            clean_uid = uid.strip()
            if clean_uid in target:
                raise ValueError(f"duplicate uid after strip in group {clean_gid}: {clean_uid}")
            if not isinstance(alias, str) or not valid_claimed_name(alias):
                raise ValueError(f"invalid alias in group {clean_gid} for uid {clean_uid}")
            if _alias_taken(target, clean_uid, alias):
                raise ValueError(f"alias collision in group {clean_gid}: {alias}")
            target[clean_uid] = alias.strip()
        result["groups"][clean_gid] = target
    raw_users = raw.get("users", {})
    if not isinstance(raw_users, dict):
        raise ValueError("users_map.users must be object")
    for uid, alias in raw_users.items():
        if not isinstance(uid, str) or not uid.strip() or uid.strip() == "unknown":
            raise ValueError("invalid uid in users_map.users")
        clean_uid = uid.strip()
        if clean_uid in result["users"]:
            raise ValueError(f"duplicate uid after strip in users_map.users: {clean_uid}")
        if not isinstance(alias, str) or not valid_claimed_name(alias):
            raise ValueError(f"invalid alias in users_map.users for uid {clean_uid}")
        result["users"][clean_uid] = alias.strip()
    return result


def _stat_or_none(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read_disk_strict(
    path: Path, stat_box: list[tuple[int, int]]
) -> tuple[dict[str, dict[str, Any]], tuple[int, int]] | None:
    try:
        with path.open("r", encoding="utf-8-sig") as fp:
            st = os.fstat(fp.fileno())
            file_stat = (st.st_mtime_ns, st.st_size)
            stat_box.append(file_stat)
            raw_text = fp.read()
    except FileNotFoundError:
        return None
    data = json.loads(raw_text)
    return (_validate_map_data(data), file_stat)


def _write_unlocked(path: Path, mapping: dict[str, dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(mapping, ensure_ascii=False, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


def _set_empty_cache(state: dict[str, Any], path_key: str) -> dict[str, dict[str, Any]]:
    empty = empty_map()
    state["cache_data"] = empty
    state["cache_key"] = (path_key, -1, -1)
    state["corrupt_key"] = None
    return _clone_map(empty)


async def load_map(bot: Bot) -> dict[str, dict[str, Any]]:
    state = _state(bot)
    path = map_path(bot)
    path_key = str(path)
    async with _get_lock(state):
        stat_box: list[tuple[int, int]] = []
        try:
            cur_stat = await asyncio.to_thread(_stat_or_none, path)
            if cur_stat is None:
                return _set_empty_cache(state, path_key)
            key = (path_key, cur_stat[0], cur_stat[1])
            if state["cache_data"] is not None and state["cache_key"] == key:
                return _clone_map(state["cache_data"])
            loaded = await asyncio.to_thread(_read_disk_strict, path, stat_box)
        except (OSError, UnicodeDecodeError, ValueError):
            empty = empty_map()
            state["cache_data"] = empty
            if stat_box:
                disk_key = (path_key, stat_box[-1][0], stat_box[-1][1])
                state["cache_key"] = disk_key
                if state["corrupt_key"] != disk_key:
                    state["corrupt_key"] = disk_key
                    logger.opt(exception=True).error(
                        "用户映射文件损坏, 本次按空表处理 path={}", path
                    )
            else:
                state["cache_key"] = None
                state["corrupt_key"] = None
                logger.opt(exception=True).error(
                    "用户映射文件读取失败, 本次按空表处理 path={}", path
                )
            return _clone_map(empty)
        if loaded is None:
            return _set_empty_cache(state, path_key)
        data, disk_stat = loaded
        state["corrupt_key"] = None
        state["cache_data"] = data
        state["cache_key"] = (path_key, disk_stat[0], disk_stat[1])
        return _clone_map(data)


async def set_alias(
    bot: Bot,
    user_id: str,
    alias: str,
    group_openid: str = "",
    wake_words: Sequence[str] = (),
) -> bool:
    if not isinstance(user_id, str) or not isinstance(alias, str):
        return False
    uid = user_id.strip()
    cleaned = alias.strip()
    gid = group_openid.strip() if isinstance(group_openid, str) else ""
    if not uid or uid == "unknown" or not valid_claimed_name(cleaned, wake_words):
        return False
    state = _state(bot)
    path = map_path(bot)
    path_key = str(path)
    async with _get_lock(state):
        stat_box: list[tuple[int, int]] = []
        try:
            loaded = await asyncio.to_thread(_read_disk_strict, path, stat_box)
        except (OSError, UnicodeDecodeError, ValueError):
            if stat_box:
                disk_key = (path_key, stat_box[-1][0], stat_box[-1][1])
                if state["corrupt_key"] != disk_key:
                    state["corrupt_key"] = disk_key
                    logger.opt(exception=True).error(
                        "用户映射文件损坏, 拒绝覆写写入 path={}", path
                    )
            else:
                state["corrupt_key"] = None
                logger.opt(exception=True).error(
                    "用户映射文件读取失败, 拒绝覆写写入 path={}", path
                )
            return False
        state["corrupt_key"] = None
        if loaded is None:
            current = empty_map()
            disk_stat: tuple[int, int] | None = None
        else:
            current, disk_stat = loaded
        if gid:
            existing_group = current["groups"].get(gid)
            if isinstance(existing_group, dict) and _alias_taken(existing_group, uid, cleaned):
                return False
            group_map = current["groups"].setdefault(gid, {})
            existing = group_map.get(uid)
            group_map[uid] = cleaned
        else:
            existing = current["users"].get(uid)
            current["users"][uid] = cleaned
        if existing == cleaned and disk_stat is not None:
            state["cache_data"] = current
            state["cache_key"] = (path_key, disk_stat[0], disk_stat[1])
            return True
        new_stat = await asyncio.to_thread(_write_unlocked, path, current)
        state["cache_data"] = current
        state["cache_key"] = (path_key, new_stat[0], new_stat[1])
    logger.info("用户备注已更新 group={} user={} alias={}", gid or "c2c", uid[-6:], cleaned)
    return True


def extract_user_id(event: Event) -> str:
    if isinstance(event, GroupAtMessage) and isinstance(event.raw, dict):
        raw_author = event.raw.get("author")
        if isinstance(raw_author, dict):
            member_openid = str(raw_author.get("member_openid") or "").strip()
            if member_openid:
                return member_openid
    author = getattr(event, "author", None)
    author_id = str(getattr(author, "id", "") or "").strip()
    return author_id or "unknown"


def extract_group_id(event: Event) -> str:
    return event.group_openid.strip() if isinstance(event, GroupAtMessage) else ""


def extract_member_nick(event: Event, wake_words: Sequence[str] = ()) -> str | None:
    if isinstance(event, GroupAtMessage) and isinstance(event.member.nick, str):
        nick = event.member.nick.strip()
        if valid_claimed_name(nick, wake_words):
            return nick
    return None


def fallback_label(user_id: str) -> str:
    uid = user_id.strip() if isinstance(user_id, str) else ""
    if not uid or uid == "unknown":
        return "未知用户"
    suffix = uid[-6:] if len(uid) >= 6 else uid
    return f"用户_{suffix}"


def custom_name(
    mapping: dict[str, Any],
    user_id: str,
    group_openid: str = "",
    wake_words: Sequence[str] = (),
) -> str | None:
    if not isinstance(mapping, dict) or not isinstance(user_id, str):
        return None
    uid = user_id.strip()
    gid = group_openid.strip() if isinstance(group_openid, str) else ""
    if not uid:
        return None
    if gid:
        groups = mapping.get("groups")
        source = groups.get(gid) if isinstance(groups, dict) else None
    else:
        source = mapping.get("users")
    if isinstance(source, dict):
        alias = source.get(uid)
        if isinstance(alias, str) and valid_claimed_name(alias, wake_words):
            return alias.strip()
    return None


def resolve_name(
    mapping: dict[str, Any],
    user_id: str,
    group_openid: str = "",
    member_nick: str | None = None,
    wake_words: Sequence[str] = (),
) -> str:
    uid = user_id.strip() if isinstance(user_id, str) else ""
    gid = group_openid.strip() if isinstance(group_openid, str) else ""
    alias = custom_name(mapping, uid, gid, wake_words=wake_words)
    if alias:
        return alias
    if isinstance(member_nick, str) and valid_claimed_name(member_nick, wake_words):
        groups = mapping.get("groups") if isinstance(mapping, dict) else None
        group_map = groups.get(gid) if gid and isinstance(groups, dict) else None
        if not (isinstance(group_map, dict) and _alias_taken(group_map, uid, member_nick)):
            return member_nick.strip()
    return fallback_label(uid)
