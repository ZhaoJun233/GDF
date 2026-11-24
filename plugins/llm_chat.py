import asyncio
import base64
import json
import re
import shutil
import time
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from core.bot import Bot, flag_config
from core.protocol.events import C2CMessage, Event, GroupAtMessage
from plugins import mcp_server, user_map

Handler = Callable[..., Awaitable[None]]

PLUGIN_DIR = Path(__file__).resolve().parent
SYSTEM_PATH = PLUGIN_DIR / "prompt_system.txt"
WORLD_PATH = PLUGIN_DIR / "prompt_world.txt"

DEFAULT_HISTORY_PATH = "data/history.json"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CONTEXT_LIMIT = 10
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_CONCURRENCY_WINDOW = 60.0
DEFAULT_MEMORY_GC_DAYS = 7
DEFAULT_MEMORY_GC_INTERVAL = 86400.0
DEFAULT_REPLY_MODE = "smart"
DEFAULT_WAKE_KEYWORDS = ("诺喵莉", "喵莉")
DEFAULT_EVAL_PROMPT = "你需要判断是否需要回复这句话。只输出 YES 或 NO。"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_TIMEOUT = 15.0
MAX_TOOL_ROUNDS = 3
MCP_TIMEOUT = 10.0
DEFAULT_SUBCONSCIOUS_PROMPT = (
    "你是后台观察者。当前时间是：{current_time}。"
    "请提取关于【{user_name}】的重要事实或情绪，"
    "提取事实时，必须带上明确的时间节点（基于系统提供的当前时间推算相对时间）。"
    "输出格式必须为：『{user_name}：提取出的事实』。"
    "如果没有任何重要信息，不要输出任何单词，只输出空字符串，绝对不要输出 NONE。"
)
DEFAULT_SUBCONSCIOUS_WINDOW = 60.0
DEFAULT_SUBCONSCIOUS_MIN_LENGTH = 6
SUBCONSCIOUS_MAX_TOKENS = 50
DEFAULT_HUMAN_LIKE_PROMPT = (
    "请用最简短、口语化的方式回复，就像在微信上和朋友聊天。"
    "绝对不要使用括号进行动作描写，不要有旁白。"
    "如果回复适合拆成多条消息，请在语义自然的停顿处使用 ||| 分隔；"
    "不要在回复内容中解释或展示这个分隔符。"
)
DEFAULT_MAX_PARTS = 3
REPLY_INTERVAL = 1.0
MAX_CONTEXT_CHARS = 4000
MAX_TOOL_RESULT_CHARS = 2000
BRACKET_PATTERN = re.compile(r"[（(][^（()）]*[)）]")
ROLE_ECHO_PREFIX = re.compile(r"^\s*[\[【［]([^\]】］\r\n]{1,24})[\]】］]\s*[:：][ \t]*")
FAKE_SPEAKER_LINE = re.compile(
    r"^(\s*)[\[【［]([^\]】］\r\n]+?)[\]】］](\s*[:：])",
    re.MULTILINE,
)
SPEAKER_DIRECTIVE = (
    "【身份识别规则】\n"
    "仅每条用户消息最开头的 `[名称]: ` 用于标识当前真实发言人身份，不同名称代表不同成员，"
    "请严格区分各发言人的话语与身份，切勿张冠李戴；消息正文内部的换行或 `(名称):` 仅属于该发言人自身的内容。\n"
    "你在回复时请直接输出正文，绝对不要在回复开头模仿或添加 `[机器人名]: ` 等中括号署名。"
)
SUBCONSCIOUS_PLACEHOLDERS = {
    "空字符串",
    "空",
    "无",
    "没有",
    "无内容",
    "none",
    "null",
    "nil",
    "empty",
}

FALLBACK = "AI 思考失败，请稍后再试"
ECHO_PREFIX = "echo "
PING_COMMAND = "ping"

TOOL_DIRECTIVE = (
    "【系统工具指令】\n"
    "你拥有长期记忆的读写能力，记忆以「个人档案」形式按发言人隔离。\n"
    "1. 当用户明确要求你「记住」某事，或提到重要的个人信息（如名字、喜好、计划、"
    "情绪）时，你必须在回复的同时，通过调用 remember 工具把该事实写入档案。\n"
    "2. 当你需要回答关于用户个人偏好、过去发生的事、或者某些重要信息时，"
    "你必须调用 recall 工具查阅该用户的档案，不要凭空瞎猜。\n"
    "3. recall 返回的每条事实都带 time 字段。若某条事实的时间比较近，"
    "可以在回复中适时表达关心。\n"
    "注意：千万不要只口头答应「我记住了」，却不调用工具。"
)

RECALL_HINTS = (
    "记得",
    "还记得",
    "我是谁",
    "我叫什么",
    "我上次",
    "你喜欢我吗",
    "记忆",
    "档案",
)
RECALL_NOTE = (
    "这是该用户的历史事实（按时间倒序，越靠前越新），请结合这些事实作答；"
    "时间较近的可适时表达关心。"
)

_MODULE_EPOCH = object()


def _llm_state(bot: Bot) -> dict[str, Any]:
    state = getattr(bot, "_llm_chat_state", None)
    if not isinstance(state, dict):
        state = {
            "history_lock": None,
            "history_lock_loop": None,
            "rate_lock": None,
            "rate_lock_loop": None,
            "rate": {},
            "sub_tasks": set(),
            "sub_clock": {},
            "cleanup_added": False,
            "system_cache": None,
            "system_key": None,
            "epoch": None,
        }
        setattr(bot, "_llm_chat_state", state)
    return state


def _bot_lock(state: dict[str, Any], name: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    loop_key = f"{name}_loop"
    lock = state.get(name)
    if lock is None or state.get(loop_key) is not loop:
        lock = asyncio.Lock()
        state[name] = lock
        state[loop_key] = loop
    return lock


async def _cancel_subtasks(tasks: set[asyncio.Task[None]]) -> None:
    for task in list(tasks):
        task.cancel()
    for task in list(tasks):
        with suppress(asyncio.CancelledError):
            await task
    tasks.clear()


def on_group_at(func: Handler) -> Handler:
    setattr(func, "__event__", "GROUP_AT_MESSAGE_CREATE")
    return func


def on_c2c(func: Handler) -> Handler:
    setattr(func, "__event__", "C2C_MESSAGE_CREATE")
    return func


@on_group_at
async def llm_group(event: GroupAtMessage, bot: Bot) -> None:
    await _chat(event, bot)


@on_c2c
async def llm_c2c(event: C2CMessage, bot: Bot) -> None:
    await _chat(event, bot)


async def _chat(event: Event, bot: Bot) -> None:
    prompt = event.content.strip()
    if not prompt and not event.images:
        return
    has_text = bool(prompt)
    if not prompt:
        prompt = "[图片]"
    if prompt.startswith(ECHO_PREFIX) or prompt == PING_COMMAND:
        return
    conv_id = _conv_id(event)
    user_id = user_map.extract_user_id(event)
    group_openid = user_map.extract_group_id(event)
    member_nick = user_map.extract_member_nick(event, bot)
    _ensure_gc(bot)
    intro_task = (
        user_map.spawn_extract(event, bot, prompt, conv_id, user_id, _complete_intro)
        if has_text
        else None
    )
    if not await _should_reply(event, bot, prompt):
        logger.info("唤醒判断未通过，跳过回复 conv={}", conv_id)
        _spawn_subconscious(
            event, bot, prompt, conv_id, user_id, group_openid, member_nick, intro_task
        )
        return
    if not await _allow(bot, conv_id):
        logger.warning("触发频控, 静默跳过 conv={}", conv_id)
        _spawn_subconscious(
            event, bot, prompt, conv_id, user_id, group_openid, member_nick, intro_task
        )
        return
    mapping = await user_map.load_map(bot)
    wakes = user_map.wake_keywords(bot)
    display_name = user_map.resolve_name(
        mapping,
        user_id,
        group_openid=group_openid,
        member_nick=member_nick,
        wake_words=wakes,
    )
    logger.info("收到 [{}] 消息: {}", event.type, prompt[:50])
    limit = _int_config(bot, "llm_context_limit", DEFAULT_CONTEXT_LIMIT)
    history_lock = _bot_lock(_llm_state(bot), "history_lock")
    async with history_lock:
        history = await _load(bot)
        turns = _messages_of(history.get(conv_id))
    custom_alias = user_map.custom_name(mapping, user_id, group_openid=group_openid)
    use_speaker_tag = bool(group_openid or custom_alias)
    effective_prompt = (
        f"[{display_name}]: {_escape_speaker_lines(prompt)}"
        if use_speaker_tag
        else prompt
    )
    user_content = await _user_content(bot, effective_prompt, event.images)
    messages = _build_messages(
        bot,
        user_content,
        turns,
        limit,
        mapping=mapping,
        group_openid=group_openid,
        include_speaker_rule=use_speaker_tag,
        wake_words=wakes,
    )
    model = str(bot.config.get("llm_model") or DEFAULT_MODEL)
    logger.info("正在请求 LLM, model={}, 图片数={}", model, len(event.images))
    try:
        reply = await _complete(
            bot,
            messages,
            conv_id=conv_id,
            user_id=user_id,
            display_name=display_name,
            prompt=prompt,
        )
    except Exception:
        logger.opt(exception=True).error(
            "LLM 请求失败 conv={} event={}", conv_id, event.type
        )
        await bot.reply(event, FALLBACK)
        return
    reply = _strip_role_echo(reply, bot)
    logger.info("LLM 回复成功: {}...", reply[:30])
    async with history_lock:
        history = await _load(bot)
        turns = _messages_of(history.get(conv_id))
        user_turn: dict[str, Any] = {
            "role": "user",
            "user_id": user_id,
            "content": prompt,
        }
        if member_nick:
            user_turn["nick"] = member_nick
        turns.append(user_turn)
        turns.append({"role": "assistant", "content": reply})
        history[conv_id] = {
            "last_active": time.time(),
            "messages": turns[-2 * limit :] if limit > 0 else [],
        }
        await _save(bot, history)
    await _send_reply(event, bot, reply)


async def _complete_intro(
    bot: Bot, messages: list[dict[str, Any]], max_tokens: int
) -> str:
    data = await _post_completion(
        bot,
        messages,
        max_tokens=max_tokens,
        temperature=0.1,
        disable_thinking=True,
    )
    return _extract_text(data)


def _escape_speaker_lines(text: str) -> str:
    return FAKE_SPEAKER_LINE.sub(r"\1(\2)\3", text)


def _strip_single_echo(text: str, known_tags: set[str]) -> str:
    cleaned = text
    while True:
        match = ROLE_ECHO_PREFIX.match(cleaned)
        if not match:
            return cleaned
        tag = unicodedata.normalize("NFKC", match.group(1)).strip().casefold()
        if tag not in known_tags:
            return cleaned
        stripped = cleaned[match.end() :]
        if not stripped:
            return cleaned
        cleaned = stripped


def _strip_role_echo(reply: str, bot: Bot) -> str:
    wake_items = {
        unicodedata.normalize("NFKC", item).strip().casefold()
        for item in user_map.wake_keywords(bot)
    }
    known_tags = wake_items | {"assistant", "ai", "bot", "机器人", "助手"}
    if not flag_config(bot.config, "llm_human_like_mode"):
        return _strip_single_echo(reply, known_tags)
    cleaned_parts: list[str] = []
    for part in reply.split("|||"):
        step = _strip_single_echo(part.strip(), known_tags)
        step = _clean_brackets(step)
        step = _strip_single_echo(step, known_tags).strip()
        if step:
            cleaned_parts.append(step)
    return " ||| ".join(cleaned_parts) if cleaned_parts else _strip_single_echo(reply, known_tags)


async def _complete(
    bot: Bot,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    conv_id: str = "",
    user_id: str = "",
    display_name: str = "",
    prompt: str = "",
) -> str:
    if conv_id and await mcp_server.ensure_started(bot):
        return await _request_with_tools(
            bot, messages, user_id, display_name, prompt, max_tokens
        )
    data = await _post_completion(bot, messages, max_tokens)
    return _extract_text(data)


async def _post_completion(
    bot: Bot,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    temperature: float | None = None,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    api_key = str(bot.config.get("llm_api_key") or "")
    if not api_key:
        raise RuntimeError("config.toml 缺少 llm_api_key")
    base_url = str(bot.config.get("llm_base_url") or DEFAULT_BASE_URL).rstrip("/")
    timeout = _float_config(bot, "llm_timeout", DEFAULT_TIMEOUT)
    payload = _payload(
        bot,
        messages,
        max_tokens,
        temperature=temperature,
        disable_thinking=disable_thinking,
    )
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    session = await bot.api.session()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with session.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        data: Any = await resp.json(content_type=None)
        if resp.status >= 400:
            logger.error(
                "LLM 请求失败 status={} data={}", resp.status, str(data)[:200]
            )
            raise RuntimeError(f"LLM status={resp.status} body={str(data)[:200]}")
    if not isinstance(data, dict):
        raise RuntimeError(f"LLM 响应非对象: {data}")
    return data


def _extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM 响应缺少 choices: {data}")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip() or FALLBACK


async def _request_with_tools(
    bot: Bot,
    messages: list[dict[str, Any]],
    user_id: str = "",
    display_name: str = "",
    prompt: str = "",
    max_tokens: int | None = None,
) -> str:
    working = list(messages)
    choice = _tool_choice(prompt)
    for _ in range(MAX_TOOL_ROUNDS):
        try:
            data = await _post_completion(
                bot, working, max_tokens, mcp_server.LLM_TOOLS, choice
            )
        except RuntimeError as exc:
            if choice == "required" and "tool_choice" in str(exc).lower():
                logger.warning("模型不支持 tool_choice=required, 降级为 auto")
                choice = "auto"
                continue
            raise
        choice = "auto"
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM 响应缺少 choices: {data}")
        message = choices[0].get("message") or {}
        calls = message.get("tool_calls") or []
        if not calls:
            return str(message.get("content") or "").strip() or FALLBACK
        working.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": calls,
            }
        )
        for call in calls:
            working.append(await _exec_tool(bot, user_id, display_name, call))
    raise RuntimeError("工具调用轮次超限")


def _tool_choice(prompt: str) -> str:
    return "required" if any(word in prompt for word in RECALL_HINTS) else "auto"


async def _exec_tool(
    bot: Bot, user_id: str, display_name: str, call: dict[str, Any]
) -> dict[str, Any]:
    call_id = str(call.get("id") or "")
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw = function.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) and raw else {}
    except json.JSONDecodeError:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": call_id or name,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments,
            "user_id": user_id,
            "display_name": display_name,
        },
    }
    result = await _mcp_post(bot, payload)
    logger.info("MCP 调用 {} user={} -> {}", name, display_name, str(result)[:120])
    content = _tool_text(result)
    if len(content) > MAX_TOOL_RESULT_CHARS:
        content = (
            f"{content[:MAX_TOOL_RESULT_CHARS]}"
            f"\n...[已截断，原长 {len(content)} 字符]"
        )
    if name == "recall":
        content = f"{content}\n{RECALL_NOTE}"
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }


async def _mcp_post(bot: Bot, payload: dict[str, Any]) -> Any:
    host = str(bot.config.get("mcp_host") or "127.0.0.1")
    port = _int_config(bot, "mcp_port", 8080)
    token = str(bot.config.get("mcp_token") or "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session = await bot.api.session()
    try:
        async with session.post(
            f"http://{host}:{port}/mcp",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=MCP_TIMEOUT),
        ) as resp:
            return await resp.json(content_type=None)
    except Exception:
        logger.opt(exception=True).error(
            "MCP 请求失败 tool={}", (payload.get("params") or {}).get("name")
        )
        return {"error": "MCP 请求失败"}


def _tool_text(result: Any) -> str:
    if isinstance(result, dict):
        body = result.get("result")
        if isinstance(body, dict):
            texts = [
                str(item.get("text"))
                for item in body.get("content") or []
                if isinstance(item, dict) and item.get("text")
            ]
            if texts:
                return "\n".join(texts)
        if "error" in result:
            return json.dumps(result["error"], ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def _spawn_subconscious(
    event: Event,
    bot: Bot,
    prompt: str,
    conv_id: str,
    user_id: str,
    group_openid: str = "",
    member_nick: str | None = None,
    intro_task: asyncio.Task[None] | None = None,
) -> asyncio.Task[None] | None:
    if not flag_config(bot.config, "llm_subconscious_enabled"):
        return None
    if not flag_config(bot.config, "mcp_enabled"):
        return None
    min_length = _int_config(
        bot, "llm_subconscious_min_length", DEFAULT_SUBCONSCIOUS_MIN_LENGTH
    )
    if len(prompt) < min_length:
        return None
    window = _float_config(
        bot, "llm_subconscious_window", DEFAULT_SUBCONSCIOUS_WINDOW
    )
    state = _llm_state(bot)
    sub_clock = state["sub_clock"]
    now = time.monotonic()
    if conv_id in sub_clock and now - sub_clock[conv_id] < window:
        return None
    sub_clock[conv_id] = now
    if not state["cleanup_added"]:
        state["cleanup_added"] = True
        bot.on_shutdown(lambda: _cancel_subtasks(state["sub_tasks"]))
    task = asyncio.create_task(
        _subconscious_memory(
            bot, prompt, conv_id, user_id, group_openid, member_nick, intro_task
        )
    )
    state["sub_tasks"].add(task)
    task.add_done_callback(state["sub_tasks"].discard)
    return task


async def _subconscious_memory(
    bot: Bot,
    prompt: str,
    conv_id: str,
    user_id: str,
    group_openid: str = "",
    member_nick: str | None = None,
    intro_task: asyncio.Task[None] | None = None,
) -> None:
    try:
        if intro_task is not None:
            with suppress(Exception):
                await intro_task
        mapping = await user_map.load_map(bot)
        display_name = user_map.resolve_name(
            mapping,
            user_id,
            group_openid=group_openid,
            member_nick=member_nick,
            wake_words=user_map.wake_keywords(bot),
        )
        template = str(
            bot.config.get("llm_subconscious_prompt") or DEFAULT_SUBCONSCIOUS_PROMPT
        )
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        system = template.replace("{user_name}", display_name).replace(
            "{current_time}", current_time
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": _escape_speaker_lines(prompt)},
        ]
        data = await _post_completion(
            bot, messages, SUBCONSCIOUS_MAX_TOKENS, disable_thinking=True
        )
        finding = _extract_text(data).strip()
        if _is_placeholder(finding):
            return
        await _remember(bot, user_id, display_name, finding)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.opt(exception=True).error("潜意识分析失败 conv={}", conv_id)


def _is_placeholder(finding: str) -> bool:
    if not finding or finding == FALLBACK:
        return True
    cleaned = finding.strip().strip("\"'“”「」").strip()
    if not cleaned:
        return True
    if cleaned.lower() in SUBCONSCIOUS_PLACEHOLDERS:
        return True
    return "NONE" in cleaned.upper()


async def _remember(
    bot: Bot, user_id: str, display_name: str, content: str
) -> None:
    if not await mcp_server.ensure_started(bot):
        logger.warning("MCP 未启用或启动失败, 放弃潜意识写入 user={}", display_name)
        return
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": f"sub-{int(time.time() * 1000)}",
        "method": "tools/call",
        "params": {
            "name": "remember",
            "arguments": {"fact": content},
            "user_id": user_id,
            "display_name": display_name,
        },
    }
    result = await _mcp_post(bot, payload)
    if isinstance(result, dict) and "error" in result:
        logger.warning("潜意识写入失败 user={} -> {}", display_name, str(result)[:120])
        return
    logger.info("潜意识记忆已入库 user={} -> {}", display_name, str(result)[:120])


def _clean_brackets(text: str) -> str:
    cleaned = BRACKET_PATTERN.sub("", text)
    return " ".join(cleaned.split())


def _reply_parts(text: str) -> list[str]:
    return [part.strip() for part in text.split("|||") if part.strip()]


async def _send_reply(event: Event, bot: Bot, reply: str) -> None:
    if not flag_config(bot.config, "llm_human_like_mode"):
        await bot.reply(event, reply)
        return
    cleaned = _clean_brackets(reply)
    parts = _reply_parts(cleaned)
    if not parts:
        await bot.reply(event, reply)
        return
    limit = _int_config(bot, "llm_reply_max_parts", DEFAULT_MAX_PARTS)
    if limit > 0 and len(parts) > limit:
        parts = parts[: limit - 1] + ["".join(parts[limit - 1 :])]
    for index, sentence in enumerate(parts):
        if index:
            await asyncio.sleep(REPLY_INTERVAL)
        await bot.reply(event, sentence)


def _payload(
    bot: Bot,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float | None = None,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": str(bot.config.get("llm_model") or DEFAULT_MODEL),
        "messages": messages,
        "stream": False,
        "temperature": (
            temperature
            if temperature is not None
            else _float_config(bot, "llm_temperature", DEFAULT_TEMPERATURE)
        ),
        "top_p": _float_config(bot, "llm_top_p", DEFAULT_TOP_P),
        "max_tokens": (
            max_tokens
            if max_tokens is not None
            else _int_config(bot, "llm_max_tokens", DEFAULT_MAX_TOKENS)
        ),
    }
    thinking = str(bot.config.get("llm_thinking") or "").strip()
    if disable_thinking:
        if thinking:
            payload["thinking"] = {"type": "disabled"}
    else:
        if thinking:
            payload["thinking"] = {"type": thinking}
        effort = str(bot.config.get("llm_reasoning_effort") or "").strip()
        if effort:
            payload["reasoning_effort"] = effort
        if flag_config(bot.config, "llm_json_mode"):
            payload["response_format"] = {"type": "json_object"}
    if flag_config(bot.config, "llm_logprobs"):
        payload["logprobs"] = True
    return payload


def _chars_of(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(item.get("content") or "")) for item in messages)


def _render_turn(
    item: dict[str, Any],
    mapping: dict[str, Any] | None,
    group_openid: str,
    wake_words: Sequence[str] = (),
) -> tuple[dict[str, str], bool]:
    role = str(item.get("role") or "user")
    content = str(item.get("content") or "")
    if role != "user" or mapping is None:
        return ({"role": role, "content": content}, False)
    uid = str(item.get("user_id") or "").strip()
    if not uid:
        escaped = _escape_speaker_lines(content) if group_openid else content
        return ({"role": role, "content": escaped}, False)
    nick = str(item.get("nick") or "").strip() or None
    if group_openid:
        name = user_map.resolve_name(
            mapping,
            uid,
            group_openid=group_openid,
            member_nick=nick,
            wake_words=wake_words,
        )
        return ({"role": role, "content": f"[{name}]: {_escape_speaker_lines(content)}"}, True)
    alias = user_map.custom_name(mapping, uid, group_openid="")
    if alias:
        return ({"role": role, "content": f"[{alias}]: {_escape_speaker_lines(content)}"}, True)
    return ({"role": role, "content": content}, False)


def _build_messages(
    bot: Bot,
    user_content: Any,
    turns: list[dict[str, Any]],
    limit: int,
    mapping: dict[str, Any] | None = None,
    group_openid: str = "",
    include_speaker_rule: bool = False,
    wake_words: Sequence[str] = (),
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    raw_selected = turns[-2 * limit :] if limit > 0 else []
    selected: list[dict[str, str]] = []
    any_tagged = False
    for item in raw_selected:
        if not isinstance(item, dict):
            continue
        rendered, tagged = _render_turn(item, mapping, group_openid, wake_words)
        selected.append(rendered)
        any_tagged = any_tagged or tagged
    system = _system_prompt(bot)
    if include_speaker_rule or any_tagged:
        system = f"{system}\n\n{SPEAKER_DIRECTIVE}" if system else SPEAKER_DIRECTIVE
    if system:
        messages.append({"role": "system", "content": system})
    while selected and _chars_of(selected) > MAX_CONTEXT_CHARS:
        selected = selected[1:]
    while selected and selected[0].get("role") != "user":
        selected = selected[1:]
    messages.extend(selected)
    messages.append({"role": "user", "content": user_content})
    return messages


async def _should_reply(event: Event, bot: Bot, prompt: str) -> bool:
    if isinstance(event, C2CMessage):
        return True
    mode = str(bot.config.get("llm_reply_mode") or DEFAULT_REPLY_MODE).strip().lower()
    if mode == "always":
        return True
    if mode == "keyword":
        return _has_keyword(bot, prompt)
    return await _eval_reply(bot, prompt)


def _has_keyword(bot: Bot, prompt: str) -> bool:
    return any(word in prompt for word in user_map.wake_keywords(bot))


async def _eval_reply(bot: Bot, prompt: str) -> bool:
    system = str(bot.config.get("llm_eval_prompt") or DEFAULT_EVAL_PROMPT)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        data = await _post_completion(
            bot, messages, max_tokens=8, temperature=0.0, disable_thinking=True
        )
        answer = _extract_text(data)
    except Exception:
        logger.opt(exception=True).error("唤醒判断失败, 本轮按不回复处理")
        return False
    return answer.strip().upper().startswith("YES")


async def _download_image(bot: Bot, url: str) -> str | None:
    logger.info("开始下载图片: {}", url[:50])
    session = await bot.api.session()
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=IMAGE_TIMEOUT)
        ) as resp:
            if resp.status >= 400:
                logger.warning("图片下载失败 status={}", resp.status)
                return None
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            data = await resp.content.read(MAX_IMAGE_BYTES + 1)
    except Exception:
        logger.opt(exception=True).error("图片下载异常")
        return None
    if not data:
        return None
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning("图片超过 {} 字节上限, 已忽略", MAX_IMAGE_BYTES)
        return None
    logger.info("图片下载完成, 大小: {} 字节", len(data))
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


async def _user_content(bot: Bot, prompt: str, images: list[str]) -> Any:
    if not images or not flag_config(bot.config, "llm_vision_enabled"):
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for url in images:
        data_url = await _download_image(bot, url)
        if data_url is None:
            continue
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
    if len(parts) == 1:
        return prompt
    return parts


def _system_prompt(bot: Bot) -> str:
    state = _llm_state(bot)
    human_like = flag_config(bot.config, "llm_human_like_mode")
    human = (
        str(bot.config.get("llm_human_like_prompt") or DEFAULT_HUMAN_LIKE_PROMPT).strip()
        if human_like
        else ""
    )
    key = (flag_config(bot.config, "mcp_enabled"), human_like, human)
    if state["system_cache"] is None or state["epoch"] is not _MODULE_EPOCH or state["system_key"] != key:
        parts: list[str] = []
        for path in (SYSTEM_PATH, WORLD_PATH):
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                logger.warning("提示词文件缺失: {}", path.name)
                continue
            if text:
                parts.append(text)
        if key[0]:
            parts.append(TOOL_DIRECTIVE)
        if human:
            parts.append(human)
        state["system_cache"] = "\n\n".join(parts)
        state["system_key"] = key
        state["epoch"] = _MODULE_EPOCH
    return str(state["system_cache"])


def _conv_id(event: Event) -> str:
    if isinstance(event, GroupAtMessage):
        return f"group:{event.group_openid}"
    if isinstance(event, C2CMessage):
        return f"user:{event.author.id}"
    return f"event:{event.id}"


def _history_path(bot: Bot) -> Path:
    raw = str(bot.config.get("llm_history_path") or DEFAULT_HISTORY_PATH)
    path = Path(raw)
    return path if path.is_absolute() else PLUGIN_DIR.parent / path


def _messages_of(entry: Any) -> list[dict[str, Any]]:
    if isinstance(entry, dict):
        messages = entry.get("messages")
        return messages if isinstance(messages, list) else []
    if isinstance(entry, list):
        return entry
    return []


async def _load(bot: Bot) -> dict[str, dict[str, Any]]:
    path = _history_path(bot)
    if not path.exists():
        return {}
    try:
        raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        logger.opt(exception=True).error("历史文件损坏, 本次按空历史处理")
        return {}
    if not isinstance(data, dict):
        return {}
    history: dict[str, dict[str, Any]] = {}
    legacy = False
    for key, value in data.items():
        if isinstance(value, list):
            history[key] = {"last_active": time.time(), "messages": value}
            legacy = True
        elif isinstance(value, dict):
            history[key] = value
    if legacy:
        logger.info("检测到旧格式历史, 已迁移为带时间戳结构")
    return history


async def _save(bot: Bot, history: dict[str, dict[str, Any]]) -> None:
    path = _history_path(bot)
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await _backup_legacy(path)
    text = json.dumps(history, ensure_ascii=False, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    await asyncio.to_thread(tmp.write_text, text, encoding="utf-8")
    await asyncio.to_thread(tmp.replace, path)


async def _backup_legacy(path: Path) -> None:
    bak = path.with_name(path.name + ".bak")
    if not path.exists() or bak.exists():
        return
    try:
        await asyncio.to_thread(shutil.copy2, path, bak)
        logger.info("已备份历史文件到 {}", bak.name)
    except OSError:
        logger.opt(exception=True).error("历史文件备份失败")


async def _allow(bot: Bot, conv_id: str) -> bool:
    limit = _int_config(bot, "llm_max_concurrency", DEFAULT_MAX_CONCURRENCY)
    window = _float_config(bot, "llm_concurrency_window", DEFAULT_CONCURRENCY_WINDOW)
    if limit <= 0 or window <= 0:
        return True
    now = time.monotonic()
    state = _llm_state(bot)
    rate_lock = _bot_lock(state, "rate_lock")
    rate = state["rate"]
    async with rate_lock:
        stamps = [stamp for stamp in rate.get(conv_id, []) if now - stamp < window]
        if len(stamps) >= limit:
            rate[conv_id] = stamps
            return False
        stamps.append(now)
        rate[conv_id] = stamps
        return True


def _ensure_gc(bot: Bot) -> None:
    task = getattr(bot, "_gc_task", None)
    if task is None or task.done():
        setattr(bot, "_gc_task", asyncio.create_task(_gc_loop(bot)))


async def _gc_loop(bot: Bot) -> None:
    while True:
        interval = _float_config(
            bot, "llm_memory_gc_interval", DEFAULT_MEMORY_GC_INTERVAL
        )
        if interval <= 0:
            return
        await asyncio.sleep(interval)
        await _gc_once(bot)


async def _gc_once(bot: Bot) -> None:
    days = _int_config(bot, "llm_memory_gc_days", DEFAULT_MEMORY_GC_DAYS)
    if days <= 0:
        return
    cutoff = time.time() - days * 86400
    history_lock = _bot_lock(_llm_state(bot), "history_lock")
    async with history_lock:
        history = await _load(bot)
        keep = {
            key: value for key, value in history.items() if _active_at(value) >= cutoff
        }
        removed = len(history) - len(keep)
        if removed <= 0:
            return
        await _save(bot, keep)
    logger.info("记忆 GC 清理 {} 个会话", removed)


def _active_at(value: dict[str, Any]) -> float:
    try:
        return float(value.get("last_active", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _float_config(bot: Bot, key: str, default: float) -> float:
    try:
        return float(bot.config.get(key, default))
    except (TypeError, ValueError):
        return default


def _int_config(bot: Bot, key: str, default: int) -> int:
    try:
        return int(bot.config.get(key, default))
    except (TypeError, ValueError):
        return default
