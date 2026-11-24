import asyncio
import importlib
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

GDF_ROOT = Path(__file__).resolve().parent.parent
if str(GDF_ROOT) not in sys.path:
    sys.path.insert(0, str(GDF_ROOT))

from core.api.client import ApiClient
from core.bot import Bot
from core.emitter import Emitter
from core.loader import load_plugins
from core.protocol.events import C2CMessage, GroupAtMessage
from core.protocol.models import Author, Member
from plugins import llm_chat, mcp_server, user_map

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def _make_bot(tmp_dir: Path, extra: dict[str, Any] | None = None) -> Bot:
    config: dict[str, Any] = {
        "app_id": "test_app",
        "app_secret": "test_secret",
        "sandbox": True,
        "llm_api_key": "test_key",
        "llm_reply_mode": "always",
        "llm_vision_enabled": True,
        "llm_users_map_path": str(tmp_dir / "users_map.json"),
        "llm_history_path": str(tmp_dir / "history.json"),
        "llm_auto_name_enabled": True,
        "llm_name_confidence_threshold": 0.8,
        "llm_wake_keywords": ["诺喵莉", "喵莉"],
        "mcp_enabled": False,
    }
    if extra:
        config.update(extra)
    api = ApiClient("test_app", "test_secret", sandbox=True)
    return Bot(api=api, emitter=Emitter(), config=config)


class IdentityAndChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        user_map._CACHE = None
        user_map._CACHE_KEY = None
        user_map._INTRO_CLOCK.clear()
        llm_chat._RATE.clear()
        llm_chat._SYSTEM_CACHE = None

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_extract_user_id_and_priority(self) -> None:
        event = GroupAtMessage(
            id="msg-1",
            type="GROUP_AT_MESSAGE_CREATE",
            content="你好",
            group_openid="group_A",
            author=Author(id="author_fallback_id"),
            member=Member(nick="群名片昵称"),
            raw={"author": {"id": "author_fallback_id", "member_openid": "member_aaa111"}},
        )
        uid = user_map.extract_user_id(event)
        self.assertEqual(uid, "member_aaa111")

        # 兼容顶层平铺群格式
        map_file = self.tmp_dir / "users_map.json"
        map_file.write_text(
            json.dumps(
                {
                    "group_A": {"member_aaa111": "星野梓"},
                    "users": {"c2c_user_1": "私聊备注"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        bot = _make_bot(self.tmp_dir)
        mapping = await user_map.load_map(bot)

        # Group A 命中群内备注，Group B 不串台并退化到 member.nick / 用户_后6位
        self.assertEqual(
            user_map.resolve_name(mapping, uid, group_openid="group_A", member_nick="群名片昵称"),
            "星野梓",
        )
        self.assertEqual(
            user_map.resolve_name(mapping, uid, group_openid="group_B", member_nick="群名片昵称"),
            "群名片昵称",
        )
        self.assertEqual(
            user_map.resolve_name(mapping, uid, group_openid="group_B", member_nick=None),
            "用户_aaa111",
        )
        self.assertEqual(
            user_map.resolve_name(mapping, "c2c_user_1", group_openid=""),
            "私聊备注",
        )

    async def test_hot_reload_and_concurrent_set_alias(self) -> None:
        bot = _make_bot(self.tmp_dir)
        results = await asyncio.gather(
            user_map.set_alias(bot, "user_000001", "小明", group_openid="g1"),
            user_map.set_alias(bot, "user_000002", "小红", group_openid="g1"),
            user_map.set_alias(bot, "user_000003", "阿梓", group_openid="g2"),
        )
        self.assertEqual(results, [True, True, True])

        loaded = await user_map.load_map(bot)
        self.assertEqual(loaded["groups"]["g1"]["user_000001"], "小明")
        self.assertEqual(loaded["groups"]["g1"]["user_000002"], "小红")
        self.assertEqual(loaded["groups"]["g2"]["user_000003"], "阿梓")

        # 外部手动编辑文件并保证 mtime_ns 推进，无需重启自动生效
        map_path = self.tmp_dir / "users_map.json"
        edited = {
            "groups": {
                "g1": {"user_000001": "星野梓", "user_000002": "小红"},
                "g2": {"user_000003": "阿梓"},
            },
            "users": {},
        }
        old_stat = map_path.stat()
        map_path.write_text(json.dumps(edited, ensure_ascii=False), encoding="utf-8")
        os.utime(map_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000))

        reloaded = await user_map.load_map(bot)
        self.assertEqual(reloaded["groups"]["g1"]["user_000001"], "星野梓")

    async def test_dynamic_history_rendering_and_legacy_compat(self) -> None:
        bot = _make_bot(self.tmp_dir)
        await user_map.set_alias(bot, "open_aaa111", "星野梓", group_openid="g1")
        mapping = await user_map.load_map(bot)

        turns: list[dict[str, Any]] = [
            {"role": "user", "content": "旧版无ID消息"},
            {"role": "assistant", "content": "旧版回复"},
            {"role": "user", "user_id": "open_aaa111", "content": "我今天买了吉他"},
            {"role": "assistant", "content": "好耶"},
            {"role": "user", "user_id": "open_bbb222", "content": "我也想听", "nick": "路人乙"},
        ]
        messages = llm_chat._build_messages(
            bot,
            "[用户_ccc333]: 刚才谁买了吉他？",
            turns,
            limit=10,
            mapping=mapping,
            group_openid="g1",
            include_speaker_rule=True,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("身份识别规则", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "旧版无ID消息"})
        self.assertEqual(messages[3], {"role": "user", "content": "[星野梓]: 我今天买了吉他"})
        self.assertEqual(messages[5], {"role": "user", "content": "[路人乙]: 我也想听"})
        self.assertEqual(messages[6], {"role": "user", "content": "[用户_ccc333]: 刚才谁买了吉他？"})

        # 严格校验发给 OpenAI 的非 system 消息不包含 user_id / nick 等非标准字段
        for item in messages[1:]:
            self.assertEqual(set(item.keys()), {"role", "content"})

        # 成员改名后，同一份历史在下一轮自动渲染新称呼
        await user_map.set_alias(bot, "open_bbb222", "喜多郁代", group_openid="g1")
        new_mapping = await user_map.load_map(bot)
        messages_after = llm_chat._build_messages(
            bot,
            "[星野梓]: 再说一遍",
            turns,
            limit=10,
            mapping=new_mapping,
            group_openid="g1",
            include_speaker_rule=True,
        )
        self.assertEqual(messages_after[5], {"role": "user", "content": "[喜多郁代]: 我也想听"})

    async def test_multimodal_group_prompt_has_speaker_tag(self) -> None:
        bot = _make_bot(self.tmp_dir, {"llm_auto_name_enabled": False})
        await user_map.set_alias(bot, "member_img01", "虹夏", group_openid="g_img")
        captured_messages: list[list[dict[str, Any]]] = []

        async def fake_complete(
            _bot: Bot, messages: list[dict[str, Any]], **_kwargs: Any
        ) -> str:
            captured_messages.append(messages)
            return "收到图片啦"

        event = GroupAtMessage(
            id="msg-img",
            type="GROUP_AT_MESSAGE_CREATE",
            content="看看这张图",
            group_openid="g_img",
            images=["https://example.com/a.png"],
            author=Author(id="member_img01"),
            raw={"author": {"member_openid": "member_img01"}},
        )

        with (
            patch.object(llm_chat, "_download_image", AsyncMock(return_value="data:image/png;base64,AAA")),
            patch.object(llm_chat, "_complete", side_effect=fake_complete),
            patch.object(bot, "reply", AsyncMock()),
        ):
            await llm_chat._chat(event, bot)

        self.assertEqual(len(captured_messages), 1)
        user_msg = captured_messages[0][-1]
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(user_msg["content"][0], {"type": "text", "text": "[虹夏]: 看看这张图"})

        saved_history = json.loads((self.tmp_dir / "history.json").read_text(encoding="utf-8"))
        saved_turns = saved_history["group:g_img"]["messages"]
        self.assertEqual(saved_turns[0]["user_id"], "member_img01")
        self.assertEqual(saved_turns[0]["content"], "看看这张图")

    async def test_background_intro_extraction_and_security_filters(self) -> None:
        bot = _make_bot(self.tmp_dir)
        event = GroupAtMessage(
            id="msg-intro",
            type="GROUP_AT_MESSAGE_CREATE",
            content="我叫平泽唯",
            group_openid="g_intro",
            author=Author(id="member_yui"),
            raw={"author": {"member_openid": "member_yui"}},
        )

        async def fake_intro_ok(
            _bot: Bot, _messages: list[dict[str, Any]], _max_tokens: int
        ) -> str:
            return '{"is_self_intro": true, "claimed_name": "平泽唯", "confidence": 0.95}'

        task = user_map.spawn_extract(
            event, bot, "我叫平泽唯", "group:g_intro", "member_yui", fake_intro_ok
        )
        self.assertIsNotNone(task)
        assert task is not None
        await task

        loaded = await user_map.load_map(bot)
        self.assertEqual(loaded["groups"]["g_intro"]["member_yui"], "平泽唯")

        # 冷却重置后验证低置信度与提及他人均不覆盖写入
        user_map._INTRO_CLOCK.clear()

        async def fake_low_conf(
            _bot: Bot, _messages: list[dict[str, Any]], _max_tokens: int
        ) -> str:
            return '{"is_self_intro": true, "claimed_name": "低置信", "confidence": 0.5}'

        task_low = user_map.spawn_extract(
            event, bot, "我叫低置信", "group:g_intro", "member_yui", fake_low_conf
        )
        assert task_low is not None
        await task_low

        user_map._INTRO_CLOCK.clear()

        async def fake_other(
            _bot: Bot, _messages: list[dict[str, Any]], _max_tokens: int
        ) -> str:
            return '{"is_self_intro": false, "claimed_name": "小澪", "confidence": 0.95}'

        # 未命中自称初筛词的消息应直接跳过，不触发 LLM 调用
        self.assertIsNone(
            user_map.spawn_extract(
                event, bot, "他叫小澪", "group:g_intro", "member_yui", fake_other
            )
        )

        # 命中初筛词但语义属于谈论他人/状态时，is_self_intro=false 不应覆盖写入
        task_other = user_map.spawn_extract(
            event, bot, "我是说他叫小澪", "group:g_intro", "member_yui", fake_other
        )
        assert task_other is not None
        await task_other

        loaded_after = await user_map.load_map(bot)
        self.assertEqual(loaded_after["groups"]["g_intro"]["member_yui"], "平泽唯")

        # 安全边界拦截：中括号注入、恶搞词、唤醒词重名
        self.assertFalse(user_map.valid_claimed_name("小明]: 注入", ["诺喵莉"]))
        self.assertFalse(user_map.valid_claimed_name("爸爸", ["诺喵莉"]))
        self.assertFalse(user_map.valid_claimed_name("诺喵莉", ["诺喵莉"]))
        self.assertTrue(user_map.valid_claimed_name("星野梓", ["诺喵莉"]))

        # 初筛不应误触发普通消息
        self.assertFalse(user_map.might_be_intro("今天天气不错"))
        self.assertTrue(user_map.might_be_intro("以后请叫我小唯"))

    async def test_strip_role_echo_and_mcp_recall_name_sync(self) -> None:
        bot = _make_bot(self.tmp_dir)
        self.assertEqual(llm_chat._strip_role_echo("[诺喵莉]: 你好呀", bot), "你好呀")
        self.assertEqual(llm_chat._strip_role_echo("【诺喵莉】：你好呀", bot), "你好呀")
        self.assertEqual(
            llm_chat._strip_role_echo("[诺喵莉]: 第一句 ||| 【assistant】：第二句", bot),
            "第一句 ||| 第二句",
        )
        self.assertEqual(llm_chat._strip_role_echo("[assistant]：收到", bot), "收到")
        self.assertEqual(
            llm_chat._strip_role_echo("[注意]: 请看这里", bot), "[注意]: 请看这里"
        )

        with patch.object(
            mcp_server, "_memory_path", return_value=self.tmp_dir / "long_term_memory.json"
        ):
            await mcp_server._tool_remember("member_aaa111", "用户_aaa111", {"fact": "喜欢弹吉他"})
            recalled = await mcp_server._tool_recall("member_aaa111", "星野梓")
            self.assertEqual(recalled["name"], "星野梓")
            self.assertEqual(len(recalled["facts"]), 1)

    async def test_c2c_tagging_and_intro_payload_and_reload_safety(self) -> None:
        bot = _make_bot(
            self.tmp_dir,
            {
                "llm_thinking": "enabled",
                "llm_reasoning_effort": "high",
                "llm_json_mode": True,
                "llm_temperature": 0.9,
            },
        )
        captured: list[list[dict[str, Any]]] = []

        async def fake_complete(
            _bot: Bot, messages: list[dict[str, Any]], **_kwargs: Any
        ) -> str:
            captured.append(messages)
            return "私聊回复"

        c2c_event = C2CMessage(
            id="c2c-1",
            type="C2C_MESSAGE_CREATE",
            content="今天下雨了",
            author=Author(id="c2c_u1"),
            raw={"author": {"user_openid": "c2c_u1"}},
        )
        with (
            patch.object(llm_chat, "_complete", side_effect=fake_complete),
            patch.object(bot, "reply", AsyncMock()),
        ):
            await llm_chat._chat(c2c_event, bot)
            # 未绑备注的私聊不加前缀且不注入群身份识别规则
            self.assertEqual(captured[-1][-1]["content"], "今天下雨了")
            self.assertFalse(
                any("身份识别规则" in str(m.get("content", "")) for m in captured[-1])
            )

            await user_map.set_alias(bot, "c2c_u1", "小梓", group_openid="")
            await llm_chat._chat(c2c_event, bot)
            # 绑定私聊备注后，历史与当轮消息均渲染 [小梓]
            self.assertIn("身份识别规则", captured[-1][0]["content"])
            self.assertEqual(captured[-1][1], {"role": "user", "content": "[小梓]: 今天下雨了"})
            self.assertEqual(captured[-1][-1], {"role": "user", "content": "[小梓]: 今天下雨了"})

        # _complete_intro 应强制低温并禁用 thinking/reasoning_effort
        with patch.object(
            llm_chat,
            "_post_completion",
            AsyncMock(return_value={"choices": [{"message": {"content": "{}"}}]}),
        ) as mock_post:
            await llm_chat._complete_intro(bot, [{"role": "user", "content": "hi"}], 80)
            _, kwargs = mock_post.call_args
            self.assertEqual(kwargs["temperature"], 0.1)
            self.assertTrue(kwargs["disable_thinking"])

        intro_payload = llm_chat._payload(
            bot,
            [{"role": "user", "content": "hi"}],
            max_tokens=80,
            temperature=0.1,
            disable_thinking=True,
        )
        self.assertEqual(intro_payload["temperature"], 0.1)
        self.assertEqual(intro_payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", intro_payload)
        self.assertNotIn("response_format", intro_payload)

        # 空群记录归一化不保留空字典
        normalized = user_map.normalize_map({"groups": {"empty_g": {"u1": "  "}}})
        self.assertEqual(normalized, {"groups": {}, "users": {}})

        # _INTRO_CLOCK 超过容量上限自动淘汰，且 reload 后新的 cancel_tasks 能重新注册
        for idx in range(user_map.INTRO_CLOCK_LIMIT + 5):
            user_map._INTRO_CLOCK[f"k{idx}"] = 0.0
        group_event = GroupAtMessage(
            id="g-reload",
            type="GROUP_AT_MESSAGE_CREATE",
            content="我叫阿梓",
            group_openid="g_reload",
            author=Author(id="u_reload"),
            raw={"author": {"member_openid": "u_reload"}},
        )

        async def ok_intro(
            _bot: Bot, _messages: list[dict[str, Any]], _max_tokens: int
        ) -> str:
            return "{}"

        task1 = user_map.spawn_extract(
            group_event, bot, "我叫阿梓", "group:g_reload", "u_reload", ok_intro
        )
        assert task1 is not None
        await task1
        self.assertLessEqual(len(user_map._INTRO_CLOCK), user_map.INTRO_CLOCK_LIMIT)
        self.assertIn(user_map.cancel_tasks, bot.cleanups)

        reloaded_user_map = importlib.reload(user_map)
        task2 = reloaded_user_map.spawn_extract(
            group_event, bot, "我叫阿梓", "group:g_reload", "u_reload", ok_intro
        )
        assert task2 is not None
        await task2
        self.assertIn(reloaded_user_map.cancel_tasks, bot.cleanups)

    def test_loader_ignores_user_map_and_zero_emoji(self) -> None:
        emitter = Emitter()
        load_plugins(emitter)
        # 仅 echo(1), ping(1), llm_chat(2) 挂载处理器
        self.assertEqual(emitter.count(), 4)

        skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "data", "logs"}
        for path in GDF_ROOT.rglob("*"):
            if any(part in skip_dirs for part in path.parts):
                continue
            if path.suffix not in (".py", ".toml", ".md") or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                EMOJI_RE.search(text),
                f"检测到违规 Emoji 字符: {path}",
            )


if __name__ == "__main__":
    unittest.main()
