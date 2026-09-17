import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mt import EN_ZH, HUNYUAN, HY_DENSE, ZH_EN, GlossaryTerm, Turn, build_prompt, format_chat  # noqa: E402

ZH_INSTRUCTION = "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释："


def test_plain_prompt_is_official_template():
    assert build_prompt("你好", []) == f"{ZH_INSTRUCTION}\n\n你好"


def test_only_relevant_terms_included():
    terms = [GlossaryTerm("回滚", "rollback"), GlossaryTerm("上线", "release")]
    prompt = build_prompt("准备回滚方案", terms)
    assert prompt.startswith("参考下面的翻译：\n回滚 翻译成 rollback\n\n")
    assert "上线" not in prompt


def test_english_to_chinese_prompt_and_terms():
    terms = [GlossaryTerm("回滚", "rollback"), GlossaryTerm("上线", "release")]
    prompt = build_prompt("Prepare a rollback plan", terms, EN_ZH)
    assert prompt.startswith("Reference the following translations:\nrollback translates to 回滚\n\n")
    assert prompt.endswith("Translate the following text into Chinese. Note that you must ONLY output the "
                           "translated result without any additional explanation:\n\nPrepare a rollback plan")


def test_hy_dense_history_rendered_as_previous_turns():
    history = [Turn("Redis 延迟很高。", "Redis latency is high.")]
    assert format_chat(HY_DENSE, "这个修好了吗？", history, []) == (
        "<｜hy_begin▁of▁sentence｜>"
        f"<｜hy_User｜>{ZH_INSTRUCTION}\n\nRedis 延迟很高。<｜hy_Assistant｜>Redis latency is high.<｜hy_place▁holder▁no▁2｜>"
        f"<｜hy_User｜>{ZH_INSTRUCTION}\n\n这个修好了吗？<｜hy_Assistant｜>"
    )


def test_mixed_direction_history_keeps_each_turns_wording():
    history = [Turn("Is it fixed?", "修好了吗？", EN_ZH)]
    chat = format_chat(HY_DENSE, "已经修好了。", history, [], ZH_EN)
    assert "Translate the following text into Chinese" in chat and chat.endswith(f"{ZH_INSTRUCTION}\n\n已经修好了。<｜hy_Assistant｜>")


def test_hunyuan_format_single_and_multi_turn():
    assert format_chat(HUNYUAN, "你好", [], []) == f"<|startoftext|>{ZH_INSTRUCTION}\n\n你好<|extra_0|>"
    history = [Turn("你好", "Hello")]
    assert format_chat(HUNYUAN, "再见", history, []) == (
        f"<|startoftext|>{ZH_INSTRUCTION}\n\n你好<|extra_0|>Hello<|eos|>"
        f"<|startoftext|>{ZH_INSTRUCTION}\n\n再见<|extra_0|>"
    )
