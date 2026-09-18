import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import direction_for, is_chinese, is_filler  # noqa: E402
from mt import EN_ZH, ZH_EN  # noqa: E402


def test_chinese_labels_and_dialects():
    assert is_chinese("Chinese", "我们今天开会")
    assert is_chinese("Cantonese", "我哋今日開會")
    assert is_chinese("chinese", "")


def test_mixed_sentence_counts_as_chinese_even_if_mislabeled():
    assert is_chinese("English", "我们把服务迁移到了 Kubernetes")


def test_other_languages_are_not_chinese():
    assert not is_chinese("English", "Let us start the meeting")
    assert not is_chinese("Japanese", "はじめましょう")
    assert not is_chinese("", "")


def test_direction_by_language():
    assert direction_for("Chinese", "我们开会") is ZH_EN
    assert direction_for("English", "Let us start") is EN_ZH
    assert direction_for("Japanese", "はじめましょう") is None


def test_fillers_are_dropped_but_real_speech_is_not():
    assert is_filler("嗯。")
    assert is_filler("呃嗯。")
    assert is_filler("Uh-huh.")
    assert is_filler("Mm-hmm")
    assert not is_filler("嗯，然后确认。")
    assert not is_filler("对。")
    assert not is_filler("Yes, understood.")
