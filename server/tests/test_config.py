from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_config_sections_are_well_formed():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    for section in ("server", "asr", "mt", "caption", "speakers", "segmenter"):
        assert isinstance(config.get(section), dict), section
    assert (ROOT / config["speakers"]["model"]).exists() or not config["speakers"]["enabled"]


def test_glossary_is_well_formed():
    glossary = yaml.safe_load((ROOT / "glossary.yaml").read_text())
    assert all(isinstance(h, str) for h in glossary["asr_hints"])
    assert all({"zh", "en"} <= set(t) for t in glossary["terms"])
