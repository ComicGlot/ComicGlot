"""Label parsing, deduplication and the class-to-guide lookup."""
import json
from pathlib import Path

from comicglot.translation.style_processor import StyleTranslator
from research.common import parse_label_line
from research.text_vs_image import evaluate, surface_features

ROOT = Path(__file__).resolve().parents[1]


def test_polygon_label_becomes_its_bounding_box():
    b = parse_label_line("5 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6")
    assert b.cls == 5 and abs(b.xc - 0.2) < 1e-9 and abs(b.yc - 0.4) < 1e-9
    assert abs(b.w - 0.2) < 1e-9 and abs(b.h - 0.4) < 1e-9


def test_every_detector_class_has_its_own_style_guide():
    guides = json.load(open(ROOT / "data" / "prompts" / "style_guides.json", encoding="utf-8"))
    t = StyleTranslator.__new__(StyleTranslator)
    t.style_guides = guides
    detector_classes = ["background_text", "cloud_bubble", "hexagon_bubble", "scream_bubble", "sfx_sound",
                        "speech_bubble", "spiked_bubble", "sqare_bubble", "standalone_text", "system_bubble",
                        "think_bubble", "title_panel", "wiggle_bubble"]
    for c in detector_classes:
        assert t._guide_for(c) != guides["default"], c


def test_ablation_conditions():
    assert StyleTranslator.guide_class("think_bubble", "true") == "think_bubble"
    assert StyleTranslator.guide_class("think_bubble", "none") == "default"
    assert StyleTranslator.guide_class("think_bubble", "shuffled") not in ("think_bubble", "default")


def test_text_classifier_runs_on_toy_rows():
    rows = [{"group": f"g{i % 6}", "class": "sfx_sound" if i % 2 else "think_bubble",
             "text": "쾅!!" if i % 2 else "그런가...", "emb_row": None} for i in range(40)]
    rep = evaluate(rows, None, folds=3, min_count=5, C=1.0)
    assert rep["models"]["text_char_ngrams"]["accuracy"] > 0.9
    assert surface_features(["a!!", "..."]).shape == (2, 10)
