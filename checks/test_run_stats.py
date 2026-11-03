"""Arbiter outcome and marker counting in research/run_stats.py."""
from research.run_stats import arbiter_outcome, kind_of, marker_table


def block(comic, general, corrected, cls="speech_bubble", en=""):
    return {"text": comic, "secondary_text": general, "corrected_text": corrected,
            "class_name": cls, "translated_text": en}


def test_agreement_ignores_spacing_case_and_tone_marks():
    assert arbiter_outcome(block("불꺼, 인마.", "불 꺼 인마!", "불꺼, 인마."), "pipeline") == "agree"
    assert arbiter_outcome(block("불꺼, 인마.", "불 꺼 인마!", "불꺼, 인마."), "space") != "agree"


def test_choice_uses_the_same_rule():
    b = block("잌지?", "있지", "있지?")
    assert arbiter_outcome(b, "pipeline") == "general"
    assert arbiter_outcome(block("ㅡㅡ", "카마", "카마코마"), "pipeline") == "neither"
    assert arbiter_outcome(block("", "카마", "카마"), "pipeline") == "one_empty"


def test_kinds_and_markers():
    assert kind_of("sqare_bubble") == "Boxes and system messages"
    assert kind_of("wiggle_bubble") == "Balloons"
    rows = [block("", "", "대체 누구지?", "think_bubble", "Who could it be...?"),
            block("", "", "크아아앙!!", "think_bubble", "GRAAAH!!!")]
    m = marker_table(rows, min_blocks=1)["think_bubble"]
    assert m["en_ellipsis"] == 0.5 and m["en_caps"] == 0.5 and m["ko_exclaim"] == 0.5
