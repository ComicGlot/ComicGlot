"""Pairwise analysis of the translator study (eval/translation_study/analyse_pairs.py)."""
import math

from eval.translation_study.analyse_pairs import alpha_nominal, base_condition, judgements, sign_test, summarise


def test_sign_test_matches_known_values():
    assert sign_test(0, 0) == 1.0
    assert abs(sign_test(9, 10) - 22 / 1024) < 1e-12          # 2 * (1 + 10) / 2^10
    assert abs(sign_test(5, 10) - 1.0) < 1e-12
    assert sign_test(1, 10) == sign_test(9, 10)


def test_alpha_nominal_perfect_and_chance():
    assert alpha_nominal([["a", "a"], ["b", "b"], ["a", "a"]]) == 1.0
    # Krippendorff (2011) nominal example reduced: total disagreement gives a negative alpha
    assert alpha_nominal([["a", "b"], ["b", "a"]]) < 0
    assert alpha_nominal([["a"], ["b"]]) is None


def test_base_condition():
    assert base_condition("true_class+page@openai/gpt-x") == "true_class"
    assert base_condition("no_class") == "no_class"


def row(rater, uid, choice, typ="speech_bubble", tone="true_class:fine;no_class:fine", att="False", wrong="False"):
    return {"rater": rater, "rater_kind": "person", "block_uid": uid, "model": "m", "detector_type": typ,
            "conditions": "no_class@m|true_class@m", "choice_condition": choice, "tone_by_condition": tone,
            "is_attention_check": att, "source_wrong": wrong}


def test_texts_count_once_by_majority():
    rows = [row("r1", "t1", "true_class@m"), row("r2", "t1", "true_class@m"),
            row("r1", "t2", "no_class@m"), row("r2", "t2", "true_class@m"),   # split: drops out
            row("r1", "t3", "tie"), row("r2", "t3", "no_class@m"),
            row("r1", "t4", "true_class@m", att="True")]                     # attention checks are left out
    js = judgements(rows)
    assert len(js) == 6
    s = summarise(js)
    assert s["texts"] == 3 and s["texts_decided"] == 2 and s["texts_prefer_guide"] == 1
    assert math.isclose(s["guide"], 3 / 6) and math.isclose(s["tie"], 1 / 6)
