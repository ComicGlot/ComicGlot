"""Ranking round of the translator study (eval/translation_study/rank_analysis.py)."""
from eval.translation_study.rank_analysis import attention, bradley_terry, judgements, pairs


def row(item, model, cond, rank, decoy=False, last=""):
    rater, uid = item.split("|")
    return {"item": item, "rater": rater, "rater_kind": "person", "block_uid": uid, "detector_type": "speech_bubble",
            "model": model, "condition": f"{cond}@{model}" if model != "attention" else "decoy", "rank": str(rank),
            "is_decoy": str(decoy), "decoy_last": last, "source_wrong": "false"}


ROWS = [
    # r1 on t1: X guide 1st, X plain 2nd, Y guide and Y plain the same card (tied 3rd), decoy last
    row("r1|t1", "x", "true_class", 1), row("r1|t1", "x", "no_class", 2),
    row("r1|t1", "y", "true_class", 3), row("r1|t1", "y", "no_class", 3),
    row("r1|t1", "attention", "decoy", 5, True, "True"),
    # r2 on t1: Y plain first, decoy not last
    row("r2|t1", "y", "no_class", 1), row("r2|t1", "x", "no_class", 2), row("r2|t1", "x", "true_class", 3),
    row("r2|t1", "attention", "decoy", 4, True, "False"), row("r2|t1", "y", "true_class", 5),
]


def test_judgements_within_model():
    js = {(j["rater"], j["model"]): j["choice"] for j in judgements(ROWS)}
    assert js == {("r1", "x"): "guide", ("r1", "y"): "tie", ("r2", "x"): "plain", ("r2", "y"): "plain"}


def test_attention_counts_decoy_last():
    assert attention(ROWS) == {"r1": {"checks": 1, "passed": 1}, "r2": {"checks": 1, "passed": 0}}


def test_pairs_skip_same_model_and_decoy():
    ps = pairs(ROWS[:5], lambda r: r["model"])
    assert len(ps) == 4 and all({a, b} == {"x", "y"} for _, a, b, _ in ps)
    assert all(s == 1.0 for _, a, _, s in ps if a == "x")


def test_bradley_terry_orders_by_wins():
    ps = [("t", "a", "b", 1.0)] * 8 + [("t", "a", "b", 0.0)] * 2 + [("t", "b", "c", 1.0)] * 7 + [("t", "b", "c", 0.0)] * 3
    s = bradley_terry(ps, ["a", "b", "c"])
    assert s[0] > s[1] > s[2] and abs(s.sum()) < 1e-6
