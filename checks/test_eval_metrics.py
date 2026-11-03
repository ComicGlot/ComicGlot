"""Scoring code used for the paper's tables."""
import numpy as np

from eval import detect_eval as de
from eval.ocr_eval import levenshtein, norm_chars
from eval.translation_study.analyse_ratings import holm, krippendorff_alpha_ordinal


def test_perfect_detection_scores_full_ap():
    box = np.array([[0.1, 0.1, 0.4, 0.4]])
    tp = de.match(box, np.array([3]), box, np.array([3]))
    res = de.per_class_ap(tp, np.array([0.9]), np.array([3]), np.array([3]), 13)
    assert res[3]["ap50"] > 0.99 and res[3]["ap"] > 0.99


def test_wrong_class_is_a_miss_but_counts_class_agnostic():
    box = np.array([[0.1, 0.1, 0.4, 0.4]])
    assert not de.match(box, np.array([5]), box, np.array([10])).any()
    assert de.match(box, np.array([5]), box, np.array([10]), agnostic=True).all()


def test_levenshtein_and_normalisation():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("kitten", "sitting") == 3
    assert norm_chars(" 안녕 하세요! ", strip_punct=True) == "안녕하세요"


def test_alpha_is_one_for_perfect_agreement_and_low_for_noise():
    assert abs(krippendorff_alpha_ordinal([[1, 1, 1], [3, 3, 3], [5, 5, 5], [2, 2, 2]]) - 1.0) < 1e-9
    rng = np.random.default_rng(0)
    noise = [list(rng.integers(1, 6, 3).astype(float)) for _ in range(200)]
    assert krippendorff_alpha_ordinal(noise) < 0.2


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.2])
    assert all(0 <= a <= 1 for a in adj)
    assert adj[0] <= adj[2] <= adj[1] <= adj[3]
