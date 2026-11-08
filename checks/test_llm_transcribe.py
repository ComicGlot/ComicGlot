"""Reference transcriptions from several models (eval/llm_transcribe.py): consensus only, no API calls."""
import csv
from types import SimpleNamespace

from eval.llm_transcribe import clean, consensus


def test_consensus_needs_agreement_and_lists_the_rest(tmp_path):
    with open(tmp_path / "readings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_uid", "m1", "m2", "m3"])
        w.writerow(["a", "안녕 하세요!", "안녕하세요!", "안녕하세요"])      # two agree once spaces go
        w.writerow(["b", "쾅", "꽝", "콰앙"])                              # nobody agrees
        w.writerow(["c", "#ILLEGIBLE", "#ILLEGIBLE", "아"])
    consensus(SimpleNamespace(out=tmp_path, min_agree=2))
    ref = {r["block_uid"]: r for r in csv.DictReader(open(tmp_path / "reference.csv", encoding="utf-8"))}
    assert ref["a"]["gt_text"] in ("안녕 하세요!", "안녕하세요!") and ref["a"]["agree"] == "2"
    assert ref["c"]["gt_text"] == "#ILLEGIBLE"
    assert "b" not in ref
    assert [r["block_uid"] for r in csv.DictReader(open(tmp_path / "disagreements.csv", encoding="utf-8"))] == ["b"]


def test_clean_strips_wrapping():
    assert clean('"안녕"') == "안녕" and clean("```\n안녕\n```") == "안녕"
