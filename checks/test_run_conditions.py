"""The translation runner (eval/translation_study/run_conditions.py) without any API call."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_never_marks_calls_as_done(tmp_path):
    run = tmp_path / "runX"
    run.mkdir()
    (run / "chapter_full_results.json").write_text(json.dumps({"frames": [{"id": 0, "blocks": [
        {"id": "0", "frame_id": "0", "class_name": "think_bubble", "text": "뭐지?", "corrected_text": "뭐지?"}]}]}), encoding="utf-8")
    sample = tmp_path / "s.csv"
    sample.write_text("block_uid\nrunX/0/0\n", encoding="utf-8")
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "eval.translation_study.run_conditions", "--runs", str(run / "chapter_full_results.json"),
           "--sample", str(sample), "--source", "corrected", "--conditions", "true_class", "no_class",
           "--model", "m/x", "--provider", "openrouter", "--dry-run", "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    lines = [json.loads(x) for x in (out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["condition"] for x in lines] == ["true_class", "no_class"]
    assert "think_bubble" in lines[0]["prompt"] and "think_bubble" not in lines[1]["prompt"]
    assert not (out / "calls.jsonl").exists()   # a later real run starts from scratch
