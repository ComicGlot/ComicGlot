"""Offline smoke test of the patched pipeline with the LLM calls mocked.

Runs process_chapter on data/frames with dual OCR, arbitration and translation
switched on, but with LLMProcessor replaced by a stub that records prompts and
returns canned answers. Checks that:
  - the arbiter receives the block class and previous lines,
  - each class_condition sends the expected style guide,
  - provenance is written into the results.

Needs the v7 detector weights: set YOLO_MODEL_PATH. No API key needed.
    YOLO_MODEL_PATH=/path/cg-manhua_bubbles-3k-v7.pt python -m research.smoke_pipeline
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

PROMPTS = []


class StubLLM:
    def __init__(self, model_name="stub", temperature=0.0, **_):
        self.model_name = f"stub:{model_name}"
        self.temperature = temperature

    def process_with_claude_sync(self, prompt, **_):
        PROMPTS.append(prompt)
        return self._answer(prompt)

    async def batch_process_with_claude(self, prompts, **_):
        PROMPTS.extend(prompts)
        return [self._answer(p) for p in prompts]

    @staticmethod
    def _answer(prompt):
        if "CORRECTED_TEXT" in prompt:
            return "CORRECTED_TEXT: stub-corrected"
        guide = prompt.split("[STYLE GUIDE]:", 1)[1].split("\n", 1)[0] if "[STYLE GUIDE]:" in prompt else ""
        return f"TRANSLATED_TEXT: <{guide[:40]}>"


def run(condition: str):
    import comicglot.translation.ocr_correction as oc
    import comicglot.translation.style_processor as sp
    oc.LLMProcessor = StubLLM
    sp.LLMProcessor = StubLLM
    from comicglot.comic_processor.comic_processor import ComicProcessor
    tmp = Path(tempfile.mkdtemp())
    for f in Path("data/frames").glob("*.jpg"):
        shutil.copy(f, tmp / f.name)
    proc = ComicProcessor(detection_model="yolo", recognition_model="trocr-base", translator="claude",
                          dual_ocr=True, use_correction=True, use_segmentation=False, class_condition=condition)
    # make the two readers disagree so every block goes through the arbiter
    real = proc.recognizer.process_image_with_blocks

    def altered(*a, **k):
        out = real(*a, **k)
        for r in out:
            r["text"] = "다른 읽기 " + (r.get("text") or "")[::-1]
        return out
    proc.secondary_recognizer.process_image_with_blocks = altered
    PROMPTS.clear()
    res = asyncio.run(proc.process_chapter(str(tmp)))
    return res, list(PROMPTS)


def main():
    assert os.getenv("YOLO_MODEL_PATH"), "set YOLO_MODEL_PATH"
    res, prompts = run("true")
    blocks = [b for f in res["frames"] for b in f["blocks"]]
    corr = [p for p in prompts if "CORRECTED_TEXT" in p]
    trans = [p for p in prompts if "TRANSLATED_TEXT" in p]
    print(f"blocks {len(blocks)}, arbiter calls {len(corr)}, translation calls {len(trans)}")
    assert corr, "no arbiter calls were made"
    for p in corr:
        assert "Text block type:" in p, "arbiter prompt lacks the block class"
    for b in blocks:
        if b.get("translated_text"):
            assert b["translation_guide_class"] == b["class_name"]
    assert res["provenance"]["detector_sha256_16"], "provenance missing detector hash"
    print("true: ok; provenance", res["provenance"])
    for cond in ("none", "shuffled"):
        res, _ = run(cond)
        blocks = [b for f in res["frames"] for b in f["blocks"] if b.get("translated_text")]
        guides = {b["class_name"]: b["translation_guide_class"] for b in blocks}
        print(cond, guides)
        if cond == "none":
            assert set(guides.values()) == {"default"}
        else:
            assert all(g != c for c, g in guides.items())
    print("all checks passed")


if __name__ == "__main__":
    main()
