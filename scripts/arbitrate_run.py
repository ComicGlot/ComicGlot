"""Apply the OCR arbiter (pipeline step 4) to a run that was made without it.

Detection and both readers are the slow part of a run and need the GPU; the arbiter needs only an
LLM. This script reads a saved chapter_full_results.json, runs exactly the pipeline's step 4 on its
blocks (ComicProcessor._arbitrate_blocks: agreeing readings are kept, an empty reading gives way to
the other, the rest go to the arbiter with the block type and previous lines), and writes the result
back. The file as it was is kept next to it as chapter_full_results.noarbiter.json, and the run's
config and provenance record the arbiter's provider, model and the date it was applied.

    python scripts/arbitrate_run.py data/output/<runs>/*/chapter_full_results.json --dry-run
    python scripts/arbitrate_run.py data/output/<runs>/*/chapter_full_results.json \
        --llm-provider openrouter --llm-model anthropic/claude-sonnet-4.5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comicglot.comic_processor.comic_processor import ComicProcessor  # noqa: E402


class _Counter:
    """Stands in for the arbiter in --dry-run: keeps the primary reading and counts the calls."""

    def __init__(self):
        self.calls = 0

    async def batch_correct_texts(self, inputs, contexts=None, language="ko"):
        self.calls += len(inputs)
        return [a for a, _ in inputs]


def arbiter_processor(corrector, correction_context: bool, arbitrate_punctuation: bool, source_lang: str) -> ComicProcessor:
    """A ComicProcessor with only what step 4 uses; no detection or recognition models are loaded."""
    p = ComicProcessor.__new__(ComicProcessor)
    p.logger = logging.getLogger("arbitrate_run")
    p.corrector = corrector
    p.correction_context = correction_context
    p.arbitrate_punctuation = arbitrate_punctuation
    p.source_lang = source_lang
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="chapter_full_results.json files")
    ap.add_argument("--llm-provider", choices=["anthropic", "openrouter"], default=None)
    ap.add_argument("--llm-model", default=None, help="OpenRouter slug or Claude name (default: the published arbiter)")
    ap.add_argument("--dry-run", action="store_true", help="count the blocks that would go to the arbiter; no LLM calls")
    ap.add_argument("--force", action="store_true", help="arbitrate again a run that already has arbiter output")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    total = 0
    for path in a.runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        if cfg.get("use_correction") and not a.force and not a.dry_run:
            print(f"{path.parent.name}: already arbitrated, skipped (use --force)")
            continue
        if not cfg.get("dual_ocr"):
            print(f"{path.parent.name}: made without the second reader, nothing to arbitrate")
            continue
        blocks = [b for fr in data.get("frames", []) for b in fr.get("blocks", [])]
        for b in blocks:  # as in a fresh run: no block has an arbiter output before step 4
            b.pop("corrected_text", None)

        if a.dry_run:
            corrector = _Counter()
        else:
            from comicglot.translation import OCRCorrector
            corrector = OCRCorrector(model_name=a.llm_model or "ocr_correction", provider=a.llm_provider)
        proc = arbiter_processor(corrector, cfg.get("correction_context", True), cfg.get("arbitrate_punctuation", False),
                                 cfg.get("source_lang", "ko"))
        asyncio.run(proc._arbitrate_blocks(blocks))

        if a.dry_run:
            print(f"{path.parent.name}: {len(blocks)} blocks, {corrector.calls} would go to the arbiter")
            total += corrector.calls
            continue

        backup = path.with_name("chapter_full_results.noarbiter.json")
        if not backup.exists():
            shutil.copyfile(path, backup)
        llm = corrector.llm_processor
        cfg.update(use_correction=True, llm_provider=llm.provider, llm_model=a.llm_model,
                   arbiter_applied_later=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        data["config"] = cfg
        data.setdefault("provenance", {}).setdefault("llm", {})["correction"] = {
            "provider": llm.provider, "model": llm.model_name, "temperature": corrector.temperature}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.parent.name}: arbitrated, {len(blocks)} blocks")
    if a.dry_run:
        print(f"total: {total} arbiter calls")


if __name__ == "__main__":
    main()
