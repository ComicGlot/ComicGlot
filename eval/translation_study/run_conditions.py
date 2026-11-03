"""Produce translations of the same blocks under controlled conditions.

Conditions (each a separate LLM call per block, same model and temperature):
  true_class   the pipeline as published: class style guide + 3 previous blocks
  no_class     the default style guide for every block (class withheld) + context
  wrong_class  the style guide of a deliberately mismatched class (WRONG_CLASS) + context
  no_context   true class, no previous blocks
  page_vlm     whole page image + the source text, no class (needs a vision model)

Source text comes from --source: "corrected" uses the pipeline's corrected_text,
"gt" uses a hand transcription CSV (block_uid, gt_text) so OCR errors cannot
mix into translation scores. The paper should use "gt".

Every call is logged to <out>/calls.jsonl with the full prompt, model id,
temperature and raw response, so the study can be audited and re-run.

Needs ANTHROPIC_API_KEY. --dry-run writes the prompts without calling the API.

Usage:
    python -m eval.translation_study.run_conditions --runs RUN.json ... --sample sample.csv \
        --source gt --gt gt.csv --model claude-sonnet-4-5 --out study_round1
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
STYLE_GUIDES = json.load(open(ROOT / "data" / "prompts" / "style_guides.json", encoding="utf-8"))

# A mismatched guide per class, chosen to push tone in a clearly different
# direction (calm <-> intense, inner <-> spoken). Fixed so the study is repeatable.
WRONG_CLASS: Dict[str, str] = {
    "speech_bubble": "scream_bubble", "scream_bubble": "think_bubble", "think_bubble": "scream_bubble",
    "cloud_bubble": "spiked_bubble", "spiked_bubble": "cloud_bubble", "wiggle_bubble": "scream_bubble",
    "hexagon_bubble": "wiggle_bubble", "sqare_bubble": "scream_bubble", "system_bubble": "wiggle_bubble",
    "standalone_text": "system_bubble", "background_text": "scream_bubble", "title_panel": "think_bubble",
    "sfx_sound": "speech_bubble",
}

TEMPLATE = (
    "[TASK]: Translate text from {src} to {tgt} with appropriate style.\n\n"
    "{context}"
    "[ORIGINAL TEXT]: {text}\n\n"
    "[STYLE GUIDE]: {guide}\n\n"
    "[INSTRUCTIONS]: Translate the original text from {src} to {tgt} following the style guide{for_type}.\n\n"
    "Analyze the original text, taking into account any context provided and the style guide. "
    "Think carefully about cultural nuances, idioms, and emotional tone.\n\n"
    "After your analysis, provide ONLY the translated text with no additional explanation, "
    "starting with 'TRANSLATED_TEXT:' followed by the translation."
)
PAGE_TEMPLATE = (
    "This is a page from a Korean webtoon. Translate the following text, which appears on this page, "
    "from {src} to {tgt}. Use the page to judge tone and who is speaking.\n\n"
    "{context}[ORIGINAL TEXT]: {text}\n\n"
    "Provide ONLY the translated text, starting with 'TRANSLATED_TEXT:' followed by the translation."
)


def load_blocks(paths: List[Path]):
    """Blocks in reading order per run, keyed by block_uid."""
    order, blocks = [], {}
    for p in paths:
        run = p.parent.name
        data = json.load(open(p, encoding="utf-8"))
        for frame in data.get("frames", []):
            for b in frame.get("blocks", []):
                uid = f"{run}/{b.get('frame_id', frame.get('id'))}/{b.get('id')}"
                b = dict(b, _uid=uid, _run=run, _frame_image=frame.get("image_path"))
                blocks[uid] = b
                order.append(uid)
    return order, blocks


def context_lines(order, blocks, uid, text_of, n=3, with_class=True) -> str:
    i = order.index(uid)
    run = blocks[uid]["_run"]
    prev = [u for u in order[max(0, i - n):i] if blocks[u]["_run"] == run]
    lines = []
    for u in prev:
        t = text_of(u)
        if not t:
            continue
        tag = f" ({blocks[u].get('class_name')})" if with_class else ""
        lines.append(f"- Prev text {len(lines) + 1}{tag}: {t}")
    return f"[CONTEXT]: Surrounding texts:\n" + "\n".join(lines) + "\n\n" if lines else ""


def build_prompt(cond, uid, order, blocks, text_of, src, tgt) -> Dict:
    b = blocks[uid]
    cls = b.get("class_name", "default")
    text = text_of(uid)
    if cond == "page_vlm":
        return {"text": PAGE_TEMPLATE.format(src=src, tgt=tgt, text=text,
                                             context=context_lines(order, blocks, uid, text_of, with_class=False)),
                "image": b.get("_frame_image"), "guide_class": None}
    guide_cls = {"true_class": cls, "no_context": cls, "no_class": "default", "wrong_class": WRONG_CLASS.get(cls, cls)}[cond]
    guide = STYLE_GUIDES.get({"sqare_bubble": "square_bubble"}.get(guide_cls, guide_cls), STYLE_GUIDES["default"])
    ctx = "" if cond == "no_context" else context_lines(order, blocks, uid, text_of, with_class=cond != "no_class")
    for_type = "" if cond == "no_class" else f" appropriate for a {guide_cls}"
    return {"text": TEMPLATE.format(src=src, tgt=tgt, context=ctx, text=text, guide=guide, for_type=for_type),
            "image": None, "guide_class": guide_cls}


def call_claude(client, model, prompt: Dict, temperature: float, image_root: Optional[Path]) -> str:
    content = []
    if prompt["image"]:
        if image_root is None:
            raise SystemExit("page_vlm needs --project-root to find segment images")
        from eval.make_ocr_gt_sheet import local_path
        data = base64.b64encode(open(local_path(prompt["image"], image_root), "rb").read()).decode()
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    content.append({"type": "text", "text": prompt["text"]})
    msg = client.messages.create(model=model, max_tokens=1024, temperature=temperature,
                                 messages=[{"role": "user", "content": content}])
    return "".join(part.text for part in msg.content if getattr(part, "type", "") == "text")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--sample", type=Path, required=True, help="CSV with a block_uid column (the items to translate)")
    ap.add_argument("--source", choices=["gt", "corrected"], default="gt")
    ap.add_argument("--gt", type=Path, help="hand transcription CSV (block_uid, gt_text) for --source gt")
    ap.add_argument("--conditions", nargs="+",
                    default=["true_class", "no_class", "wrong_class", "no_context", "page_vlm"])
    ap.add_argument("--model", required=True, help="exact model id; recorded with every call")
    ap.add_argument("--temperature", type=float, default=0.7, help="pipeline default is 0.7")
    ap.add_argument("--src", default="ko")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--project-root", type=Path, help="local project root, for page images")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    order, blocks = load_blocks(args.runs)
    gt = {}
    if args.source == "gt":
        if not args.gt:
            raise SystemExit("--source gt needs --gt")
        gt = {r["block_uid"]: r["gt_text"] for r in csv.DictReader(open(args.gt, encoding="utf-8-sig")) if r["gt_text"].strip()}

    def text_of(uid):
        return gt.get(uid) if args.source == "gt" else blocks[uid].get("corrected_text") or blocks[uid].get("text")

    items = [r["block_uid"] for r in csv.DictReader(open(args.sample, encoding="utf-8-sig"))]
    items = [u for u in items if text_of(u)]
    args.out.mkdir(parents=True, exist_ok=True)
    client = None
    if not args.dry_run:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    done = set()
    log = args.out / "calls.jsonl"
    if log.exists():
        done = {(json.loads(l)["block_uid"], json.loads(l)["condition"]) for l in open(log, encoding="utf-8")}
    with open(log, "a", encoding="utf-8") as fh:
        for uid in items:
            for cond in args.conditions:
                if (uid, cond) in done:
                    continue
                prompt = build_prompt(cond, uid, order, blocks, text_of, args.src, args.tgt)
                raw = "" if args.dry_run else call_claude(client, args.model, prompt, args.temperature, args.project_root)
                out = raw.split("TRANSLATED_TEXT:", 1)[1].strip() if "TRANSLATED_TEXT:" in raw else raw.strip()
                fh.write(json.dumps({"block_uid": uid, "condition": cond, "class_name": blocks[uid].get("class_name"),
                                     "guide_class": prompt["guide_class"], "source_text": text_of(uid),
                                     "model": args.model, "temperature": args.temperature,
                                     "prompt": prompt["text"], "used_image": bool(prompt["image"]),
                                     "raw": raw, "translation": out, "time": time.time()}, ensure_ascii=False) + "\n")
                fh.flush()
    print(f"{len(items)} items x {len(args.conditions)} conditions -> {log}")


if __name__ == "__main__":
    main()
