"""Prompt lab: develop the translation prompt and the type guides on a development set, judged blind.

A variant (data/prompts/lab/<name>.json) is a template plus one guide per type. Each variant yields two prompts per
block: guided (the type's description and guide) and plain (the same template with the type left out), so the
comparison measures what the type adds and nothing else. The published prompts are the variant "v0".

The development set holds blocks of the study chapters (and of the top-up frames) that are not among the 302 test
texts, so prompts tuned here are tested on texts they never saw. Its Korean is the pipeline's arbiter reading; both
prompts of a pair get the same text.

    python -m eval.translation_study.prompt_lab devset --per-type 12
    python -m eval.translation_study.prompt_lab translate --variant v1 --models google/gemini-3.5-flash-lite
    python -m eval.translation_study.prompt_lab judge --a v1:guided --b v1:plain --models google/gemini-3.5-flash-lite \
        --judge google/gemini-4-argon
    python -m eval.translation_study.prompt_lab report

The judge sees what a translator in the study sees: the frame with the block in red, the Korean and the two
versions in random order (eval/translation_study/judge_pairs.py, same prompt and parser). Calls resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.sample_frames import FAMILIES, local_path
from eval.translation_study.judge_pairs import PROMPT as JUDGE_PROMPT, ask, frame_jpeg, parse
from eval.translation_study.run_conditions import call_openrouter, context_lines, load_blocks

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "data" / "prompts" / "lab"
OUT = ROOT / "data" / "study2026" / "prompt_lab"
# LAB_SET=test: the 302 texts of the translator study with their checked Korean, never used for tuning; outputs go
# to prompt_lab/test/ so the development results stay apart.
TEST = __import__("os").environ.get("LAB_SET") == "test"
DEV_OUT = OUT
if TEST:
    OUT = OUT / "test"
S = ROOT / "data" / "study2026"
RUNS = sorted((ROOT / "data" / "output" / "study2026" / "runs").glob("*/chapter_full_results.json")) + [
    ROOT / "data" / "output" / "study2026" / "topup" / "dataset_v9_run1" / "chapter_full_results.json",
    ROOT / "data" / "output" / "study2026" / "topup2" / "dataset_v9_run2" / "chapter_full_results.json",
    # dataset frames run for the top-ups and not used there (rare types for development): a copy with the frame
    # paths fixed after the folder moved and without the 6 frames that are also in the test top-ups
    ROOT / "data" / "study2026" / "prompt_lab" / "runs" / "topup_v7_unused__dataset_v7_run1" / "chapter_full_results.json"]
GUIDE_KEY = {"sqare_bubble": "square_bubble"}


def test_sources() -> dict:
    """The checked Korean of the study texts, as the study translated it."""
    out = {}
    for d in ("translations_gt", "translations_gt_topup", "translations_gt_topup2"):
        for f in (S / d).glob("*/calls.jsonl"):
            for l in open(f, encoding="utf-8"):
                c = json.loads(l)
                if c.get("source_text"):
                    out.setdefault(c["block_uid"], c["source_text"])
    return out


def load():
    """Blocks of RUNS; blocks without an arbiter reading take the reading of llm_ocr/readings.csv (rare types).
    In the test set every study text takes its checked Korean."""
    order, blocks = load_blocks(RUNS)
    if TEST:
        for u, t in test_sources().items():
            if u in blocks:
                blocks[u]["corrected_text"] = t
    rd = DEV_OUT / "llm_ocr" / "readings.csv"
    if rd.exists():
        for r in csv.DictReader(open(rd, encoding="utf-8")):
            t = (r.get("google/gemini-4-argon") or "").strip()
            b = blocks.get(r["block_uid"])
            if b is not None and t and t != "#ILLEGIBLE" and not (b.get("corrected_text") or "").strip():
                b["corrected_text"] = t
    return order, blocks


def study_uids() -> set:
    out = set()
    for d in ("translations_gt", "translations_gt_topup", "translations_gt_topup2"):
        for f in (S / d).glob("*/calls.jsonl"):
            out |= {json.loads(l)["block_uid"] for l in open(f, encoding="utf-8")}
    return out


def devset(a) -> None:
    order, blocks = load()
    test = study_uids()
    pool = defaultdict(list)
    for u in order:
        b = blocks[u]
        src = (b.get("corrected_text") or "").strip()
        if u in test or not src or not b.get("class_name") or len(src) > 200:
            continue
        pool[b["class_name"]].append(u)
    rng = random.Random(a.seed)
    rows = []
    for cls in sorted(pool):
        pick = rng.sample(pool[cls], min(a.per_type, len(pool[cls])))
        rows += [{"block_uid": u, "class_name": cls} for u in pick]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "dev.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["block_uid", "class_name"])
        w.writeheader()
        w.writerows(rows)
    print(Counter(r["class_name"] for r in rows), len(rows), "blocks ->", OUT / "dev.csv")


def load_dev():
    if TEST:
        # exactly the texts the translators see: the blocks of the rating app's round2 export
        keep = set()
        for f in (ROOT.parent / "comicglot-rating" / "data").glob("round2*/chapters.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            keep |= {b["id"] for c in d["chapters"] if not c.get("practice") for pg in c["pages"] for b in pg["blocks"]}
        return sorted(u for u in test_sources() if u in keep)
    return [r["block_uid"] for r in csv.DictReader(open(OUT / "dev.csv", encoding="utf-8"))]


def variant(name: str) -> dict:
    return json.loads((LAB / f"{name}.json").read_text(encoding="utf-8"))


def build(v: dict, mode: str, uid: str, order, blocks) -> str:
    """mode: guided or plain. A variant fills {text}, {context}, {type}, {guide} (empty for plain)."""
    b = blocks[uid]
    cls = GUIDE_KEY.get(b["class_name"], b["class_name"])
    text_of = lambda u: blocks[u].get("corrected_text") or blocks[u].get("text")
    guided = mode == "guided"
    ctx = context_lines(order, blocks, uid, text_of, n=v.get("context_lines", 3), with_class=guided and v.get("context_types", True))
    if guided:
        t, g = v["types"].get(cls, ""), v["guides"].get(cls, v["guides"].get("default", ""))
        typ = v["type_line"].format(type=t) if t and v.get("type_line") else ""
        guide = v["guide_line"].format(guide=g) if g else ""
    else:
        typ, guide = "", v.get("plain_guide_line", "")
    return v["template"].format(text=text_of(uid), context=ctx, type=typ, guide=guide).replace("\n\n\n", "\n\n")


def translate(a) -> None:
    order, blocks = load()
    v = variant(a.variant)
    for model in a.models:
        out = OUT / "tr" / a.variant / (model.replace("/", "__") + ".jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {(r["block_uid"], r["mode"]) for r in map(json.loads, open(out, encoding="utf-8")) if r.get("translation")}
        modes = ["guided", "plain"] if not v.get("guided_only") else ["guided"]
        jobs = [(u, m) for u in load_dev() for m in modes if (u, m) not in done]

        def run(job):
            u, m = job
            p = build(v, m, u, order, blocks)
            raw, err = "", None
            try:
                raw = call_openrouter(model, {"text": p, "image": None}, a.temperature, None, a.max_tokens)
            except Exception as e:
                err = str(e)[:300]
            tr = raw.split("TRANSLATED_TEXT:", 1)[1].strip() if "TRANSLATED_TEXT:" in raw else raw.strip()
            return {"block_uid": u, "mode": m, "variant": a.variant, "model": model, "prompt": p, "raw": raw,
                    "translation": " ".join(tr.split()), "error": err, **getattr(call_openrouter, "last", {})}

        with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(run, jobs):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{a.variant} {model}: {len(jobs)} calls")


def versions(spec: str, model: str) -> dict:
    """'v1:guided' -> {uid: translation}."""
    name, mode = spec.split(":")
    f = OUT / "tr" / name / (model.replace("/", "__") + ".jsonl")
    out = {}
    for r in map(json.loads, open(f, encoding="utf-8")):
        if r["mode"] == mode and r.get("translation"):
            out[r["block_uid"]] = r["translation"]
    return out


def judge(a) -> None:
    order, blocks = load()
    frames = {}
    for rf in RUNS:
        run = rf.parent.name
        for fr in json.loads(rf.read_text(encoding="utf-8")).get("frames", []):
            for b in fr.get("blocks", []):
                frames[f"{run}/{b.get('frame_id', fr.get('id'))}/{b.get('id')}"] = (fr["image_path"], b["box"])
    for model in a.models:
        va, vb = versions(a.a, model), versions(a.b, model)
        out = OUT / "judge" / f"{a.a}_vs_{a.b}".replace(":", "-") / f"{model.replace('/', '__')}__by__{a.judge.replace('/', '__')}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {r["block_uid"] for r in map(json.loads, open(out, encoding="utf-8")) if r.get("parsed") or r.get("identical")}
        items = [u for u in load_dev() if u in va and u in vb and u not in done]

        def run(u):
            x, y = va[u], vb[u]
            rec = {"block_uid": u, "class_name": blocks[u]["class_name"], "a": x, "b": y, "model": model, "judge": a.judge}
            if x == y:
                return {**rec, "identical": True}
            flip = random.Random(f"{u}:{a.a}:{a.b}").random() < 0.5
            A, B = (y, x) if flip else (x, y)
            img = frame_jpeg(local_path(frames[u][0], ROOT), frames[u][1])
            src = blocks[u].get("corrected_text") or blocks[u].get("text")
            res = ask(a.judge, JUDGE_PROMPT.format(source=src, a=A, b=B), img, a.max_tokens)
            p = parse(res.get("raw", ""))
            if p:
                back = {"A": "b" if flip else "a", "B": "a" if flip else "b", "equal": "equal"}
                p = {"preferred": back[p["choice"]], "tone_a": p["tone"]["B" if flip else "A"], "tone_b": p["tone"]["A" if flip else "B"], "reason": p["reason"]}
            return {**rec, "flip": flip, "raw": res.get("raw", "")[:600], "parsed": p, "usage": res.get("usage"), "error": res.get("error")}

        with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(run, items):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{a.a} vs {a.b}, {model}, judge {a.judge}: {len(items)} judged")
        summarise(out)


LETTERED_PROMPT = """You are an experienced editor of English webtoon localisations, checking two candidate translations of one line.
Image 1 is the original Korean page; the line is inside the red box. Images 2 and 3 show the same spot lettered with English version A and version B, as a reader would see it.

Korean: {source}
A: {a}
B: {b}

Answer each question in turn:
1. styling: which version carries more expressive styling (emphatic or colourful wording, interjections, stretched sounds, !, ?, ..., stutters, sound-word conventions)? "A", "B" or "same".
2. fit: for each version, does its styling suit this balloon, this lettering and this scene? "helps", "neutral" or "hurts".
3. meaning: for each version, against the Korean: "ok", "minor" (a small shift or loss) or "major" (wrong or missing meaning).
4. natural: which lettered panel reads more like a professionally localised English webtoon? "A", "B" or "equal".
5. choice: which version would you publish in this panel? "A", "B" or "equal".

Answer with JSON only: {{"styling": "...", "fit": {{"A": "...", "B": "..."}}, "meaning": {{"A": "...", "B": "..."}}, "natural": "...", "choice": "...", "reason": "<one sentence>"}}"""


def parse_lettered(raw: str) -> dict | None:
    import re
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    norm = lambda v, ok: (lambda s: s if s in ok else None)({"a": "A", "b": "B", "tie": "equal", "same": "same"}.get(str(v).strip().lower(), str(v).strip().lower()))
    ab = {"A", "B", "equal"}
    out = {"styling": norm(j.get("styling"), {"A", "B", "same"}), "natural": norm(j.get("natural"), ab), "choice": norm(j.get("choice"), ab),
           "fit": {k: norm((j.get("fit") or {}).get(k), {"helps", "neutral", "hurts"}) for k in "AB"},
           "meaning": {k: norm((j.get("meaning") or {}).get(k), {"ok", "minor", "major"}) for k in "AB"},
           "reason": str(j.get("reason", ""))[:300]}
    if None in (out["styling"], out["natural"], out["choice"], *out["fit"].values(), *out["meaning"].values()):
        return None
    return out


def judge_lettered(a) -> None:
    """The judge sees the page and both versions lettered into the block, and answers descriptive questions (more
    styling? does it suit the panel? meaning kept? natural?) before choosing. --swap judges with A and B the other way
    round, to measure how often the judge changes its mind with the order."""
    from eval.translation_study.letter import jpeg_b64, letter
    order, blocks = load()
    frames = {}
    for rf in RUNS:
        run = rf.parent.name
        for fr in json.loads(rf.read_text(encoding="utf-8")).get("frames", []):
            for b in fr.get("blocks", []):
                frames[f"{run}/{b.get('frame_id', fr.get('id'))}/{b.get('id')}"] = (fr["image_path"], b["box"])
    for model in a.models:
        va, vb = versions(a.a, model), versions(a.b, model)
        tag = "__swap" if a.swap else ""
        out = OUT / "judge2" / f"{a.a}_vs_{a.b}".replace(":", "-") / f"{model.replace('/', '__')}__by__{a.judge.replace('/', '__')}{tag}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {r["block_uid"] for r in map(json.loads, open(out, encoding="utf-8")) if r.get("parsed") or r.get("identical")}
        items = [u for u in load_dev() if u in va and u in vb and u not in done]
        if a.limit:
            items = items[: a.limit]

        def run(u):
            x, y = va[u], vb[u]
            cls = blocks[u]["class_name"]
            rec = {"block_uid": u, "class_name": cls, "a": x, "b": y, "model": model, "judge": a.judge}
            if x == y:
                return {**rec, "identical": True}
            flip = (random.Random(f"{u}:{a.a}:{a.b}").random() < 0.5) != a.swap
            A, B = (y, x) if flip else (x, y)
            path = local_path(frames[u][0], ROOT)
            imgs = [frame_jpeg(path, frames[u][1], 1024), jpeg_b64(letter(path, frames[u][1], A, cls)), jpeg_b64(letter(path, frames[u][1], B, cls))]
            src = blocks[u].get("corrected_text") or blocks[u].get("text")
            res = ask_images(a.judge, LETTERED_PROMPT.format(source=src, a=A, b=B), imgs, a.max_tokens)
            p = parse_lettered(res.get("raw", ""))
            if p:
                m = {"A": "b" if flip else "a", "B": "a" if flip else "b", "equal": "equal", "same": "same"}
                ka, kb = ("B", "A") if flip else ("A", "B")
                p = {"preferred": m[p["choice"]], "natural": m[p["natural"]], "styling": m[p["styling"]],
                     "fit_a": p["fit"][ka], "fit_b": p["fit"][kb], "meaning_a": p["meaning"][ka], "meaning_b": p["meaning"][kb], "reason": p["reason"]}
            return {**rec, "flip": flip, "raw": res.get("raw", "")[:800], "parsed": p, "usage": res.get("usage"), "error": res.get("error")}

        with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(run, items):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[lettered] {a.a} vs {a.b}, {model}, judge {a.judge}{' (swapped)' if a.swap else ''}: {len(items)} judged")
        summarise2(out)


def ask_images(judge: str, text: str, images: list[str], max_tokens: int) -> dict:
    import os
    import requests
    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images]
    body = {"model": judge, "max_tokens": max_tokens, "temperature": 0, "messages": [{"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "ComicGlot"}
    last = ""
    for attempt in range(5):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers, timeout=240)
            if r.status_code == 200 and not r.json().get("error"):
                j = r.json()
                ch = (j.get("choices") or [{}])[0]
                return {"raw": ch.get("message", {}).get("content") or "", "usage": j.get("usage")}
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            last = str(e)
        time.sleep(min(30, 2 ** attempt))
    return {"raw": "", "error": last}


def summarise2(path: Path) -> dict:
    rows = {}
    for r in map(json.loads, open(path, encoding="utf-8")):
        if r.get("parsed") or r.get("identical"):
            rows[r["block_uid"]] = r
    by = defaultdict(Counter)
    cost = 0.0
    for r in rows.values():
        for key in (FAMILIES.get(r["class_name"], "?"), "all"):
            c = by[key]
            if r.get("identical"):
                c["same_text"] += 1
                continue
            p = r["parsed"]
            c["choice_" + p["preferred"]] += 1
            c["natural_" + p["natural"]] += 1
            c["styled_" + p["styling"]] += 1
            for s in "ab":
                c[f"fit_{s}_{p['fit_' + s]}"] += 1
                c[f"meaning_{s}_{p['meaning_' + s]}"] += 1
        cost += float((r.get("usage") or {}).get("cost") or 0)
    print(f"  {path.parent.name} / {path.stem}  (cost ${cost:.2f})")
    out = {}
    for k in ("outlined_balloon", "box_or_free_text", "sound_effect", "all"):
        c = by.get(k)
        if not c:
            continue
        dec = c["choice_a"] + c["choice_b"]
        out[k] = dict(c)
        print(f"    {k:18s} choice A {c['choice_a']:3d} B {c['choice_b']:3d} eq {c['choice_equal']:3d} (A {c['choice_a'] / max(1, dec):.2f})"
              f" | natural A {c['natural_a']} B {c['natural_b']} | more styled A {c['styled_a']} B {c['styled_b']}"
              f" | fit helps {c['fit_a_helps']}/{c['fit_b_helps']} hurts {c['fit_a_hurts']}/{c['fit_b_hurts']}"
              f" | major {c['meaning_a_major']}/{c['meaning_b_major']}")
    return out


def summarise(path: Path) -> dict:
    rows = {}
    for r in map(json.loads, open(path, encoding="utf-8")):
        if r.get("parsed") or r.get("identical"):
            rows[r["block_uid"]] = r
    by = defaultdict(Counter)
    cost = 0.0
    for r in rows.values():
        fam = FAMILIES.get(r["class_name"], "?")
        for key in (fam, "all", r["class_name"]):
            c = by[key]
            if r.get("identical"):
                c["same"] += 1
                continue
            p = r["parsed"]
            c[p["preferred"]] += 1
            c["strong_a"] += p["tone_a"] == "strong"
            c["strong_b"] += p["tone_b"] == "strong"
            c["flat_a"] += p["tone_a"] == "flat"
            c["flat_b"] += p["tone_b"] == "flat"
        cost += float((r.get("usage") or {}).get("cost") or 0)
    out = {}
    for k, c in by.items():
        n = c["a"] + c["b"] + c["equal"]
        out[k] = {"a": c["a"], "b": c["b"], "equal": c["equal"], "same": c["same"],
                  "a_share_decided": round(c["a"] / max(1, c["a"] + c["b"]), 3),
                  "strong_a": c["strong_a"], "strong_b": c["strong_b"], "flat_a": c["flat_a"], "flat_b": c["flat_b"], "n": n}
    fams = ["outlined_balloon", "box_or_free_text", "sound_effect", "all"]
    print(f"  {path.parent.name} / {path.stem}  (cost ${cost:.2f})")
    for k in fams + sorted(x for x in out if x not in fams and "_" in x and x not in FAMILIES.values()):
        if k in out:
            o = out[k]
            print(f"    {k:18s} A {o['a']:3d}  B {o['b']:3d}  eq {o['equal']:3d}  same {o['same']:3d}  A share {o['a_share_decided']:.2f}"
                  f"  strong {o['strong_a']}/{o['strong_b']}  flat {o['flat_a']}/{o['flat_b']}")
    return out


def report(a) -> None:
    for f in sorted((OUT / "judge").glob("*/*.jsonl")):
        summarise(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("devset")
    d.add_argument("--per-type", type=int, default=12)
    d.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("translate")
    t.add_argument("--variant", required=True)
    t.add_argument("--models", nargs="+", required=True)
    t.add_argument("--temperature", type=float, default=0.7)
    t.add_argument("--max-tokens", type=int, default=3000)
    t.add_argument("--workers", type=int, default=12)
    j = sub.add_parser("judge")
    j.add_argument("--a", required=True, help="variant:mode, e.g. v1:guided")
    j.add_argument("--b", required=True)
    j.add_argument("--models", nargs="+", required=True)
    j.add_argument("--judge", default="google/gemini-4-argon")
    j.add_argument("--max-tokens", type=int, default=2000)
    j.add_argument("--workers", type=int, default=12)
    j2 = sub.add_parser("judge2", help="judge with both versions lettered into the page and descriptive questions first")
    j2.add_argument("--a", required=True)
    j2.add_argument("--b", required=True)
    j2.add_argument("--models", nargs="+", required=True)
    j2.add_argument("--judge", default="google/gemini-4-argon")
    j2.add_argument("--max-tokens", type=int, default=3000)
    j2.add_argument("--workers", type=int, default=10)
    j2.add_argument("--limit", type=int, default=0)
    j2.add_argument("--swap", action="store_true", help="the same pairs with A and B the other way round")
    sub.add_parser("report")
    a = ap.parse_args()
    {"devset": devset, "translate": translate, "judge": judge, "judge2": judge_lettered, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
