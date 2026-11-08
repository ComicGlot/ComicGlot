"""Applying the arbiter to a saved run (scripts/arbitrate_run.py) follows the pipeline's step 4."""
import asyncio
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("arbitrate_run", Path(__file__).resolve().parents[1] / "scripts" / "arbitrate_run.py")
arb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arb)


class Echo:
    """Arbiter stand-in that records what it was sent."""
    def __init__(self):
        self.inputs, self.contexts = [], []

    async def batch_correct_texts(self, inputs, contexts=None, language="ko"):
        self.inputs += inputs
        self.contexts += contexts
        return [f"ARB({a}|{b})" for a, b in inputs]


def test_step4_on_saved_blocks():
    blocks = [
        {"id": "0", "class_name": "speech_bubble", "text": "안녕!", "secondary_text": "안녕"},     # equal up to punctuation
        {"id": "1", "class_name": "sfx_sound", "text": "", "secondary_text": "쾅"},              # one reading empty
        {"id": "2", "class_name": "think_bubble", "text": "있다", "secondary_text": "잌다"},     # differ: arbiter
    ]
    e = Echo()
    p = arb.arbiter_processor(e, correction_context=True, arbitrate_punctuation=False, source_lang="ko")
    asyncio.run(p._arbitrate_blocks(blocks))
    assert [b["corrected_text"] for b in blocks] == ["안녕!", "쾅", "ARB(있다|잌다)"]
    assert e.inputs == [["있다", "잌다"]]
    assert e.contexts == ["Text block type: think_bubble. Previous lines on the page: 안녕! / 쾅"]


def test_arbiter_answer_after_the_last_marker():
    from comicglot.translation.ocr_correction import corrected_from
    assert corrected_from("CORRECTED_TEXT: 있다") == "있다"
    assert corrected_from("CORRECTED_TEXT: influ\n\nWait, correcting this properly:\n\nCORRECTED_TEXT: 인듯") == "인듯"
