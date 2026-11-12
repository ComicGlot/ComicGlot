"""Translation prompt variants of the prompt lab (StyleTranslator prompt_variant, eval/translation_study/prompt_lab.py)."""
from comicglot.translation.style_processor import StyleTranslator
from eval.translation_study import prompt_lab


def translator(variant):
    t = StyleTranslator.__new__(StyleTranslator)  # no LLM client needed to build prompts
    t.prompt_variant = variant
    t.variant = t._load_variant(variant) if variant else None
    t.style_guides = {"default": StyleTranslator.DEFAULT_FALLBACK, "square_bubble": "SQUARE GUIDE"}
    t.prompt_template = "{bubble_type}|{style_guide}|{text}|{context_section}"
    return t


CTX = [{"context_strategy": "prev", "bubbles": [{"text": "앞줄", "bubble_type": "speech_bubble"}]}]


def test_published_prompt_unchanged_without_variant():
    t = translator(None)
    assert t._build_prompt("말", "ko", "en", "sqare_bubble", None) == "sqare_bubble|SQUARE GUIDE|말|"


def test_variant_guided_and_plain():
    t = translator("v5")
    v = t.variant
    guided = t._build_prompt("말", "ko", "en", "sqare_bubble", CTX)
    plain = t._build_prompt("말", "ko", "en", "default", CTX)
    assert v["types"]["square_bubble"] in guided and v["guides"]["square_bubble"] in guided
    assert v["types"]["square_bubble"] not in plain and "How to translate it" not in plain
    assert "Korean: 말" in guided and "Prev text 1: 앞줄" in guided and "(speech_bubble)" not in guided
    assert "\n\n\n" not in guided and "\n\n\n" not in plain


def test_pipeline_prompt_matches_the_lab():
    """The pipeline builds the same prompt the lab tested, for a guided and a plain block."""
    order, blocks = ["r/0/1", "r/0/2"], {
        "r/0/1": {"_run": "r", "class_name": "speech_bubble", "corrected_text": "앞줄"},
        "r/0/2": {"_run": "r", "class_name": "sqare_bubble", "corrected_text": "말"},
    }
    v = prompt_lab.variant("v5")
    t = translator("v5")
    assert prompt_lab.build(v, "guided", "r/0/2", order, blocks) == t._build_prompt("말", "ko", "en", "sqare_bubble", CTX)
    assert prompt_lab.build(v, "plain", "r/0/2", order, blocks) == t._build_prompt("말", "ko", "en", "default", CTX)


def test_selective_guides_use_the_plain_prompt_for_other_types():
    t = translator("v9")
    assert t._build_prompt("말", "ko", "en", "speech_bubble", CTX) == t._build_prompt("말", "ko", "en", "default", CTX)
    assert t._build_prompt("쾅", "ko", "en", "sfx_sound", CTX) != t._build_prompt("쾅", "ko", "en", "default", CTX)
