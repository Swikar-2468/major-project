"""
Preprocessing utilities for Nepali hate speech detection
(ML / GRU baselines only)
"""

import re
import emoji
import regex

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False
    print("Warning: indic_transliteration not available. Romanization disabled.")

# Nepali stopwords (Devanagari)
NEPALI_STOPWORDS = set([
    "र", "मा", "कि", "भने", "त", "छ", "हो", "लाई", "ले",
    "गरेको", "गर्छ", "गर्छन्", "हुन्", "गरे", "न", "नभएको",
    "को", "का", "की", "ने", "पनि", "नै", "थियो", "थिए"
])

# Dirghikaran normalization
DIRGHIKARAN_MAP = {
    "उ": "ऊ", "इ": "ई", "ऋ": "रि", "ए": "ऐ", "अ": "आ",
    "\u200d": "", "\u200c": "",
    "।": ".", "॥": ".",
    "ि": "ी", "ु": "ू"
}

_roman_stopwords_cache = None


def is_devanagari(text: str) -> bool:
    """Return True if more than 50% of letters are Devanagari."""
    if not isinstance(text, str) or not text.strip():
        return False

    dev_chars = len(regex.findall(r'\p{Devanagari}', text))
    total_chars = len(regex.findall(r'\p{L}', text))
    return total_chars > 0 and (dev_chars / total_chars) > 0.5


def devanagari_to_roman(text: str) -> str:
    """Convert Devanagari text to Roman (ITRANS)."""
    if not TRANSLITERATION_AVAILABLE:
        return text
    try:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:
        return text


def normalize_dirghikaran(text: str) -> str:
    """Normalize orthographic variants in Devanagari."""
    for src, tgt in DIRGHIKARAN_MAP.items():
        text = text.replace(src, tgt)
    return text


def clean_text(text: str) -> str:
    """
    Aggressive cleaning for ML/GRU:
    - lowercase
    - remove URLs, mentions, hashtags
    - remove emojis
    - remove digits and punctuation
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords_devanagari(text: str) -> str:
    words = text.split()
    return " ".join(w for w in words if w not in NEPALI_STOPWORDS)


def remove_stopwords_roman(text: str) -> str:
    global _roman_stopwords_cache

    if _roman_stopwords_cache is None:
        _roman_stopwords_cache = {
            devanagari_to_roman(w) for w in NEPALI_STOPWORDS
        }

    words = text.split()
    return " ".join(w for w in words if w not in _roman_stopwords_cache)


def preprocess_for_ml_gru(text: str) -> str:
    """
    ML / GRU preprocessing pipeline:
    1. Aggressive cleaning
    2. Dirghikaran normalization (if Devanagari)
    3. Stopword removal
    4. Romanization
    """
    if not isinstance(text, str):
        return ""

    text = clean_text(text)

    if is_devanagari(text):
        text = normalize_dirghikaran(text)
        text = remove_stopwords_devanagari(text)
        text = devanagari_to_roman(text)
    else:
        text = remove_stopwords_roman(text)

    return text


def batch_preprocess(texts):
    """Batch preprocessing for ML / GRU."""
    return [preprocess_for_ml_gru(t) for t in texts]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    examples = [
        "Musalman haru aatankbadi hun",
        "यो नेपाली पाठ हो",
        "Nepal ko राजधानी Kathmandu ho",
        "@user123 check this https://example.com 🔥",
    ]

    print("=" * 60)
    print("ML / GRU PREPROCESSING TEST")
    print("=" * 60)

    for t in examples:
        print(f"Original : {t}")
        print(f"Processed: {preprocess_for_ml_gru(t)}\n")
