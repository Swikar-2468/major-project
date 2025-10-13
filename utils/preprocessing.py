"""
Preprocessing utilities for Nepali hate speech detection
"""
import re
import emoji
import regex
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Nepali stopwords
NEPALI_STOPWORDS = set([
    "र", "मा", "कि", "भने", "त", "छ", "हो", "लाई", "ले",
    "गरेको", "गर्छ", "गर्छन्", "हुन्", "गरे", "न", "नभएको",
    "को", "का", "की", "ने", "पनि", "नै", "थियो", "थिए"
])

# Dirghikaran normalization mapping
DIRGHIKARAN_MAP = {
    "उ": "ऊ", "इ": "ई", "ऋ": "रि", "ए": "ऐ", "अ": "आ",
    "\u200d": "", "\u200c": "",  # Zero-width characters
    "।": ".", "॥": ".",  # Devanagari punctuation
    "ि": "ी", "ु": "ू"  # Vowel signs
}

# Cache for Roman stopwords
_roman_stopwords_cache = None


def is_devanagari(text: str) -> bool:
    """Detect if text contains Devanagari characters."""
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(regex.search(r'\p{Devanagari}', text))


def devanagari_to_roman(text: str) -> str:
    """Convert Devanagari script to Roman (ITRANS)."""
    try:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:
        return text


def roman_to_devanagari(text: str) -> str:
    """Convert Roman script to Devanagari (ITRANS)."""
    try:
        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    except Exception:
        return text


def normalize_dirghikaran(text: str) -> str:
    """Apply dirghikaran normalization to reduce orthographic variants."""
    for original, replacement in DIRGHIKARAN_MAP.items():
        text = text.replace(original, replacement)
    return text


def clean_text(text: str, aggressive: bool = True) -> str:
    """
    Clean text with various preprocessing steps.
    
    Args:
        text: Input text
        aggressive: If True, removes punctuation and numbers
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace="")
    
    if aggressive:
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        
        # Remove punctuation (keep Devanagari characters)
        text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def remove_stopwords_devanagari(text: str) -> str:
    """Remove Devanagari stopwords."""
    words = text.split()
    filtered = [w for w in words if w not in NEPALI_STOPWORDS]
    return ' '.join(filtered)


def remove_stopwords_roman(text: str) -> str:
    """Remove Romanized stopwords."""
    global _roman_stopwords_cache
    
    # Initialize cache on first use
    if _roman_stopwords_cache is None:
        _roman_stopwords_cache = set([
            devanagari_to_roman(w) for w in NEPALI_STOPWORDS
        ])
    
    words = text.split()
    filtered = [w for w in words if w not in _roman_stopwords_cache]
    return ' '.join(filtered)


def preprocess_for_ml_gru(text: str) -> str:
    """
    Preprocess for ML/GRU models: Romanized, cleaned, stopwords removed.
    
    Pipeline:
    1. Clean text
    2. Remove stopwords (script-dependent)
    3. Normalize dirghikaran (if Devanagari)
    4. Transliterate to Roman
    """
    if not isinstance(text, str):
        return ""
    
    if is_devanagari(text):
        text = clean_text(text, aggressive=True)
        text = normalize_dirghikaran(text)
        text = remove_stopwords_devanagari(text)
        text = devanagari_to_roman(text)
    else:
        text = clean_text(text, aggressive=True)
        text = remove_stopwords_roman(text)
    
    return text


def preprocess_for_transformer(text: str) -> str:
    """
    Preprocess for Transformer models (XLM-R): Devanagari, lightly cleaned.
    
    Pipeline:
    1. Transliterate Roman → Devanagari (if needed)
    2. Light cleaning (preserve punctuation for tokenizer)
    3. Normalize dirghikaran
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to Devanagari if needed
    if not is_devanagari(text):
        text = roman_to_devanagari(text)
    
    # Light cleaning (preserve punctuation)
    text = clean_text(text, aggressive=False)
    
    # Normalize orthographic variants
    text = normalize_dirghikaran(text)
    
    return text


def batch_preprocess(texts, mode='ml'):
    """
    Batch preprocess texts.
    
    Args:
        texts: List of text strings
        mode: 'ml' for ML/GRU, 'transformer' for XLM-R
        
    Returns:
        List of preprocessed texts
    """
    if mode == 'ml':
        return [preprocess_for_ml_gru(t) for t in texts]
    elif mode == 'transformer':
        return [preprocess_for_transformer(t) for t in texts]
    else:
        raise ValueError(f"Unknown mode: {mode}")