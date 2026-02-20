"""
Transformer Data Preprocessing Module
======================================
Enhanced preprocessing pipeline for Nepali hate speech classification.

This module provides:
- Script detection (Devanagari/Romanized/English/Mixed)
- Transliteration (Romanized → Devanagari)
- Translation (English → Nepali)
- Emoji semantic mapping with feature extraction
- Text normalization

Usage:
------
from scripts.transformer_data_preprocessing import HateSpeechPreprocessor

# Initialize preprocessor
preprocessor = HateSpeechPreprocessor(
    model_type="xlmr",
    translate_english=True
)

# Preprocess single text
processed_text, emoji_features = preprocessor.preprocess("Your text here")

# Preprocess batch
texts_list = ["text1", "text2", "text3"]
processed_texts, features_list = preprocessor.preprocess_batch(texts_list)
"""

import re
import emoji
import regex
from typing import Any, Literal, Optional, Tuple, Dict, List
from deep_translator import GoogleTranslator
from functools import lru_cache
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Try to import transliteration (optional)
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False
    logger.warning("indic_transliteration not available. Transliteration disabled.")


# ============================================================================
# COMPREHENSIVE EMOJI MAPPINGS
# ============================================================================

EMOJI_TO_NEPALI = {
    # Positive emotions
    '😂': 'हाँसो', '🤣': 'ठूलो_हाँसो', '😀': 'खुशी', '😁': 'खुशी', '😃': 'खुशी',
    '😄': 'खुशी', '😅': 'नर्भस_हाँसो', '😆': 'हाँसो', '😊': 'मुस्कान', '☺️': 'मुस्कान',
    '😉': 'आँखा_झिम्काउने', '🙂': 'मुस्कान', '🙃': 'उल्टो_मुस्कान', '😌': 'शान्त',
    '😍': 'माया', '🥰': 'माया', '😘': 'चुम्बन', '😗': 'चुम्बन', '😙': 'चुम्बन', '😚': 'चुम्बन',
    '🤗': 'अँगालो', '🤩': 'चकित', '🥳': 'उत्सव', '🤤': 'लालसा',
    
    # Mockery & Sarcasm
    '😏': 'व्यंग्य', '😜': 'जिब्रो_देखाउने', '😝': 'जिब्रो_देखाउने', '😛': 'जिब्रो',
    '🙄': 'आँखा_घुमाउने', '😤': 'निराश', '😑': 'अभिव्यक्तिहीन', '😐': 'तटस्थ',
    '😬': 'तनाव', '🤨': 'शंकास्पद', '🤫': 'चुपचाप', '🤭': 'हात_छोप्ने', 
    '🤥': 'झूठ', '😶': 'मौन',
    
    # Anger & Hate
    '😠': 'रिस', '😡': 'ठूलो_रिस', '🤬': 'गाली', '😈': 'खराब', '👿': 'खराब',
    '💢': 'क्रोध', '🔪': 'हिंसा', '💣': 'हिंसा', '🗡️': 'तरवार', '⚔️': 'युद्ध',
    '💥': 'विस्फोट', '🔫': 'बन्दुक', '🧨': 'विस्फोटक', '☠️': 'मृत्यु', '💀': 'खोपडी',
    '👹': 'राक्षस', '👺': 'दानव', '🤡': 'जोकर', '🖤': 'कालो_मन',
    '😾': 'रिसाएको', '👊': 'मुक्का', '✊': 'मुक्का',
    
    # Offensive Gestures
    '🖕': 'अपमान', '👎': 'नकारात्मक', '👎🏻': 'नकारात्मक', '👎🏼': 'नकारात्मक',
    '👎🏽': 'नकारात्मक', '👎🏾': 'नकारात्मक', '👎🏿': 'नकारात्मक',
    '🖕🏻': 'अपमान', '🖕🏼': 'अपमान', '🖕🏽': 'अपमान', '🖕🏾': 'अपमान', '🖕🏿': 'अपमान',
    
    # Sadness
    '😭': 'रुवाइ', '😢': 'रुवाइ', '😿': 'रुवाइ', '😔': 'उदास', '😞': 'उदास',
    '😒': 'उदास', '😓': 'चिन्तित', '😟': 'चिन्तित', '😕': 'अलमलिएको',
    '🙁': 'तल्लो_मुख', '☹️': 'दुःखी', '😩': 'थकित', '😫': 'थकित',
    '😖': 'भ्रमित', '😣': 'अडिग', '😥': 'निराश', '🥺': 'बिन्ती',
    
    # Fear & Shock
    '😨': 'डर', '😰': 'चिन्तित_पसिना', '😱': 'चिच्याउने', '😳': 'लजाउने',
    '🤯': 'मन_उडेको', '😵': 'चक्कर', '😲': 'चकित', '😯': 'छक्क',
    
    # Disgust
    '🤢': 'बान्ता', '🤮': 'बान्ता', '🤧': 'हाच्छ्यूँ', '😷': 'बिरामी',
    '🤒': 'ज्वरो', '🤕': 'घाइते', '🥴': 'मात्तिएको', '😪': 'निद्रा',
    
    # Positive Gestures
    '👍': 'सकारात्मक', '👍🏻': 'सकारात्मक', '👍🏼': 'सकारात्मक', 
    '👍🏽': 'सकारात्मक', '👍🏾': 'सकारात्मक', '👍🏿': 'सकारात्मक',
    '👏': 'तालि', '🙌': 'उत्सव', '👌': 'ठीक_छ', '🤝': 'हात_मिलाउनु',
    '🙏': 'नमस्कार', '🤲': 'प्रार्थना', '💪': 'शक्ति', '✌️': 'शान्ति',
    
    # Hearts
    '❤️': 'माया', '🧡': 'माया', '💛': 'माया', '💚': 'माया', '💙': 'माया',
    '💜': 'माया', '🤍': 'सेतो_मन', '🤎': 'खैरो_मन', '❣️': 'माया', 
    '💕': 'माया', '💞': 'माया', '💓': 'माया', '💗': 'माया',
    '💖': 'माया', '💘': 'माया', '💝': 'माया', '💔': 'टुटेको_मन',
    
    # Symbols
    '🔥': 'आगो', '💯': 'पूर्ण', '💨': 'हावा', '💫': 'चमक',
    '⭐': 'तारा', '✨': 'चमक', '🌟': 'चम्किलो_तारा',
    '🚫': 'निषेध', '⛔': 'प्रवेश_निषेध', '❌': 'रद्द', '❎': 'गलत',
    
    # People
    '👫': 'जोडी', '👬': 'पुरुष_जोडी', '👭': 'महिला_जोडी', '👨\u200d👩\u200d👧\u200d👦': 'परिवार',
    '👶': 'बच्चा', '👦': 'केटा', '👧': 'केटी', '👨': 'पुरुष', '👩': 'महिला',
    '👴': 'बूढो', '👵': 'बूढी', '🧒': 'बालक', '👱': 'गोरो', '🧔': 'दाह्री',
    
    # Country
    '🇳🇵': 'नेपाल', '🇮🇳': 'भारत', '🇵🇰': 'पाकिस्तान', '🇧🇩': 'बंगलादेश',
    '🇨🇳': 'चीन', '🇺🇸': 'अमेरिका', '🏴': 'झण्डा',
    
    # Animals
    '🐕': 'कुकुर', '🐖': 'सुँगुर', '🐀': 'मुसा', '🐍': 'सर्प', '🦂': 'बिच्छी',
    '🐒': 'बाँदर', '🐵': 'बाँदर_अनुहार', '🦍': 'गोरिल्ला', '🐗': 'जङ्गली_सुँगुर',
    
    # Other
    '🤔': 'सोच', '🧐': 'अनुसन्धान', '😴': 'सुत्ने', '💩': 'मल',
    '👻': 'भूत', '🤖': 'रोबोट', '👽': 'विदेशी', '🎭': 'मुखौटा',
    
    # === EXPANDED COMMON EMOJIS ===
    
    # Celebrations & Party
    '🎉': 'उत्सव', '🎊': 'पार्टी', '🎈': 'बेलुन', '🎁': 'उपहार',
    '🎂': 'केक', '🍰': 'मिठाई', '🥂': 'चश्मा', '🍾': 'शराब',
    
    # Food & Drink (common in casual/hate contexts)
    '🍕': 'पिज्जा', '🍔': 'बर्गर', '🍗': 'चिकन', '🍖': 'मासु',
    '🍺': 'बियर', '🍻': 'पार्टी', '☕': 'चिया', '🍵': 'चिया',
    '🍜': 'नूडल', '🍛': 'करी', '🍲': 'खाना', '🥘': 'परिकार',
    
    # Sports & Activities
    '⚽': 'फुटबल', '🏏': 'क्रिकेट', '🏀': 'बास्केटबल', '🎮': 'खेल',
    '🏆': 'ट्रफी', '🥇': 'स्वर्ण', '🥈': 'रजत', '🥉': 'कांस्य',
    
    # Weather & Nature
    '☀️': 'घाम', '🌙': 'चन्द्रमा', '🌧️': 'पानी', '⛈️': 'आँधी',
    '❄️': 'हिउँ', '🌈': 'इन्द्रेणी', '⚡': 'बिजुली', '🌪️': 'बतास',
    
    # Technology & Modern
    '📱': 'मोबाइल', '💻': 'कम्प्युटर', '📷': 'क्यामेरा', '🎥': 'भिडियो',
    '🖥️': 'कम्प्युटर', '⌨️': 'किबोर्ड', '🖱️': 'माउस', '📡': 'एन्टेना',
    
    # Time & Clock
    '⏰': 'घडी', '⏳': 'समय', '⌛': 'बालुवा_घडी', '🕐': 'एक_बजे',
    
    # Objects
    '📚': 'किताब', '📖': 'खुल्ला_किताब', '✏️': 'पेन्सिल', '📝': 'लेख',
    '🎤': 'माइक', '🎧': 'हेडफोन', '📢': 'घोषणा', '📣': 'चिल्लाउने',
    
    # Miscellaneous Common
    '✅': 'ठीक', '☑️': 'जाँच', '💯': 'सय', '🆗': 'ठीक',
    '🆕': 'नयाँ', '🆓': 'मुक्त', '🔴': 'रातो', '🟢': 'हरियो',
}

# Emoji categories for feature extraction
HATE_RELATED_EMOJIS = {
    '😠', '😡', '🤬', '😈', '👿', '💢', '👊', '✊',
    '🔪', '💣', '🗡️', '⚔️', '💥', '🔫', '🧨', '☠️', '💀',
    '🖕', '🖕🏻', '🖕🏼', '🖕🏽', '🖕🏾', '🖕🏿',
    '👎', '👎🏻', '👎🏼', '👎🏽', '👎🏾', '👎🏿',
    '👹', '👺', '🤡', '🖤', '💔',
    '🐕', '🐖', '🐀', '🐍', '🦂', '🐒', '🐵', '🦍', '🐗',
    '💩', '😾',
}

MOCKERY_EMOJIS = {
    '😏', '😜', '😝', '😛', '🙄', '😤', '🙃',
    '😑', '😐', '😬', '🤨', '🤫', '🤭', '🤥',
    '🤡', '👻', '🎭',
}

POSITIVE_EMOJIS = {
    '😊', '😀', '😁', '😃', '😄', '☺️', '🙂', '😌', '🥰', '😍',
    '❤️', '🧡', '💛', '💚', '💙', '💜', '🤍', '🤎',
    '💕', '💞', '💓', '💗', '💖', '💘', '💝', '❣️',
    '👍', '👍🏻', '👍🏼', '👍🏽', '👍🏾', '👍🏿',
    '🙏', '👏', '🙌', '👌', '🤝', '✌️',
    '🥳', '🎉', '🎊', '⭐', '✨', '🌟',
}

SADNESS_EMOJIS = {
    '😭', '😢', '😿', '😔', '😞', '😒', '😓', '😟', '😕',
    '🙁', '☹️', '😩', '😫', '😖', '😣', '😥', '🥺',
}

FEAR_EMOJIS = {
    '😨', '😰', '😱', '😳', '🤯', '😵', '😲', '😯',
}

DISGUST_EMOJIS = {
    '🤢', '🤮', '🤧', '😷', '🤒', '🤕', '🥴',
}


# ============================================================================
# NORMALIZATION MAPPINGS
# ============================================================================

DIRGHIKARAN_MAP = {
    "\u200d": "",  # Zero-width joiner
    "\u200c": "",  # Zero-width non-joiner
    "।": ".",      # Devanagari danda
    "॥": ".",      # Double danda
}


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

ScriptType = Literal["devanagari", "romanized_nepali", "english", "mixed", "other"]


# ============================================================================
# EMOJI FEATURE EXTRACTION
# ============================================================================

def extract_emoji_features(text: str) -> Dict[str, int]:
    """
    Extract comprehensive emoji-based semantic features
    
    Returns 18 features:
    - 6 binary flags (has_X_emoji)
    - 6 count features (X_emoji_count)
    - 6 derived features (total, ratio, mixed_sentiment, unknown tracking)
    """
    emojis_found = [c for c in text if c in emoji.EMOJI_DATA]
    
    hate_count = sum(1 for e in emojis_found if e in HATE_RELATED_EMOJIS)
    mockery_count = sum(1 for e in emojis_found if e in MOCKERY_EMOJIS)
    positive_count = sum(1 for e in emojis_found if e in POSITIVE_EMOJIS)
    sadness_count = sum(1 for e in emojis_found if e in SADNESS_EMOJIS)
    fear_count = sum(1 for e in emojis_found if e in FEAR_EMOJIS)
    disgust_count = sum(1 for e in emojis_found if e in DISGUST_EMOJIS)
    
    # Track unknown emojis (not in our mapping)
    known_emojis = set(EMOJI_TO_NEPALI.keys())
    unknown_emojis = [e for e in emojis_found if e not in known_emojis]
    unknown_count = len(unknown_emojis)
    
    return {
        # Binary flags
        'has_hate_emoji': 1 if hate_count > 0 else 0,
        'has_mockery_emoji': 1 if mockery_count > 0 else 0,
        'has_positive_emoji': 1 if positive_count > 0 else 0,
        'has_sadness_emoji': 1 if sadness_count > 0 else 0,
        'has_fear_emoji': 1 if fear_count > 0 else 0,
        'has_disgust_emoji': 1 if disgust_count > 0 else 0,
        
        # Count features
        'hate_emoji_count': hate_count,
        'mockery_emoji_count': mockery_count,
        'positive_emoji_count': positive_count,
        'sadness_emoji_count': sadness_count,
        'fear_emoji_count': fear_count,
        'disgust_emoji_count': disgust_count,
        'total_emoji_count': len(emojis_found),
        
        # Derived features
        'hate_to_positive_ratio': hate_count / max(positive_count, 1),
        'has_mixed_sentiment': 1 if (hate_count > 0 and positive_count > 0) else 0,
        
        # NEW: Unknown emoji tracking
        'unknown_emoji_count': unknown_count,
        'has_unknown_emoji': 1 if unknown_count > 0 else 0,
        'known_emoji_ratio': (len(emojis_found) - unknown_count) / max(len(emojis_found), 1),
    }


def remove_emojis_for_detection(text: str) -> str:
    """Remove emojis temporarily for script detection"""
    return emoji.replace_emoji(text, replace="")


# ============================================================================
# SCRIPT DETECTION
# ============================================================================

def detect_script_type(text: str) -> Tuple[ScriptType, dict]:
    """
    Detect the dominant script type ignoring emojis
    
    Returns:
        Tuple of (script_type, detection_details)
    """
    if not text or not text.strip():
        return "other", {"confidence": 0.0, "reason": "empty_text"}
    
    # Remove emojis before detection
    text_no_emoji = remove_emojis_for_detection(text)
    
    if not text_no_emoji.strip():
        return "other", {"confidence": 0.5, "reason": "emoji_only"}
    
    letters = regex.findall(r"\p{L}", text_no_emoji)
    letter_count = len(letters)
    
    if letter_count == 0:
        return "other", {"confidence": 0.0, "reason": "no_letters"}
    
    devanagari_chars = regex.findall(r"\p{Devanagari}", text_no_emoji)
    dev_count = len(devanagari_chars)
    dev_ratio = dev_count / letter_count
    
    latin_chars = regex.findall(r"[a-zA-Z]", text_no_emoji)
    latin_count = len(latin_chars)
    latin_ratio = latin_count / letter_count
    
    # Romanized Nepali patterns
    romanized_nepali_patterns = [
        # Common words
        r'\b[xX]u\b', r'\b[xX]um?\b', r'\bhajur\b', r'\bdai\b', r'\bbhai\b', r'\bdidi\b',
        r'\bbahini\b', r'\bsanghai\b', r'\bsunu\b', r'\bhera\b', r'\bsun\b',
        
        # Particles & Postpositions
        r'\bko\b', r'\bki\b', r'\bka\b', r'\bho\b', r'\btyo\b', r'\byo\b', r'\bta\b',
        r'\bma\b', r'\bma?i\b', r'\bla[ie]?\b', r'\bnai?\b', r'\bpani\b', r'\bni\b',
        
        # Verbs
        r'\bhun[ae]\b', r'\bhunchha\b', r'\bhunuhunchha\b', r'\bgar\w+\b', r'\bgarna\b',
        r'\bx[ao]\b', r'\bxa\b', r'\bxan\b', r'\bxaina\b', r'\bxu\b',
        r'\bchain\b', r'\bchaina\b', r'\bthiy[oe]\b', r'\bhola\b', r'\bhos\b',
        r'\bbhan\w*\b', r'\bbol\w*\b', r'\bher\w*\b',
        
        # Common adjectives/states
        r'\bkh[ou]s[hi]?\b', r'\bkhusi\b', r'\bkhushi\b', r'\bramro\b', r'\bnaramro\b',
        r'\bthulo\b', r'\bsano\b', r'\brasilo\b', r'\bmitho\b', r'\btikhi\b',
        r'\bdherei\b', r'\baliali\b', r'\bastai\b', r'\blastai\b',
        
        # Question words
        r'\bkina\b', r'\bkasari\b', r'\bkahile\b', r'\bkaha[n]?\b', r'\bke\b', r'\bko\b',
        
        # Pronouns
        r'\bma\b', r'\btimi\b', r'\btapai\b', r'\buha\b', r'\buni\b', r'\byini\b',
        r'\bmero\b', r'\btimro\b', r'\buhako\b', r'\buniko\b', r'\bhamro\b',
        
        # Common nouns
        r'\bmanxe\b', r'\bmanchhe\b', r'\bmanche\b', r'\bharu\b', r'\bdes[ha]?\b',
        r'\bgha?r\b', r'\bthau\b', r'\bsamay\b', r'\bbela\b',
        
        # Nepali-specific endings (transliterated)
        r'\w+[ae]ko\b', r'\w+[ae]ki\b', r'\w+dai\b', r'\w+lai\b',
        r'\w+ma\b', r'\w+xa\b', r'\w+hun[ae]\b', r'\w+thiyo\b',
    ]
    
    romanized_indicators = sum(1 for pattern in romanized_nepali_patterns 
                               if re.search(pattern, text_no_emoji, re.IGNORECASE))
    
    # Calculate Romanized Nepali score
    romanized_score = 0.0
    if latin_ratio > 0.5 and dev_ratio < 0.3:
        if romanized_indicators > 0:
            romanized_score = min(0.5 + (romanized_indicators * 0.15), 0.95)
        else:
            # Check for typical Romanized Nepali patterns
            romanized_patterns = re.findall(r'\b\w*[aeiou](?:h)?\b', text_no_emoji.lower())
            if any(word.endswith(('xu', 'ro', 'no', 'lo', 'ko', 'ho')) 
                   for word in romanized_patterns):
                romanized_score = 0.4
            else:
                romanized_score = 0.3
    
    # English indicators (EXPANDED)
    english_indicators = [
        # Articles & Determiners
        'the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any', 'all', 'every',
        
        # Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
        'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
        
        # Common verbs (be, have, do)
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'done',
        'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
        
        # Common verbs (action)
        'get', 'got', 'go', 'went', 'gone', 'make', 'made', 'take', 'took', 'taken',
        'come', 'came', 'see', 'saw', 'seen', 'know', 'knew', 'known', 'say', 'said',
        'tell', 'told', 'think', 'thought', 'give', 'gave', 'given', 'find', 'found',
        
        # Question words
        'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
        
        # Prepositions
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'by', 'about', 'as',
        'into', 'through', 'over', 'under', 'after', 'before', 'between', 'among',
        
        # Conjunctions
        'and', 'or', 'but', 'so', 'yet', 'nor', 'because', 'if', 'when', 'while',
        'although', 'though', 'unless', 'since', 'until', 'where', 'whether',
        
        # Negations
        'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere', 'neither',
        
        # Common adjectives
        'good', 'bad', 'great', 'big', 'small', 'long', 'short', 'high', 'low',
        'old', 'new', 'young', 'early', 'late', 'right', 'wrong', 'true', 'false',
        'hot', 'cold', 'happy', 'sad', 'angry', 'nice', 'beautiful', 'ugly',
        
        # Sentiment words (hate speech relevant)
        'hate', 'love', 'like', 'dislike', 'stupid', 'dumb', 'idiot', 'fool',
        'kill', 'die', 'dead', 'death', 'fuck', 'shit', 'ass', 'damn', 'hell',
        'worst', 'terrible', 'horrible', 'awful', 'disgusting', 'pathetic',
        
        # Common nouns
        'man', 'woman', 'people', 'person', 'thing', 'time', 'day', 'year',
        'way', 'work', 'life', 'world', 'country', 'place', 'home', 'hand',
        
        # Very & Adverbs
        'very', 'really', 'quite', 'too', 'so', 'just', 'only', 'even', 'also',
        'well', 'much', 'more', 'most', 'less', 'least', 'still', 'already',
    ]
    english_words = [w.lower() for w in re.findall(r'\b\w+\b', text_no_emoji)]
    english_count = sum(1 for w in english_words if w in english_indicators)
    english_ratio = english_count / len(english_words) if english_words else 0
    
    # Detection details
    details = {
        "devanagari_count": dev_count,
        "devanagari_ratio": dev_ratio,
        "latin_count": latin_count,
        "latin_ratio": latin_ratio,
        "romanized_indicators": romanized_indicators,
        "english_ratio": english_ratio,
        "letter_count": letter_count
    }
    
    # Decision logic
    if dev_ratio >= 0.8:
        return "devanagari", {**details, "confidence": dev_ratio, "reason": "dominant_devanagari"}
    
    elif dev_ratio >= 0.4:
        return "mixed", {**details, "confidence": 0.7, "reason": "mixed_with_devanagari"}
    
    elif romanized_score > 0.5 and dev_ratio < 0.2:
        return "romanized_nepali", {**details, "confidence": romanized_score, "reason": "romanized_nepali_detected"}
    
    elif english_ratio > 0.2 and romanized_score < 0.4:
        return "english", {**details, "confidence": min(english_ratio + 0.3, 0.9), "reason": "english_detected"}
    
    elif latin_ratio > 0.5 and romanized_score > 0.3:
        return "romanized_nepali", {**details, "confidence": romanized_score, "reason": "likely_romanized_nepali"}
    
    elif latin_ratio > 0.8:
        if english_ratio > 0.1:
            return "english", {**details, "confidence": 0.6, "reason": "likely_english"}
        else:
            return "romanized_nepali", {**details, "confidence": 0.5, "reason": "ambiguous_latin_script"}
    
    else:
        return "other", {**details, "confidence": 0.3, "reason": "insufficient_indicators"}



# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

def clean_text_basic(text: str) -> str:
    """Basic text cleaning"""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtag symbol but keep text
    text = re.sub(r"#(\w+)", r"\1", text)
    # Remove quotes (single and double, including smart quotes)
    text = text.replace('"', '').replace("'", '').replace('"', '').replace('"', '').replace(''', '').replace(''', '')
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_devanagari(text: str) -> str:
    """Normalize Devanagari-specific characters"""
    for k, v in DIRGHIKARAN_MAP.items():
        text = text.replace(k, v)
    return text


def replace_emojis_semantic(text: str, preserve_spacing: bool = True, 
                           preserve_unknown: bool = True) -> str:
    """
    Replace emojis with Nepali text
    
    Args:
        text: Input text with emojis
        preserve_spacing: Add spaces around Nepali replacements
        preserve_unknown: Keep emojis not in EMOJI_TO_NEPALI mapping (default: True)
                         If False, unknown emojis will be removed
    
    Returns:
        Text with emojis replaced (or preserved)
    """
    # Replace known emojis with Nepali translations
    for emoji_char, nepali_text in EMOJI_TO_NEPALI.items():
        if preserve_spacing:
            text = text.replace(emoji_char, f" {nepali_text} ")
        else:
            text = text.replace(emoji_char, nepali_text)
    
    # Handle unknown emojis
    if not preserve_unknown:
        # OLD BEHAVIOR: Remove all remaining emojis
        text = emoji.replace_emoji(text, replace=" ")
    # else: NEW BEHAVIOR: Keep unknown emojis as-is
    # This allows the model to process them directly
    
    return text


def transliterate_romanized_nepali(text: str) -> str:
    """Transliterate Romanized Nepali to Devanagari"""
    if not TRANSLITERATION_AVAILABLE:
        return text
    
    try:
        result = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        return result if result else text
    except Exception as e:
        logger.warning(f"Transliteration failed: {e}")
        return text


# ============================================================================
# CACHED TRANSLATOR
# ============================================================================

class CachedNepaliTranslator:
    """Translator with LRU cache for efficiency"""
    
    def __init__(self, cache_size: int = 2000):
        self.translator = GoogleTranslator(source='en', target='ne')
        self.cache_size = cache_size
        self._translate_cached = lru_cache(maxsize=cache_size)(self._translate_single)
    
    def _translate_single(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        try:
            result = self.translator.translate(text.strip())
            return result if result else text
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return text
    
    def translate(self, text: str, fallback_to_original: bool = True) -> str:
        if not text or not text.strip():
            return ""
        try:
            return self._translate_cached(text.strip())
        except Exception as e:
            if fallback_to_original:
                logger.warning(f"Translation failed, using original: {str(e)}")
                return text
            raise
    
    def get_cache_info(self) -> dict:
        """Get cache statistics"""
        cache_info = self._translate_cached.cache_info()
        return {
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'size': cache_info.currsize,
            'max_size': cache_info.maxsize,
            'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) 
                       if (cache_info.hits + cache_info.misses) > 0 else 0.0
        }


def translate_latin_spans(text: str, translator: CachedNepaliTranslator) -> str:
    """Translate Latin word spans in Devanagari text"""
    def repl(match):
        latin_text = match.group(0)
        translated = translator.translate(latin_text, fallback_to_original=True)
        return f" {translated} "
    
    return re.sub(r"[A-Za-z][A-Za-z\s]{2,}", repl, text)


# ============================================================================
# MAIN PREPROCESSOR CLASS
# ============================================================================

class HateSpeechPreprocessor:
    """
    Main preprocessing pipeline for Nepali hate speech classification
    
    Pipeline:
    1. Extract emoji features (before any processing)
    2. Detect script type (ignoring emojis)
    3. Apply script-specific processing
    4. Replace emojis with Nepali text
    5. Normalize Devanagari
    """

    def __init__(
        self,
        model_type: Literal["xlmr", "mbert", "nepalibert"] = "xlmr",
        translate_english: bool = True,
        cache_size: int = 2000
    ):
        self.model_type = model_type
        self.translate_english = translate_english
        self.translator = CachedNepaliTranslator(cache_size) if translate_english else None

    def preprocess(self, text: str, verbose: bool = False) -> Tuple[str, Dict[str, int]]:
        """
        Preprocess a single text
        
        Args:
            text: Input text
            verbose: Print processing steps
        
        Returns:
            Tuple of (preprocessed_text, emoji_features)
        """
        if not isinstance(text, str) or not text.strip():
            return "", {
                'has_hate_emoji': 0, 'has_mockery_emoji': 0, 'has_positive_emoji': 0,
                'has_sadness_emoji': 0, 'has_fear_emoji': 0, 'has_disgust_emoji': 0,
                'hate_emoji_count': 0, 'mockery_emoji_count': 0, 'positive_emoji_count': 0,
                'sadness_emoji_count': 0, 'fear_emoji_count': 0, 'disgust_emoji_count': 0,
                'total_emoji_count': 0, 'hate_to_positive_ratio': 0.0, 'has_mixed_sentiment': 0
            }

        original_text = text
        
        # Step 1: Extract emoji features
        emoji_features = extract_emoji_features(original_text)
        
        # Step 2: Detect script type
        script_type, details = detect_script_type(text)
        
        if verbose:
            print(f"Script detected: {script_type} (confidence: {details.get('confidence', 0):.2%})")
        
        # Step 3: Basic cleaning
        text = clean_text_basic(text)
        
        # Step 4: Script-specific processing
        if script_type == "devanagari":
            processed = text
            if self.translate_english and self.translator:
                processed = translate_latin_spans(processed, self.translator)
        
        elif script_type == "romanized_nepali":
            processed = transliterate_romanized_nepali(text)
        
        elif script_type == "english":
            if self.translate_english and self.translator:
                processed = self.translator.translate(text, fallback_to_original=True)
            else:
                processed = text
        
        elif script_type == "mixed":
            processed = transliterate_romanized_nepali(text)
            if self.translate_english and self.translator:
                processed = translate_latin_spans(processed, self.translator)
        else:
            processed = text
        
        # Step 5: Replace emojis
        processed = replace_emojis_semantic(processed)
        
        # Step 6: Normalize
        final = normalize_devanagari(processed)
        final = re.sub(r"\s+", " ", final).strip()
        
        if verbose:
            print(f"Original: {original_text}")
            print(f"Processed: {final}")
            print(f"Emoji features: {emoji_features}")
        
        return final, emoji_features
    
    def preprocess_batch(self, texts: List[str], verbose: bool = False, show_progress: bool = False) -> Tuple[List[str], List[Dict[str, int]]]:
        """
        Preprocess multiple texts
        
        Args:
            texts: List of input texts
            verbose: Print processing steps for each text
            show_progress: Show progress bar (requires tqdm)
        
        Returns:
            Tuple of (preprocessed_texts, emoji_features_list)
        """
        if show_progress:
            try:
                from tqdm import tqdm
                results = [self.preprocess(text, verbose=verbose) for text in tqdm(texts, desc="Preprocessing")]
            except ImportError:
                results = [self.preprocess(text, verbose=verbose) for text in texts]
        else:
            results = [self.preprocess(text, verbose=verbose) for text in texts]
        
        texts_processed = [r[0] for r in results]
        features = [r[1] for r in results]
        return texts_processed, features
    
    def get_stats(self) -> dict:
        """Get preprocessor statistics"""
        stats = {
            'model_type': self.model_type,
            'translation_enabled': self.translate_english,
            'transliteration_available': TRANSLITERATION_AVAILABLE,
        }
        if self.translator:
            stats['cache_info'] = self.translator.get_cache_info()
        return stats


# ============================================================================
# CONVENIENCE FUNCTIONS FOR STREAMLIT
# ============================================================================

def preprocess_text(
    text: str,
    model_type: str = "xlmr",
    translate_english: bool = True,
    verbose: bool = False
) -> Tuple[str, Dict[str, int]]:
    """
    Quick preprocessing function for single text (Streamlit-friendly)
    
    Args:
        text: Input text
        model_type: Model type (xlmr, mbert, nepalibert)
        translate_english: Whether to translate English
        verbose: Print processing steps
    
    Returns:
        Tuple of (preprocessed_text, emoji_features)
    """
    preprocessor = HateSpeechPreprocessor(
        model_type=model_type,
        translate_english=translate_english
    )
    return preprocessor.preprocess(text, verbose=verbose)


def get_script_info(text: str) -> Dict[str, any]:
    """
    Get detailed script detection info (useful for Streamlit display)
    
    Returns:
        Dictionary with script type, confidence, and details
    """
    script_type, details = detect_script_type(text)
    return {
        'script_type': script_type,
        'confidence': details.get('confidence', 0),
        'details': details
    }


def get_emoji_info(text: str) -> Dict[str, Any]:
    """Get detailed information about emojis in text"""
    emojis_found = [c for c in text if c in emoji.EMOJI_DATA]
    known_emojis = set(EMOJI_TO_NEPALI.keys())
    unknown_emojis = [e for e in emojis_found if e not in known_emojis]
    known_emojis_found = [e for e in emojis_found if e in known_emojis]
    
    return {
        'emojis_found': emojis_found,
        'total_count': len(emojis_found),
        'known_emojis': known_emojis_found,
        'known_count': len(known_emojis_found),
        'unknown_emojis': unknown_emojis,
        'unknown_count': len(unknown_emojis),
        'coverage': len(known_emojis_found) / len(emojis_found) if emojis_found else 1.0
    }