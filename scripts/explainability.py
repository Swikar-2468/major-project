"""
Explainability Module - LIME & SHAP
===================================
Model-agnostic explainability for Nepali hate speech classification.

This module provides:
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Emoji-aware visualization with Nepali font support

Usage:
------
from scripts.explainability import LIMEExplainer, SHAPExplainer, create_explainer_wrapper

# Create model wrapper
wrapper = create_explainer_wrapper(model, tokenizer, label_encoder, preprocessor)

# LIME explanation
lime = LIMEExplainer(wrapper, nepali_font=font)
lime.explain_and_visualize(original_text, preprocessed_text, save_path="lime.png")

# SHAP explanation
shap_exp = SHAPExplainer(wrapper, nepali_font=font)
shap_exp.explain_and_visualize(original_text, preprocessed_text, save_path="shap.png")
"""

import os
import numpy as np
import torch
import re
import emoji
import regex
import warnings
warnings.filterwarnings("ignore")

from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Explainability libraries
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not installed. Install with: pip install lime")

try:
    import shap
    from shap import Explainer, maskers
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")


# ============================================================================
# MODEL WRAPPER CLASS
# ============================================================================

class ModelExplainerWrapper:
    """
    Wrapper class for model + preprocessing
    Makes model compatible with LIME/SHAP
    """
    
    def __init__(self, model, tokenizer, label_encoder, preprocessor, device=None):
        """
        Args:
            model: Trained model
            tokenizer: Model tokenizer
            label_encoder: Label encoder
            preprocessor: HateSpeechPreprocessor instance
            device: torch device (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = label_encoder.classes_.tolist()
        self.preprocessor = preprocessor
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
    
    def preprocess_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Preprocess text using the HateSpeechPreprocessor"""
        return self.preprocessor.preprocess(text, verbose=False)
    
    def predict_proba(self, texts):
        """
        Predict probabilities for texts
        
        Args:
            texts: Single text or list of texts (already preprocessed)
        
        Returns:
            numpy array of probabilities
        """
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist() if texts.ndim > 0 else [str(texts)]
        
        # Convert to strings and filter empty
        texts = [str(t).strip() for t in texts if str(t).strip()]
        
        if not texts:
            # Return uniform probabilities for empty input
            return np.ones((1, len(self.class_names))) / len(self.class_names)
        
        # Tokenize
        enc = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
        
        return probs.cpu().numpy()
    
    def predict_with_analysis(self, text: str) -> Dict:
        """
        Predict with full analysis
        
        Returns:
            Dictionary with original text, preprocessed text, predictions, etc.
        """
        # Preprocess
        preprocessed, emoji_features = self.preprocess_text(text)
        
        # Predict
        probs = self.predict_proba(preprocessed)[0]
        pred_idx = int(np.argmax(probs))
        
        return {
            "original_text": text,
            "preprocessed_text": preprocessed,
            "emoji_features": emoji_features,
            "predicted_label": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {label: float(prob) for label, prob in zip(self.class_names, probs)}
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def apply_nepali_font(ax, nepali_font: Optional[FontProperties] = None,
                     texts: Optional[list] = None, is_tick_labels: bool = True):
    """
    Apply Nepali font to Devanagari text while preserving emojis
    
    Args:
        ax: Matplotlib axes
        nepali_font: Nepali font properties
        texts: Text objects to apply font to (if not tick labels)
        is_tick_labels: Whether to apply to tick labels
    """
    if nepali_font is None:
        return
    
    if is_tick_labels or texts is None:
        for txt in ax.get_yticklabels():
            text_content = txt.get_text()
            # Only apply if has Devanagari AND no emojis
            has_devanagari = bool(regex.search(r'\p{Devanagari}', text_content))
            has_emoji = any(c in emoji.EMOJI_DATA for c in text_content)
            
            if has_devanagari and not has_emoji:
                txt.set_fontproperties(nepali_font)
                txt.set_fontsize(11)
    else:
        for txt in texts:
            text_content = txt.get_text()
            has_devanagari = bool(regex.search(r'\p{Devanagari}', text_content))
            has_emoji = any(c in emoji.EMOJI_DATA for c in text_content)
            
            if has_devanagari and not has_emoji:
                txt.set_fontproperties(nepali_font)


def create_display_text_with_emojis(original_text: str, preprocessed_text: str) -> Tuple[List[str], List[str]]:
    """
    Create aligned display tokens preserving emojis
    
    Handles multi-word emoji translations like: 😡 → "ठूलो रिस" (2 words)
    
    Args:
        original_text: Original text with emojis
        preprocessed_text: Preprocessed text (emojis replaced with Nepali)
    
    Returns:
        Tuple of (display_tokens, model_tokens)
    """
    from scripts.transformer_data_preprocessing import EMOJI_TO_NEPALI
    
    original_tokens = original_text.split()
    preprocessed_tokens = preprocessed_text.split()
    
    # Build emoji to word count mapping (how many words each emoji becomes)
    emoji_word_counts = {}
    for emoji_char, nepali_text in EMOJI_TO_NEPALI.items():
        word_count = len(nepali_text.split())
        emoji_word_counts[emoji_char] = word_count
    
    display_tokens = []
    model_tokens = []
    
    orig_idx = 0
    proc_idx = 0
    
    while orig_idx < len(original_tokens):
        orig_token = original_tokens[orig_idx]
        
        # Check if token contains emoji
        has_emoji = any(c in emoji.EMOJI_DATA for c in orig_token)
        
        if has_emoji:
            # Display: keep original emoji
            display_tokens.append(orig_token)
            
            # Model: use Nepali translation (may be multiple words!)
            # Count how many emojis in this token
            emojis_in_token = [c for c in orig_token if c in emoji.EMOJI_DATA]
            
            if emojis_in_token:
                # Calculate total words needed for all emojis in this token
                total_words_needed = sum(
                    emoji_word_counts.get(e, 1) for e in emojis_in_token
                )
                
                # Collect that many preprocessed tokens
                nepali_words = []
                for _ in range(total_words_needed):
                    if proc_idx < len(preprocessed_tokens):
                        nepali_words.append(preprocessed_tokens[proc_idx])
                        proc_idx += 1
                
                # Join them as the model token
                if nepali_words:
                    model_tokens.append(' '.join(nepali_words))
                else:
                    model_tokens.append(orig_token)
            else:
                # Shouldn't happen, but fallback
                if proc_idx < len(preprocessed_tokens):
                    model_tokens.append(preprocessed_tokens[proc_idx])
                    proc_idx += 1
                else:
                    model_tokens.append(orig_token)
        else:
            # No emoji: use preprocessed for both
            if proc_idx < len(preprocessed_tokens):
                display_tokens.append(preprocessed_tokens[proc_idx])
                model_tokens.append(preprocessed_tokens[proc_idx])
                proc_idx += 1
            else:
                display_tokens.append(orig_token)
                model_tokens.append(orig_token)
        
        orig_idx += 1
    
    # Handle remaining preprocessed tokens
    while proc_idx < len(preprocessed_tokens):
        token = preprocessed_tokens[proc_idx]
        display_tokens.append(token)
        model_tokens.append(token)
        proc_idx += 1
    
    return display_tokens, model_tokens


# ============================================================================
# LIME EXPLAINER
# ============================================================================

class LIMEExplainer:
    """LIME explainer with emoji support"""
    
    def __init__(self, model_wrapper: ModelExplainerWrapper, nepali_font: Optional[FontProperties] = None):
        """
        Args:
            model_wrapper: ModelExplainerWrapper instance
            nepali_font: Nepali font properties for visualization
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME not installed. Install with: pip install lime")
        
        self.model_wrapper = model_wrapper
        self.nepali_font = nepali_font
        self.explainer = LimeTextExplainer(
            class_names=model_wrapper.class_names,
            random_state=42
        )
    
    def explain(self, original_text: str, preprocessed_text: str, num_samples: int = 200) -> Dict:
        """
        Generate LIME explanation
        
        Args:
            original_text: Original text with emojis
            preprocessed_text: Preprocessed text for model
            num_samples: Number of samples for LIME
        
        Returns:
            Dictionary with explanation data
        """
        # Get LIME explanation
        exp = self.explainer.explain_instance(
            preprocessed_text,
            self.model_wrapper.predict_proba,
            num_samples=num_samples
        )
        
        # Get token weights
        token_weights = dict(exp.as_list())
        
        # Create aligned tokens
        display_tokens, model_tokens = create_display_text_with_emojis(
            original_text, preprocessed_text
        )
        
        # Map weights to display tokens
        word_scores = []
        for display_tok, model_tok in zip(display_tokens, model_tokens):
            score = 0.0
            for lime_token, weight in token_weights.items():
                if lime_token in model_tok or model_tok in lime_token:
                    score += weight
            word_scores.append((display_tok, score))
        
        # Merge multi-word emoji attributions
        word_scores = self._merge_multi_word_emojis(word_scores)
        
        return {
            'word_scores': word_scores,
            'display_tokens': display_tokens,
            'model_tokens': model_tokens,
            'lime_explanation': exp
        }
    
    def _merge_multi_word_emojis(self, word_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Merge attributions for multi-word emoji translations like: ठूलो रिस → 😡
        
        Args:
            word_scores: List of (word, score) tuples
        
        Returns:
            Merged list with multi-word emojis combined
        """
        from scripts.transformer_data_preprocessing import EMOJI_TO_NEPALI
        
        # Build set of multi-word phrases
        multi_word_phrases = {}
        for emoji_char, nepali_text in EMOJI_TO_NEPALI.items():
            if ' ' in nepali_text:
                multi_word_phrases[nepali_text] = emoji_char
        
        # Merge consecutive words that form multi-word emoji phrases
        merged_scores = []
        i = 0
        while i < len(word_scores):
            word, score = word_scores[i]
            
            # Check if this word + next word(s) form a multi-word emoji phrase
            merged = False
            for phrase, emoji_char in multi_word_phrases.items():
                phrase_words = phrase.split()
                if i + len(phrase_words) <= len(word_scores):
                    # Check if consecutive words match the phrase
                    candidate_words = [word_scores[i + j][0] for j in range(len(phrase_words))]
                    candidate_phrase = ' '.join(candidate_words)
                    
                    if candidate_phrase == phrase:
                        # Found a multi-word emoji phrase - merge their scores
                        total_score = sum(word_scores[i + j][1] for j in range(len(phrase_words)))
                        
                        merged_scores.append((emoji_char, total_score))
                        i += len(phrase_words)  # Skip all words in the phrase
                        merged = True
                        break
            
            if not merged:
                merged_scores.append((word, score))
                i += 1
        
        return merged_scores
    
    def visualize(self, word_scores: List[Tuple[str, float]], save_path: Optional[str] = None,
                 show: bool = True, figsize: Tuple[int, int] = None):
        """
        Visualize LIME explanation
        
        Args:
            word_scores: List of (word, score) tuples
            save_path: Path to save figure
            show: Whether to display figure
            figsize: Figure size (auto if None)
        
        Returns:
            matplotlib figure
        """
        if not word_scores:
            print("⚠️ No words to visualize")
            return None
        
        # Replace underscores with spaces for display
        word_scores_display = [(w.replace('_', ' '), score) for w, score in word_scores]
        
        features, weights = zip(*word_scores_display)
        y_pos = range(len(features))
        
        if figsize is None:
            figsize = (10, max(6, len(features) * 0.4))
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = ['red' if w < 0 else 'green' for w in weights]
        ax.barh(y_pos, weights, color=colors, alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution to Prediction", fontsize=12)
        ax.set_title("LIME Feature Importance (Red=Against, Green=For)", fontsize=14)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        # Apply Nepali font
        apply_nepali_font(ax, self.nepali_font)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ LIME visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def explain_and_visualize(self, original_text: str, preprocessed_text: str,
                            save_path: Optional[str] = None, show: bool = True,
                            num_samples: int = 200):
        """
        Explain and visualize in one step
        
        Args:
            original_text: Original text with emojis
            preprocessed_text: Preprocessed text for model
            save_path: Path to save figure
            show: Whether to display figure
            num_samples: Number of LIME samples
        
        Returns:
            Dictionary with explanation and figure
        """
        # Generate explanation
        explanation = self.explain(original_text, preprocessed_text, num_samples)
        
        # Visualize
        fig = self.visualize(explanation['word_scores'], save_path, show)
        
        return {
            'explanation': explanation,
            'figure': fig
        }


# ============================================================================
# SHAP EXPLAINER
# ============================================================================

class SHAPExplainer:
    """SHAP explainer with emoji support and fallback methods"""
    
    def __init__(self, model_wrapper: ModelExplainerWrapper, nepali_font: Optional[FontProperties] = None):
        """
        Args:
            model_wrapper: ModelExplainerWrapper instance
            nepali_font: Nepali font properties for visualization
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        self.model_wrapper = model_wrapper
        self.nepali_font = nepali_font
    
    def explain(self, original_text: str, preprocessed_text: str, use_fallback: bool = True) -> Dict:
        """
        Generate SHAP explanation
        
        Args:
            original_text: Original text with emojis
            preprocessed_text: Preprocessed text for model
            use_fallback: Use fallback method if SHAP fails
        
        Returns:
            Dictionary with explanation data
        """
        try:
            # Try SHAP with text masker
            def predict_masked(masked_texts):
                if isinstance(masked_texts, np.ndarray):
                    if masked_texts.ndim == 1:
                        texts = [' '.join(str(t) for t in masked_texts if str(t).strip())]
                    else:
                        texts = [' '.join(str(t) for t in row if str(t).strip()) for row in masked_texts]
                elif isinstance(masked_texts, str):
                    texts = [masked_texts]
                elif isinstance(masked_texts, list):
                    texts = masked_texts
                else:
                    texts = [str(masked_texts)]
                
                return self.model_wrapper.predict_proba(texts)
            
            explainer = Explainer(predict_masked, maskers.Text(preprocessed_text))
            sv = explainer([preprocessed_text])[0]
            
            shap_tokens = list(sv.data)
            values_array = np.array(sv.values)
            
            # Validate that we got meaningful results
            if len(shap_tokens) == 0 or values_array.size == 0:
                raise ValueError("SHAP returned empty results")
            
            method_used = "shap"
            
        except Exception as e:
            if not use_fallback:
                raise e
            
            # Use fallback silently (only show in debug mode)
            import logging
            logging.debug(f"SHAP failed: {e}, using gradient fallback")
            
            shap_tokens, values_array = self._gradient_based_attribution(preprocessed_text)
            method_used = "gradient"
        
        # Get predicted class
        pred_probs = self.model_wrapper.predict_proba([preprocessed_text])[0]
        class_idx = int(np.argmax(pred_probs))
        
        # Extract values for predicted class
        if values_array.ndim == 1:
            token_values = values_array
        elif values_array.ndim == 2:
            token_values = values_array[:, class_idx]
        elif values_array.ndim == 3:
            token_values = values_array[0, :, class_idx]
        else:
            token_values = values_array.flatten()[:len(shap_tokens)]
        
        # Create aligned tokens
        display_tokens, model_tokens = create_display_text_with_emojis(
            original_text, preprocessed_text
        )
        
        # Map SHAP values to display tokens
        word_scores = self._align_shap_values(
            display_tokens, model_tokens, shap_tokens, token_values
        )
        
        # Merge multi-word emoji attributions
        word_scores = self._merge_multi_word_emojis(word_scores)
        
        return {
            'word_scores': word_scores,
            'display_tokens': display_tokens,
            'model_tokens': model_tokens,
            'shap_tokens': shap_tokens,
            'token_values': token_values,
            'class_idx': class_idx,
            'method_used': method_used
        }
    
    def _gradient_based_attribution(self, text: str) -> Tuple[List[str], np.ndarray]:
        """
        Fallback: Word-level attribution using occlusion
        
        Masks each word and measures prediction change
        """
        words = text.split()
        base_probs = self.model_wrapper.predict_proba([text])[0]
        base_pred_idx = int(np.argmax(base_probs))
        base_score = base_probs[base_pred_idx]
        
        attributions = []
        for i in range(len(words)):
            # Mask the word
            masked_words = words[:i] + words[i+1:]
            masked_text = ' '.join(masked_words)
            
            if not masked_text.strip():
                attributions.append(base_score)
                continue
            
            # Get prediction without this word
            masked_probs = self.model_wrapper.predict_proba([masked_text])[0]
            masked_score = masked_probs[base_pred_idx]
            
            # Attribution = score drop when word removed
            attribution = base_score - masked_score
            attributions.append(attribution)
        
        # Ensure we have at least one attribution
        if len(attributions) == 0:
            attributions = [0.0] * len(words)
        
        return words, np.array(attributions)
    
    def _merge_multi_word_emojis(self, word_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Merge attributions for multi-word emoji translations like: ठूलो रिस → 😡
        
        Args:
            word_scores: List of (word, score) tuples
        
        Returns:
            Merged list with multi-word emojis combined
        """
        from scripts.transformer_data_preprocessing import EMOJI_TO_NEPALI
        
        # Build set of multi-word phrases
        multi_word_phrases = {}
        for emoji_char, nepali_text in EMOJI_TO_NEPALI.items():
            if ' ' in nepali_text:
                multi_word_phrases[nepali_text] = emoji_char
        
        # Merge consecutive words that form multi-word emoji phrases
        merged_scores = []
        i = 0
        while i < len(word_scores):
            word, score = word_scores[i]
            
            # Check if this word + next word(s) form a multi-word emoji phrase
            merged = False
            for phrase, emoji_char in multi_word_phrases.items():
                phrase_words = phrase.split()
                if i + len(phrase_words) <= len(word_scores):
                    # Check if consecutive words match the phrase
                    candidate_words = [word_scores[i + j][0] for j in range(len(phrase_words))]
                    candidate_phrase = ' '.join(candidate_words)
                    
                    if candidate_phrase == phrase:
                        # Found a multi-word emoji phrase - merge their scores
                        total_score = sum(word_scores[i + j][1] for j in range(len(phrase_words)))
                        
                        merged_scores.append((emoji_char, total_score))
                        i += len(phrase_words)  # Skip all words in the phrase
                        merged = True
                        break
            
            if not merged:
                merged_scores.append((word, score))
                i += 1
        
        return merged_scores
    
    def _align_shap_values(self, display_tokens: List[str], model_tokens: List[str],
                          shap_tokens: List[str], token_values: np.ndarray) -> List[Tuple[str, float]]:
        """Align SHAP values with display tokens"""
        word_scores = []
        
        if len(display_tokens) == len(model_tokens):
            # Direct alignment
            for display_tok, model_tok in zip(display_tokens, model_tokens):
                score = 0.0
                for j, shap_tok in enumerate(shap_tokens):
                    if j < len(token_values) and (shap_tok in model_tok or model_tok in shap_tok):
                        score += float(token_values[j])
                word_scores.append((display_tok, score))
        else:
            # Fallback: distribute evenly
            for display_tok in display_tokens:
                score = np.mean(token_values) if len(token_values) > 0 else 0.0
                word_scores.append((display_tok, score))
        
        return word_scores
    
    def visualize(self, word_scores: List[Tuple[str, float]], class_name: str,
                 save_path: Optional[str] = None, show: bool = True,
                 figsize: Tuple[int, int] = None):
        """
        Visualize SHAP explanation with highlighted text
        
        Args:
            word_scores: List of (word, score) tuples
            class_name: Predicted class name
            save_path: Path to save figure
            show: Whether to display figure
            figsize: Figure size (auto if None)
        
        Returns:
            matplotlib figure
        """
        if not word_scores:
            print("⚠️ No words to visualize")
            return None
        
        # Safe max calculation with fallback
        abs_vals = [abs(v) for _, v in word_scores]
        if not abs_vals or all(v == 0 for v in abs_vals):
            max_val = 1.0  # Default to 1.0 if all values are zero
        else:
            max_val = max(abs_vals) + 1e-6
        
        if figsize is None:
            figsize = (max(10, 0.5 * len(word_scores)), 3)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        
        x, y = 0.01, 0.5
        text_objs = []
        
        for word, val in word_scores:
            # Replace underscores with spaces for display
            display_word = word.replace('_', ' ')
            
            # Color intensity
            intensity = min(abs(val) / max_val, 1.0)
            
            # Red=negative, Green=positive
            if val < 0:
                color = (1.0, 1.0 - intensity * 0.7, 1.0 - intensity * 0.7)
            else:
                color = (1.0 - intensity * 0.7, 1.0, 1.0 - intensity * 0.7)
            
            txt = ax.text(
                x, y, f" {display_word} ",
                fontsize=14,
                bbox=dict(
                    facecolor=color,
                    edgecolor='gray',
                    alpha=0.8,
                    boxstyle="round,pad=0.4"
                )
            )
            text_objs.append(txt)
            
            # Update position (emojis take less space)
            char_width = 0.025 if any(c in emoji.EMOJI_DATA for c in display_word) else 0.04
            x += char_width * len(display_word) + 0.01
            
            if x > 0.92:
                x = 0.01
                y -= 0.35
        
        # Apply Nepali font
        apply_nepali_font(ax, self.nepali_font, texts=text_objs, is_tick_labels=False)
        
        ax.text(0.5, 0.95, f"SHAP Explanation (Predicted: {class_name})",
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ SHAP visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def explain_and_visualize(self, original_text: str, preprocessed_text: str,
                            save_path: Optional[str] = None, show: bool = True,
                            use_fallback: bool = True):
        """
        Explain and visualize in one step
        
        Args:
            original_text: Original text with emojis
            preprocessed_text: Preprocessed text for model
            save_path: Path to save figure
            show: Whether to display figure
            use_fallback: Use fallback if SHAP fails
        
        Returns:
            Dictionary with explanation and figure
        """
        # Generate explanation
        explanation = self.explain(original_text, preprocessed_text, use_fallback)
        
        # Get class name
        class_name = self.model_wrapper.class_names[explanation['class_idx']]
        
        # Visualize
        fig = self.visualize(explanation['word_scores'], class_name, save_path, show)
        
        return {
            'explanation': explanation,
            'figure': fig
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_explainer_wrapper(model, tokenizer, label_encoder, preprocessor, device=None):
    """
    Convenience function to create model wrapper
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        label_encoder: Label encoder
        preprocessor: HateSpeechPreprocessor instance
        device: torch device (auto if None)
    
    Returns:
        ModelExplainerWrapper instance
    """
    return ModelExplainerWrapper(model, tokenizer, label_encoder, preprocessor, device)


def explain_prediction(text: str, model_wrapper: ModelExplainerWrapper,
                      method: str = "both", nepali_font: Optional[FontProperties] = None,
                      save_dir: Optional[str] = None, show: bool = True) -> Dict:
    """
    Explain a prediction using LIME and/or SHAP
    
    Args:
        text: Input text
        model_wrapper: ModelExplainerWrapper instance
        method: "lime", "shap", or "both"
        nepali_font: Nepali font for visualization
        save_dir: Directory to save figures
        show: Whether to display figures
    
    Returns:
        Dictionary with explanations and figures
    """
    # Get analysis
    analysis = model_wrapper.predict_with_analysis(text)
    original_text = analysis['original_text']
    preprocessed_text = analysis['preprocessed_text']
    
    results = {
        'analysis': analysis,
        'lime': None,
        'shap': None
    }
    
    # LIME
    if method in ["lime", "both"] and LIME_AVAILABLE:
        lime = LIMEExplainer(model_wrapper, nepali_font)
        save_path = os.path.join(save_dir, f"lime_{abs(hash(text)) % 10**8}.png") if save_dir else None
        results['lime'] = lime.explain_and_visualize(
            original_text, preprocessed_text, save_path, show
        )
    
    # SHAP
    if method in ["shap", "both"] and SHAP_AVAILABLE:
        shap_exp = SHAPExplainer(model_wrapper, nepali_font)
        save_path = os.path.join(save_dir, f"shap_{abs(hash(text)) % 10**8}.png") if save_dir else None
        results['shap'] = shap_exp.explain_and_visualize(
            original_text, preprocessed_text, save_path, show
        )
    
    return results


# ============================================================================
# AVAILABILITY CHECK
# ============================================================================

def check_availability() -> Dict[str, bool]:
    """Check which explainability methods are available"""
    return {
        'lime': LIME_AVAILABLE,
        'shap': SHAP_AVAILABLE
    }