"""
Captum Explainer Module
========================
Gradient-based explainability using Captum's Integrated Gradients.

This module provides:
- Layer Integrated Gradients attribution
- Token-level importance visualization
- Emoji-aware visualization with Nepali font support
- Heatmap and bar chart visualizations

Usage:
------
from scripts.captum_explainer import CaptumExplainer, explain_with_captum

# Create explainer
explainer = CaptumExplainer(model, tokenizer, label_encoder, preprocessor)

# Explain prediction
result = explainer.explain(
    original_text="Your text here",
    n_steps=50,
    nepali_font=font
)

# Visualize
explainer.visualize_bar_chart(result, save_path="ig_bar.png")
explainer.visualize_heatmap(result, save_path="ig_heatmap.png")

# All-in-one
result = explainer.explain_and_visualize(
    original_text="Your text",
    save_dir="./explanations",
    show=True
)
"""

import os
import numpy as np
import torch
import re
import emoji
import regex
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Optional
from matplotlib import pyplot as plt, cm
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors

# Captum
try:
    from captum.attr import LayerIntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠️ Captum not installed. Install with: pip install captum")


# ============================================================================
# TOKEN ALIGNMENT WITH EMOJI PRESERVATION
# ============================================================================

def create_display_tokens_from_subwords(
    original_text: str,
    preprocessed_text: str,
    tokenizer_tokens: List[str],
    emoji_to_nepali_map: Dict[str, str],
    remove_special: bool = True
) -> List[str]:
    """
    Create display tokens that preserve emojis from original text
    
    Maps preprocessed tokens (with emoji translations) back to original tokens (with actual emojis)
    
    Args:
        original_text: Original text with emojis (e.g., "तेरी कसम 😀😀")
        preprocessed_text: Preprocessed text (e.g., "तेरी कसम खुशी खुशी")
        tokenizer_tokens: Tokenized output from model
        emoji_to_nepali_map: Emoji to Nepali mapping dictionary
        remove_special: Whether to remove special tokens
    
    Returns:
        List of display tokens with emojis preserved (e.g., ["तेरी", "कसम", "😀", "😀"])
    """
    # Build reverse emoji mapping (Nepali text → emoji)
    reverse_emoji_map = {nepali: emoji_char for emoji_char, nepali in emoji_to_nepali_map.items()}
    
    # Clean and group tokenizer output into words
    word_pieces = []
    current_word = ""
    
    for tok in tokenizer_tokens:
        # Skip special tokens if requested
        if remove_special and tok in ['<s>', '</s>', '[CLS]', '[SEP]', '<pad>', '[PAD]']:
            continue
        
        if tok.startswith("▁"):
            # New word
            if current_word:
                word_pieces.append(current_word)
            current_word = tok.replace("▁", "")
        else:
            # Continue current word
            current_word += tok.replace("▁", "")
    
    if current_word:
        word_pieces.append(current_word)
    
    # Get original words
    original_words = original_text.split()
    
    # Map word_pieces back to original with emojis
    display_tokens = []
    orig_idx = 0
    
    for word in word_pieces:
        # Check if this word is an emoji translation
        if word in reverse_emoji_map:
            # This is a Nepali emoji translation → use the actual emoji
            display_tokens.append(reverse_emoji_map[word])
        else:
            # Regular word - try to match with original
            matched = False
            
            # Look for matching word in original
            while orig_idx < len(original_words):
                orig_word = original_words[orig_idx]
                
                # Skip emojis in original (they're handled by reverse_emoji_map)
                if any(c in emoji.EMOJI_DATA for c in orig_word):
                    orig_idx += 1
                    continue
                
                # Check if words match
                orig_clean = emoji.replace_emoji(orig_word, replace="").strip()
                if orig_clean and (word in orig_clean or orig_clean in word or word == orig_clean):
                    display_tokens.append(orig_word)
                    matched = True
                    orig_idx += 1
                    break
                
                orig_idx += 1
            
            if not matched:
                # Couldn't match - use the word as-is
                display_tokens.append(word)
    
    return display_tokens


# ============================================================================
# FONT HANDLING
# ============================================================================

def apply_nepali_font(ax_or_text, nepali_font: Optional[FontProperties] = None,
                     is_axis: bool = True):
    """
    Apply Nepali font to text containing Devanagari
    
    Args:
        ax_or_text: Matplotlib axis or text object
        nepali_font: Nepali font properties
        is_axis: Whether ax_or_text is an axis (True) or text object (False)
    """
    if nepali_font is None:
        return
    
    if is_axis:
        # Apply to axis tick labels
        for lbl in ax_or_text.get_xticklabels():
            text_content = lbl.get_text()
            if regex.search(r'\p{Devanagari}', text_content):
                lbl.set_fontproperties(nepali_font)
                lbl.set_fontsize(11)
    else:
        # Apply to single text object
        text_content = ax_or_text.get_text()
        if regex.search(r'\p{Devanagari}', text_content):
            ax_or_text.set_fontproperties(nepali_font)


# ============================================================================
# CAPTUM EXPLAINER CLASS
# ============================================================================

class CaptumExplainer:
    """
    Captum Integrated Gradients explainer with emoji support
    """
    
    def __init__(self, model, tokenizer, label_encoder, preprocessor, 
                 emoji_to_nepali_map: Optional[Dict[str, str]] = None,
                 device=None, max_length: int = 256):
        """
        Args:
            model: Trained model
            tokenizer: Model tokenizer
            label_encoder: Label encoder
            preprocessor: HateSpeechPreprocessor instance
            emoji_to_nepali_map: Emoji to Nepali mapping (optional)
            device: torch device (auto-detected if None)
            max_length: Maximum sequence length
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum not installed. Install with: pip install captum")
        
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.preprocessor = preprocessor
        self.class_names = label_encoder.classes_.tolist()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.emoji_to_nepali_map = emoji_to_nepali_map or {}
        
        self.model.to(self.device).eval()
        
        # Get embedding layer (model-specific)
        self.embedding_layer = self._get_embedding_layer()
    
    def _get_embedding_layer(self):
        """Get the embedding layer from the model"""
        # Try different model architectures
        if hasattr(self.model, 'roberta'):
            # XLM-RoBERTa
            return self.model.roberta.embeddings.word_embeddings
        elif hasattr(self.model, 'bert'):
            # BERT-based
            return self.model.bert.embeddings.word_embeddings
        elif hasattr(self.model, 'transformer'):
            # Generic transformer
            return self.model.transformer.wte
        else:
            raise AttributeError("Could not find embedding layer. Please specify manually.")
    
    def explain(self, original_text: str, target: Optional[int] = None, 
               n_steps: int = 50) -> Dict:
        """
        Generate Integrated Gradients explanation
        
        Args:
            original_text: Original text with emojis
            target: Target class index (None = predicted class)
            n_steps: Number of IG steps
        
        Returns:
            Dictionary with explanation results
        """
        # Preprocess
        preprocessed, emoji_features = self.preprocessor.preprocess(original_text, verbose=False)
        
        if not preprocessed:
            raise ValueError("Preprocessing resulted in empty text")
        
        # Tokenize
        encoding = self.tokenizer(
            preprocessed,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_label = self.class_names[pred_idx]
            pred_conf = float(probs[pred_idx])
        
        if target is None:
            target = pred_idx
        
        # Forward function for Captum
        def forward_func(input_ids_arg, attention_mask_arg):
            """Forward function that takes input_ids"""
            return self.model(input_ids=input_ids_arg, attention_mask=attention_mask_arg).logits[:, target]
        
        # Initialize Integrated Gradients
        lig = LayerIntegratedGradients(forward_func, self.embedding_layer)
        
        # Baseline: all pad tokens
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        
        # Calculate attributions
        attributions, delta = lig.attribute(
            input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True,
            n_steps=n_steps
        )
        
        # Sum across embedding dimension
        attributions_sum = attributions.sum(dim=-1).squeeze(0)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].cpu().tolist(),
            skip_special_tokens=False
        )
        
        # Create display tokens with emojis preserved
        display_tokens = create_display_tokens_from_subwords(
            original_text,
            preprocessed,
            tokens,
            self.emoji_to_nepali_map,
            remove_special=True
        )
        
        # Aggregate word-level attributions
        word_attributions = self._aggregate_word_attributions(
            tokens, attributions_sum, display_tokens
        )
        
        return {
            "original_text": original_text,
            "preprocessed_text": preprocessed,
            "emoji_features": emoji_features,
            "predicted_label": pred_label,
            "predicted_index": pred_idx,
            "confidence": pred_conf,
            "probabilities": {label: float(prob) for label, prob in zip(self.class_names, probs)},
            "word_attributions": word_attributions,
            "convergence_delta": float(delta.sum().cpu().numpy()),
            "tokens": tokens,
            "display_tokens": display_tokens
        }
    
    def _aggregate_word_attributions(self, tokens: List[str], attributions_sum: torch.Tensor,
                                    display_tokens: List[str]) -> List[Tuple[str, float, float]]:
        """
        Aggregate subword attributions to word-level
        
        Returns:
            List of (word, abs_score, signed_score) tuples
        """
        word_attributions = []
        current_indices = []
        
        for i, tok in enumerate(tokens):
            # Skip special tokens
            if tok in ['<s>', '</s>', '[CLS]', '[SEP]', '<pad>', '[PAD]']:
                continue
            
            if tok.startswith("▁"):
                # New word starts
                if current_indices:
                    # Save previous word
                    grp_vals = attributions_sum[current_indices].detach().cpu().numpy()
                    score = float(np.sum(np.abs(grp_vals)))
                    signed_score = float(np.sum(grp_vals))
                    word = "".join([tokens[j].replace("▁", "") for j in current_indices])
                    word_attributions.append((word, score, signed_score))
                
                current_indices = [i]
            else:
                # Continue current word
                current_indices.append(i)
        
        # Don't forget last word
        if current_indices:
            grp_vals = attributions_sum[current_indices].detach().cpu().numpy()
            score = float(np.sum(np.abs(grp_vals)))
            signed_score = float(np.sum(grp_vals))
            word = "".join([tokens[j].replace("▁", "") for j in current_indices])
            word_attributions.append((word, score, signed_score))
        
        # Align with display tokens
        if len(display_tokens) == len(word_attributions):
            aligned_attributions = [
                (display_tok, score, signed_score)
                for display_tok, (_, score, signed_score) in zip(display_tokens, word_attributions)
            ]
        else:
            aligned_attributions = word_attributions
        
        return aligned_attributions
    
    def visualize_bar_chart(self, explanation: Dict, save_path: Optional[str] = None,
                           show: bool = True, nepali_font: Optional[FontProperties] = None,
                           figsize: Tuple[int, int] = None):
        """
        Create bar chart visualization
        
        Args:
            explanation: Explanation dictionary from explain()
            save_path: Path to save figure
            show: Whether to display figure
            nepali_font: Nepali font properties
            figsize: Figure size (auto if None)
        
        Returns:
            matplotlib figure
        """
        word_attributions = explanation['word_attributions']
        pred_label = explanation['predicted_label']
        pred_conf = explanation['confidence']
        
        scores = [s for _, s, _ in word_attributions]
        words = [w for w, _, _ in word_attributions]
        signed_scores = [ss for _, _, ss in word_attributions]
        
        if figsize is None:
            figsize = (max(8, 0.6 * len(words)), 5)
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = ['green' if ss > 0 else 'red' for ss in signed_scores]
        
        ax.bar(range(len(words)), scores, tick_label=words, color=colors, alpha=0.7)
        ax.set_ylabel("Attribution (sum abs)", fontsize=12)
        ax.set_title(
            f"Integrated Gradients → Pred: {pred_label} ({pred_conf:.2%})",
            fontsize=14,
            fontweight='bold'
        )
        
        # Apply Nepali font
        if nepali_font:
            apply_nepali_font(ax, nepali_font, is_axis=True)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Bar chart saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def visualize_heatmap(self, explanation: Dict, save_path: Optional[str] = None,
                         show: bool = True, nepali_font: Optional[FontProperties] = None,
                         figsize: Tuple[int, int] = None):
        """
        Create heatmap visualization with colored text boxes
        
        Args:
            explanation: Explanation dictionary from explain()
            save_path: Path to save figure
            show: Whether to display figure
            nepali_font: Nepali font properties
            figsize: Figure size (auto if None)
        
        Returns:
            matplotlib figure
        """
        word_attributions = explanation['word_attributions']
        pred_label = explanation['predicted_label']
        
        scores = [s for _, s, _ in word_attributions]
        max_score = max(scores) if scores else 1.0
        
        cmap = cm.get_cmap("RdYlGn")
        
        if figsize is None:
            figsize = (max(10, 0.6 * len(word_attributions)), 3)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        x, y = 0.01, 0.6
        text_objs = []
        
        for word, score, signed_score in word_attributions:
            # Normalize for color
            intensity = min(score / max_score, 1.0) if max_score > 0 else 0.0
            
            # Color based on signed score
            if signed_score > 0:
                color = cmap(0.5 + intensity * 0.5)  # Green side
            else:
                color = cmap(0.5 - intensity * 0.5)  # Red side
            
            txt = ax.text(
                x, y, f" {word} ",
                fontsize=13,
                bbox=dict(
                    facecolor=mcolors.to_hex(color),
                    alpha=0.8,
                    boxstyle="round,pad=0.3",
                    edgecolor='gray'
                )
            )
            
            # Apply Nepali font only to Devanagari text
            if nepali_font and regex.search(r'\p{Devanagari}', word):
                txt.set_fontproperties(nepali_font)
            
            text_objs.append(txt)
            
            # Update position - emojis take less horizontal space
            char_width = 0.025 if any(c in emoji.EMOJI_DATA for c in word) else 0.04
            x += char_width * len(word) + 0.01
            
            if x > 0.92:
                x = 0.01
                y -= 0.35
        
        # Title
        ax.text(
            0.5, 0.95,
            f"Token Attributions (Predicted: {pred_label})",
            ha='center',
            va='top',
            fontsize=14,
            fontweight='bold',
            transform=ax.transAxes
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmap saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def explain_and_visualize(self, original_text: str, target: Optional[int] = None,
                            n_steps: int = 50, save_dir: Optional[str] = None,
                            show: bool = True, nepali_font: Optional[FontProperties] = None):
        """
        Explain and visualize in one step
        
        Args:
            original_text: Original text with emojis
            target: Target class index (None = predicted)
            n_steps: Number of IG steps
            save_dir: Directory to save figures
            show: Whether to display figures
            nepali_font: Nepali font properties
        
        Returns:
            Dictionary with explanation and figures
        """
        # Generate explanation
        explanation = self.explain(original_text, target, n_steps)
        
        # Generate file paths if save_dir provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            hash_suffix = abs(hash(original_text)) % 10**8
            bar_path = os.path.join(save_dir, f"ig_bar_{explanation['predicted_label']}_{hash_suffix}.png")
            heatmap_path = os.path.join(save_dir, f"ig_heatmap_{explanation['predicted_label']}_{hash_suffix}.png")
        else:
            bar_path = None
            heatmap_path = None
        
        # Visualize
        bar_fig = self.visualize_bar_chart(explanation, bar_path, show, nepali_font)
        heatmap_fig = self.visualize_heatmap(explanation, heatmap_path, show, nepali_font)
        
        return {
            'explanation': explanation,
            'bar_chart': bar_fig,
            'heatmap': heatmap_fig
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def explain_with_captum(text: str, model, tokenizer, label_encoder, preprocessor,
                       emoji_to_nepali_map: Optional[Dict[str, str]] = None,
                       n_steps: int = 50, nepali_font: Optional[FontProperties] = None,
                       save_dir: Optional[str] = None, show: bool = True) -> Dict:
    """
    Convenience function to explain a text with Captum
    
    Args:
        text: Input text
        model: Trained model
        tokenizer: Model tokenizer
        label_encoder: Label encoder
        preprocessor: HateSpeechPreprocessor instance
        emoji_to_nepali_map: Emoji mapping dictionary
        n_steps: Number of IG steps
        nepali_font: Nepali font properties
        save_dir: Directory to save figures
        show: Whether to display figures
    
    Returns:
        Dictionary with explanation and visualizations
    """
    explainer = CaptumExplainer(
        model, tokenizer, label_encoder, preprocessor,
        emoji_to_nepali_map=emoji_to_nepali_map
    )
    
    return explainer.explain_and_visualize(
        text, n_steps=n_steps, save_dir=save_dir, show=show, nepali_font=nepali_font
    )


def check_availability() -> bool:
    """Check if Captum is available"""
    return CAPTUM_AVAILABLE


# ============================================================================
# DEFAULT EMOJI MAPPING (For standalone usage)
# ============================================================================

DEFAULT_EMOJI_TO_NEPALI = {
    '😀': 'खुशी', '😁': 'खुशी', '😂': 'हाँसो', '😃': 'खुशी', '😄': 'खुशी',
    '😅': 'नर्भस हाँसो', '😆': 'हाँसो', '😊': 'मुस्कान', '😍': 'माया',
    '😠': 'रिस', '😡': 'ठूलो रिस', '🤬': 'गाली', '😈': 'खराब',
    '🖕': 'अपमान', '👎': 'नकारात्मक', '👍': 'सकारात्मक', '🙏': 'नमस्कार',
    '❤️': 'माया', '💔': 'टुटेको मन', '🔥': 'आगो', '💯': 'पूर्ण',
}