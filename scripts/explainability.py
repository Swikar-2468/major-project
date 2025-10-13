"""
Model Explainability using LIME and SHAP
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not installed. Install with: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import preprocess_for_transformer


class XLMRobertaExplainer:
    """Wrapper for XLM-RoBERTa model with explainability."""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize explainer.
        
        Args:
            model_path: Path to saved model directory
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label encoder
        le_path = os.path.join(model_path, 'label_encoder.pkl')
        self.label_encoder = joblib.load(le_path)
        self.class_names = self.label_encoder.classes_.tolist()
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.class_names}")
    
    def predict_proba(self, texts):
        """
        Predict probabilities for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of shape (n_samples, n_classes)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess
        preprocessed = [preprocess_for_transformer(t) for t in texts]
        
        # Tokenize
        encodings = self.tokenizer(
            preprocessed,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probas = torch.softmax(outputs.logits, dim=-1)
        
        return probas.cpu().numpy()
    
    def predict(self, texts):
        """Predict class labels."""
        probas = self.predict_proba(texts)
        return np.argmax(probas, axis=1)
    
    def get_prediction_details(self, text):
        """Get detailed prediction information."""
        proba = self.predict_proba([text])[0]
        pred_class_idx = np.argmax(proba)
        pred_class = self.class_names[pred_class_idx]
        confidence = proba[pred_class_idx]
        
        return {
            'text': text,
            'predicted_class': pred_class,
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(proba[i])
                for i in range(len(self.class_names))
            }
        }


class LIMEExplainer:
    """LIME explainer for hate speech model."""
    
    def __init__(self, model_explainer):
        """
        Initialize LIME explainer.
        
        Args:
            model_explainer: XLMRobertaExplainer instance
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed")
        
        self.model_explainer = model_explainer
        self.explainer = LimeTextExplainer(
            class_names=model_explainer.class_names,
            random_state=42
        )
    
    def explain_instance(self, text, num_features=10, num_samples=1000):
        """
        Explain a single prediction using LIME.
        
        Args:
            text: Input text to explain
            num_features: Number of features to show
            num_samples: Number of samples for LIME
            
        Returns:
            LIME explanation object
        """
        # Get prediction
        pred_details = self.model_explainer.get_prediction_details(text)
        print(f"\nPredicted: {pred_details['predicted_class']} "
              f"(Confidence: {pred_details['confidence']:.4f})")
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            text,
            self.model_explainer.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return explanation
    
    def visualize_explanation(self, explanation, save_path=None):
        """Visualize LIME explanation."""
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation saved to: {save_path}")
        
        plt.show()
    
    def explain_and_visualize(self, text, save_path=None):
        """Convenience method to explain and visualize."""
        explanation = self.explain_instance(text)
        self.visualize_explanation(explanation, save_path)
        return explanation


class SHAPExplainer:
    """SHAP explainer for hate speech model."""
    
    def __init__(self, model_explainer, background_texts=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model_explainer: XLMRobertaExplainer instance
            background_texts: Sample texts for background distribution
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed")
        
        self.model_explainer = model_explainer
        
        # Use a small background dataset
        if background_texts is None:
            background_texts = ["यो राम्रो छ", "यो नराम्रो छ"]
        
        # Create SHAP explainer
        self.explainer = shap.Explainer(
            self.model_explainer.predict_proba,
            shap.maskers.Text(self.model_explainer.tokenizer)
        )
    
    def explain_instance(self, text):
        """
        Explain a single prediction using SHAP.
        
        Args:
            text: Input text to explain
            
        Returns:
            SHAP explanation object
        """
        # Get prediction
        pred_details = self.model_explainer.get_prediction_details(text)
        print(f"\nPredicted: {pred_details['predicted_class']} "
              f"(Confidence: {pred_details['confidence']:.4f})")
        
        # Generate SHAP values
        shap_values = self.explainer([text])
        
        return shap_values
    
    def visualize_explanation(self, shap_values, save_path=None):
        """Visualize SHAP explanation."""
        shap.plots.text(shap_values[0])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP explanation saved to: {save_path}")
    
    def explain_and_visualize(self, text, save_path=None):
        """Convenience method to explain and visualize."""
        shap_values = self.explain_instance(text)
        self.visualize_explanation(shap_values, save_path)
        return shap_values


def explain_predictions(model_path, test_samples, save_dir='results/explanations'):
    """
    Generate explanations for test samples.
    
    Args:
        model_path: Path to saved model
        test_samples: List of (text, true_label) tuples
        save_dir: Directory to save explanations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(" Generating Model Explanations")
    print("="*60 + "\n")
    
    # Initialize model explainer
    model_explainer = XLMRobertaExplainer(model_path)
    
    # Initialize LIME explainer
    if LIME_AVAILABLE:
        print("Initializing LIME explainer...")
        lime_explainer = LIMEExplainer(model_explainer)
    else:
        lime_explainer = None
        print("LIME not available. Skipping LIME explanations.")
    
    # Process each sample
    for idx, (text, true_label) in enumerate(test_samples):
        print(f"\n{'='*60}")
        print(f" Sample {idx + 1}/{len(test_samples)}")
        print(f"{'='*60}")
        print(f"Text: {text[:100]}...")
        print(f"True Label: {true_label}")
        
        # Get prediction
        pred_details = model_explainer.get_prediction_details(text)
        print(f"\nPrediction Details:")
        print(f"  Predicted: {pred_details['predicted_class']}")
        print(f"  Confidence: {pred_details['confidence']:.4f}")
        print(f"  All Probabilities:")
        for class_name, prob in pred_details['probabilities'].items():
            print(f"    {class_name}: {prob:.4f}")
        
        # LIME explanation
        if lime_explainer:
            print("\nGenerating LIME explanation...")
            try:
                lime_save_path = os.path.join(save_dir, f'lime_sample_{idx+1}.png')
                explanation = lime_explainer.explain_and_visualize(text, lime_save_path)
                
                # Print top features
                print("\nTop contributing features (LIME):")
                for feature, weight in explanation.as_list()[:5]:
                    print(f"  '{feature}': {weight:.4f}")
            except Exception as e:
                print(f"Error generating LIME explanation: {str(e)}")
        
        print("\n" + "-"*60)


def batch_explain(model_path, test_df, num_samples=10, save_dir='results/explanations'):
    """
    Generate explanations for a batch of samples from each class.
    
    Args:
        model_path: Path to saved model
        test_df: Test dataframe
        num_samples: Number of samples per class
        save_dir: Directory to save explanations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample from each class
    samples = []
    for label in test_df['Label_Multiclass'].unique():
        class_df = test_df[test_df['Label_Multiclass'] == label]
        sampled = class_df.sample(min(num_samples, len(class_df)), random_state=42)
        
        for _, row in sampled.iterrows():
            samples.append((row['Comment'], row['Label_Multiclass']))
    
    print(f"Selected {len(samples)} samples for explanation")
    
    # Generate explanations
    explain_predictions(model_path, samples, save_dir)


def interactive_explainer(model_path):
    """
    Interactive explainer for user input.
    
    Args:
        model_path: Path to saved model
    """
    print("\n" + "="*60)
    print(" Interactive Hate Speech Explainer")
    print("="*60 + "\n")
    
    # Initialize explainers
    model_explainer = XLMRobertaExplainer(model_path)
    
    if LIME_AVAILABLE:
        lime_explainer = LIMEExplainer(model_explainer)
    else:
        lime_explainer = None
        print("LIME not available.")
    
    print("\nEnter Nepali text to analyze (or 'quit' to exit):")
    print("Example: यो राम्रो छैन")
    
    while True:
        print("\n" + "-"*60)
        text = input("\nEnter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        try:
            # Get prediction
            print("\n" + "="*60)
            pred_details = model_explainer.get_prediction_details(text)
            
            print(f"Prediction: {pred_details['predicted_class']}")
            print(f"Confidence: {pred_details['confidence']:.4f}")
            print(f"\nAll Class Probabilities:")
            for class_name, prob in pred_details['probabilities'].items():
                bar = "█" * int(prob * 50)
                print(f"  {class_name}: {prob:.4f} {bar}")
            
            # LIME explanation
            if lime_explainer:
                explain = input("\nGenerate LIME explanation? (y/n): ").strip().lower()
                if explain == 'y':
                    print("\nGenerating explanation...")
                    explanation = lime_explainer.explain_instance(text, num_features=8)
                    
                    print("\nTop Contributing Features:")
                    for feature, weight in explanation.as_list()[:8]:
                        direction = "↑" if weight > 0 else "↓"
                        print(f"  {direction} '{feature}': {weight:.4f}")
                    
                    viz = input("\nShow visualization? (y/n): ").strip().lower()
                    if viz == 'y':
                        lime_explainer.visualize_explanation(explanation)
        
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Explain hate speech predictions')
    parser.add_argument('--model_path', type=str, 
                       default='models/saved_models/xlm_roberta_final',
                       help='Path to saved model')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'],
                       default='interactive',
                       help='Explanation mode')
    parser.add_argument('--test_data', type=str,
                       default='data/test.json',
                       help='Path to test data (for batch mode)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples per class (batch mode)')
    parser.add_argument('--save_dir', type=str,
                       default='results/explanations',
                       help='Directory to save explanations')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_explainer(args.model_path)
    else:
        # Load test data
        test_df = pd.read_json(args.test_data)
        batch_explain(args.model_path, test_df, args.num_samples, args.save_dir)