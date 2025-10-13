"""
Evaluation utilities for hate speech detection models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, accuracy_score, precision_recall_fscore_support
)


def compute_metrics(y_true, y_pred, labels=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        
    Returns:
        Dictionary of metrics
    """
    if labels is None:
        labels = ['NO', 'OO', 'OR', 'OS']
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(labels))
    )
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': {
            labels[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
            for i in range(len(labels))
        }
    }
    
    return metrics


def print_metrics(metrics, title="Model Performance"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"Overall Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print(f"\nPer-Class Performance:")
    print(f"{'-'*60}")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print(f"{'-'*60}")
    
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<10} "
              f"{class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} "
              f"{class_metrics['f1']:<12.4f} "
              f"{int(class_metrics['support'])}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=True, 
                         save_path=None, title="Confusion Matrix"):
    """
    Plot confusion matrix with proper formatting.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        normalize: Whether to normalize by true class
        save_path: Path to save figure
        title: Plot title
    """
    if labels is None:
        labels = ['NO', 'OO', 'OR', 'OS']
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        cbar_label = 'Proportion'
    else:
        fmt = 'd'
        cbar_label = 'Count'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': cbar_label},
                linewidths=0.5, linecolor='gray')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'train_acc' in history and 'val_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.show()


def compare_models(results_dict, metric='macro_f1', save_path=None):
    """
    Compare multiple models on a specific metric.
    
    Args:
        results_dict: Dict mapping model names to metrics dicts
        metric: Metric to compare ('macro_f1', 'accuracy', etc.)
        save_path: Path to save figure
    """
    model_names = list(results_dict.keys())
    scores = [results_dict[name][metric] for name in model_names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, scores, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Model Comparison: {metric.replace("_", " ").title()}', 
             fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(scores) * 1.15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def error_analysis(y_true, y_pred, texts, labels=None, save_path=None):
    """
    Analyze prediction errors in detail.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: Original texts
        labels: Label names
        save_path: Path to save error analysis CSV
        
    Returns:
        DataFrame with error analysis
    """
    if labels is None:
        labels = ['NO', 'OO', 'OR', 'OS']
    
    errors = []
    for i, (true_label, pred_label, text) in enumerate(zip(y_true, y_pred, texts)):
        if true_label != pred_label:
            errors.append({
                'index': i,
                'text': text,
                'true_label': labels[true_label],
                'pred_label': labels[pred_label],
                'text_length': len(text.split()),
                'error_type': categorize_error(true_label, pred_label)
            })
    
    errors_df = pd.DataFrame(errors)
    
    if len(errors_df) > 0:
        print(f"\n{'='*60}")
        print(" Error Analysis Summary")
        print(f"{'='*60}")
        print(f"Total errors: {len(errors_df)} out of {len(y_true)} samples")
        print(f"Error rate: {len(errors_df)/len(y_true)*100:.2f}%")
        print(f"\nError distribution by type:")
        print(errors_df['error_type'].value_counts())
        print(f"{'='*60}\n")
        
        if save_path:
            errors_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"Error analysis saved to: {save_path}")
    
    return errors_df


def categorize_error(true_label, pred_label):
    """Categorize prediction errors."""
    # OR=2, OS=3 are minority classes
    if true_label in [2, 3] and pred_label == 0:
        return 'Critical Miss (Hate→Non-Offensive)'
    elif true_label in [2, 3] and pred_label == 1:
        return 'Minority Confusion (Hate→Other-Offensive)'
    elif true_label == 1 and pred_label == 0:
        return 'False Negative (Offensive→Non-Offensive)'
    elif true_label == 0 and pred_label in [1, 2, 3]:
        return 'False Positive (Non-Offensive→Offensive)'
    else:
        return 'Other Misclassification'