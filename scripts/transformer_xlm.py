"""
XLM-RoBERTa Transformer Model for Nepali Hate Speech Detection
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import preprocess_for_transformer
from utils.evaluation import compute_metrics, print_metrics, plot_confusion_matrix


class NepaliHateDataset(Dataset):
    """Dataset for XLM-RoBERTa."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class WeightedTrainer(Trainer):
    """Custom Trainer with class weights for handling imbalance."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics_for_trainer(eval_pred):
    """Compute metrics for Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_xlm_roberta(train_df, val_df, test_df, 
                      model_name='xlm-roberta-base',
                      save_dir='models/saved_models'):
    """
    Train XLM-RoBERTa model for hate speech detection.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        model_name: Pretrained model name
        save_dir: Directory to save model
    """
    print("\n" + "="*60)
    print(" Training XLM-RoBERTa Model")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    output_dir = os.path.join(save_dir, 'xlm_roberta_checkpoints')
    final_model_dir = os.path.join(save_dir, 'xlm_roberta_final')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess data
    print("\n1. Preprocessing data for transformer...")
    for df in [train_df, val_df, test_df]:
        if 'transformer_input' not in df.columns:
            df['transformer_input'] = df['Comment'].apply(preprocess_for_transformer)
    
    print(f"Sample preprocessed text: {train_df['transformer_input'].iloc[0][:100]}...")
    
    # Prepare labels
    le = LabelEncoder()
    le.fit(train_df['Label_Multiclass'])
    
    y_train = le.transform(train_df['Label_Multiclass'])
    y_val = le.transform(val_df['Label_Multiclass'])
    y_test = le.transform(test_df['Label_Multiclass'])
    
    print(f"\nLabel mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Class weights: {dict(zip(le.classes_, class_weights))}")
    
    # Initialize tokenizer
    print(f"\n2. Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset = NepaliHateDataset(
        train_df['transformer_input'].tolist(),
        y_train,
        tokenizer,
        max_length=128
    )
    
    val_dataset = NepaliHateDataset(
        val_df['transformer_input'].tolist(),
        y_val,
        tokenizer,
        max_length=128
    )
    
    test_dataset = NepaliHateDataset(
        test_df['transformer_input'].tolist(),
        y_test,
        tokenizer,
        max_length=128
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    print(f"\n4. Loading model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(le.classes_),
        problem_type="single_label_classification"
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,  # Effective batch size: 32
        
        # Optimization
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Scheduler
        lr_scheduler_type='cosine',
        
        # Regularization
        label_smoothing_factor=0.1,
        
        # Evaluation and saving
        evaluation_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        
        # Logging
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        report_to='none',  # Disable wandb/tensorboard
        
        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        
        # Reproducibility
        seed=42
    )
    
    # Initialize trainer with class weights
    print("\n5. Initializing trainer...")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights_tensor
    )
    
    # Train model
    print("\n6. Training model...")
    print("This may take 30-60 minutes depending on your hardware...\n")
    
    train_result = trainer.train()
    
    # Save final model
    print("\n7. Saving final model...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Save label encoder
    import joblib
    joblib.dump(le, os.path.join(final_model_dir, 'label_encoder.pkl'))
    
    print(f"Model saved to: {final_model_dir}")
    
    # Evaluate on test set
    print("\n8. Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    
    test_preds = np.argmax(predictions.predictions, axis=-1)
    test_labels = predictions.label_ids
    
    # Compute detailed metrics
    test_metrics = compute_metrics(test_labels, test_preds, labels=le.classes_)
    
    # Print results
    print_metrics(test_metrics, title="XLM-RoBERTa - Test Set Performance")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds,
        labels=le.classes_,
        save_path=os.path.join(save_dir, 'xlm_confusion_matrix.png'),
        title="XLM-RoBERTa - Confusion Matrix (Test Set)"
    )
    
    # Save results
    results = {
        'model': 'XLM-RoBERTa',
        'model_name': model_name,
        'test_metrics': test_metrics,
        'train_time': train_result.metrics['train_runtime'],
        'best_val_f1': trainer.state.best_metric
    }
    
    print("\n" + "="*60)
    print(" Training Complete!")
    print("="*60)
    print(f"Best Val F1: {results['best_val_f1']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Training time: {results['train_time']:.2f}s")
    
    return model, tokenizer, results


def test_alternative_models(train_df, val_df, test_df, save_dir='models/saved_models'):
    """
    Test alternative transformer models (MuRIL, IndicBERT).
    """
    alternative_models = [
        'google/muril-base-cased',
        'ai4bharat/indic-bert'
    ]
    
    results_comparison = {}
    
    for model_name in alternative_models:
        print(f"\n{'='*60}")
        print(f" Testing: {model_name}")
        print(f"{'='*60}")
        
        try:
            model, tokenizer, results = train_xlm_roberta(
                train_df, val_df, test_df,
                model_name=model_name,
                save_dir=save_dir
            )
            results_comparison[model_name] = results
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    return results_comparison


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_df = pd.read_json("data/train.json")
    test_df = pd.read_json("data/test.json")
    
    # Create validation split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df["Label_Multiclass"],
        random_state=42
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train XLM-RoBERTa (main model)
    model, tokenizer, results = train_xlm_roberta(train_df, val_df, test_df)
    
    print("\n✓ XLM-RoBERTa training complete!")
    
    # Optional: Test alternative models
    # results_comparison = test_alternative_models(train_df, val_df, test_df)