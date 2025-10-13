"""
Optimized GRU Model for Nepali Hate Speech Detection
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import preprocess_for_ml_gru
from utils.evaluation import compute_metrics, print_metrics, plot_confusion_matrix, plot_training_history


class HateSpeechDataset(Dataset):
    """PyTorch Dataset for hate speech classification."""
    
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class ImprovedGRUClassifier(nn.Module):
    """
    Optimized Bidirectional GRU with attention mechanism.
    
    Improvements:
    - Attention mechanism for better context capture
    - Batch normalization for stable training
    - Residual connections
    - Optimized dropout rates
    """
    
    def __init__(self, embedding_matrix, hidden_dim=256, output_dim=4, 
                 dropout=0.4, num_layers=2):
        super(ImprovedGRUClassifier, self).__init__()
        
        num_embeddings, embedding_dim = embedding_matrix.shape
        
        # Embedding layer (trainable)
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), 
            freeze=False
        )
        self.embedding_dropout = nn.Dropout(dropout * 0.5)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        
        # Classification head with residual
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def attention_net(self, gru_output):
        """Apply attention mechanism."""
        # gru_output: [batch_size, seq_len, hidden_dim*2]
        
        # Compute attention scores
        attn_scores = self.attention(gru_output)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Apply attention weights
        context = torch.sum(attn_weights * gru_output, dim=1)  # [batch, hidden_dim*2]
        
        return context, attn_weights
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding_dropout(self.embedding(x))  # [batch, seq_len, emb_dim]
        
        # GRU
        gru_output, _ = self.gru(embedded)  # [batch, seq_len, hidden_dim*2]
        
        # Attention pooling
        context, attn_weights = self.attention_net(gru_output)
        
        # Batch normalization
        context = self.batch_norm(context)
        
        # Classification
        logits = self.classifier(context)
        
        return logits


def encode_and_pad(tokens, word2idx, max_len=50):
    """Convert tokens to indices and pad."""
    indices = [word2idx.get(tok, 0) for tok in tokens]  # 0 for <UNK>
    
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [0] * (max_len - len(indices))
    
    return indices


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, class_weights=None):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard examples.
    """
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')(logits, targets)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def train_epoch(model, dataloader, optimizer, criterion, device, class_weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # Compute loss (using focal loss for better imbalance handling)
        loss = focal_loss(logits, labels, class_weights=class_weights)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, f1


def evaluate(model, dataloader, criterion, device, class_weights):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids)
            loss = focal_loss(logits, labels, class_weights=class_weights)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, f1, all_preds, all_labels


def train_gru_model(train_df, val_df, test_df, save_dir='models/saved_models'):
    """
    Complete training pipeline for GRU model.
    """
    print("\n" + "="*60)
    print(" Training Optimized GRU Model")
    print("="*60 + "\n")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess data
    print("\n1. Preprocessing data...")
    for df in [train_df, val_df, test_df]:
        if 'clean_comment' not in df.columns:
            df['clean_comment'] = df['Comment'].apply(preprocess_for_ml_gru)
        if 'tokens' not in df.columns:
            df['tokens'] = df['clean_comment'].apply(str.split)
    
    # Build Word2Vec embeddings
    print("\n2. Training Word2Vec embeddings...")
    all_tokens = train_df['tokens'].tolist()
    w2v_model = Word2Vec(
        sentences=all_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
    )
    
    # Create vocabulary
    vocab = {word: idx+1 for idx, word in enumerate(w2v_model.wv.index_to_key)}
    vocab['<UNK>'] = 0
    vocab['<PAD>'] = 0
    
    # Create embedding matrix
    embedding_matrix = np.zeros((len(vocab), 100))
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    # Encode sequences
    print("\n3. Encoding sequences...")
    max_len = 50
    for df in [train_df, val_df, test_df]:
        df['input_ids'] = df['tokens'].apply(lambda x: encode_and_pad(x, vocab, max_len))
    
    # Prepare labels
    le = LabelEncoder()
    le.fit(train_df['Label_Multiclass'])
    
    y_train = le.transform(train_df['Label_Multiclass'])
    y_val = le.transform(val_df['Label_Multiclass'])
    y_test = le.transform(test_df['Label_Multiclass'])
    
    print(f"Label classes: {le.classes_}")
    
    # Create datasets
    train_dataset = HateSpeechDataset(train_df['input_ids'].tolist(), y_train)
    val_dataset = HateSpeechDataset(val_df['input_ids'].tolist(), y_val)
    test_dataset = HateSpeechDataset(test_df['input_ids'].tolist(), y_test)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"\nClass weights: {dict(zip(le.classes_, class_weights))}")
    
    # Initialize model
    print("\n4. Initializing model...")
    model = ImprovedGRUClassifier(
        embedding_matrix=embedding_matrix,
        hidden_dim=256,
        output_dim=len(le.classes_),
        dropout=0.4,
        num_layers=2
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Training loop
    print("\n5. Training model...")
    epochs = 30
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, class_weights_tensor
        )
        
        # Validate
        val_loss, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device, class_weights_tensor
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'vocab': vocab,
                'label_encoder': le
            }, os.path.join(save_dir, 'best_gru_model.pt'))
            
            print(f"✓ Best model saved! (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print("\n⚠ Early stopping triggered!")
            break
    
    # Plot training history
    print("\n6. Plotting training history...")
    history['train_acc'] = history['train_f1']  # For plotting
    history['val_acc'] = history['val_f1']
    plot_training_history(history, save_path=os.path.join(save_dir, 'gru_training_history.png'))
    
    # Load best model for evaluation
    print("\n7. Evaluating best model on test set...")
    checkpoint = torch.load(os.path.join(save_dir, 'best_gru_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, class_weights_tensor
    )
    
    # Compute detailed metrics
    test_metrics = compute_metrics(test_labels, test_preds, labels=le.classes_)
    
    # Print results
    print_metrics(test_metrics, title="GRU Model - Test Set Performance")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds, 
        labels=le.classes_,
        save_path=os.path.join(save_dir, 'gru_confusion_matrix.png'),
        title="GRU Model - Confusion Matrix (Test Set)"
    )
    
    # Save results
    results = {
        'model': 'Optimized GRU',
        'test_loss': test_loss,
        'test_f1': test_f1,
        'best_val_f1': best_val_f1,
        'total_epochs': len(history['train_loss']),
        'metrics': test_metrics
    }
    
    print("\n" + "="*60)
    print(" Training Complete!")
    print("="*60)
    
    return model, results, history


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
    
    # Train model
    model, results, history = train_gru_model(train_df, val_df, test_df)
    
    print("\n✓ GRU model training complete!")
    print(f"Final Test Macro F1: {results['test_f1']:.4f}")