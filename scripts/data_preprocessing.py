"""
Data Preprocessing Script for Nepali Hate Speech Detection
Handles the exact format from your dataset with ID, Comment, Label_Binary, Label_Multiclass
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import (
    preprocess_for_ml_gru, 
    preprocess_for_transformer,
    is_devanagari
)


def load_and_explore_data(train_path, test_path):
    """Load and explore the dataset."""
    print("\n" + "="*70)
    print(" LOADING AND EXPLORING DATASET")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data files...")
    train_df = pd.read_json(train_path)
    test_df = pd.read_json(test_path)
    
    print(f"✓ Train data loaded: {train_df.shape}")
    print(f"✓ Test data loaded: {test_df.shape}")
    
    # Display sample
    print("\n" + "-"*70)
    print("Sample data from training set:")
    print("-"*70)
    for idx, row in train_df.head(3).iterrows():
        print(f"\nID: {row['ID']}")
        print(f"Comment: {row['Comment'][:100]}...")
        print(f"Label Binary: {row['Label_Binary']}")
        print(f"Label Multiclass: {row['Label_Multiclass']}")
        print("-"*70)
    
    # Check columns
    print("\nDataset columns:")
    print(f"Train: {list(train_df.columns)}")
    print(f"Test: {list(test_df.columns)}")
    
    # Check for missing values
    print("\nMissing values:")
    print("Train set:")
    print(train_df.isnull().sum())
    print("\nTest set:")
    print(test_df.isnull().sum())
    
    # Check duplicates
    train_duplicates = train_df.duplicated(subset=['Comment']).sum()
    test_duplicates = test_df.duplicated(subset=['Comment']).sum()
    print(f"\nDuplicate comments:")
    print(f"  Train: {train_duplicates}")
    print(f"  Test: {test_duplicates}")
    
    return train_df, test_df


def analyze_class_distribution(train_df, test_df, save_dir='results/eda'):
    """Analyze and visualize class distribution."""
    print("\n" + "="*70)
    print(" CLASS DISTRIBUTION ANALYSIS")
    print("="*70 + "\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Multiclass distribution
    print("Label_Multiclass Distribution:")
    print("-"*70)
    
    train_counts = train_df['Label_Multiclass'].value_counts()
    test_counts = test_df['Label_Multiclass'].value_counts()
    
    print("\nTraining Set:")
    for label, count in train_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f"  {label}: {count:>5} ({percentage:>5.2f}%)")
    
    print("\nTest Set:")
    for label, count in test_counts.items():
        percentage = (count / len(test_df)) * 100
        print(f"  {label}: {count:>5} ({percentage:>5.2f}%)")
    
    # Calculate imbalance ratio
    max_class = train_counts.max()
    min_class = train_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"\n⚠️  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training set distribution
    axes[0].bar(train_counts.index, train_counts.values, color='skyblue', edgecolor='navy')
    axes[0].set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class Label')
    axes[0].set_ylabel('Number of Samples')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (label, count) in enumerate(train_counts.items()):
        axes[0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Test set distribution
    axes[1].bar(test_counts.index, test_counts.values, color='lightcoral', edgecolor='darkred')
    axes[1].set_title('Test Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class Label')
    axes[1].set_ylabel('Number of Samples')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (label, count) in enumerate(test_counts.items()):
        axes[1].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_dir}/class_distribution.png")
    plt.show()
    
    return train_counts, test_counts


def analyze_text_characteristics(train_df, test_df, save_dir='results/eda'):
    """Analyze text length and script distribution."""
    print("\n" + "="*70)
    print(" TEXT CHARACTERISTICS ANALYSIS")
    print("="*70 + "\n")
    
    # Text length analysis
    print("Analyzing text lengths...")
    for df, name in [(train_df, 'Train'), (test_df, 'Test')]:
        df['text_length_chars'] = df['Comment'].apply(lambda x: len(str(x)))
        df['text_length_words'] = df['Comment'].apply(lambda x: len(str(x).split()))
    
    print("\nText Length Statistics (words):")
    print("-"*70)
    for df, name in [(train_df, 'Train'), (test_df, 'Test')]:
        print(f"\n{name} Set:")
        print(df['text_length_words'].describe())
    
    # Script distribution
    print("\n\nScript Distribution:")
    print("-"*70)
    for df, name in [(train_df, 'Train'), (test_df, 'Test')]:
        df['is_devanagari'] = df['Comment'].apply(is_devanagari)
        devanagari_count = df['is_devanagari'].sum()
        roman_count = len(df) - devanagari_count
        
        print(f"\n{name} Set:")
        print(f"  Devanagari: {devanagari_count} ({devanagari_count/len(df)*100:.1f}%)")
        print(f"  Romanized:  {roman_count} ({roman_count/len(df)*100:.1f}%)")
    
    # Visualization - Text Length Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training set - words
    axes[0, 0].hist(train_df['text_length_words'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Train Set - Word Length Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Words')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(train_df['text_length_words'].median(), color='red', 
                       linestyle='--', label=f'Median: {train_df["text_length_words"].median():.0f}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Test set - words
    axes[0, 1].hist(test_df['text_length_words'], bins=30, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Test Set - Word Length Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Words')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(test_df['text_length_words'].median(), color='red', 
                       linestyle='--', label=f'Median: {test_df["text_length_words"].median():.0f}')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Word length by class - Training
    for label in train_df['Label_Multiclass'].unique():
        subset = train_df[train_df['Label_Multiclass'] == label]['text_length_words']
        axes[1, 0].hist(subset, bins=20, alpha=0.5, label=label)
    axes[1, 0].set_title('Train Set - Length by Class', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Words')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Script distribution pie chart
    script_data = [train_df['is_devanagari'].sum(), len(train_df) - train_df['is_devanagari'].sum()]
    axes[1, 1].pie(script_data, labels=['Devanagari', 'Romanized'], autopct='%1.1f%%',
                   colors=['#66b3ff', '#ff9999'], startangle=90)
    axes[1, 1].set_title('Train Set - Script Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'text_characteristics.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_dir}/text_characteristics.png")
    plt.show()


def create_validation_split(train_df, val_size=0.15, random_state=42):
    """Create validation split from training data."""
    print("\n" + "="*70)
    print(" CREATING VALIDATION SPLIT")
    print("="*70 + "\n")
    
    print(f"Splitting training data (validation size: {val_size*100}%)...")
    
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df['Label_Multiclass'],
        random_state=random_state
    )
    
    print(f"✓ Split complete!")
    print(f"  New train size: {len(train_split)}")
    print(f"  Validation size: {len(val_split)}")
    
    # Check stratification
    print("\nValidation set class distribution:")
    val_counts = val_split['Label_Multiclass'].value_counts()
    for label, count in val_counts.items():
        percentage = (count / len(val_split)) * 100
        print(f"  {label}: {count:>5} ({percentage:>5.2f}%)")
    
    return train_split, val_split


def apply_preprocessing(train_df, val_df, test_df):
    """Apply preprocessing to all datasets."""
    print("\n" + "="*70)
    print(" APPLYING PREPROCESSING")
    print("="*70 + "\n")
    
    datasets = [
        (train_df, 'Train'),
        (val_df, 'Validation'),
        (test_df, 'Test')
    ]
    
    for df, name in datasets:
        print(f"Processing {name} set...")
        
        # ML/GRU preprocessing (Romanized)
        print(f"  - Creating ML/GRU features (Romanized)...")
        df['clean_comment'] = df['Comment'].apply(preprocess_for_ml_gru)
        df['tokens'] = df['clean_comment'].apply(str.split)
        
        # Transformer preprocessing (Devanagari)
        print(f"  - Creating Transformer features (Devanagari)...")
        df['transformer_input'] = df['Comment'].apply(preprocess_for_transformer)
        
        print(f"  ✓ {name} set preprocessed")
    
    # Show examples
    print("\n" + "-"*70)
    print("Preprocessing Examples:")
    print("-"*70)
    
    for idx in range(min(3, len(train_df))):
        row = train_df.iloc[idx]
        print(f"\n[{idx+1}] Label: {row['Label_Multiclass']}")
        print(f"Original:     {row['Comment'][:80]}...")
        print(f"ML/GRU:       {row['clean_comment'][:80]}...")
        print(f"Transformer:  {row['transformer_input'][:80]}...")
        print(f"Tokens:       {row['tokens'][:10]}")
        print("-"*70)
    
    print("\n✓ All preprocessing complete!")
    
    return train_df, val_df, test_df


def save_preprocessed_data(train_df, val_df, test_df, output_dir='data/processed'):
    """Save preprocessed datasets."""
    print("\n" + "="*70)
    print(" SAVING PREPROCESSED DATA")
    print("="*70 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    train_df.to_json(os.path.join(output_dir, 'train_preprocessed.json'), 
                     orient='records', force_ascii=False, indent=2)
    val_df.to_json(os.path.join(output_dir, 'val_preprocessed.json'), 
                   orient='records', force_ascii=False, indent=2)
    test_df.to_json(os.path.join(output_dir, 'test_preprocessed.json'), 
                    orient='records', force_ascii=False, indent=2)
    
    print(f"✓ Saved to {output_dir}/")
    print(f"  - train_preprocessed.json ({len(train_df)} samples)")
    print(f"  - val_preprocessed.json ({len(val_df)} samples)")
    print(f"  - test_preprocessed.json ({len(test_df)} samples)")
    
    # Save preprocessing statistics
    stats = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_size': len(train_df) + len(val_df) + len(test_df),
        'class_distribution_train': train_df['Label_Multiclass'].value_counts().to_dict(),
        'class_distribution_val': val_df['Label_Multiclass'].value_counts().to_dict(),
        'class_distribution_test': test_df['Label_Multiclass'].value_counts().to_dict(),
        'avg_text_length_words': {
            'train': float(train_df['text_length_words'].mean()),
            'val': float(val_df['text_length_words'].mean()),
            'test': float(test_df['text_length_words'].mean())
        },
        'devanagari_percentage': {
            'train': float(train_df['is_devanagari'].mean() * 100),
            'val': float(val_df['is_devanagari'].mean() * 100),
            'test': float(test_df['is_devanagari'].mean() * 100)
        }
    }
    
    import json
    with open(os.path.join(output_dir, 'preprocessing_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Statistics saved to {output_dir}/preprocessing_stats.json")


def main():
    """Main preprocessing pipeline."""
    print("\n" + "="*70)
    print(" NEPALI HATE SPEECH DETECTION - DATA PREPROCESSING")
    print("="*70)
    
    # Paths (adjust these to your actual paths)
    train_path = "D:/major project/nepali-offensive-lang-detection-dataset/train.json"
    test_path = "D:/major project/nepali-offensive-lang-detection-dataset/test.json"
    
    # Check if files exist
    if not os.path.exists(train_path):
        print(f"\n❌ Error: {train_path} not found!")
        print("Please update the path in the script or place your data files in the correct location.")
        return
    
    if not os.path.exists(test_path):
        print(f"\n❌ Error: {test_path} not found!")
        print("Please update the path in the script or place your data files in the correct location.")
        return
    
    # Step 1: Load and explore
    train_df, test_df = load_and_explore_data(train_path, test_path)
    
    # Step 2: Analyze class distribution
    analyze_class_distribution(train_df, test_df)
    
    # Step 3: Analyze text characteristics
    analyze_text_characteristics(train_df, test_df)
    
    # Step 4: Create validation split
    train_df, val_df = create_validation_split(train_df)
    
    # Step 5: Apply preprocessing
    train_df, val_df, test_df = apply_preprocessing(train_df, val_df, test_df)
    
    # Step 6: Save preprocessed data
    save_preprocessed_data(train_df, val_df, test_df)
    
    # Final summary
    print("\n" + "="*70)
    print(" PREPROCESSING COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print(f"  Total samples: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    print(f"\n  Classes: {sorted(train_df['Label_Multiclass'].unique())}")
    print(f"  Avg text length: {train_df['text_length_words'].mean():.1f} words")
    print(f"\n✓ Preprocessed data saved to: data/processed/")
    print(f"✓ Visualizations saved to: results/eda/")
    print("\n🚀 Ready for model training!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()