"""
Complete training pipeline for Nepali Hate Speech Detection
Runs ML baselines → GRU → XLM-RoBERTa → Explainability
"""
import os
import sys
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("\n" + "="*70)
    print(" NEPALI HATE SPEECH DETECTION - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Load data
    print("Step 1: Loading dataset...")
    train_path = "data/train.json"
    test_path = "data/test.json"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Data files not found!")
        print(f"Please ensure {train_path} and {test_path} exist")
        return
    
    train_df = pd.read_json(train_path)
    test_df = pd.read_json(test_path)
    
    # Create validation split
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df["Label_Multiclass"],
        random_state=42
    )
    
    print(f"✓ Data loaded successfully!")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Classes: {train_df['Label_Multiclass'].unique()}")
    
    # Create directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Step 2: ML Baselines
    print("\n" + "="*70)
    print("Step 2: Training ML Baseline Models")
    print("="*70)
    
    try:
        from utils.preprocessing import preprocess_for_ml_gru
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.utils.class_weight import compute_class_weight
        from imblearn.over_sampling import SMOTE
        from utils.evaluation import compute_metrics, print_metrics, plot_confusion_matrix
        import numpy as np
        import joblib
        
        # Preprocess
        print("\nPreprocessing data for ML models...")
        for df in [train_df, val_df, test_df]:
            df['clean_comment'] = df['Comment'].apply(preprocess_for_ml_gru)
        
        # Vectorize
        print("Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
        X_train = vectorizer.fit_transform(train_df['clean_comment'])
        X_val = vectorizer.transform(val_df['clean_comment'])
        X_test = vectorizer.transform(test_df['clean_comment'])
        
        # Labels
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['Label_Multiclass'])
        y_val = le.transform(val_df['Label_Multiclass'])
        y_test = le.transform(test_df['Label_Multiclass'])
        
        # Class weights
        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(y_train), 
                                            y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        # SMOTE for Naive Bayes
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Train models
        ml_models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                class_weight=class_weight_dict, 
                random_state=42
            ),
            'SVM': LinearSVC(
                class_weight=class_weight_dict, 
                random_state=42, 
                max_iter=2000
            ),
            'Naive Bayes': MultinomialNB()
        }
        
        for name, model in ml_models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Naive Bayes':
                model.fit(X_train_res, y_train_res)
            else:
                model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred, labels=le.classes_)
            
            print_metrics(metrics, title=f"{name} - Test Results")
            
            # Save results
            all_results[name] = metrics
            
            # Save model
            model_path = f'models/saved_models/{name.lower().replace(" ", "_")}_model.pkl'
            joblib.dump(model, model_path)
            print(f"✓ Model saved to {model_path}")
        
        # Save vectorizer and label encoder
        joblib.dump(vectorizer, 'models/saved_models/tfidf_vectorizer.pkl')
        joblib.dump(le, 'models/saved_models/ml_label_encoder.pkl')
        
        print("\n✓ ML Baselines complete!")
        
    except Exception as e:
        print(f"Error in ML baseline training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Step 3: GRU Model
    print("\n" + "="*70)
    print("Step 3: Training GRU Model")
    print("="*70)
    
    try:
        from scripts.gru_model import train_gru_model
        
        model, gru_results, history = train_gru_model(train_df, val_df, test_df)
        all_results['GRU'] = gru_results['metrics']
        
        print("\n✓ GRU training complete!")
        
    except Exception as e:
        print(f"Error in GRU training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Step 4: XLM-RoBERTa
    print("\n" + "="*70)
    print("Step 4: Training XLM-RoBERTa Transformer")
    print("="*70)
    
    try:
        from scripts.transformer_xlm import train_xlm_roberta
        
        model, tokenizer, xlm_results = train_xlm_roberta(
            train_df, val_df, test_df
        )
        all_results['XLM-RoBERTa'] = xlm_results['test_metrics']
        
        print("\n✓ XLM-RoBERTa training complete!")
        
    except Exception as e:
        print(f"Error in XLM-RoBERTa training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Model Comparison
    print("\n" + "="*70)
    print("Step 5: Model Comparison")
    print("="*70 + "\n")
    
    if all_results:
        # Create comparison table
        comparison_df = pd.DataFrame({
            model_name: {
                'Accuracy': results['accuracy'],
                'Macro F1': results['macro_f1'],
                'Weighted F1': results['weighted_f1'],
                'NO F1': results['per_class']['NO']['f1'],
                'OO F1': results['per_class']['OO']['f1'],
                'OR F1': results['per_class']['OR']['f1'],
                'OS F1': results['per_class']['OS']['f1']
            }
            for model_name, results in all_results.items()
        }).T
        
        print("Model Performance Comparison:")
        print("="*70)
        print(comparison_df.to_string())
        
        # Save comparison
        comparison_df.to_csv('results/model_comparison.csv')
        print(f"\n✓ Comparison saved to results/model_comparison.csv")
        
        # Find best model
        best_model = comparison_df['Macro F1'].idxmax()
        best_f1 = comparison_df.loc[best_model, 'Macro F1']
        
        print(f"\n🏆 Best Model: {best_model} (Macro F1: {best_f1:.4f})")
        
        # Visualize comparison
        try:
            from utils.evaluation import compare_models as plot_compare
            
            metrics_dict = {name: results['macro_f1'] 
                          for name, results in all_results.items()}
            plot_compare(
                {name: {'macro_f1': f1} for name, f1 in metrics_dict.items()},
                metric='macro_f1',
                save_path='results/model_comparison.png'
            )
        except Exception as e:
            print(f"Could not create comparison plot: {str(e)}")
    
    # Step 6: Generate Explanations
    print("\n" + "="*70)
    print("Step 6: Generating Model Explanations")
    print("="*70)
    
    try:
        from scripts.explainability import batch_explain
        
        print("\nGenerating LIME explanations for sample predictions...")
        batch_explain(
            model_path='models/saved_models/xlm_roberta_final',
            test_df=test_df,
            num_samples=3,
            save_dir='results/explanations'
        )
        
        print("\n✓ Explanations generated!")
        
    except Exception as e:
        print(f"Error generating explanations: {str(e)}")
        print("You can run explanations manually using scripts/5_explainability.py")
    
    # Final Summary
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print(f"  - Models trained: {len(all_results)}")
    print(f"  - Best model: {best_model if 'best_model' in locals() else 'N/A'}")
    print(f"  - Results saved to: results/")
    print(f"  - Models saved to: models/saved_models/")
    
    print("\n🚀 Next Steps:")
    print("  1. Review model comparison: results/model_comparison.csv")
    print("  2. Check explanations: results/explanations/")
    print("  3. Run web app: streamlit run scripts/6_streamlit_app.py")
    print("  4. Run interactive explainer: python scripts/5_explainability.py --mode interactive")
    
    print("\n" + "="*70 + "\n")
    
    # Save final summary
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'models_trained': list(all_results.keys()),
        'best_model': best_model if 'best_model' in locals() else None,
        'best_macro_f1': float(best_f1) if 'best_f1' in locals() else None,
        'results': {
            name: {
                'accuracy': float(results['accuracy']),
                'macro_f1': float(results['macro_f1']),
                'weighted_f1': float(results['weighted_f1'])
            }
            for name, results in all_results.items()
        }
    }
    
    with open('results/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Training summary saved to results/training_summary.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Complete training pipeline for Nepali hate speech detection'
    )
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip ML baseline training')
    parser.add_argument('--skip-gru', action='store_true',
                       help='Skip GRU training')
    parser.add_argument('--skip-transformer', action='store_true',
                       help='Skip transformer training')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: train only XLM-RoBERTa with reduced epochs')
    
    args = parser.parse_args()
    
    if args.quick:
        print("\n⚡ Quick mode enabled - training only XLM-RoBERTa with reduced settings")
        # You can modify training parameters in transformer_xlm.py for quick mode
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()