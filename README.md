# 🇳🇵 Nepali Hate Content Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

A state-of-the-art hate content detection system for Nepali social media text, featuring advanced preprocessing for mixed-script content, transformer-based classification, and comprehensive explainability methods.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Performance](#model-performance)
- [Demo & Screenshots](#demo--screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements a multi-class text classification system for detecting hate content in Nepali social media content. The system classifies text into four categories:

| Class | Label | Description |
|-------|-------|-------------|
| **NO** | Non-Offensive | Content without offensive language |
| **OO** | Other-Offensive | General offensive content (profanity, insults) |
| **OR** | Offensive-Racist | Content targeting ethnicity, race, or caste |
| **OS** | Offensive-Sexist | Content targeting gender or sexual orientation |

### 🌟 Challenges Addressed

- ✅ **Mixed-Script Text**: Handles Devanagari, Romanized Nepali, and English
- ✅ **Code-Mixing**: Processes multilingual utterances seamlessly
- ✅ **Emoji Semantics**: Maps 180+ emojis to Nepali words with contextual meaning
- ✅ **Class Imbalance**: Severe imbalance (14.98× ratio) addressed through weighted metrics
- ✅ **Low-Resource Language**: Leverages cross-lingual transfer learning

---

## ✨ Key Features

### 🔧 Advanced Preprocessing
- **Script Detection**: Automatic identification of Devanagari, Romanized, or English text
- **Transliteration**: ITRANS-based conversion of Romanized Nepali to Devanagari
- **Translation**: English-to-Nepali translation for code-mixed content
- **Emoji Processing**: Semantic mapping of 180+ emojis to Nepali words
- **18 Emoji Features**: Statistical features capturing emoji usage patterns
- **Unknown Emoji Preservation**: Retains unmapped emojis to prevent information loss

### 🤖 Multiple Model Architectures
- **Traditional ML Baselines**: Logistic Regression, SVM, Naive Bayes, Random Forest
- **Deep Learning**: GRU with Word2Vec embeddings (26K vocabulary)
- **Transformers**: XLM-RoBERTa Large (560M params) & NepaliBERT (110M params)

### 📊 Comprehensive Explainability
- **LIME**: Local interpretable model-agnostic explanations
- **SHAP**: Shapley additive explanations with game theory foundations
- **Captum (Integrated Gradients)**: Gradient-based token attributions

### 🌐 Production-Ready Web Application
- **Single & Batch Prediction**: Process individual texts or CSV files
- **Real-Time Processing**: Instant predictions with confidence scores
- **Interactive Visualizations**: Charts, confusion matrices, and attribution heatmaps
- **Prediction History**: Track and download prediction logs
- **Session Statistics**: Live analytics and class distribution

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT TEXT (Mixed Script)                    │
│           "yo ramro chainau 😡" / "यो राम्रो छैन"               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Script Detection  →  2. Transliteration/Translation         │
│  3. Emoji Processing  →  4. Text Cleaning  →  5. Features       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL INFERENCE                              │
│   XLM-RoBERTa Large (560M params) / NepaliBERT (110M params)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREDICTION + CONFIDENCE                        │
│         NO: 15% | OO: 72% | OR: 10% | OS: 3%                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXPLAINABILITY (Optional)                      │
│              LIME | SHAP | Integrated Gradients                 │
└─────────────────────────────────────────────────────────────────┘
```

### Complete System Block Diagram

![System Architecture](notebooks/models/saved_models/system_architecture.png)

The system follows a 9-stage workflow from requirement analysis to deployment.

---

## 🔄 Preprocessing Pipeline

Our preprocessing pipeline is specifically designed to handle the unique challenges of Nepali social media text.

### Pipeline Overview

![Preprocessing Pipeline](notebooks/models/saved_models/preprocessing_flowchart.png)

### Detailed Stages

#### Stage 1: Script Detection
```python
confidence_devanagari = count(devanagari_chars) / total_chars
confidence_ascii = count(ascii_chars) / total_chars

if confidence_devanagari > 0.7:    → Pure Devanagari
elif confidence_ascii > 0.7:       → Romanized/English
else:                              → Code-Mixed
```

#### Stage 2: Script Unification

**Romanized Nepali → Devanagari (ITRANS)**
```
Input:  "yo ramro cha"
Output: "यो राम्रो छ"
```

**English → Nepali (Translation)**
```
Input:  "This is good"
Output: "यो राम्रो छ"
```

#### Stage 3: Emoji Processing

![Emoji Processing Module](notebooks/models/saved_models/emoji_processing.jpg)

**Emoji-to-Nepali Semantic Mapping (180+ emojis)**

| Emoji | Nepali Mapping | Category |
|-------|----------------|----------|
| 😡 | ठूलो रिस (great anger) | Hate |
| 😀 | खुशी (happiness) | Positive |
| 😏 | व्यंग्य (sarcasm) | Mockery |
| 💩 | मल (feces) | Disgust |
| ❤️ | माया (love) | Positive |
| 🖕 | अपमान (insult) | Offensive |

**18 Emoji Features Extracted:**
- Binary Indicators (6): has_hate_emoji, has_mockery_emoji, has_positive_emoji, etc.
- Count Features (6): hate_emoji_count, positive_emoji_count, etc.
- Aggregate Features (3): total_emoji_count, hate_to_positive_ratio, has_mixed_sentiment
- Coverage Features (3): unknown_emoji_count, has_unknown_emoji, known_emoji_ratio

**Example Transformation:**
```
Input:  "yo ramro chainau 😡😡"
After:  "यो राम्रो छैनौ ठूलो रिस ठूलो रिस"
Features: {hate_emoji_count: 2, total_emoji_count: 2, ...}
```

#### Stage 4: Text Cleaning
- ❌ Remove URLs: `https://example.com` → removed
- ❌ Remove mentions: `@username` → removed
- ✂️ Clean hashtags: `#राम्रो` → `राम्रो`
- 🧹 Normalize whitespace: Multiple spaces → Single space

---

## 📈 Model Performance

### Results Summary

![Model Comparison](notebooks/models/saved_models/ml_comparison_chart.png)

### Traditional Machine Learning Baselines

**Configuration:**
- Features: TF-IDF (5,000 features, n-grams: 1-3)
- Sparsity: 99.78%
- Class Weighting: Applied to handle imbalance

| Model | Accuracy | Macro F1 | Weighted F1 | NO F1 | OO F1 | OR F1 | OS F1 |
|-------|----------|----------|-------------|-------|-------|-------|-------|
| **Logistic Regression** | **75.38%** | **0.5701** | 0.7542 | 0.8225 | 0.6722 | 0.5000 | 0.2857 |
| SVM | 75.52% | 0.5502 | 0.7542 | 0.8288 | 0.6659 | 0.4660 | 0.2400 |
| Naive Bayes (+ SMOTE) | 72.14% | 0.4947 | 0.7288 | 0.7989 | 0.6599 | 0.3448 | 0.1754 |
| Random Forest | 70.69% | 0.4751 | 0.6895 | 0.8032 | 0.5278 | 0.4231 | 0.1463 |


**Key Observations:**
- ✅ Logistic Regression achieves best performance among baselines
- ⚠️ Struggles with minority classes (OR, OS) due to severe imbalance
- 📊 Strong NO class performance (>0.80 F1) across all models

---

### Deep Learning: GRU + Word2Vec

**Architecture:**
- Embedding: Word2Vec (100D, 26,554 vocab)
- GRU Layer: 128 units with dropout (0.3)
- Dense Layer: 64 units (ReLU)
- Output: 4 classes (Softmax)

**5-Fold Cross-Validation Results:**

| Fold | Validation F1 | Early Stop Epoch |
|------|---------------|------------------|
| Fold 1 | 0.6358 | 28 |
| Fold 2 | **0.6434** | 31 |
| Fold 3 | 0.5996 | 26 |
| Fold 4 | **0.6999** | 33 |
| Fold 5 | 0.6053 | 23 |
| **Mean ± Std** | **0.6368 ± 0.0358** | **28.2 ± 3.5** |


**Performance Across Stages:**

| Stage | Macro F1 | Dataset Size | Note |
|-------|----------|--------------|------|
| Cross-Validation | 0.6368 ± 0.0358 | 5,555 (train) | 5-fold CV |
| Final Validation | 0.5243 | 619 | After retraining |
| **Final Test** | **0.3307** | 1,443 | ⚠️ Generalization issue |

![GRU Training Curves](notebooks/models/saved_models/gru_final_training_history.png)

**Analysis:**
- ✅ Strong cross-validation performance (0.6368)
- ⚠️ Significant performance drop on test set (0.3307)
- 📉 Indicates overfitting and generalization challenges
- 🔄 High variance across folds (0.0358 std)

---

### Transformer Models

#### XLM-RoBERTa Large (Best Model ⭐)

**Model Specifications:**
- Parameters: 560M
- Layers: 24
- Hidden Size: 1024
- Pre-training: 2.5TB CommonCrawl, 100 languages
- Fine-tuning: AdamW (LR: 2e-5), 5 epochs

**Test Set Performance:**

| Metric | Score |
|--------|-------|
| **Accuracy** | **70.34%** |
| **Macro Precision** | 0.5174 |
| **Macro Recall** | 0.5978 |
| **Macro F1** | **0.5465** |
| **Weighted F1** | 0.7127 |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **NO** | 0.8344 | 0.7366 | **0.7825** | 896 |
| **OO** | 0.5949 | 0.6708 | **0.6306** | 486 |
| **OR** | 0.2941 | 0.5102 | **0.3731** | 49 |
| **OS** | 0.3462 | 0.4737 | **0.4000** | 19 |

![XLM-RoBERTa Test Confusion Matrix](notebooks/models/saved_models/xlmr_test_confusion_matrix.png)
![XLM-RoBERTa Validation Confusion Matrix](notebooks/models/saved_models/xlmr_val_confusion_matrix.png)

![XLM-RoBERTa training](notebooks/models/saved_models/xlmr_training_curves.png)

**Strengths:**
- ✅ Excellent performance on majority classes (NO: 0.78, OO: 0.63)
- ✅ Best overall macro F1 (0.5465) across all models
- ✅ Robust to code-mixed and multilingual text
- ✅ Strong cross-lingual transfer capabilities

**Challenges:**
- ⚠️ Lower performance on minority classes (OR: 0.37, OS: 0.40)
- 📊 Severe class imbalance impacts minority class learning
- 🎯 Future work: Data augmentation for OR/OS classes

---

#### NepaliBERT

**Model Specifications:**
- Parameters: 110M
- Layers: 12
- Hidden Size: 768
- Pre-training: Nepali corpus
- Fine-tuning: Same configuration as XLM-RoBERTa

**Test Set Performance:**

| Metric | Score |
|--------|-------|
| Accuracy | 69.72% |
| Macro Precision | 0.4933 |
| Macro Recall | 0.5413 |
| **Macro F1** | **0.5126** |
| Weighted F1 | 0.6994 |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NO | 0.7805 | 0.7701 | 0.7753 | 896 |
| OO | 0.6102 | 0.5926 | 0.6013 | 486 |
| **OR** | 0.4085 | 0.5918 | **0.4833** | 49 |
| OS | 0.1739 | 0.2105 | 0.1905 | 19 |

**Validation Confusion Matrix**  
![NepaliBERT Validation Confusion Matrix](notebooks/models/saved_models/nepalibert_val_confusion_matrix.png)

**Test Confusion Matrix**  
![NepaliBERT Test Confusion Matrix](notebooks/models/saved_models/nepalibert_test_confusion_matrix.png)

**Training Curves**  
![NepaliBERT Training Curves](notebooks/models/saved_models/nepalibert_training_curves.png)

**Comparison with XLM-RoBERTa:**
- ⚠️ Lower overall macro F1 (0.5126 vs 0.5465)
- ✅ **Better OR class performance** (0.4833 vs 0.3731)
- ⚠️ Weaker OS class performance (0.1905 vs 0.4000)
- 📊 Smaller model size (110M vs 560M parameters)

---

### Overall Model Comparison

| Approach | Model | Accuracy | Macro F1 | Parameters | Training Time |
|----------|-------|----------|----------|------------|---------------|
| Traditional ML | Logistic Regression | 75.38% | 0.5701 | 5K features | < 1 min |
| Deep Learning | GRU + Word2Vec | - | 0.3307* | 2.6M | ~30 min |
| **Transformers** | **XLM-RoBERTa Large** | **70.34%** | **0.5465** | **560M** | **~2 hours** |
| Transformers | NepaliBERT | 69.72% | 0.5126 | 110M | ~1.5 hours |

*GRU test F1 (CV F1 was 0.6368 ± 0.0358)

![Overall Comparison Chart](notebooks/models/saved_models/overall_comparison.png)

**Key Insights:**
1. 🏆 **XLM-RoBERTa achieves best overall performance**
2. 📊 Transformers significantly outperform traditional ML on minority classes
3. ⚠️ GRU suffers from severe overfitting despite strong CV performance
4. 🎯 Class imbalance remains primary challenge (14.98× ratio)
5. 💡 Multilingual pre-training (XLM-R) more effective than language-specific (NepaliBERT)

---

## 🖼️ Demo & Screenshots

### Web Application Interface

![Main Interface](notebooks/models/saved_models/app_main_interface.png)

**Single Text Prediction:**

![Single Prediction](notebooks/models/saved_models/app_single_prediction.png)

Features:
- Real-time preprocessing visualization
- Confidence scores for all classes
- Probability distribution chart
- Optional history saving

---

**Batch Processing:**

![Batch Processing](notebooks/models/saved_models/app_batch_processing.png)

Features:
- CSV file upload (up to 10MB)
- Text area input (multiple lines)
- Progress tracking
- Results table with downloadable CSV
- Summary statistics and visualizations

---

**Explainability - LIME:**

![LIME Explanation](notebooks/models/saved_models/app_lime_explanation.png)

Shows token-level importance with color-coded visualization:
- 🟢 Green: Words supporting the predicted class
- 🔴 Red: Words arguing against the predicted class
- Intensity: Proportional to importance magnitude

---

**Explainability - SHAP:**

![SHAP Explanation](notebooks/models/saved_models/app_shap_explanation.png)

Shapley value-based attributions with theoretical guarantees.

---

**Explainability - Captum (Integrated Gradients):**

![Captum Explanation](notebooks/models/saved_models/app_captum_explanation.png)

Gradient-based token attributions with convergence metrics.

---

**Prediction History & Analytics:**

![History Dashboard](notebooks/models/saved_models/app_history_dashboard.png)

Features:
- Session statistics (live tracking)
- Persistent history (saved predictions)
- Class distribution visualization
- Download history as CSV
- Clear history functionality

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)
- 8GB RAM minimum (16GB recommended)

### Clone Repository

```bash
git clone https://github.com/UddavRajbhandari/major-project.git
cd major-project
```

### Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Model (Optional)

The model will be automatically downloaded from HuggingFace Hub on first run. To pre-download:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "UDHOV/xlm-roberta-large-nepali-hate-classification"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## 💻 Usage

### Web Application

Launch the Streamlit web interface:

```bash
streamlit run main_app.py
```

The application will open in your browser at `http://localhost:8501`


---

## 📁 Project Structure

```
major-project/
├── main_app.py                      # Streamlit web application
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
│
├── data/                           # Dataset directory
│   ├── processed/                  # Preprocessed data
│   └── prediction_history.json    # Saved predictions
│   └── train.json 
│   └── train_final.json
│   └── val_final.json
│   └── test.json   
│   └── example_batch.csv # input sample for inference
│   └── example.txt # input samples
│
├── models/                         # Trained models
│   ├── saved_models/   # xlm-roberta model           
│
├── scripts/                        # Core modules
│   ├── transformer_data_preprocessing.py  # Preprocessing pipeline
│   ├── explainability.py                  # LIME & SHAP
│   ├── captum_explainer.py              # Integrated Gradients
│
├── notebooks/                      # Jupyter notebooks
│   ├── models/
│       ├── saved_models/ # contains results img and model xlmr
│   ├── analysis_output/       # per class analysis
│   ├── emoji_analysis/
│   ├── results/               # lime / shap / captum results
│   ├── saved_models/
│   ├── 01_dataset.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ml_results.ipynb
│   ├── 04_gru-exp.ipynb
│   ├── 05_xlm-roberta.ipynb
│   ├── 06_nepalibert.ipynb
│   └── 07_explainability.ipynb
│   ├── 08_captum-xlm.ipynb
│   ├── 09_llm_prompting.ipynb
│
├── fonts/                          # Nepali fonts
│   └── Kalimati.ttf               # For Devanagari rendering
│
└── utils/                          
    ├── preprocessing.py
    ├── visualization.py
    └── evaluation.py
```

---

## 📊 Dataset

### Source

The dataset is from Niraula et al. (2021), containing 7,625 manually annotated Nepali social media comments.

**Citation:**
```bibtex
@inproceedings{niraula2021offensive,
  title={Offensive Language Detection in Nepali Social Media},
  author={Niraula, Nobal B and others},
  booktitle={Proceedings of LREC},
  year={2021}
}
```

### Statistics

| Split | NO | OO | OR | OS | Total |
|-------|-----|-----|-----|-----|-------|
| Train | 3,206 (57.7%) | 1,759 (31.6%) | 376 (6.8%) | 214 (3.8%) | 5,555 |
| Val | 356 (57.5%) | 195 (31.5%) | 42 (6.8%) | 27 (4.4%) | 620 |
| Test | 896 (62.1%) | 486 (33.7%) | 49 (3.4%) | 19 (1.3%) | 1,450 |
| **Total** | **4,458** | **2,440** | **467** | **260** | **7,625** |

**Class Imbalance:**
- NO vs OR: 8.52× ratio
- NO vs OS: 14.98× ratio (most severe)

### Data Format

```json
{
  "ID": "comment_001",
  "Comment": "यो राम्रो छैन 😡",
  "Label_Binary": "offensive",
  "Label_Multiclass": "OO"
}
```

---

## 🏋️ Training

### Traditional ML Baselines


**Hyperparameters:**
- TF-IDF: max_features=5000, ngram_range=(1,3)
- Logistic Regression: C=1.0, solver='liblinear'
- SVM: C=1.0, kernel='linear'
- Random Forest: n_estimators=100, max_depth=None

---

### GRU Model



**Key Parameters:**
- Word2Vec: dim=100, window=5, min_count=2
- GRU: units=128, dropout=0.3, recurrent_dropout=0.3
- Optimizer: Adam (lr=0.001)
- Early stopping: patience=5

---

### Transformer Models



**Supported Models:**
- `xlm-roberta-large` (560M params) - **Recommended**
- `xlm-roberta-base` (270M params)
- `nepali-bert` (110M params)

**Training Configuration:**
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Batch size: 16 (with gradient accumulation=2)
- Mixed precision: FP16
- Warmup steps: 10% of total steps

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 scripts/ main_app.py
black --check scripts/ main_app.py

# Format code
black scripts/ main_app.py
```

### Contribution Areas

- 🐛 Bug fixes
- ✨ New features (e.g., additional explainability methods)
- 📝 Documentation improvements
- 🧪 Additional test coverage
- 🌐 UI/UX enhancements
- 📊 New visualization types
- 🔧 Performance optimizations

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{nepali-hate-content-2025,
  author = {Uddav Rajbhandari},
  title = {Nepali Hate content Detection System with Transformer-based Models},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/UddavRajbhandari/major-project}
}
```

**Dataset Citation:**
```bibtex
@inproceedings{niraula2021offensive,
  title={Offensive Language Detection in Nepali Social Media},
  author={Niraula, Nobal B and others},
  booktitle={Proceedings of LREC},
  year={2021}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 Acknowledgments

### Research & Datasets
- **Niraula et al. (2021)** for the Nepali hate speech dataset
- **Hugging Face** for transformer models and infrastructure
- **Google Translate API** for translation services

### Libraries & Frameworks
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - Pre-trained models
- [Streamlit](https://streamlit.io/) - Web application framework
- [LIME](https://github.com/marcotcr/lime) - Model explainability
- [SHAP](https://github.com/slundberg/shap) - Shapley values
- [Captum](https://captum.ai/) - Model interpretability

### Fonts
- **Kalimati** - Nepali Devanagari font

### Community
- Nepali NLP community for feedback and support
- Contributors and testers

---


## 🗺️ Roadmap

### Completed ✅
- [x] Multi-class hate content classification
- [x] Mixed-script preprocessing pipeline
- [x] Emoji semantic mapping (180+ emojis)
- [x] Traditional ML baselines
- [x] GRU with Word2Vec
- [x] Transformer fine-tuning (XLM-R, NepaliBERT)
- [x] Three explainability methods (LIME, SHAP, Captum)
- [x] Web application deployment
- [x] Batch processing
- [x] Prediction history tracking


### Future Work 🔮
- [ ] Real-time social media monitoring
- [ ] Multi-lingual expansion (Hindi, Bengali)
- [ ] Temporal hate content evolution analysis
- [ ] User feedback integration
- [ ] Bias detection and mitigation
- [ ] Adversarial robustness testing

---

## ⚠️ Disclaimer

This tool is designed for research and educational purposes. Automated hate content detection systems are not perfect and should not be used as the sole basis for content moderation decisions. Human oversight and context consideration are essential for responsible deployment.

**Limitations:**
- Model performance varies across classes (especially minority classes)
- Context-dependent meanings may be misclassified
- Emerging slang and new emojis may not be recognized
- Cultural nuances may be missed
- Should be used as an assistive tool, not autonomous system

---

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ for the Nepali NLP community

[Report Bug](https://github.com/UddavRajbhandari/major-project/issues) • [Request Feature](https://github.com/UddavRajbhandari/major-project/issues) 

---