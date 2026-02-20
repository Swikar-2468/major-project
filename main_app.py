"""
Nepali Hate Speech Detection - Streamlit Application
=====================================================
Complete application with preprocessing, prediction, and explainability (LIME/SHAP/Captum)

Run with: streamlit run main_app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Matplotlib for Nepali font support
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Import custom modules
try:
    from scripts.transformer_data_preprocessing import (
        HateSpeechPreprocessor, 
        preprocess_text,
        get_script_info,
        get_emoji_info,
        EMOJI_TO_NEPALI
    )
    from scripts.explainability import (
        create_explainer_wrapper,
        LIMEExplainer,
        SHAPExplainer,
        check_availability as check_explainability
    )
    from scripts.captum_explainer import (
        CaptumExplainer,
        check_availability as check_captum_availability
    )
    CUSTOM_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Custom modules not found: {e}")
    CUSTOM_MODULES_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Nepali Hate Content Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        font-weight: 600;
    }
    
    .no-box { background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%); }
    .oo-box { background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%); }
    .or-box { background: linear-gradient(135deg, #dc3545 0%, #a71d2a 100%); }
    .os-box { background: linear-gradient(135deg, #6f42c1 0%, #4a1f9e 100%); }
    
    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# NEPALI FONT LOADING
# ============================================================================

@st.cache_resource
def load_nepali_font():
    """Load Nepali font for matplotlib visualizations."""
    # Common font paths to try
    font_paths = [
        'fonts/Kalimati.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
        '/System/Library/Fonts/Supplemental/DevanagariSangamMN.ttc',
        'C:\\Windows\\Fonts\\NirmalaUI.ttf',
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                fontManager.addfont(font_path)
                fp = FontProperties(fname=font_path)
                st.info(f"✅ Loaded Nepali font: {fp.get_name()}")
                return fp
            except Exception as e:
                continue
    
    # If no font found, warn user but continue
    st.warning("⚠️ Nepali font not found. Devanagari text may display as squares. "
              "Place Kalimati.ttf in 'fonts/' directory for proper display.")
    return None


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_text' not in st.session_state:
    st.session_state.last_text = ""
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'batch_mode' not in st.session_state:
    st.session_state.batch_mode = None
if 'csv_text_column' not in st.session_state:
    st.session_state.csv_text_column = None
if 'explainability_results' not in st.session_state:
    st.session_state.explainability_results = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model_wrapper' not in st.session_state:
    st.session_state.model_wrapper = None
if 'nepali_font' not in st.session_state:
    st.session_state.nepali_font = None

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model_and_preprocessor():
    """Load model, tokenizer, label encoder, and preprocessor."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import joblib
    
    local_model_path = 'models/saved_models/xlm_roberta_results/large_final'
    hf_model_id = "UDHOV/xlm-roberta-large-nepali-hate-classification"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize label encoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(['NO', 'OO', 'OR', 'OS'])
    
    # Try LOCAL model first
    if os.path.exists(local_model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            model.to(device).eval()
            
            le_path = os.path.join(local_model_path, 'label_encoder.pkl')
            if os.path.exists(le_path):
                le = joblib.load(le_path)
            
            st.success(f"✅ Model loaded from LOCAL path on {device}")
            
        except Exception as e:
            st.warning(f"⚠️ Local model failed: {e}")
            st.info("Trying HuggingFace Hub...")
            
            # Fallback to HuggingFace
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
            model.to(device).eval()
            
            try:
                from huggingface_hub import hf_hub_download
                le_file = hf_hub_download(repo_id=hf_model_id, filename="label_encoder.pkl")
                le = joblib.load(le_file)
            except:
                pass
            
            st.success(f"✅ Model loaded from HuggingFace Hub on {device}")
    else:
        # HuggingFace only
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
        model.to(device).eval()
        
        try:
            from huggingface_hub import hf_hub_download
            le_file = hf_hub_download(repo_id=hf_model_id, filename="label_encoder.pkl")
            le = joblib.load(le_file)
        except:
            pass
        
        st.success(f"✅ Model loaded from HuggingFace Hub on {device}")
    
    # Initialize preprocessor
    if CUSTOM_MODULES_AVAILABLE:
        preprocessor = HateSpeechPreprocessor(
            model_type="xlmr",
            translate_english=True,
            cache_size=2000
        )
    else:
        preprocessor = None
    
    return model, tokenizer, le, preprocessor, device


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_text(text, model, tokenizer, label_encoder, preprocessor, max_length=256):
    """Make prediction with preprocessing."""
    device = next(model.parameters()).device
    
    # Preprocess
    if preprocessor:
        preprocessed, emoji_features = preprocessor.preprocess(text, verbose=False)
    else:
        preprocessed = text
        emoji_features = {}
    
    if not preprocessed.strip():
        return {
            'prediction': 'NO',
            'confidence': 0.0,
            'probabilities': {label: 0.0 for label in label_encoder.classes_},
            'preprocessed_text': '',
            'emoji_features': emoji_features,
            'error': 'Empty text after preprocessing'
        }
    
    # Tokenize
    inputs = tokenizer(
        preprocessed,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    probs_np = probs.cpu().numpy()
    pred_idx = np.argmax(probs_np)
    pred_label = label_encoder.classes_[pred_idx]
    confidence = probs_np[pred_idx]
    
    return {
        'prediction': pred_label,
        'confidence': float(confidence),
        'probabilities': {
            label_encoder.classes_[i]: float(probs_np[i])
            for i in range(len(label_encoder.classes_))
        },
        'preprocessed_text': preprocessed,
        'emoji_features': emoji_features
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_probabilities(probabilities):
    """Create probability bar chart."""
    labels = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = {
        'NO': '#28a745',
        'OO': '#ffc107',
        'OR': '#dc3545',
        'OS': '#6f42c1'
    }
    bar_colors = [colors.get(label, '#6c757d') for label in labels]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probs,
            marker_color=bar_colors,
            text=[f'{p:.2%}' for p in probs],
            textposition='outside',
            hovertemplate='%{x}<br>Probability: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1.1],
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def get_label_description(label):
    """Get description for each label."""
    descriptions = {
        'NO': '✅ Non-Offensive: The text does not contain hate speech or offensive content.',
        'OO': '⚠️ Other-Offensive: Contains general offensive language but not targeted hate.',
        'OR': '🚫 Offensive-Racist: Contains hate speech targeting race, ethnicity, or religion.',
        'OS': '🚫 Offensive-Sexist: Contains hate speech targeting gender or sexuality.'
    }
    return descriptions.get(label, 'Unknown category')


# ============================================================================
# HISTORY MANAGEMENT
# ============================================================================

def save_prediction_to_history(text, result, feedback=None):
    """Save prediction to history file."""
    history_file = 'data/prediction_history.json'
    os.makedirs('data', exist_ok=True)
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text,
        'prediction': result.get('prediction'),
        'confidence': result.get('confidence'),
        'probabilities': result.get('probabilities'),
        'preprocessed_text': result.get('preprocessed_text'),
        'emoji_features': result.get('emoji_features', {}),
        'feedback': feedback
    }
    
    # Load existing history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    # Append and save
    history.append(entry)
    
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save history: {e}")
        return False


# ============================================================================
# BATCH EXPLAINABILITY HELPER
# ============================================================================

def render_batch_explainability(results_df, text_column, model, tokenizer, label_encoder, 
                                preprocessor, nepali_font, explainability_available, 
                                captum_available, mode_key="batch"):
    """
    Render explainability UI for batch results.
    
    Args:
        results_df: DataFrame with batch results
        text_column: Name of column containing full text
        model: Model instance
        tokenizer: Tokenizer instance
        label_encoder: Label encoder
        preprocessor: Preprocessor instance
        nepali_font: Nepali font properties
        explainability_available: Dict with lime/shap availability
        captum_available: Whether Captum is available
        mode_key: Unique key prefix for widgets ("batch" or "csv")
    """
    if not CUSTOM_MODULES_AVAILABLE:
        st.warning("⚠️ Explainability not available.")
        return
    
    if not (explainability_available['lime'] or explainability_available['shap'] or captum_available):
        st.warning("⚠️ No explainability methods available.")
        return
    
    with st.expander("💡 Explain Individual Results", expanded=False):
        st.markdown("**Select a text from the batch to explain:**")
        
        # Create selection dropdown
        text_options = [f"Row {idx}: {str(row[text_column])[:50]}..." for idx, row in results_df.iterrows()]
        selected_idx = st.selectbox(
            "Choose text:", 
            range(len(text_options)), 
            format_func=lambda x: text_options[x], 
            key=f"{mode_key}_select"
        )
        
        selected_text = str(results_df.iloc[selected_idx][text_column])
        selected_pred = results_df.iloc[selected_idx]['Prediction']
        
        st.write(f"**Selected:** {selected_text}")
        st.write(f"**Prediction:** {selected_pred}")
        
        # Method selection
        available_methods = []
        if explainability_available['lime']:
            available_methods.append("LIME")
        if explainability_available['shap']:
            available_methods.append("SHAP")
        if captum_available:
            available_methods.append("Captum (IG)")
        
        if not available_methods:
            st.warning("⚠️ No explainability methods available.")
            return
        
        explain_method = st.selectbox(
            "Explanation method:", 
            available_methods, 
            key=f"{mode_key}_method"
        )
        
        if st.button("🔍 Generate Explanation", key=f"{mode_key}_explain_btn"):
            with st.spinner("Generating explanation..."):
                try:
                    # Create model wrapper if needed
                    if st.session_state.model_wrapper is None:
                        st.session_state.model_wrapper = create_explainer_wrapper(
                            model, tokenizer, label_encoder, preprocessor
                        )
                    
                    wrapper = st.session_state.model_wrapper
                    
                    # Clean text of quotes before processing
                    clean_selected = selected_text.replace('"', '').replace("'", '').replace('"', '').replace('"', '')
                    
                    # Preprocess and analyze
                    preprocessed, emoji_features = preprocessor.preprocess(clean_selected)
                    analysis = wrapper.predict_with_analysis(clean_selected)
                    
                    if explain_method == "LIME":
                        lime_exp = LIMEExplainer(wrapper, nepali_font=nepali_font)
                        result = lime_exp.explain_and_visualize(
                            analysis['original_text'],
                            analysis['preprocessed_text'],
                            save_path=None,
                            show=False,
                            num_samples=200
                        )
                        
                        st.subheader("LIME Explanation")
                        st.pyplot(result['figure'])
                        
                        # Show details directly without nested expander
                        st.markdown("---")
                        st.markdown("**📊 Feature Importance Details:**")
                        word_scores = result['explanation']['word_scores']
                        if word_scores:
                            df = pd.DataFrame(word_scores, columns=['Word', 'Score'])
                            df = df.sort_values('Score', ascending=False)
                            st.dataframe(df, hide_index=True, use_container_width=True)
                        else:
                            st.warning("No word scores available")
                    
                    elif explain_method == "SHAP":
                        shap_exp = SHAPExplainer(wrapper, nepali_font=nepali_font)
                        result = shap_exp.explain_and_visualize(
                            analysis['original_text'],
                            analysis['preprocessed_text'],
                            save_path=None,
                            show=False,
                            use_fallback=True
                        )
                        
                        st.subheader("SHAP Explanation")
                        st.pyplot(result['figure'])
                        
                        # Show details directly without nested expander
                        st.markdown("---")
                        st.markdown("**📊 Attribution Details:**")
                        st.write(f"**Method used:** {result['explanation']['method_used']}")
                        word_scores = result['explanation']['word_scores']
                        if word_scores:
                            df = pd.DataFrame(word_scores, columns=['Word', 'Score'])
                            df = df.sort_values('Score', key=lambda x: abs(x), ascending=False)
                            st.dataframe(df, hide_index=True, use_container_width=True)
                        else:
                            st.warning("No word scores available")
                    
                    elif explain_method == "Captum (IG)":
                        captum_exp = CaptumExplainer(
                            model, tokenizer, label_encoder, preprocessor,
                            emoji_to_nepali_map=EMOJI_TO_NEPALI
                        )
                        result = captum_exp.explain_and_visualize(
                            analysis['original_text'],
                            target=None,
                            n_steps=50,
                            save_dir=None,
                            show=False,
                            nepali_font=nepali_font
                        )
                        
                        st.subheader("Captum Integrated Gradients")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Bar Chart**")
                            st.pyplot(result['bar_chart'])
                        
                        with col2:
                            st.markdown("**Heatmap**")
                            st.pyplot(result['heatmap'])
                        
                        # Show details directly without nested expander
                        st.markdown("---")
                        st.markdown("**📊 Attribution Details:**")
                        st.write(f"**Convergence Delta:** {result['explanation']['convergence_delta']:.6f}")
                        
                        word_attrs = result['explanation']['word_attributions']
                        if word_attrs:
                            df = pd.DataFrame(
                                word_attrs,
                                columns=['Word', 'Abs Score', 'Signed Score']
                            )
                            df = df.sort_values('Abs Score', ascending=False)
                            st.dataframe(df, hide_index=True, use_container_width=True)
                        else:
                            st.warning("No word attributions available")
                
                except Exception as e:
                    st.error(f"❌ Explanation failed: {str(e)}")
                    # Show error details directly without nested expander
                    st.markdown("**🐛 Error Details:**")
                    st.code(str(e))
                    import traceback
                    st.code(traceback.format_exc())


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application."""
    
    # Load Nepali font
    if st.session_state.nepali_font is None:
        st.session_state.nepali_font = load_nepali_font()
    
    nepali_font = st.session_state.nepali_font
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Nepali Hate Content Detector</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
    AI-powered hate speech detection for Nepali text with advanced explainability
    <br>
    <strong>XLM-RoBERTa Large</strong> fine-tuned on Nepali social media data
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Model**: XLM-RoBERTa Large  
        **Task**: Multi-class hate speech detection  
        **Language**: Nepali (Devanagari & Romanized)
        
        **Classes:**
        - **NO**: Non-offensive
        - **OO**: General offensive
        - **OR**: Racist/ethnic hate
        - **OS**: Sexist/gender hate
        """)
        
        st.markdown("---")
        
        st.header("🔧 Features")
        st.markdown("""
        ✅ **Preprocessing**
        - Script detection
        - Transliteration
        - Translation
        - Emoji mapping
        
        ✅ **Explainability**
        - LIME
        - SHAP
        - Captum (IG)
        
        ✅ **Batch Analysis**
        - CSV upload
        - Text area input
        """)
        
        st.markdown("---")
        
        st.header("🎨 Font Settings")
        with st.expander("Nepali Font Info", expanded=False):
            st.markdown(f"""
            **Status:** {'✅ Loaded' if nepali_font else '❌ Not loaded'}
            
            **Fix squares in Devanagari:**
            1. Download Kalimati.ttf
            2. Create `fonts/` directory
            3. Place font file there
            4. Restart app
            """)
        
        st.markdown("---")
        
        st.header("📊 Statistics")
        if os.path.exists('data/prediction_history.json'):
            try:
                with open('data/prediction_history.json', 'r') as f:
                    history = json.load(f)
                st.metric("Total Predictions", len(history))
                
                if history:
                    pred_counts = pd.Series([h['prediction'] for h in history]).value_counts()
                    st.write("**Distribution:**")
                    for label, count in pred_counts.items():
                        st.write(f"• {label}: {count}")
            except:
                pass
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.9rem; color: #666;'>
        <a href='https://huggingface.co/UDHOV/xlm-roberta-large-nepali-hate-classification' target='_blank'>
        Model on HuggingFace 🤗
        </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, label_encoder, preprocessor, device = load_model_and_preprocessor()
    
    if model is None:
        st.error("❌ Failed to load model!")
        st.stop()
    
    # Check explainability availability
    explainability_available = check_explainability() if CUSTOM_MODULES_AVAILABLE else {'lime': False, 'shap': False}
    captum_available = check_captum_availability() if CUSTOM_MODULES_AVAILABLE else False
    
    # Main interface
    tabs = st.tabs([
        "🔍 Single Prediction",
        "💡 Explainability",
        "📝 Batch Analysis",
        "📈 History"
    ])
    
    # ========================================================================
    # TAB 1: SINGLE PREDICTION
    # ========================================================================
    
    with tabs[0]:
        st.subheader("🔍 Single Text Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter Nepali Text",
                height=200,
                placeholder="यहाँ आफ्नो पाठ लेख्नुहोस्...\nOr enter romanized Nepali: ma khusi xu\nOr English: This is a test",
                help="Enter text in Devanagari, Romanized Nepali, or English. The system will automatically detect and convert."
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
            with col_b:
                save_to_history = st.checkbox("Save to history", value=True)
        
        with col2:
            st.markdown("##### 💡 Quick Info")
            st.info("""
            **Supported:**
            - Devanagari: नेपाली
            - Romanized: ma nepali xu
            - English: I am Nepali
            - Mixed scripts
            - Emojis: 😀😡🙏
            
            **Auto-processing:**
            - Script detection
            - Transliteration
            - Translation
            - Emoji → Nepali words
            - URL/mention removal
            """)
        
        # Analysis
        if analyze_button and text_input.strip():
            with st.spinner("🔄 Analyzing text..."):
                # Predict
                result = predict_text(
                    text_input, model, tokenizer, 
                    label_encoder, preprocessor
                )
                
                st.session_state.last_prediction = result
                st.session_state.last_text = text_input
                
                # Save to history
                if save_to_history:
                    save_prediction_to_history(text_input, result)
            
            if 'error' in result:
                st.warning(f"⚠️ {result['error']}")
                st.stop()
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Prediction box
            pred_label = result['prediction']
            confidence = result['confidence']
            
            box_class = {
                'NO': 'no-box',
                'OO': 'oo-box',
                'OR': 'or-box',
                'OS': 'os-box'
            }.get(pred_label, 'no-box')
            
            st.markdown(f"""
            <div class='prediction-box {box_class}'>
                <h2 style='margin:0;'>Prediction: {pred_label}</h2>
                <p style='font-size:1.3rem; margin:0.5rem 0;'>
                    Confidence: <strong>{confidence:.2%}</strong>
                </p>
                <p style='margin:0; font-size:1rem;'>{get_label_description(pred_label)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability chart
            st.plotly_chart(plot_probabilities(result['probabilities']), use_container_width=True)
            
            # Details
            with st.expander("🔍 Preprocessing Details", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Original Text:**")
                    st.code(text_input, language=None)
                
                with col2:
                    st.markdown("**Preprocessed:**")
                    st.code(result['preprocessed_text'], language=None)
                
                with col3:
                    # Script info
                    if CUSTOM_MODULES_AVAILABLE and preprocessor:
                        script_info = get_script_info(text_input)
                        st.markdown("**Script Detected:**")
                        st.write(f"• Type: {script_info['script_type']}")
                        # Cap confidence at 100%
                        confidence_pct = min(script_info['confidence'] * 100, 100.0)
                        st.write(f"• Confidence: {confidence_pct:.1f}%")
            
            # Emoji features
            if result.get('emoji_features', {}).get('total_emoji_count', 0) > 0:
                with st.expander("😊 Emoji Analysis", expanded=False):
                    features = result['emoji_features']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Emojis", features['total_emoji_count'])
                        st.metric("Hate Emojis", features['hate_emoji_count'])
                    
                    with col2:
                        st.metric("Positive Emojis", features['positive_emoji_count'])
                        st.metric("Mockery Emojis", features['mockery_emoji_count'])
                    
                    with col3:
                        st.metric("Sadness Emojis", features['sadness_emoji_count'])
                        st.metric("Fear Emojis", features['fear_emoji_count'])
                    
                    if CUSTOM_MODULES_AVAILABLE:
                        emoji_info = get_emoji_info(text_input)
                        if emoji_info['emojis_found']:
                            st.markdown("**Emojis Found:**")
                            st.write(" ".join(emoji_info['emojis_found']))
            
            # All probabilities
            with st.expander("📊 Detailed Probabilities", expanded=False):
                prob_df = pd.DataFrame({
                    'Class': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                })
                prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.4f}")
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
    
    # ========================================================================
    # TAB 2: EXPLAINABILITY
    # ========================================================================
    
    with tabs[1]:
        st.subheader("💡 Model Explainability")
        
        if not CUSTOM_MODULES_AVAILABLE:
            st.error("❌ Explainability modules not available. Please check scripts directory.")
            st.stop()
        
        # Check what's available
        st.info(f"""
        **Available Methods:**
        - LIME: {'✅' if explainability_available['lime'] else '❌ (install: pip install lime)'}
        - SHAP: {'✅' if explainability_available['shap'] else '❌ (install: pip install shap)'}
        - Captum: {'✅' if captum_available else '❌ (install: pip install captum)'}
        """)
        
        # Text input
        explain_text = st.text_area(
            "Enter text to explain",
            height=150,
            value=st.session_state.last_text if st.session_state.last_text else "",
            placeholder="Enter Nepali text..."
        )
        
        # Method selection
        available_methods = []
        if explainability_available['lime']:
            available_methods.append("LIME")
        if explainability_available['shap']:
            available_methods.append("SHAP")
        if captum_available:
            available_methods.append("Captum (IG)")
        
        if not available_methods:
            st.warning("⚠️ No explainability methods available. Please install required packages.")
            st.code("pip install lime shap captum", language="bash")
            st.stop()
        
        method = st.selectbox("Select explanation method", available_methods)
        
        # Configuration
        with st.expander("⚙️ Configuration", expanded=False):
            if method == "LIME":
                num_samples = st.slider("Number of samples", 100, 500, 200, 50)
            elif method == "SHAP":
                use_fallback = st.checkbox("Use fallback if SHAP fails", value=True)
            elif method == "Captum (IG)":
                n_steps = st.slider("Integration steps", 10, 100, 50, 10)
        
        explain_button = st.button("🔍 Generate Explanation", type="primary", use_container_width=True)
        
        if explain_button and explain_text.strip():
            with st.spinner("Generating explanation..."):
                # Create model wrapper
                if st.session_state.model_wrapper is None:
                    st.session_state.model_wrapper = create_explainer_wrapper(
                        model, tokenizer, label_encoder, preprocessor
                    )
                
                wrapper = st.session_state.model_wrapper
                
                # Get preprocessing
                preprocessed, emoji_features = preprocessor.preprocess(explain_text)
                
                # Prediction
                analysis = wrapper.predict_with_analysis(explain_text)
                
                # Display prediction first
                st.success(f"**Prediction:** {analysis['predicted_label']} ({analysis['confidence']:.2%})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original:**", explain_text)
                with col2:
                    st.write("**Preprocessed:**", preprocessed)
                
                st.markdown("---")
                
                # Generate explanation
                try:
                    if method == "LIME":
                        lime_exp = LIMEExplainer(wrapper, nepali_font=nepali_font)
                        result = lime_exp.explain_and_visualize(
                            analysis['original_text'],
                            analysis['preprocessed_text'],
                            save_path=None,
                            show=False,
                            num_samples=num_samples
                        )
                        
                        st.subheader("LIME Explanation")
                        st.pyplot(result['figure'])
                        
                        with st.expander("📊 Feature Importance Details"):
                            word_scores = result['explanation']['word_scores']
                            df = pd.DataFrame(word_scores, columns=['Word', 'Score'])
                            df = df.sort_values('Score', ascending=False)
                            st.dataframe(df, hide_index=True, use_container_width=True)
                    
                    elif method == "SHAP":
                        shap_exp = SHAPExplainer(wrapper, nepali_font=nepali_font)
                        result = shap_exp.explain_and_visualize(
                            analysis['original_text'],
                            analysis['preprocessed_text'],
                            save_path=None,
                            show=False,
                            use_fallback=use_fallback
                        )
                        
                        st.subheader("SHAP Explanation")
                        st.pyplot(result['figure'])
                        
                        with st.expander("📊 Attribution Details"):
                            st.write(f"**Method used:** {result['explanation']['method_used']}")
                            word_scores = result['explanation']['word_scores']
                            df = pd.DataFrame(word_scores, columns=['Word', 'Score'])
                            df = df.sort_values('Score', key=lambda x: abs(x), ascending=False)
                            st.dataframe(df, hide_index=True, use_container_width=True)
                    
                    elif method == "Captum (IG)":
                        captum_exp = CaptumExplainer(
                            model, tokenizer, label_encoder, preprocessor,
                            emoji_to_nepali_map=EMOJI_TO_NEPALI
                        )
                        
                        result = captum_exp.explain_and_visualize(
                            analysis['original_text'],
                            target=None,
                            n_steps=n_steps,
                            save_dir=None,  # Changed from save_path
                            show=False,
                            nepali_font=nepali_font
                        )
                        
                        st.subheader("Captum Integrated Gradients")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Bar Chart**")
                            st.pyplot(result['bar_chart'])
                        
                        with col2:
                            st.markdown("**Heatmap**")
                            st.pyplot(result['heatmap'])
                        
                        with st.expander("📊 Attribution Details"):
                            st.write(f"**Convergence Delta:** {result['explanation']['convergence_delta']:.6f}")
                            
                            word_attrs = result['explanation']['word_attributions']
                            df = pd.DataFrame(
                                word_attrs,
                                columns=['Word', 'Abs Score', 'Signed Score']
                            )
                            df = df.sort_values('Abs Score', ascending=False)
                            st.dataframe(df, hide_index=True, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Explanation failed: {str(e)}")
                    with st.expander("🐛 Error Details"):
                        st.exception(e)
    
    # ========================================================================
    # TAB 3: BATCH ANALYSIS
    # ========================================================================
    
    with tabs[2]:
        st.subheader("📝 Batch Analysis")
        
        # Example files
        st.markdown("### 📥 Download Example Files")
        col1, col2 = st.columns(2)
        
        with col1:
            example_csv_data = {
                'text': [
                    'यो राम्रो छ',
                    'तिमी मुर्ख हौ',
                    'मुस्लिम हरु सबै खराब छन्',
                    'केटीहरु घरमा बस्नु पर्छ',
                    'नमस्ते, कस्तो छ?'
                ]
            }
            example_csv = pd.DataFrame(example_csv_data).to_csv(index=False)
            
            st.download_button(
                label="📄 Download Example CSV",
                data=example_csv,
                file_name="example_batch.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            example_text = """यो राम्रो छ
तिमी मुर्ख हौ
मुस्लिम हरु सबै खराब छन्
केटीहरु घरमा बस्नु पर्छ
नमस्ते, कस्तो छ?"""
            
            st.download_button(
                label="📝 Download Example Text",
                data=example_text,
                file_name="example_batch.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Input method
        input_method = st.radio("Input method:", ["Text Area", "CSV Upload"])
        
        if input_method == "Text Area":
            st.info("💡 Enter one text per line")
            
            batch_text = st.text_area(
                "Enter texts (one per line)",
                height=250,
                placeholder="यो राम्रो छ\nतिमी मुर्ख हौ\n..."
            )
            
            if st.button("🚀 Analyze Batch", type="primary"):
                if batch_text.strip():
                    texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
                    
                    with st.spinner(f"Analyzing {len(texts)} texts..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, text in enumerate(texts):
                            try:
                                result = predict_text(
                                    text, model, tokenizer,
                                    label_encoder, preprocessor
                                )
                                results.append({
                                    'Text': text[:60] + '...' if len(text) > 60 else text,
                                    'Full_Text': text,
                                    'Prediction': result['prediction'],
                                    'Confidence': result['confidence'],
                                    'Preprocessed': result['preprocessed_text']
                                })
                            except Exception as e:
                                results.append({
                                    'Text': text[:60],
                                    'Full_Text': text,
                                    'Prediction': 'Error',
                                    'Confidence': 0.0,
                                    'Preprocessed': str(e)
                                })
                            
                            progress_bar.progress((idx + 1) / len(texts))
                        
                        # Store in session state with mode flag
                        st.session_state.batch_results = pd.DataFrame(results)
                        st.session_state.batch_mode = 'text_area'
                        st.rerun()
                else:
                    st.warning("Please enter some texts.")
            
            # Display results OUTSIDE button block if they exist
            if (st.session_state.batch_results is not None and 
                st.session_state.get('batch_mode') == 'text_area'):
                
                results_df = st.session_state.batch_results
                
                st.success(f"✅ Analyzed {len(results_df)} texts!")
                
                display_df = results_df[['Text', 'Prediction', 'Confidence']].copy()
                display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
                
                # Summary
                st.markdown("---")
                st.subheader("📊 Summary Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Texts", len(results_df))
                    st.metric("Avg Confidence", f"{results_df['Confidence'].mean():.2%}")
                
                with col2:
                    summary = results_df['Prediction'].value_counts()
                    fig = px.pie(
                        values=summary.values,
                        names=summary.index,
                        title="Prediction Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    # Class breakdown
                    st.markdown("**Class Breakdown:**")
                    for label, count in summary.items():
                        pct = count / len(results_df) * 100
                        st.write(f"• {label}: {count} ({pct:.1f}%)")
                
                # Download - MOVED OUTSIDE col3
                st.markdown("---")
                download_df = results_df[['Full_Text', 'Prediction', 'Confidence', 'Preprocessed']]
                download_df.columns = ['Text', 'Prediction', 'Confidence', 'Preprocessed']
                csv = download_df.to_csv(index=False)
                
                col_download, col_explain = st.columns(2)
                
                with col_download:
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_batch_text"
                    )
                
                with col_explain:
                    if st.button("💡 Explain Selected", use_container_width=True, key="hint_batch_text"):
                        st.info("👇 Select a text below to explain")
                
                # Add explainability for batch results using helper function
                render_batch_explainability(
                    results_df=results_df,
                    text_column='Full_Text',
                    model=model,
                    tokenizer=tokenizer,
                    label_encoder=label_encoder,
                    preprocessor=preprocessor,
                    nepali_font=nepali_font,
                    explainability_available=explainability_available,
                    captum_available=captum_available,
                    mode_key="text_area"
                )
        
        else:  # CSV Upload
            st.info("💡 Upload CSV with a 'text' column")
            
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("📄 **File Preview:**")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    text_column = st.selectbox("Select text column:", df.columns)
                    
                    if st.button("🚀 Analyze CSV", type="primary"):
                        texts = df[text_column].astype(str).tolist()
                        
                        with st.spinner(f"Analyzing {len(texts)} texts..."):
                            predictions = []
                            confidences = []
                            preprocessed_list = []
                            progress_bar = st.progress(0)
                            
                            for idx, text in enumerate(texts):
                                try:
                                    result = predict_text(
                                        str(text), model, tokenizer,
                                        label_encoder, preprocessor
                                    )
                                    predictions.append(result['prediction'])
                                    confidences.append(result['confidence'])
                                    preprocessed_list.append(result['preprocessed_text'])
                                except Exception as e:
                                    predictions.append('Error')
                                    confidences.append(0.0)
                                    preprocessed_list.append(str(e))
                                
                                progress_bar.progress((idx + 1) / len(texts))
                            
                            df['Prediction'] = predictions
                            df['Confidence'] = confidences
                            df['Preprocessed'] = preprocessed_list
                            
                            # Store in session state with mode and column info
                            st.session_state.batch_results = df
                            st.session_state.batch_mode = 'csv'
                            st.session_state.csv_text_column = text_column
                            st.rerun()
                    
                    # Display results OUTSIDE button block if they exist
                    if (st.session_state.batch_results is not None and 
                        st.session_state.get('batch_mode') == 'csv'):
                        
                        df_results = st.session_state.batch_results
                        text_col = st.session_state.get('csv_text_column', text_column)
                        
                        st.success("✅ Analysis complete!")
                        st.dataframe(df_results, use_container_width=True, height=400)
                        
                        # Summary
                        st.markdown("---")
                        st.subheader("📊 Summary")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            summary = df_results['Prediction'].value_counts()
                            fig = px.bar(
                                x=summary.index,
                                y=summary.values,
                                title="Prediction Distribution",
                                labels={'x': 'Class', 'y': 'Count'},
                                color=summary.index,
                                color_discrete_map={
                                    'NO': '#28a745',
                                    'OO': '#ffc107',
                                    'OR': '#dc3545',
                                    'OS': '#6f42c1'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.metric("Total Texts", len(df_results))
                            st.metric("Avg Confidence", f"{df_results['Confidence'].mean():.2%}")
                            
                            st.markdown("**Class Distribution:**")
                            for label, count in summary.items():
                                st.write(f"• {label}: {count}")
                        
                        # Download - MOVED OUTSIDE col2
                        st.markdown("---")
                        csv_data = df_results.to_csv(index=False)
                        
                        col_download, col_explain = st.columns(2)
                        
                        with col_download:
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv_data,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key="download_csv_results"
                            )
                        
                        with col_explain:
                            if st.button("💡 Explain Selected", use_container_width=True, key="csv_explain_hint"):
                                st.info("👇 Use expander below to explain")
                        
                        # Add explainability for CSV results using helper function
                        render_batch_explainability(
                            results_df=df_results,
                            text_column=text_col,
                            model=model,
                            tokenizer=tokenizer,
                            label_encoder=label_encoder,
                            preprocessor=preprocessor,
                            nepali_font=nepali_font,
                            explainability_available=explainability_available,
                            captum_available=captum_available,
                            mode_key="csv"
                        )

                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    with st.expander("🐛 Error Details"):
                        st.exception(e)
    
    # ========================================================================
    # TAB 4: HISTORY
    # ========================================================================
    
    with tabs[3]:
        st.subheader("📈 Prediction History")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("View and analyze your prediction history")
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        history_file = 'data/prediction_history.json'
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                if history:
                    history_df = pd.DataFrame(history)
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    
                    # Metrics
                    st.markdown("### 📊 Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Predictions", len(history_df))
                    with col2:
                        st.metric("Avg Confidence", f"{history_df['confidence'].mean():.2%}")
                    with col3:
                        if 'emoji_features' in history_df.columns:
                            total_emojis = sum(
                                e.get('total_emoji_count', 0) 
                                for e in history_df['emoji_features'] 
                                if isinstance(e, dict)
                            )
                            st.metric("Total Emojis", total_emojis)
                        else:
                            st.metric("Total Emojis", "N/A")
                    with col4:
                        most_common = history_df['prediction'].mode()[0]
                        st.metric("Most Common", most_common)
                    
                    # Visualizations
                    st.markdown("---")
                    st.markdown("### 📈 Trends")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Timeline
                        daily_counts = history_df.groupby(history_df['timestamp'].dt.date).size().reset_index(name='count')
                        fig = px.line(
                            daily_counts,
                            x='timestamp',
                            y='count',
                            title="Predictions Over Time",
                            labels={'timestamp': 'Date', 'count': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Distribution
                        class_dist = history_df['prediction'].value_counts()
                        fig = px.pie(
                            values=class_dist.values,
                            names=class_dist.index,
                            title="Class Distribution",
                            color=class_dist.index,
                            color_discrete_map={
                                'NO': '#28a745',
                                'OO': '#ffc107',
                                'OR': '#dc3545',
                                'OS': '#6f42c1'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recent predictions
                    st.markdown("---")
                    st.markdown("### 📋 Recent Predictions")
                    
                    num_to_show = st.slider("Number to show", 5, 50, 20, 5)
                    
                    recent = history_df.tail(num_to_show).sort_values('timestamp', ascending=False)
                    display = recent[['timestamp', 'text', 'prediction', 'confidence']].copy()
                    display['confidence'] = display['confidence'].apply(lambda x: f"{x:.2%}")
                    display['text'] = display['text'].apply(lambda x: x[:80] + '...' if len(x) > 80 else x)
                    display['timestamp'] = display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    st.dataframe(display, use_container_width=True, hide_index=True, height=400)
                    
                    # Download/Clear
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = history_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full History",
                            data=csv,
                            file_name=f"history_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                            if os.path.exists(history_file):
                                os.remove(history_file)
                                st.success("✅ History cleared!")
                                st.rerun()
                
                else:
                    st.info("📝 No predictions in history yet.")
            
            except Exception as e:
                st.error(f"❌ Error loading history: {str(e)}")
                with st.expander("🐛 Error Details"):
                    st.exception(e)
        
        else:
            st.info("📝 No history file found yet.")
            st.markdown("""
            ### How to Build History:
            1. Go to **Single Prediction** tab
            2. Enable "Save to history" checkbox
            3. Analyze some text
            4. Your predictions will appear here!
            """)


if __name__ == "__main__":
    main()