"""
Streamlit Web Application for Nepali Hate Speech Detection
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
import re
import emoji
import regex
from huggingface_hub import hf_hub_download
# Page configuration
st.set_page_config(
    page_title="Nepali Hate Content Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .no-box { background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%); }  /* Dark Green */
    .oo-box { background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%); }  /* Dark Yellow */
    .or-box { background: linear-gradient(135deg, #dc3545 0%, #a71d2a 100%); }  /* Dark Red */
    .os-box { background: linear-gradient(135deg, #6f42c1 0%, #4a1f9e 100%); }  /* Dark Purple */

    </style>
""", unsafe_allow_html=True)

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_text' not in st.session_state:
    st.session_state.last_text = ""


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False

DIRGHIKARAN_MAP = {
    "उ": "ऊ", "इ": "ई", "ऋ": "रि", "ए": "ऐ", "अ": "आ",
    "\u200d": "", "\u200c": "", "।": ".", "॥": ".",
    "ि": "ी", "ु": "ू"
}

def is_devanagari(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(regex.search(r'\p{Devanagari}', text))

def roman_to_devanagari(text: str) -> str:
    if not TRANSLITERATION_AVAILABLE:
        return text
    try:
        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    except:
        return text

def normalize_dirghikaran(text: str) -> str:
    for original, replacement in DIRGHIKARAN_MAP.items():
        text = text.replace(original, replacement)
    return text

def clean_text(text: str, aggressive: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = emoji.replace_emoji(text, replace="")
    
    if aggressive:
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_for_transformer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    if not is_devanagari(text) and TRANSLITERATION_AVAILABLE:
        text = roman_to_devanagari(text)
    
    text = clean_text(text, aggressive=False)
    text = normalize_dirghikaran(text)
    
    return text


# ============================================================================
# MODEL LOADING (LOCAL FIRST, THEN HUGGINGFACE)
# ============================================================================

@st.cache_resource
def load_model():
    """Load model from LOCAL first, then HuggingFace Hub as fallback without stopping."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.preprocessing import LabelEncoder
    import joblib
    import torch

    local_model_path = 'models/saved_models/xlm_roberta_results/large_final'
    hf_model_id = "UDHOV/xlm-roberta-large-nepali-hate-classification"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize label encoder
    le = LabelEncoder()
    le.fit(['NO', 'OO', 'OR', 'OS'])

    # Try loading LOCAL model
    if os.path.exists(local_model_path):
        try:
            with st.spinner("Loading model from local path..."):
                tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
                model.to(device)
                model.eval()

                # Load label encoder if exists
                le_path = os.path.join(local_model_path, 'label_encoder.pkl')
                if os.path.exists(le_path):
                    le = joblib.load(le_path)

                st.success(f"✅ Model loaded from LOCAL path on {device}")
                return model, tokenizer, le
        except Exception as e:
            st.warning(f"⚠️ Local model failed: {e}")
            st.info("Trying HuggingFace Hub...")

    # Fallback to HuggingFace
    try:
        with st.spinner(f"Loading from HuggingFace Hub ({hf_model_id})..."):
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
            model.to(device)
            model.eval()

            # Try to load label encoder from HF repo
            try:
                from huggingface_hub import hf_hub_download
                le_file = hf_hub_download(
                    repo_id=hf_model_id,
                    filename="label_encoder.pkl",
                    cache_dir="models/cache"
                )
                le = joblib.load(le_file)
                st.success("✅ Label encoder loaded from HF repo")
            except Exception:
                st.info("⚠️ Label encoder not found in HF repo, using default encoder")

            st.success(f"✅ Model loaded from HuggingFace Hub on {device}")
            return model, tokenizer, le

    except Exception as hf_error:
        st.error(f"❌ Could not load model from HuggingFace Hub: {hf_error}")
        return None, None, le



# ============================================================================
# PREDICTION
# ============================================================================

def predict(text, model, tokenizer, label_encoder, max_length=256):
    """Make prediction on input text."""
    device = next(model.parameters()).device
    
    preprocessed = preprocess_for_transformer(text)
    
    if not preprocessed.strip():
        return {
            'prediction': 'NO',
            'confidence': 0.0,
            'probabilities': {label: 0.0 for label in label_encoder.classes_},
            'error': 'Empty text after preprocessing'
        }
    
    inputs = tokenizer(
        preprocessed,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    probs_np = probs.cpu().numpy()
    pred_idx = np.argmax(probs_np)
    pred_label = label_encoder.classes_[pred_idx]
    confidence = probs_np[pred_idx]
    
    results = {
        'prediction': pred_label,
        'confidence': float(confidence),
        'probabilities': {
            label_encoder.classes_[i]: float(probs_np[i])
            for i in range(len(label_encoder.classes_))
        },
        'preprocessed_text': preprocessed
    }
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_probabilities(probabilities):
    """Create probability bar chart."""
    labels = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = {'NO': '#28a745', 'OO': '#ffc107', 'OR': '#dc3545', 'OS': '#c82333'}
    bar_colors = [colors.get(label, '#6c757d') for label in labels]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels, y=probs, marker_color=bar_colors,
            text=[f'{p:.2%}' for p in probs], textposition='outside',
            hovertemplate='%{x}<br>Probability: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities", xaxis_title="Class", yaxis_title="Probability",
        yaxis_range=[0, 1.1], height=400, showlegend=False, template='plotly_white'
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
# HISTORY MANAGEMENT (COMPLETELY FIXED)
# ============================================================================

def save_prediction_to_file(text, result, feedback=None):
    """Save prediction - including corrected label and correctness flag."""
    history_file = 'data/prediction_history.json'
    os.makedirs('data', exist_ok=True)

    entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text,
        'prediction': result.get('prediction'),
        'correct_label': result.get('correct_label'),
        'is_correct': result.get('is_correct'),
        'confidence': result.get('confidence'),
        'probabilities': result.get('probabilities'),
        'preprocessed_text': result.get('preprocessed_text'),
        'feedback': feedback
    }

    # Load old history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []

    # append + save
    history.append(entry)

    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return True

    except Exception as e:
        print("SAVE ERROR:", e)
        return False



# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application."""
    
    # Initialize session state
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Nepali Hate Speech Detector</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Classify Nepali social media text into four categories: 
    <strong>Non-Offensive (NO)</strong>, <strong>Other-Offensive (OO)</strong>, 
    <strong>Offensive-Racist (OR)</strong>, and <strong>Offensive-Sexist (OS)</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **XLM-RoBERTa Large** fine-tuned on Nepali hate speech data.
        
        **Classes:**
        - **NO**: Non-offensive
        - **OO**: General offensive
        - **OR**: Racist/ethnic hate
        - **OS**: Sexist/gender hate
        
        [Model on HuggingFace](https://huggingface.co/UDHOV/xlm-roberta-large-nepali-hate-classification)
        """)
        
        st.header("📊 Statistics")
        if os.path.exists('data/prediction_history.json'):
            try:
                with open('data/prediction_history.json', 'r') as f:
                    history = json.load(f)
                st.metric("Total Predictions", len(history))
                
                if history:
                    pred_counts = pd.Series([h['prediction'] for h in history]).value_counts()
                    st.write("**Recent:**")
                    for label, count in pred_counts.items():
                        st.write(f"- {label}: {count}")
            except:
                pass
    
    # Load model
    model, tokenizer, label_encoder = load_model()
    
    if model is None:
        st.stop()
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📝 Batch Analysis", "📈 History"])
    
    # ========================================================================
    # TAB 1: SINGLE PREDICTION
    # ========================================================================

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Enter Nepali Text")
            text_input = st.text_area(
                "Enter text:",
                height=150,
                placeholder="यहाँ आफ्नो पाठ लेख्नुहोस्..."
            )
            analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

        with col2:
            st.subheader("Quick Info")
            st.info("""
            **Tips:**
            - Complete sentences
            - Either script works
            - Avoid excessive emojis
            
            **Auto-processing:**
            - URL removal
            - Emoji removal  
            - Script conversion
            """)

        # Run prediction
        if analyze_button and text_input.strip():
            with st.spinner("Analyzing..."):
                result = predict(text_input, model, tokenizer, label_encoder)
                st.session_state.last_prediction = result
                st.session_state.last_text = text_input

                if 'error' in result:
                    st.warning(f"⚠️ {result['error']}")
                    st.stop()

            # Display prediction
            st.markdown("---")
            st.subheader("📊 Results")
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
                <p style='font-size:1.2rem; margin:0.5rem 0;'>
                    Confidence: <strong>{confidence:.2%}</strong>
                </p>
                <p style='margin:0; font-size:0.95rem;'>{get_label_description(pred_label)}</p>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(plot_probabilities(result['probabilities']), use_container_width=True)

            # Preprocessing details
            with st.expander("🔧 Preprocessing"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.text("Original:")
                    st.code(text_input)
                with col_b:
                    st.text("Preprocessed:")
                    st.code(result['preprocessed_text'])

            
    # ========================================================================
    # TAB 2: BATCH ANALYSIS (DOWNLOAD BUTTON AFTER ANALYSIS)
    # ========================================================================
    with tab2:
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
        
        input_method = st.radio("Input method:", ["Text Area", "CSV Upload"])
        
        if input_method == "Text Area":
            st.info("💡 Enter one text per line")
            
            batch_text = st.text_area(
                "Enter texts:",
                height=200,
                placeholder="यो राम्रो छ\nतिमी मुर्ख हौ\n..."
            )
            
            if st.button("Analyze Batch", type="primary"):
                if batch_text.strip():
                    texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
                    
                    with st.spinner(f"Analyzing {len(texts)} texts..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, text in enumerate(texts):
                            try:
                                result = predict(text, model, tokenizer, label_encoder)
                                results.append({
                                    'Text': text[:50] + '...' if len(text) > 50 else text,
                                    'Full_Text': text,
                                    'Prediction': result['prediction'],
                                    'Confidence': result['confidence']
                                })
                            except:
                                results.append({
                                    'Text': text[:50],
                                    'Full_Text': text,
                                    'Prediction': 'Error',
                                    'Confidence': 0.0
                                })
                            
                            progress_bar.progress((idx + 1) / len(texts))
                        
                        # Store in session state
                        st.session_state.batch_results = pd.DataFrame(results)
                        
                        # Display
                        results_df = st.session_state.batch_results
                        display_df = results_df[['Text', 'Prediction', 'Confidence']].copy()
                        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        st.subheader("📊 Summary")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            summary = results_df['Prediction'].value_counts()
                            fig = px.pie(values=summary.values, names=summary.index, title="Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.metric("Total", len(results_df))
                            st.metric("Avg Confidence", f"{results_df['Confidence'].mean():.2%}")
                        
                        # Download button ONLY after analysis
                        download_df = results_df[['Full_Text', 'Prediction', 'Confidence']]
                        download_df.columns = ['Text', 'Prediction', 'Confidence']
                        csv = download_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_batch_text"
                        )
                else:
                    st.warning("Please enter some texts.")
        
        else:  # CSV Upload
            st.info("💡 Upload CSV with 'text' column")
            
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("📄 Preview:")
                    st.dataframe(df.head(10))
                    
                    text_column = st.selectbox("Select text column:", df.columns)
                    
                    if st.button("Analyze CSV", type="primary"):
                        texts = df[text_column].astype(str).tolist()
                        
                        with st.spinner(f"Analyzing {len(texts)} texts..."):
                            predictions = []
                            confidences = []
                            progress_bar = st.progress(0)
                            
                            for idx, text in enumerate(texts):
                                try:
                                    result = predict(str(text), model, tokenizer, label_encoder)
                                    predictions.append(result['prediction'])
                                    confidences.append(result['confidence'])
                                except:
                                    predictions.append('Error')
                                    confidences.append(0.0)
                                
                                progress_bar.progress((idx + 1) / len(texts))
                            
                            df['Prediction'] = predictions
                            df['Confidence'] = confidences
                            
                            st.success("✅ Complete!")
                            st.dataframe(df)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                summary = pd.Series(predictions).value_counts()
                                fig = px.bar(x=summary.index, y=summary.values, title="Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.metric("Total", len(df))
                                st.metric("Avg Confidence", f"{np.mean(confidences):.2%}")
                            
                            # Download button ONLY after analysis
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_batch_csv"
                            )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ========================================================================
    # TAB 3: HISTORY
    # ========================================================================
    with tab3:
        st.subheader("📈 Prediction History")
        
        # Refresh button
        if st.button("🔄 Refresh History"):
            st.rerun()
        
        if os.path.exists('data/prediction_history.json'):
            try:
                with open('data/prediction_history.json', 'r', encoding='utf-8') as f:
                    history = json.load(f)

                
                if history:
                    history_df = pd.DataFrame(history)
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    correct_count = sum(1 for e in history if isinstance(e.get('feedback'), dict) and e['feedback'].get('feedback_type') == 'correct')
                    incorrect_count = sum(1 for e in history if isinstance(e.get('feedback'), dict) and e['feedback'].get('feedback_type') == 'incorrect')
                    
                    with col1:
                        st.metric("Total", len(history_df))
                    with col2:
                        st.metric("Avg Confidence", f"{history_df['confidence'].mean():.2%}")
                    with col3:
                        st.metric("✅ Correct", correct_count)
                    with col4:
                        st.metric("❌ Incorrect", incorrect_count)
                    
                    if correct_count + incorrect_count > 0:
                        acc = correct_count / (correct_count + incorrect_count) * 100
                        st.info(f"📊 **User-Reported Accuracy:** {acc:.1f}% (n={correct_count + incorrect_count})")
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        pred_counts = history_df.groupby(history_df['timestamp'].dt.date).size().reset_index(name='count')
                        fig = px.line(pred_counts, x='timestamp', y='count', title="Predictions Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        class_dist = history_df['prediction'].value_counts()
                        fig = px.bar(x=class_dist.index, y=class_dist.values, title="Class Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Recent Predictions")
                    recent = history_df.tail(20).sort_values('timestamp', ascending=False)
                    display = recent[['timestamp', 'text', 'prediction', 'confidence']].copy()
                    display['confidence'] = display['confidence'].apply(lambda x: f"{x:.2%}")
                    display['text'] = display['text'].apply(lambda x: x[:80] + '...' if len(x) > 80 else x)
                    st.dataframe(display, use_container_width=True, hide_index=True)
                    
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
                            try:
                                os.remove('data/prediction_history.json')
                                st.success("✅ History cleared!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to clear: {e}")
                else:
                    st.info("📝 No predictions yet. History is empty.")
                    st.markdown("""
                    ### Get Started:
                    1. Go to **Single Prediction** tab
                    2. Enter Nepali text
                    3. Click "Analyze Text"
                    4. Provide feedback
                    5. Come back here to see your history!
                    """)
            except Exception as e:
                st.error(f"Error loading history: {str(e)}")
                st.exception(e)
        else:
            st.info("📝 No history file found yet.")
            st.markdown("""
            ### How to Create History:
            1. Go to **Single Prediction** tab
            2. Analyze some text
            3. Submit feedback
            4. History will appear here automatically!
            
            **Expected file location:** `data/prediction_history.json`
            """)
            
            # Debug info
            with st.expander("🔍 Debug Information"):
                st.write("**Current directory:**", os.getcwd())
                st.write("**Data directory exists:**", os.path.exists('data'))
                st.write("**History file exists:**", os.path.exists('data/prediction_history.json'))
                
                if st.button("Create Test Entry"):
                    test_result = {
                        'prediction': 'NO',
                        'confidence': 0.95,
                        'probabilities': {'NO': 0.95, 'OO': 0.03, 'OR': 0.01, 'OS': 0.01}
                    }
                    test_feedback = {
                        'feedback_type': 'correct',
                        'correct_label': None,
                        'comment': None
                    }
                    success = save_prediction_to_file("Test entry", test_result, test_feedback)
                    if success:
                        st.success("✅ Test entry created! Refresh the page.")
                    else:
                        st.error("❌ Failed to create test entry.")


if __name__ == "__main__":
    main()