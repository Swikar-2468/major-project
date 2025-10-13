"""
Streamlit Web Application for Nepali Hate Speech Detection
Run with: streamlit run scripts/6_streamlit_app.py
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Nepali Hate Speech Detector",
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
    .no-box { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); }
    .oo-box { background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); }
    .or-box { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); }
    .os-box { background: linear-gradient(135deg, #f5c6cb 0%, #f1b0b7 100%); }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached)."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import joblib
        
        model_path = 'models/saved_models/xlm_roberta_final'
        
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please train the model first.")
            return None, None, None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Load label encoder
        le = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
        
        return model, tokenizer, le
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def preprocess_text(text):
    """Preprocess input text."""
    from utils.preprocessing import preprocess_for_transformer
    return preprocess_for_transformer(text)


def predict(text, model, tokenizer, label_encoder):
    """Make prediction on input text."""
    device = next(model.parameters()).device
    
    # Preprocess
    preprocessed = preprocess_text(text)
    
    # Tokenize
    inputs = tokenizer(
        preprocessed,
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    )
    
    # Move to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    # Get results
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
        }
    }
    
    return results


def plot_probabilities(probabilities):
    """Create probability bar chart."""
    labels = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Color mapping
    colors = {
        'NO': '#28a745',
        'OO': '#ffc107',
        'OR': '#dc3545',
        'OS': '#c82333'
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
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def save_prediction(text, result, feedback=None):
    """Save prediction to history file."""
    history_file = 'data/prediction_history.json'
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text,
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'feedback': feedback
    }
    
    # Load existing history
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(entry)
    
    # Save updated history
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Nepali Hate Speech Detector</h1>', 
                unsafe_allow_html=True)
    
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
        This application uses **XLM-RoBERTa**, a state-of-the-art 
        multilingual transformer model, fine-tuned on Nepali hate speech data.
        
        **Class Descriptions:**
        - **NO**: Non-offensive content
        - **OO**: General offensive language
        - **OR**: Racist/ethnicity-based hate
        - **OS**: Sexist/gender-based hate
        
        **Model Information:**
        - Architecture: XLM-RoBERTa Base
        - Training samples: ~7,000
        - Languages: Nepali (Devanagari & Romanized)
        """)
        
        st.header("📊 Statistics")
        if os.path.exists('data/prediction_history.json'):
            with open('data/prediction_history.json', 'r', encoding='utf-8') as f:
                history = json.load(f)
            st.metric("Total Predictions", len(history))
            
            # Class distribution
            pred_counts = pd.Series([h['prediction'] for h in history]).value_counts()
            st.write("**Recent Predictions:**")
            for label, count in pred_counts.items():
                st.write(f"- {label}: {count}")
    
    # Load model
    model, tokenizer, label_encoder = load_model()
    
    if model is None:
        st.stop()
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📝 Batch Analysis", "📈 History"])
    
    # Tab 1: Single Prediction
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter Nepali Text")
            
            # Example texts
            examples = {
                "Non-Offensive": "यो राम्रो छ, धन्यवाद",
                "Other-Offensive": "तिमी मुर्ख हौ",
                "Racist (Example)": "बाहुन हरु सबै एउटै छन",
                "Custom": ""
            }
            
            example_choice = st.selectbox("Select an example or enter custom text:", 
                                         list(examples.keys()))
            
            if example_choice == "Custom":
                text_input = st.text_area(
                    "Enter text here (in Devanagari or Romanized Nepali):",
                    height=150,
                    placeholder="यहाँ आफ्नो पाठ लेख्नुहोस्..."
                )
            else:
                text_input = st.text_area(
                    "Enter text here (in Devanagari or Romanized Nepali):",
                    value=examples[example_choice],
                    height=150
                )
            
            analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("Quick Info")
            st.info("""
            **Tips for best results:**
            - Enter complete sentences
            - Use either Devanagari or Roman script
            - Avoid excessive emojis
            - Context matters!
            """)
        
        if analyze_button and text_input.strip():
            with st.spinner("Analyzing text..."):
                try:
                    # Make prediction
                    result = predict(text_input, model, tokenizer, label_encoder)
                    
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
                        <h2 style='margin:0; color: #333;'>Prediction: {pred_label}</h2>
                        <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                            Confidence: <strong>{confidence:.2%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability chart
                    fig = plot_probabilities(result['probabilities'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    with st.expander("📋 Detailed Probabilities"):
                        prob_df = pd.DataFrame([
                            {'Class': k, 'Probability': f"{v:.4f}", 'Percentage': f"{v*100:.2f}%"}
                            for k, v in result['probabilities'].items()
                        ])
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                    # Feedback
                    st.markdown("---")
                    st.subheader("💬 Feedback")
                    feedback_col1, feedback_col2 = st.columns(2)
                    
                    with feedback_col1:
                        if st.button("✅ Correct Prediction", use_container_width=True):
                            save_prediction(text_input, result, feedback="correct")
                            st.success("Thank you for your feedback!")
                    
                    with feedback_col2:
                        if st.button("❌ Incorrect Prediction", use_container_width=True):
                            save_prediction(text_input, result, feedback="incorrect")
                            st.warning("Thank you! We'll use this to improve the model.")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        
        elif analyze_button:
            st.warning("Please enter some text to analyze.")
    
    # Tab 2: Batch Analysis
    with tab2:
        st.subheader("📝 Batch Text Analysis")
        st.markdown("Upload a CSV file or enter multiple texts (one per line)")
        
        input_method = st.radio("Input method:", ["Text Area", "CSV Upload"])
        
        if input_method == "Text Area":
            batch_text = st.text_area(
                "Enter multiple texts (one per line):",
                height=200,
                placeholder="First text\nSecond text\nThird text..."
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
                                    'Prediction': result['prediction'],
                                    'Confidence': f"{result['confidence']:.2%}"
                                })
                            except Exception as e:
                                results.append({
                                    'Text': text[:50],
                                    'Prediction': 'Error',
                                    'Confidence': str(e)
                                })
                            
                            progress_bar.progress((idx + 1) / len(texts))
                        
                        # Display results
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        # Summary
                        st.subheader("📊 Summary")
                        summary = results_df['Prediction'].value_counts()
                        
                        fig = px.pie(
                            values=summary.values,
                            names=summary.index,
                            title="Prediction Distribution"
                        )
                        st.plotly_chart(fig)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Please enter some texts to analyze.")
        
        else:  # CSV Upload
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:")
                st.dataframe(df.head())
                
                text_column = st.selectbox("Select text column:", df.columns)
                
                if st.button("Analyze CSV", type="primary"):
                    texts = df[text_column].tolist()
                    
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
                        
                        st.success("Analysis complete!")
                        st.dataframe(df)
                        
                        # Download
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
    
    # Tab 3: History
    with tab3:
        st.subheader("📈 Prediction History")
        
        if os.path.exists('data/prediction_history.json'):
            with open('data/prediction_history.json', 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if history:
                # Convert to dataframe
                history_df = pd.DataFrame(history)
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", len(history_df))
                with col2:
                    st.metric("Avg Confidence", f"{history_df['confidence'].mean():.2%}")
                with col3:
                    correct_feedback = len(history_df[history_df['feedback'] == 'correct'])
                    st.metric("Correct Feedback", correct_feedback)
                with col4:
                    incorrect_feedback = len(history_df[history_df['feedback'] == 'incorrect'])
                    st.metric("Incorrect Feedback", incorrect_feedback)
                
                # Visualizations
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Predictions over time
                    pred_counts = history_df.groupby(
                        history_df['timestamp'].dt.date
                    ).size().reset_index(name='count')
                    
                    fig = px.line(pred_counts, x='timestamp', y='count',
                                 title="Predictions Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Class distribution
                    class_dist = history_df['prediction'].value_counts()
                    fig = px.bar(x=class_dist.index, y=class_dist.values,
                                title="Class Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recent predictions
                st.markdown("---")
                st.subheader("Recent Predictions")
                recent = history_df.tail(10).sort_values('timestamp', ascending=False)
                st.dataframe(
                    recent[['timestamp', 'text', 'prediction', 'confidence', 'feedback']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Clear history button
                if st.button("🗑️ Clear History", type="secondary"):
                    os.remove('data/prediction_history.json')
                    st.success("History cleared!")
                    st.rerun()
            else:
                st.info("No prediction history yet.")
        else:
            st.info("No prediction history yet. Start making predictions!")


if __name__ == "__main__":
    main()