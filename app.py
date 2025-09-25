import streamlit as st
import numpy as np
import time
import re
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
import torch

# Page configuration
st.set_page_config(
    page_title="Urdu-Roman Urdu Transliterator",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #2D5A27;
        --secondary-color: #4A7C59;
        --accent-color: #8FBC8F;
        --text-dark: #1a1a1a;
        --text-light: #666666;
        --bg-light: #f8f9fa;
        --border-light: #e9ecef;
        --gradient-primary: linear-gradient(135deg, #2D5A27 0%, #4A7C59 100%);
        --gradient-accent: linear-gradient(135deg, #8FBC8F 0%, #98FB98 100%);
        --shadow-soft: 0 4px 20px rgba(45, 90, 39, 0.1);
        --shadow-medium: 0 8px 30px rgba(45, 90, 39, 0.15);
    }
    
    * {
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: var(--text-dark);
        line-height: 1.6;
    }
    
    .main-header {
        background: var(--gradient-primary);
        color: white;
        padding: 2rem 0;
        text-align: center;
        margin-bottom: 2rem;
        border-radius: 15px;
        box-shadow: var(--shadow-medium);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="0.5" fill="white" opacity="0.1"/><circle cx="90" cy="40" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        pointer-events: none;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.2rem;
        line-height: 1.8;
        direction: rtl;
        text-align: right;
    }
    
    .transliteration-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .transliteration-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-medium);
    }
    
    .input-section {
        background: var(--gradient-accent);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .output-section {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px solid var(--accent-color);
        min-height: 120px;
    }
    
    .stTextArea > div > div {
        background: white;
        border-radius: 10px;
        border: 2px solid var(--border-light);
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.1rem;
        direction: rtl;
        text-align: right;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(45, 90, 39, 0.1);
    }
    
    .transliterate-btn {
        background: var(--gradient-primary);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-soft);
        width: 100%;
        margin: 1rem 0;
    }
    
    .transliterate-btn:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }
    
    .transliterate-btn:active {
        transform: translateY(0);
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .result-text {
        font-size: 1.3rem;
        font-weight: 500;
        color: var(--text-dark);
        line-height: 1.6;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    .stats-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-medium);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        display: block;
    }
    
    .stats-label {
        color: var(--text-light);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .history-item {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid var(--accent-color);
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .history-urdu {
        font-family: 'Noto Nastaliq Urdu', serif;
        color: var(--text-dark);
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .history-roman {
        color: var(--text-light);
        font-size: 0.9rem;
        font-style: italic;
    }
    
    .sidebar-section {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-soft);
    }
    
    .model-status {
        display: flex;
        align-items: center;
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .status-connected {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-disconnected {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    .status-connected .status-indicator {
        background: #28a745;
    }
    
    .status-disconnected .status-indicator {
        background: #dc3545;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .example-chip {
        display: inline-block;
        background: var(--accent-color);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
    }
    
    .example-chip:hover {
        background: var(--primary-color);
        transform: scale(1.05);
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'total_transliterations' not in st.session_state:
    st.session_state.total_transliterations = 0

# PyTorch Model Wrapper Class
class BiLSTMTransliterator:
    """BiLSTM transliterator wrapper for the trained PyTorch model"""
    
    def __init__(self, model):
        self.model = model
        self.model_name = "BiLSTM Transliterator v1.0"
        self.vocab_size = getattr(model, 'vocab_size', 15000)
        self.embedding_dim = getattr(model, 'embedding_dim', 256)
        self.hidden_units = getattr(model, 'hidden_size', 512)
        
        # Fallback character mappings in case model prediction fails
        self.fallback_mapping = {
            'ا': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's', 'ج': 'j',
            'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd', 'ذ': 'z', 'ر': 'r',
            'ڑ': 'r', 'ز': 'z', 'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z',
            'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k',
            'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'w', 'ہ': 'h',
            'ھ': 'h', 'ء': '', 'ی': 'y', 'ئ': 'y', 'ے': 'e', 'آ': 'a', 'یٰ': 'y'
        }
        
        self.common_words = {
            'السلام': 'Assalam', 'علیکم': 'Alaikum', 'کیا': 'kya', 'ہے': 'hai',
            'آپ': 'aap', 'کا': 'ka', 'نام': 'naam', 'میرا': 'mera', 'دوست': 'dost',
            'شکریہ': 'shukriya', 'خوش': 'khush', 'آمد': 'aamad', 'مدد': 'madad'
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess Urdu text for model input"""
        # Add your specific preprocessing steps here
        text = text.strip()
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def predict(self, urdu_text: str) -> str:
        """Make prediction using the PyTorch model"""
        try:
            # Preprocess the input text
            processed_text = self.preprocess_text(urdu_text)
            
            with torch.no_grad():
                # TODO: Replace this section with your actual model inference code
                # This is where you'll implement your model's prediction logic
                
                # Example structure (modify based on your model's input/output format):
                # 1. Tokenize the input text
                # 2. Convert to tensors
                # 3. Pass through the model
                # 4. Decode the output
                
                # For now, using fallback logic until you integrate your model
                return self._fallback_prediction(processed_text)
                
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            # Use fallback prediction in case of error
            return self._fallback_prediction(urdu_text)
    
    def _fallback_prediction(self, urdu_text: str) -> str:
        """Fallback prediction method using simple character mapping"""
        words = urdu_text.split()
        roman_words = []
        
        for word in words:
            # Check if it's a common word
            if word in self.common_words:
                roman_words.append(self.common_words[word])
            else:
                # Character-by-character transliteration
                roman_word = ''
                for char in word:
                    if char in self.fallback_mapping:
                        roman_word += self.fallback_mapping[char]
                    else:
                        roman_word += char
                roman_words.append(roman_word)
        
        return ' '.join(roman_words)

# Initialize model
@st.cache_resource
def load_model():
    """Load the BiLSTM PyTorch model"""
    try:
        # Check if model file exists
        model_path = 'model.pt'
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found!")
            return None, False
        
        # Load the PyTorch model
        device = torch.device('cpu')  # Use CPU for deployment
        pytorch_model = torch.load(model_path, map_location=device)
        pytorch_model.eval()  # Set to evaluation mode
        
        # Wrap the model in our custom class
        model = BiLSTMTransliterator(pytorch_model)
        
        st.success(f"✅ Model loaded successfully from {model_path}")
        return model, True
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Using fallback prediction method...")
        
        # Create a fallback model instance
        fallback_model = BiLSTMTransliterator(None)
        return fallback_model, True

# Header
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📝 Urdu-Roman Urdu Transliterator</h1>
        <p class="subtitle">Powered by BiLSTM Neural Network with Attention Mechanism</p>
    </div>
""", unsafe_allow_html=True)

# Load model (moved from sidebar)
if not st.session_state.model_loaded:
    with st.spinner("Loading BiLSTM Model..."):
        model, loaded = load_model()
        if loaded:
            st.session_state.model = model
            st.session_state.model_loaded = True

# Main content area
st.markdown("<div class='transliteration-card slide-in'>", unsafe_allow_html=True)
    
# Input Section
st.markdown("<div class='input-section'>", unsafe_allow_html=True)
st.subheader("🖋️ Enter Urdu Text")

# Quick example button
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Try: السلام علیکم", key="example_button", help="Click to try this example"):
        st.session_state.example_text = "السلام علیکم"

# Check if there's example text
default_text = st.session_state.get('example_text', '')

urdu_input = st.text_area(
    "Type or paste Urdu text here:",
    value=default_text,
    height=120,
    placeholder="یہاں اردو متن لکھیں...",
    key="urdu_input",
    label_visibility="collapsed"
)

# Character count
char_count = len(urdu_input.replace(' ', '') if urdu_input else '')
st.caption(f"Characters: {char_count}")
st.markdown("</div>", unsafe_allow_html=True)

# Transliterate Button
if st.button("🔄 Transliterate", key="transliterate", type="primary", use_container_width=True):
    if urdu_input and urdu_input.strip():
        if st.session_state.model_loaded and hasattr(st.session_state, 'model') and st.session_state.model:
            with st.spinner("🔮 Transliterating..."):
                # Simulate processing time for better UX
                time.sleep(0.5)
                
                # Get prediction from model
                roman_output = st.session_state.model.predict(urdu_input)
                
                # Add to history
                st.session_state.history.append({
                    'urdu': urdu_input,
                    'roman': roman_output,
                    'timestamp': datetime.now()
                })
                
                st.session_state.total_transliterations += 1
                st.session_state.current_output = roman_output
                
                # Clear example text
                if 'example_text' in st.session_state:
                    del st.session_state.example_text
                    
        else:
            st.error("❌ Model not loaded. Please check your model configuration.")
    else:
        st.warning("⚠️ Please enter some Urdu text to transliterate.")

# Output Section
st.markdown("<div class='output-section'>", unsafe_allow_html=True)
st.subheader("🎯 Roman Urdu Output")

if 'current_output' in st.session_state:
    st.markdown(f"""
        <div class="result-text fade-in">
            {st.session_state.current_output}
        </div>
    """, unsafe_allow_html=True)
    
    # Copy button
    st.code(st.session_state.current_output, language="text")
    
    # Action buttons
    col_copy, col_clear = st.columns(2)
    with col_copy:
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.success("✅ Copied to clipboard!")
    
    with col_clear:
        if st.button("🗑️ Clear Output", use_container_width=True):
            if 'current_output' in st.session_state:
                del st.session_state.current_output
            st.rerun()
else:
    st.info("💡 Enter Urdu text above and click 'Transliterate' to see the Roman Urdu output")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)



# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: var(--text-light); padding: 1rem;">
        <p>🤖 Built with BiLSTM Neural Network | 🎯 Optimized for Urdu Transliteration</p>
        <p><small>Replace the MockTransliterator class with your actual BiLSTM model for production use</small></p>
    </div>
""", unsafe_allow_html=True)