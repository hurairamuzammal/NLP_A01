# app.py
import os, re
import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm

# -------------------- paths (relative) --------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "Model", "best_seq2seq.pth")
URDU_SPM_PATH = os.path.join(BASE, "Tokenizer", "urdu_tokenizer.model")
ROMAN_SPM_PATH = os.path.join(BASE, "Tokenizer", "roman_tokenizer.model")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(URDU_SPM_PATH) or not os.path.exists(ROMAN_SPM_PATH):
    raise FileNotFoundError("Tokenizer files not found in Tokenizer/")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- load tokenizers --------------------
urdu_sp = spm.SentencePieceProcessor(model_file=URDU_SPM_PATH)
roman_sp = spm.SentencePieceProcessor(model_file=ROMAN_SPM_PATH)

# -------------------- helper: inspect checkpoint --------------------
def inspect_ckpt(state_dict):
    # returns useful inferred sizes and keys
    info = {}
    # find decoder output linear weight key (vocab x (dec_h + enc_h*2))
    out_key = None
    for k in state_dict.keys():
        if re.search(r"(\bdec\b|\bdecoder\b).*\.out\.weight", k) or k.endswith(".out.weight"):
            out_key = k; break
    if out_key is None:
        # fallback: any 'out.weight'
        for k in state_dict.keys():
            if k.endswith(".out.weight"):
                out_key = k; break
    info['dec_out_key'] = out_key
    if out_key:
        v, in_f = state_dict[out_key].shape
        info['vocab_size'] = v
        info['dec_out_in_features'] = in_f

    # find decoder rnn weight to infer dec_hidden and num_layers
    dec_rnn_keys = sorted([k for k in state_dict.keys() if re.search(r"(\bdec\b|\bdecoder\b).*rnn.*weight_ih_l", k)])
    if not dec_rnn_keys:
        # try any 'rnn.weight_ih_l0' keys and assume decoder
        dec_rnn_keys = sorted([k for k in state_dict.keys() if k.endswith("rnn.weight_ih_l0")])
    info['dec_rnn_keys'] = dec_rnn_keys
    if dec_rnn_keys:
        key0 = dec_rnn_keys[0]
        rows, cols = state_dict[key0].shape
        dec_hid = rows // 4
        # count layers by looking for l{n} occurrences
        layers = len({int(re.search(r"weight_ih_l(\d+)", k).group(1)) for k in dec_rnn_keys})
        info['dec_hidden'] = dec_hid
        info['dec_layers'] = layers

    # find encoder rnn keys
    enc_rnn_keys = sorted([k for k in state_dict.keys() if re.search(r"(\benc\b|\bencoder\b).*rnn.*weight_ih_l", k)])
    if not enc_rnn_keys:
        enc_rnn_keys = sorted([k for k in state_dict.keys() if "enc.rnn.weight_ih_l0" in k or "encoder.rnn.weight_ih_l0" in k])
    info['enc_rnn_keys'] = enc_rnn_keys
    if enc_rnn_keys:
        rows, cols = state_dict[enc_rnn_keys[0]].shape
        enc_hid = rows // 4  # for single direction; if bidirectional encoder, this is per-direction
        info['enc_hidden_per_dir'] = enc_hid

    # find embedding dims from embedding weight
    emb_keys = [k for k in state_dict.keys() if k.endswith(".embedding.weight") or ".embedding.weight" in k]
    if emb_keys:
        info['emb_key'] = emb_keys[0]
        info['emb_dim'] = state_dict[emb_keys[0]].shape[1]

    # presence of projection layers names
    info['has_h_proj'] = any(k.endswith("h_proj.weight") or k.endswith(".h_proj.weight") for k in state_dict.keys())
    info['has_c_proj'] = any(k.endswith("c_proj.weight") or k.endswith(".c_proj.weight") for k in state_dict.keys())
    info['has_dec_attn_proj'] = any("attn_proj.weight" in k for k in state_dict.keys())

    return info

# -------------------- dynamic model builders --------------------
# We'll build classes similar to training, but using inferred sizes
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if n_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out, (h, c)

class DecoderWithLuong(nn.Module):
    def __init__(self, vocab_size, emb_dim, dec_hid, n_layers, dropout, pad_idx, enc_hid2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, dec_hid, n_layers, batch_first=True,
                           dropout=dropout if n_layers>1 else 0.0)
        # attn proj (name must match checkpoint if present)
        self.attn_proj = nn.Linear(dec_hid, enc_hid2)
        self.out = nn.Linear(dec_hid + enc_hid2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        emb = self.dropout(self.embedding(input_token))
        rnn_out, (nh, nc) = self.rnn(emb, (hidden, cell))
        rnn_out_s = rnn_out.squeeze(1)
        proj = self.attn_proj(nh[-1])
        energy = torch.bmm(encoder_outputs, proj.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        concat = torch.cat([rnn_out_s, context], dim=1)
        logits = self.out(concat)
        return logits, (nh, nc), attn_weights

class Seq2SeqAttention(nn.Module):
    def __init__(self, enc, dec, has_h_proj=False, has_c_proj=False, proj_in=None, proj_out=None):
        super().__init__()
        self.enc = enc
        self.dec = dec
        # if checkpoint had h_proj/c_proj saved as top-level modules, create those to match names
        if has_h_proj:
            # shape: (proj_out, proj_in) will be set by loading state dict later
            self.h_proj = nn.Linear(proj_in or 1, proj_out or 1)
        if has_c_proj:
            self.c_proj = nn.Linear(proj_in or 1, proj_out or 1)

    def init_decoder_states(self, eh, ec):
        enc_layers = eh.size(0) // 2
        forward_h, backward_h = eh[enc_layers - 1], eh[-1]
        forward_c, backward_c = ec[enc_layers - 1], ec[-1]
        final_h = torch.cat([forward_h, backward_h], dim=1)
        final_c = torch.cat([forward_c, backward_c], dim=1)

        # prefer using self.h_proj/self.c_proj if present (these will be re-shaped by load)
        if hasattr(self, "h_proj") and hasattr(self, "c_proj"):
            dh = self.h_proj(final_h).unsqueeze(0).repeat(self.dec.rnn.num_layers, 1, 1)
            dc = self.c_proj(final_c).unsqueeze(0).repeat(self.dec.rnn.num_layers, 1, 1)
            return dh, dc

        # else create temporary projections (non-registered)
        proj_h = nn.Linear(final_h.size(1), self.dec.rnn.hidden_size).to(final_h.device)
        proj_c = nn.Linear(final_c.size(1), self.dec.rnn.hidden_size).to(final_c.device)
        dh = proj_h(final_h).unsqueeze(0).repeat(self.dec.rnn.num_layers, 1, 1)
        dc = proj_c(final_c).unsqueeze(0).repeat(self.dec.rnn.num_layers, 1, 1)
        return dh, dc

    def forward(self, src, src_lens, tgt_in):
        enc_out, (eh, ec) = self.enc(src, src_lens)
        dh, dc = self.init_decoder_states(eh, ec)
        outputs = []
        input_tok = tgt_in[:, 0:1]
        for _ in range(tgt_in.size(1) - 1):
            logits, (dh, dc), _ = self.dec(input_tok, dh, dc, enc_out)
            input_tok = logits.argmax(dim=-1, keepdim=True)
            outputs.append(input_tok)
        return torch.cat(outputs, dim=1)

# -------------------- build model using checkpoint shapes --------------------
raw = torch.load(MODEL_PATH, map_location="cpu")
state = raw.get("model_state_dict", raw)
cfg = raw.get("config", {})

info = inspect_ckpt(state)
# infer sizes
vocab = info.get("vocab_size", roman_sp.get_piece_size())
dec_out_in = info.get("dec_out_in_features", None)
dec_hid = info.get("dec_hidden", cfg.get("hidden_size", None))
enc_hid_per_dir = info.get("enc_hidden_per_dir", cfg.get("hidden_size", None))
dec_layers = info.get("dec_layers", cfg.get("decoder_layers", 1))
emb_dim = info.get("emb_dim", cfg.get("embed_dim", 256))

# if dec_out_in and dec_hid available → compute enc_hid2 and enc_hidden
if dec_out_in is not None and dec_hid is not None:
    enc_hid2 = dec_out_in - dec_hid
    # enc_hid2 should be enc_hidden * num_directions (2)
    enc_hidden_inferred = enc_hid2 // 2
else:
    enc_hid2 = (cfg.get("hidden_size", 256) * 2)
    enc_hidden_inferred = cfg.get("hidden_size", 256)

# fill values cleanly
embed_dim = emb_dim
encoder_layers = cfg.get("encoder_layers", cfg.get("enc_layers", 1))
decoder_layers = dec_layers
decoder_hidden = dec_hid or cfg.get("hidden_size", 256)
encoder_hidden = enc_hidden_inferred or cfg.get("hidden_size", 256)
dropout = cfg.get("dropout", 0.25)

# create encoder & decoder with inferred sizes
enc = Encoder(urdu_sp.get_piece_size(), embed_dim, encoder_hidden, encoder_layers, dropout, urdu_sp.pad_id(), bidirectional=True)
dec = DecoderWithLuong(vocab, embed_dim, decoder_hidden, decoder_layers, dropout, roman_sp.pad_id(), enc_hid2=encoder_hidden*2)

model = Seq2SeqAttention(enc, dec, has_h_proj=info['has_h_proj'], has_c_proj=info['has_c_proj'],
                         proj_in=(encoder_hidden*2 if encoder_hidden else None), proj_out=decoder_hidden)

# Move model to device BEFORE adapting shapes to correct devices
model.to(DEVICE)

# -------------------- Now load state strictly but adapt any shape differences by slicing --------------------
def strict_load_with_adapt(model, state_dict):
    tgt = model.state_dict()
    new_state = {}
    adapted = {"copied":0, "sliced":0, "missing":[], "extra":[]}
    for k_src, v_src in state_dict.items():
        if k_src in tgt:
            v_tgt = tgt[k_src]
            if v_src.shape == v_tgt.shape:
                new_state[k_src] = v_src.to(v_tgt.device)
                adapted["copied"] += 1
            else:
                # try to slice to overlap
                slices = tuple(slice(0, min(s_src, s_tgt)) for s_src, s_tgt in zip(v_src.shape, v_tgt.shape))
                tmp = v_tgt.clone()
                tmp.zero_()
                try:
                    tmp[slices] = v_src[slices].to(tmp.device, dtype=tmp.dtype)
                    new_state[k_src] = tmp
                    adapted["sliced"] += 1
                except Exception as e:
                    adapted["missing"].append((k_src, str(e)))
        else:
            adapted["extra"].append(k_src)

    # now ensure all target keys are present in new_state (if not, keep original tgt param)
    final_state = {}
    for k_tgt, v_tgt in tgt.items():
        if k_tgt in new_state:
            final_state[k_tgt] = new_state[k_tgt]
        else:
            final_state[k_tgt] = v_tgt

    # load strictly now (all keys match)
    model.load_state_dict(final_state, strict=True)
    return adapted

adapt_info = strict_load_with_adapt(model, state)
model.eval()

# -------------------- UI Setup --------------------
st.set_page_config(page_title="Urdu → Roman Urdu", page_icon="�", layout="wide")

# Custom CSS for better styling with dark/light mode support
st.markdown("""
    <style>
    /* Main title styling */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.25rem !important;
        margin-bottom: 1rem !important;
        font-weight: 400 !important;
        opacity: 0.8;
    }
    
    /* Section headers */
    h2 {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Info box styling - adapts to theme */
    .info-box {
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 3px solid rgba(59, 130, 246, 0.5);
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .info-box strong {
        font-weight: 600;
    }
    
    /* Text area labels */
    .stTextArea label {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton button {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.25rem !important;
    }
    
    /* Example button styling */
    div[data-testid="column"] button {
        font-size: 1rem !important;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0 !important;
        opacity: 0.2;
    }
    
    /* Warning/Info messages */
    .stAlert {
        font-size: 1.05rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Urdu to Roman Urdu Transliteration")
st.markdown('<p class="subtitle">Transform Urdu script into Roman Urdu using deep learning</p>', unsafe_allow_html=True)

# Model architecture information
st.markdown("""
    <div class="info-box">
        <strong>Model Architecture:</strong> This system is built using a <strong>Bidirectional LSTM (BiLSTM) encoder</strong> 
        and an <strong>LSTM decoder</strong> without attention mechanism. The model learns to transliterate 
        Urdu text into Roman script through sequence-to-sequence learning.
    </div>
    """, unsafe_allow_html=True)

# -------------------- translate util --------------------
def translate_text(src_text, max_len=120):
    if not src_text.strip():
        return ""
    with torch.no_grad():
        src_ids = urdu_sp.encode(src_text, out_type=int)
        if len(src_ids) == 0:
            return ""
        src = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)
        src_lens = torch.tensor([len(src_ids)], dtype=torch.long).to(DEVICE)
        tgt_in = torch.tensor([[roman_sp.bos_id()] + [roman_sp.pad_id()]*(max_len-1)], dtype=torch.long).to(DEVICE)
        preds = model(src, src_lens, tgt_in)[0].cpu().tolist()
        out = roman_sp.decode(preds)
        return out.replace("<PAD>", "").replace("<pad>", "").strip()

# -------------------- UI --------------------
st.markdown("---")

# Example buttons
st.markdown("### Try Sample Examples")
col1, col2 = st.columns(2)

with col1:
    if st.button("Example 1", use_container_width=True):
        st.session_state['input_text'] = "عاشقی میں میرؔ جیسے خواب مت دیکھا کرو"

with col2:
    if st.button("Example 2", use_container_width=True):
        st.session_state['input_text'] = "اب کیا سوچیں کیا حالات تھے کس کارن یہ زہر پیا ہے"

st.markdown("---")

# Get input text from session state or default to empty
default_text = st.session_state.get('input_text', '')
input_text = st.text_area("Enter Urdu Text:", value=default_text, height=150, placeholder="یہاں اردو لکھیں...")

# Update session state when text area changes
st.session_state['input_text'] = input_text

if st.button("Transliterate", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some Urdu text first.")
    else:
        with st.spinner("Transliterating your text..."):
            out = translate_text(input_text.strip(), max_len=120)
        st.text_area("Roman Urdu Output:", value=out, height=150)

# Limitation note
st.markdown("---")
st.markdown("""
    <div class="info-box" style="border-left-color: rgba(239, 68, 68, 0.5); background-color: rgba(239, 68, 68, 0.06);">
        <strong>Note:</strong> The model may not accurately predict all transliterations due to the lack of an attention mechanism 
        and limited training data available for poetic Urdu text.
    </div>
    """, unsafe_allow_html=True)
