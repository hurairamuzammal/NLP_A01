# %% [markdown]
# # Urdu → Roman Urdu Character-Level Seq2Seq
# A clean, end-to-end training notebook that rebuilds the pipeline with a bidirectional encoder, unidirectional decoder, and configurable hyperparameters.

# %% [markdown]
# ## 1. Environment Setup and Imports

# %%
# Install lightweight dependencies if missing (safe to re-run)
try:
    import sacrebleu  # noqa: F401
except ModuleNotFoundError:
    pip install --quiet sacrebleu

import math
import random
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

# Display basic runtime info for reproducibility
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# %% [markdown]
# ## 2. Configuration Dictionary for Hyperparameters

# %%
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


base_dir = Path("/kaggle/input/urduromanurdu-cleaned")
local_dir = Path.cwd() / "Data"

config = {
    "data": {
        "src_path": str((base_dir / "all_urdu.txt")),
        "tgt_path": str((base_dir / "all_english_clean.txt")),
        "local_src_path": str((local_dir / "all_urdu.txt")),
        "local_tgt_path": str((local_dir / "all_english_clean.txt")),
        "max_seq_len": 100,
        "min_seq_len": 1,
        "train_ratio": 0.50,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "chunk_size": 10_000,
    },
    "vocab": {
        "special_tokens": ["<pad>", "<bos>", "<eos>", "<unk>"]
    },
    "model": {
        "embedding_size": 256,
        "hidden_size": 384,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.25,
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 20,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "teacher_forcing_ratio": 0.5,
        "grad_clip": 1.0,
        "max_steps_per_epoch": None,  # set integer to debug quickly
        "patience": 5,
        "scheduler": {
            "type": "one_cycle",
            "monitor": "val_loss",
            "params": {
                "max_lr": 5e-3,
                "pct_start": 0.1,
                "anneal_strategy": "cos",
                "div_factor": 10.0,
                "final_div_factor": 1e3,
                "three_phase": False,
            },
        },
    },
    "evaluation": {
        "max_decode_len": 110,
    },
    "artifacts": {
        "checkpoint_path": str(Path("temp") / "best_seq2seq.pt"),
        "prediction_path": str(Path("temp") / "sample_predictions.txt"),
    },
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

set_seed(config["seed"])
print("Active configuration:")
for key, value in config.items():
    if isinstance(value, dict):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {value}")

# %% [markdown]
# ## 3. Load and Inspect Parallel Corpora

# %%
def resolve_data_path(primary: str, fallback: str) -> Path:
    primary_path = Path(primary)
    if primary_path.exists():
        return primary_path
    fallback_path = Path(fallback)
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(f"Neither {primary_path} nor {fallback_path} was found.")


def stream_parallel_corpus(src_path: Path, tgt_path: Path, chunk_size: int = 10_000) -> Tuple[List[str], List[str]]:
    src_lines, tgt_lines = [], []
    with src_path.open("r", encoding="utf-8") as src_file, tgt_path.open("r", encoding="utf-8") as tgt_file:
        while True:
            src_chunk = [src_file.readline() for _ in range(chunk_size)]
            tgt_chunk = [tgt_file.readline() for _ in range(chunk_size)]
            src_chunk = [line for line in src_chunk if line]
            tgt_chunk = [line for line in tgt_chunk if line]
            if not src_chunk and not tgt_chunk:
                break
            src_lines.extend(src_chunk)
            tgt_lines.extend(tgt_chunk)
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Misaligned corpora: {len(src_lines)} Urdu vs {len(tgt_lines)} Roman Urdu lines.")
    return src_lines, tgt_lines


src_path = resolve_data_path(config["data"]["src_path"], config["data"]["local_src_path"])
tgt_path = resolve_data_path(config["data"]["tgt_path"], config["data"]["local_tgt_path"])
raw_src, raw_tgt = stream_parallel_corpus(src_path, tgt_path, config["data"]["chunk_size"])

print(f"Loaded {len(raw_src):,} parallel sentence pairs.")
for idx in range(3):
    print(f"Pair {idx + 1} → Urdu: {raw_src[idx].strip()} | Roman Urdu: {raw_tgt[idx].strip()}")

# %% [markdown]
# ## 4. Text Cleaning and Normalization Pipeline

# %%
whitespace_re = re.compile(r"\s+")

def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u200c", " ")  # zero-width non-joiner → space
    normalized = whitespace_re.sub(" ", normalized)
    normalized = normalized.strip()
    return normalized


clean_src = [clean_text(line) for line in raw_src]
clean_tgt = [clean_text(line) for line in raw_tgt]

usable_pairs = [
    (src, tgt)
    for src, tgt in zip(clean_src, clean_tgt)
    if len(src) >= config["data"]["min_seq_len"]
    and len(tgt) >= config["data"]["min_seq_len"]
]
print(f"Pairs after cleaning/filtering: {len(usable_pairs):,}")
for idx in range(3):
    src, tgt = usable_pairs[idx]
    print(f"Clean {idx + 1}: Urdu='{src}' | Roman Urdu='{tgt}'")

# %% [markdown]
# ## 5. Character Vocabulary Construction

# %%
def build_char_vocab(sentences: List[str], specials: List[str]) -> Dict[str, Dict[str, int]]:
    counter = Counter()
    for s in sentences:
        counter.update(list(s))
    stoi = {token: idx for idx, token in enumerate(specials)}
    for char in sorted(counter.keys()):
        if char not in stoi:
            stoi[char] = len(stoi)
    itos = {idx: token for token, idx in stoi.items()}
    return {"stoi": stoi, "itos": itos, "freqs": counter}


special_tokens = config["vocab"]["special_tokens"]
src_vocab = build_char_vocab([src for src, _ in usable_pairs], special_tokens)
tgt_vocab = build_char_vocab([tgt for _, tgt in usable_pairs], special_tokens)

pad_idx = src_vocab["stoi"]["<pad>"]
bos_idx = src_vocab["stoi"]["<bos>"]
eos_idx = src_vocab["stoi"]["<eos>"]
unk_idx = src_vocab["stoi"]["<unk>"]

tgt_pad_idx = tgt_vocab["stoi"]["<pad>"]
tgt_bos_idx = tgt_vocab["stoi"]["<bos>"]
tgt_eos_idx = tgt_vocab["stoi"]["<eos>"]
tgt_unk_idx = tgt_vocab["stoi"]["<unk>"]

print(f"Source vocab size: {len(src_vocab['stoi'])}")
print(f"Target vocab size: {len(tgt_vocab['stoi'])}")
print(f"Top Urdu chars: {src_vocab['freqs'].most_common(10)}")
print(f"Top Roman Urdu chars: {tgt_vocab['freqs'].most_common(10)}")

# %% [markdown]
# ## 6. Sequence Encoding and Padding Utilities

# %%
def clip_sequence(chars: List[str], max_len: int) -> List[str]:
    if len(chars) <= max_len:
        return chars
    return chars[:max_len]


def encode_sentence(sentence: str, vocab: Dict[str, Dict[str, int]], max_len: int) -> List[int]:
    stoi = vocab["stoi"]
    max_core_len = max_len - 2  # reserve spots for <bos>/<eos>
    char_tokens = clip_sequence(list(sentence), max_core_len)
    indices = [stoi.get(char, stoi["<unk>"]) for char in char_tokens]
    return [stoi["<bos>"]] + indices + [stoi["<eos>"]]


def decode_indices(indices: List[int], vocab: Dict[str, Dict[int, str]]) -> str:
    itos = vocab["itos"]
    chars = []
    for idx in indices:
        token = itos.get(idx, "")
        if token in {"<bos>", "<pad>"}:
            continue
        if token == "<eos>":
            break
        chars.append(token)
    return "".join(chars)


def collate_fn(batch):
    # batch: List[Tuple[src_tensor, tgt_tensor]]
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    src_sequences, tgt_sequences = zip(*batch)
    src_lengths = torch.tensor([len(seq) for seq in src_sequences], dtype=torch.long)
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_sequences], dtype=torch.long)

    src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_sequences, batch_first=True, padding_value=tgt_pad_idx)
    return src_padded, src_lengths, tgt_padded, tgt_lengths


encoded_pairs = [
    (
        torch.tensor(encode_sentence(src, src_vocab, config["data"]["max_seq_len"]), dtype=torch.long),
        torch.tensor(encode_sentence(tgt, tgt_vocab, config["data"]["max_seq_len"]), dtype=torch.long),
    )
    for src, tgt in usable_pairs
]
print(f"Sample encoded source indices: {encoded_pairs[0][0][:10].tolist()}")
print(f"Decoded back: {decode_indices(encoded_pairs[0][0].tolist(), src_vocab)}")

# %% [markdown]
# ## 7. Dataset Split and PyTorch DataLoaders

# %%
class ParallelCharDataset(Dataset):
    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]


def split_pairs(
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List, List, List]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    total = len(pairs)
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]

    # Ensure no split is empty
    if not all(len(split) > 0 for split in [train_pairs, val_pairs, test_pairs]):
        raise ValueError("One of the data splits is empty. Adjust ratios or dataset size.")

    return train_pairs, val_pairs, test_pairs


train_pairs, val_pairs, test_pairs = split_pairs(
    encoded_pairs,
    config["data"]["train_ratio"],
    config["data"]["val_ratio"],
    config["data"]["test_ratio"],
    config["seed"],
)

print(
    f"Train: {len(train_pairs):,}, Val: {len(val_pairs):,}, Test: {len(test_pairs):,}"
)

train_loader = DataLoader(
    ParallelCharDataset(train_pairs),
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=False,
)
val_loader = DataLoader(
    ParallelCharDataset(val_pairs),
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False,
)
test_loader = DataLoader(
    ParallelCharDataset(test_pairs),
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False,
)

print("Dataloaders ready.")

# %% [markdown]
# ## 8. Bidirectional Encoder and Unidirectional Decoder Definition

# %%
class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, padding_value=pad_idx)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=tgt_pad_idx)
        self.rnn = nn.LSTM(
            emb_dim,
            dec_hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(dec_hid_dim + emb_dim, output_dim)
        self.layer_norm = nn.LayerNorm(dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_step: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_step = input_step.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_step))

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        output = output.squeeze(1)
        output = self.layer_norm(output)
        embedded = embedded.squeeze(1)

        prediction = self.fc_out(torch.cat((output, embedded), dim=1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_idx: int, tgt_pad_idx: int, device: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

        self.enc_hid_dim = encoder.rnn.hidden_size
        self.dec_hid_dim = decoder.rnn.hidden_size
        self.enc_layers = encoder.rnn.num_layers
        self.dec_layers = decoder.rnn.num_layers

        self.fc_hidden = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
        self.fc_cell = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_idx).to(src.device)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = src.new_zeros((batch_size, trg_len, trg_vocab_size), dtype=torch.float32)

        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        hidden, cell = self._bridge_hidden(hidden, cell)

        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs

    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        batch_size = src.shape[0]
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        hidden, cell = self._bridge_hidden(hidden, cell)

        input_token = torch.full((batch_size,), tgt_bos_idx, dtype=torch.long, device=src.device)
        outputs = [input_token.unsqueeze(1)]

        for _ in range(1, max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            top1 = output.argmax(1)
            outputs.append(top1.unsqueeze(1))
            input_token = top1
        return torch.cat(outputs, dim=1)

    def _bridge_hidden(self, hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        def reshape(state: torch.Tensor) -> torch.Tensor:
            state = state.view(self.enc_layers, 2, state.size(1), state.size(2))
            forward_state = state[:, 0, :, :]
            backward_state = state[:, 1, :, :]
            return torch.cat((forward_state, backward_state), dim=2)

        hidden_cat = reshape(hidden)
        cell_cat = reshape(cell)

        hidden_proj = torch.tanh(self.fc_hidden(hidden_cat))
        cell_proj = torch.tanh(self.fc_cell(cell_cat))

        if hidden_proj.size(0) != self.dec_layers:
            multiplier = self.dec_layers // hidden_proj.size(0)
            hidden_proj = hidden_proj.repeat(multiplier, 1, 1)
        if cell_proj.size(0) != self.dec_layers:
            multiplier = self.dec_layers // cell_proj.size(0)
            cell_proj = cell_proj.repeat(multiplier, 1, 1)

        return hidden_proj, cell_proj

# %% [markdown]
# ## 9. Seq2Seq Training Loop with Teacher Forcing

# %%
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def move_batch_to_device(batch, device: str):
    src, src_lengths, tgt, tgt_lengths = batch
    return (
        src.to(device),
        src_lengths.to(device),
        tgt.to(device),
        tgt_lengths.to(device),
    )


def train_one_epoch(
    model: Seq2Seq,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    teacher_forcing_ratio: float,
    grad_clip: float,
    max_steps: Optional[int] = None,
    scheduler: Optional[object] = None,
    scheduler_step_on: str = "epoch",
) -> float:
    model.train()
    epoch_loss = 0.0
    total_steps = 0
    scheduler_mode = (scheduler_step_on or "").lower()

    for step, batch in enumerate(dataloader, start=1):
        src, src_lengths, tgt, _ = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        outputs = model(src, src_lengths, tgt, teacher_forcing_ratio)
        output_dim = outputs.shape[-1]

        output = outputs[:, 1:, :].reshape(-1, output_dim)
        target = tgt[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if scheduler is not None and scheduler_mode == "batch":
            scheduler.step()

        epoch_loss += loss.item()
        total_steps = step

        if max_steps is not None and step >= max_steps:
            break

    return epoch_loss / max(1, total_steps)


# %% [markdown]
# ## 10. Validation Metrics: Loss, BLEU, and CER

# %%
import sacrebleu


def levenshtein_distance(ref: str, hyp: str) -> int:
    if ref == hyp:
        return 0
    if len(ref) == 0:
        return len(hyp)
    if len(hyp) == 0:
        return len(ref)

    prev_row = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current_row = [i]
        for j, hyp_char in enumerate(hyp, start=1):
            insertions = current_row[j - 1] + 1
            deletions = prev_row[j] + 1
            substitutions = prev_row[j - 1] + (ref_char != hyp_char)
            current_row.append(min(insertions, deletions, substitutions))
        prev_row = current_row
    return prev_row[-1]


def character_error_rate(references: List[str], hypotheses: List[str]) -> float:
    total_chars = 0
    total_distance = 0
    for ref, hyp in zip(references, hypotheses):
        total_chars += max(len(ref), 1)
        total_distance += levenshtein_distance(ref, hyp)
    return total_distance / total_chars


def bleu_score(references: List[str], hypotheses: List[str]) -> float:
    ref_tokens = [[" ".join(list(ref)) for ref in references]]
    hyp_tokens = [" ".join(list(hyp)) for hyp in hypotheses]
    return sacrebleu.corpus_bleu(hyp_tokens, ref_tokens).score


def evaluate(
    model: Seq2Seq,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    max_decode_len: int,
) -> Dict[str, float]:
    model.eval()
    losses = []
    references, hypotheses = [], []

    with torch.no_grad():
        for batch in dataloader:
            src, src_lengths, tgt, _ = move_batch_to_device(batch, device)
            outputs = model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)

            output_dim = outputs.shape[-1]
            output = outputs[:, 1:, :].reshape(-1, output_dim)
            target = tgt[:, 1:].reshape(-1)
            loss = criterion(output, target)
            losses.append(loss.item())

            predictions = model.greedy_decode(src, src_lengths, max_decode_len)
            for idx in range(predictions.size(0)):
                pred_str = decode_indices(predictions[idx].tolist(), tgt_vocab)
                ref_str = decode_indices(tgt[idx].tolist(), tgt_vocab)
                hypotheses.append(pred_str)
                references.append(ref_str)

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    bleu = bleu_score(references, hypotheses) if hypotheses else 0.0
    cer = character_error_rate(references, hypotheses) if hypotheses else 1.0

    return {"loss": avg_loss, "bleu": bleu, "cer": cer}

# %% [markdown]
# ## 11. Model Checkpointing and Early Stopping

# %%
class EarlyStopping:
    def __init__(self, patience: int, mode: str = "min", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_state = None
        self.best_epoch = None

    def _improves(self, score: float) -> bool:
        if self.best_score is None:
            return True
        delta = score - self.best_score
        if self.mode == "min":
            return delta < -self.min_delta
        return delta > self.min_delta

    def step(self, score: float, state: Dict) -> bool:
        improved = self._improves(score)
        if improved:
            self.best_score = score
            self.best_state = state
            self.best_epoch = state.get("epoch")
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def build_model_and_optimizer() -> Tuple[Seq2Seq, torch.optim.Optimizer, nn.Module]:
    enc = Encoder(
        input_dim=len(src_vocab["stoi"]),
        emb_dim=config["model"]["embedding_size"],
        hid_dim=config["model"]["hidden_size"],
        n_layers=config["model"]["encoder_layers"],
        dropout=config["model"]["dropout"],
    )
    dec = Decoder(
        output_dim=len(tgt_vocab["stoi"]),
        emb_dim=config["model"]["embedding_size"],
        enc_hid_dim=config["model"]["hidden_size"],
        dec_hid_dim=config["model"]["hidden_size"],
        n_layers=config["model"]["decoder_layers"],
        dropout=config["model"]["dropout"],
    )
    model = Seq2Seq(enc, dec, pad_idx, tgt_pad_idx, config["device"]).to(config["device"])
    model.apply(init_weights)

    training_cfg = config["training"]
    base_lr = training_cfg["learning_rate"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=training_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    return model, optimizer, criterion


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
) -> Tuple[Optional[object], Dict[str, Any]]:
    scheduler_cfg = config["training"].get("scheduler", {}) or {}
    scheduler_type = (scheduler_cfg.get("type") or "none").lower()
    metadata: Dict[str, Any] = {
        "type": scheduler_type,
        "step_on": None,
        "monitor": scheduler_cfg.get("monitor"),
    }

    if scheduler_type in {"none", "", "null"}:
        return None, metadata

    if scheduler_type == "one_cycle":
        params = scheduler_cfg.get("params", {})
        max_lr = params.get("max_lr")
        if max_lr is None:
            raise ValueError("OneCycleLR scheduler requires 'max_lr' in params.")
        max_steps_cfg = config["training"].get("max_steps_per_epoch")
        try:
            total_batches = len(train_loader)
        except TypeError as exc:
            raise ValueError(
                "OneCycleLR scheduler requires the training DataLoader to have a defined length."
            ) from exc
        if total_batches <= 0:
            raise ValueError("Training DataLoader must contain at least one batch before scheduling.")
        steps_per_epoch = total_batches if max_steps_cfg is None else max(1, min(max_steps_cfg, total_batches))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=config["training"]["num_epochs"],
            steps_per_epoch=steps_per_epoch,
            pct_start=params.get("pct_start", 0.3),
            anneal_strategy=params.get("anneal_strategy", "cos"),
            div_factor=params.get("div_factor", 25.0),
            final_div_factor=params.get("final_div_factor", 1e4),
            three_phase=params.get("three_phase", False),
        )
        metadata["step_on"] = "batch"
        return scheduler, metadata

    if scheduler_type == "reduce_on_plateau":
        params = {
            "mode": "min",
            "factor": 0.5,
            "patience": 2,
            "threshold": 1e-4,
            "min_lr": 1e-5,
            "verbose": False,
        }
        params.update(scheduler_cfg.get("params", {}))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        metadata["step_on"] = "epoch_metric"
        if metadata["monitor"] is None:
            metadata["monitor"] = "loss"
        return scheduler, metadata

    if scheduler_type == "cosine_annealing":
        params = scheduler_cfg.get("params", {})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get("t_max", config["training"]["num_epochs"]),
            eta_min=params.get("eta_min", 0.0),
        )
        metadata["step_on"] = "epoch"
        return scheduler, metadata

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def train_model():
    model, optimizer, criterion = build_model_and_optimizer()
    scheduler, scheduler_meta = create_scheduler(optimizer, train_loader)
    scheduler_meta = scheduler_meta or {}
    scheduler_step_on = scheduler_meta.get("step_on")
    scheduler_type = scheduler_meta.get("type")
    if scheduler_type not in {None, "", "none", "null"}:
        print(f"Using {scheduler_type} learning rate scheduler.")
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")

    checkpoint_path = Path(config["artifacts"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        mode="min",
        min_delta=1e-4,
    )

    history = []
    for epoch in range(1, config["training"]["num_epochs"] + 1):
        start = time.time()
        prev_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config["device"],
            config["training"]["teacher_forcing_ratio"],
            config["training"]["grad_clip"],
            config["training"]["max_steps_per_epoch"],
            scheduler=scheduler if scheduler_step_on == "batch" else None,
            scheduler_step_on=scheduler_step_on or "epoch",
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            config["device"],
            config["evaluation"]["max_decode_len"],
        )

        if scheduler is not None:
            if scheduler_step_on == "epoch_metric":
                monitor = scheduler_meta.get("monitor", "loss")
                metric_key = monitor.replace("val_", "")
                metric_value = val_metrics.get(metric_key)
                if metric_value is None:
                    available = ", ".join(sorted(val_metrics.keys()))
                    raise KeyError(
                        f"Validation metric '{metric_key}' not found. Available metrics: {available}."
                    )
                scheduler.step(metric_value)
            elif scheduler_step_on == "epoch":
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        lr_changed = scheduler is not None and not math.isclose(
            current_lr, prev_lr, rel_tol=1e-6, abs_tol=1e-8
        )
        elapsed = time.time() - start
        improved = early_stopping.step(val_metrics["cer"], {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
        })
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "time_sec": elapsed,
            "learning_rate": current_lr,
            "scheduler_type": scheduler_type,
            "is_best": improved,
        }
        history.append(history_entry)
        train_ppl = math.exp(min(train_loss, 20.0))
        log_parts = [
            f"Epoch {epoch:02d}",
            f"train_loss={train_loss:.4f}",
            f"train_ppl={train_ppl:.2f}",
            f"val_loss={val_metrics['loss']:.4f}",
            f"val_bleu={val_metrics['bleu']:.2f}",
            f"val_cer={val_metrics['cer']:.4f}",
            f"best_val_cer={early_stopping.best_score:.4f}",
            f"lr={current_lr:.6f}",
            f"time={elapsed:.1f}s",
            f"best_epoch={early_stopping.best_epoch if early_stopping.best_epoch is not None else '-'}",
            f"is_best={'YES' if improved else 'NO'}",
        ]
        print(" | ".join(log_parts))
        if lr_changed:
            if scheduler_type == "reduce_on_plateau" and current_lr < prev_lr:
                monitor = scheduler_meta.get("monitor", "validation metric")
                print(
                    f"  ↳ Learning rate reduced from {prev_lr:.6f} to {current_lr:.6f} "
                    f"after plateau in {monitor}."
                )
            elif scheduler_type == "one_cycle":
                print(
                    f"  ↳ OneCycleLR updated learning rate from {prev_lr:.6f} to {current_lr:.6f} during epoch {epoch:02d}."
                )
            elif scheduler_type == "cosine_annealing":
                print(
                    f"  ↳ CosineAnnealingLR set learning rate to {current_lr:.6f} at epoch {epoch:02d}."
                )

        if improved:
            torch.save(early_stopping.best_state, checkpoint_path)
            print(
                f"  ↳ Saved new best model checkpoint to {checkpoint_path} at epoch {epoch:02d}"
            )

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if early_stopping.best_state is not None:
        best_val_metrics = early_stopping.best_state["val_metrics"]
        best_epoch = early_stopping.best_state.get("epoch")
        print(
            f"Best validation | epoch={best_epoch:02d} | "
            f"loss={best_val_metrics['loss']:.4f} | "
            f"BLEU={best_val_metrics['bleu']:.2f} | "
            f"CER={best_val_metrics['cer']:.4f}"
        )
        model.load_state_dict(early_stopping.best_state["model_state"])
    else:
        print("Early stopping did not save any state; keeping final model.")

    return model, history, str(checkpoint_path)


trained_model, training_history, best_checkpoint_path = train_model()


# %% [markdown]
# ## 12. Test Set Evaluation and Metric Reporting

# %%
def load_checkpoint(model: Seq2Seq, checkpoint_path: str, device: str) -> Dict:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_file} not found.")
    state = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state["model_state"])
    return state


best_state = load_checkpoint(trained_model, best_checkpoint_path, config["device"])
print(
    "Best checkpoint | "
    f"epoch={best_state['epoch']} | "
    f"val_loss={best_state['val_metrics']['loss']:.4f} | "
    f"val_BLEU={best_state['val_metrics']['bleu']:.2f} | "
    f"val_CER={best_state['val_metrics']['cer']:.4f}"
)

test_metrics = evaluate(
    trained_model,
    test_loader,
    nn.CrossEntropyLoss(ignore_index=tgt_pad_idx),
    config["device"],
    config["evaluation"]["max_decode_len"],
)

print(
    "Test metrics | "
    f"loss={test_metrics['loss']:.4f} | "
    f"BLEU={test_metrics['bleu']:.2f} | "
    f"CER={test_metrics['cer']:.4f}"
)

# %% [markdown]
# ## 13. Inference Utilities for Sample Translations

# %%
def tensorize_single(sentence: str, vocab: Dict[str, Dict[str, int]], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cleaned = clean_text(sentence)
    indices = encode_sentence(cleaned, vocab, max_len)
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    length = torch.tensor([len(indices)], dtype=torch.long)
    return tensor, length


def translate_sentence(
    model: Seq2Seq,
    sentence: str,
    max_len: int = None,
) -> str:
    model.eval()
    max_len = max_len or config["evaluation"]["max_decode_len"]
    src_tensor, src_length = tensorize_single(sentence, src_vocab, config["data"]["max_seq_len"])
    src_tensor = src_tensor.to(config["device"])
    src_length = src_length.to(config["device"])

    with torch.no_grad():
        prediction_indices = model.greedy_decode(src_tensor, src_length, max_len)
    prediction = decode_indices(prediction_indices[0].tolist(), tgt_vocab)
    return prediction


sample_sentence = usable_pairs[0][0]
print(f"Sample source: {sample_sentence}")
print(f"Model prediction: {translate_sentence(trained_model, sample_sentence)}")

# %% [markdown]
# ## 14. Batch Inference on Reference Text Files

# %%
def collect_batch_predictions(
    model: Seq2Seq,
    dataloader: DataLoader,
    limit: int = 10,
) -> List[Tuple[str, str, str]]:
    model.eval()
    collected = []
    with torch.no_grad():
        for batch in dataloader:
            src, src_lengths, tgt, _ = move_batch_to_device(batch, config["device"])
            preds = model.greedy_decode(src, src_lengths, config["evaluation"]["max_decode_len"])
            for i in range(src.size(0)):
                if len(collected) >= limit:
                    return collected
                src_str = decode_indices(src[i].cpu().tolist(), src_vocab)
                tgt_str = decode_indices(tgt[i].cpu().tolist(), tgt_vocab)
                pred_str = decode_indices(preds[i].cpu().tolist(), tgt_vocab)
                collected.append((src_str, tgt_str, pred_str))
    return collected


sampled_triplets = collect_batch_predictions(trained_model, test_loader, limit=12)
prediction_path = Path(config["artifacts"]["prediction_path"])
prediction_path.parent.mkdir(parents=True, exist_ok=True)

with prediction_path.open("w", encoding="utf-8") as fout:
    for src_text, tgt_text, pred_text in sampled_triplets:
        fout.write(f"SRC\t{src_text}\nREF\t{tgt_text}\nPRED\t{pred_text}\n\n")

print(f"Saved {len(sampled_triplets)} sample predictions to {prediction_path}.")
for idx, (src_text, tgt_text, pred_text) in enumerate(sampled_triplets[:5], start=1):
    print("-" * 80)
    print(f"Example {idx}")
    print(f"SRC : {src_text}")
    print(f"REF : {tgt_text}")
    print(f"PRED: {pred_text}")

# %% [markdown]
# ### Try your own sentence

# %%
custom_input = "یہاں اپنی اردو لائن لکھیں"  # Replace with any Urdu sentence
print(f"Input: {custom_input}")
print(f"Prediction: {translate_sentence(trained_model, custom_input)}")


