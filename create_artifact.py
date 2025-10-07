import torch
import os
from pathlib import Path

# Default configuration (adjust if necessary)
config = {
    "embed_dim": 256,
    "hidden_size": 512,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "dropout": 0.3,
    "max_len": 220,
}

# Dummy tokenizers based on what might be expected.
# This is a placeholder. If you have the actual tokenizer files,
# you should load them here.
src_tokenizer_mock = {"itos": list(" abcdefghijklmnopqrstuvwxyzاآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوھہیے")}
tgt_tokenizer_mock = {"itos": list(" abcdefghijklmnopqrstuvwxyz")}


# Define the path to your existing model and the new artifact path
EXISTING_MODEL_PATH = "urdu_romanUrdu.pt"
NEW_ARTIFACT_PATH = Path("artifacts") / "char_seq2seq.pt"

# Create the artifacts directory if it doesn't exist
NEW_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load the state dictionary from your existing model
try:
    model_state_dict = torch.load(EXISTING_MODEL_PATH, map_location=torch.device("cpu"))

    # Create the payload with the expected structure
    payload = {
        "config": config,
        "model_state_dict": model_state_dict,
        "src_tokenizer": src_tokenizer_mock,
        "tgt_tokenizer": tgt_tokenizer_mock,
    }

    # Save the new artifact
    torch.save(payload, NEW_ARTIFACT_PATH)

    print(f"Successfully created new artifact at: {NEW_ARTIFACT_PATH}")

except FileNotFoundError:
    print(f"Error: The model file was not found at {EXISTING_MODEL_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")
