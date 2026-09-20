"""Run with `python -B example.py`; entirely synthetic feature-only inference."""
from pathlib import Path
import torch
from runtime import load_model, synthetic_inputs

if __name__ == "__main__":
    predictor, metadata = load_model("pusht_unbounded_spatial_mix_s0", Path(__file__).parent)
    initial, commands = synthetic_inputs("pusht", 1)
    prediction = predictor.predict(torch.from_numpy(initial), torch.from_numpy(commands))
    print(metadata)
    print("Synthetic feature forecast shape:", tuple(prediction.shape))
