"""Speaker similarity utility functions.

Provides cosine similarity calculation and WavLM-SV embedding extraction
for speaker verification evaluation.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector (1-D).
        b: Second embedding vector (1-D).

    Returns:
        Cosine similarity in [-1.0, 1.0].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_wavlm_embedding(audio_path: str) -> np.ndarray:
    """Extract WavLM-SV speaker verification embedding from audio file.

    Uses the WavLM-Base-Plus-SV model for speaker embedding extraction.
    Lazy-imports torch and transformers to avoid loading at module scope.

    Args:
        audio_path: Path to audio file (.wav).

    Returns:
        1-D numpy array of speaker embedding features.
    """
    import os

    import torch
    import torchaudio

    # Force PyTorch backend — transformers 5.0 defaults to MLX on Apple Silicon,
    # which causes Metal shader compilation and extreme slowdown under memory pressure.
    os.environ.setdefault("TRANSFORMERS_BACKEND", "pt")
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    model_name = "microsoft/wavlm-base-plus-sv"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name)

    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Truncate to first 10 seconds — speaker identity is captured in the first few
    # seconds; longer files cause WavLM CPU inference to take many minutes.
    max_samples = 10 * 16000
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.embeddings.squeeze().numpy()
    return embedding
