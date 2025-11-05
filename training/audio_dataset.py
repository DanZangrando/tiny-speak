"""Audio dataset helpers for TinyListener training."""

from __future__ import annotations

import base64
import io
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from training.config import load_master_dataset_config
from utils import WAV2VEC_SR

try:  # pragma: no cover - defensive import
    import torchaudio
except Exception:  # noqa: BLE001
    torchaudio = None  # type: ignore


DEFAULT_AUDIO_SPLIT_RATIOS: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}


@dataclass
class AudioSample:
    """Container with decoded audio information for a single word example."""

    word: str
    waveform: torch.Tensor
    sample_rate: int
    duration_ms: float | None
    metadata: Dict[str, Any]


class AudioWordDataset(Dataset):
    """Dataset backed by audio samples coming from the master configuration."""

    def __init__(self, samples: List[AudioSample], class_names: Iterable[str]) -> None:
        self.samples = samples
        self.class_names = list(class_names)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.class_names)}

    def __len__(self) -> int:  # noqa: D401
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:  # noqa: D401
        sample = self.samples[index]
        waveform = sample.waveform.clone()  # defensive copy to avoid in-place ops downstream
        label = self.word_to_idx[sample.word]
        return {
            "waveform": waveform,
            "label": label,
            "word": sample.word,
            "duration_ms": sample.duration_ms,
            "metadata": sample.metadata,
        }

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def words(self) -> List[str]:
        return self.class_names


def _decode_waveform(audio_b64: str, target_sr: int = WAV2VEC_SR) -> Tuple[torch.Tensor, int]:
    raw = base64.b64decode(audio_b64)
    buffer = io.BytesIO(raw)

    waveform: torch.Tensor
    sample_rate: int

    waveform = None
    sample_rate = -1

    if torchaudio is not None:
        buffer.seek(0)
        try:
            waveform, sample_rate = torchaudio.load(buffer)
        except Exception:  # pragma: no cover - gracefully fallback if torchaudio backend is unavailable
            waveform = None

    if waveform is None:
        import soundfile as sf  # local import

        buffer.seek(0)
        audio_np, sample_rate = sf.read(buffer)
        waveform = torch.tensor(audio_np, dtype=torch.float32).transpose(0, -1)  # type: ignore[arg-type]

    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.to(dtype=torch.float32)

    if sample_rate != target_sr:
        if torchaudio is None:  # pragma: no cover - fallback
            raise RuntimeError("Resample requires torchaudio to be installed.")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
        sample_rate = target_sr

    return waveform, sample_rate


def _compute_split_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    if total == 0:
        return {key: 0 for key in ratios}

    raw_counts: Dict[str, int] = {key: int(math.floor(total * ratio)) for key, ratio in ratios.items()}
    remainder = total - sum(raw_counts.values())
    fractional = sorted(
        ((total * ratio) - raw_counts[key], key) for key, ratio in ratios.items()
    )
    fractional.reverse()
    for _, split in fractional:
        if remainder == 0:
            break
        raw_counts[split] += 1
        remainder -= 1
    return raw_counts


def _load_all_samples(config: Dict[str, Any], target_sr: int, seed: int) -> Tuple[List[str], Dict[str, List[AudioSample]]]:
    generated = config.get("generated_samples", {}) or {}
    words = sorted(generated.keys())
    rng = random.Random(seed)

    samples_by_word: Dict[str, List[AudioSample]] = {word: [] for word in words}
    for word, entries in generated.items():
        for entry in entries:
            audio_b64 = entry.get("audio_base64")
            if not audio_b64:
                continue
            try:
                waveform, sr = _decode_waveform(audio_b64, target_sr=target_sr)
            except Exception:  # noqa: BLE001
                continue
            duration_ms = entry.get("duracion_ms")
            metadata = {k: v for k, v in entry.items() if k != "audio_base64"}
            samples_by_word[word].append(
                AudioSample(
                    word=word,
                    waveform=waveform,
                    sample_rate=sr,
                    duration_ms=duration_ms,
                    metadata=metadata,
                )
            )
        rng.shuffle(samples_by_word[word])

    # Filter out words without samples entirely.
    words = [word for word in words if samples_by_word[word]]
    samples_by_word = {word: samples_by_word[word] for word in words}
    return words, samples_by_word


def build_audio_datasets(
    *,
    seed: int,
    split_ratios: Dict[str, float] | None = None,
    target_sr: int = WAV2VEC_SR,
) -> Dict[str, AudioWordDataset]:
    config = load_master_dataset_config()
    ratios = split_ratios or DEFAULT_AUDIO_SPLIT_RATIOS
    ratio_total = sum(ratios.values())
    if not math.isclose(ratio_total, 1.0, rel_tol=1e-3):
        raise ValueError("Las proporciones de split de audio deben sumar 1.0")

    words, samples_by_word = _load_all_samples(config, target_sr=target_sr, seed=seed)
    if not words:
        raise ValueError("El master dataset de audio no contiene muestras generadas.")

    splits: Dict[str, List[AudioSample]] = {key: [] for key in ratios.keys()}
    for word in words:
        samples = samples_by_word[word]
        counts = _compute_split_counts(len(samples), ratios)
        cursor = 0
        for split_name in ratios.keys():
            take = counts.get(split_name, 0)
            if take <= 0:
                continue
            splits[split_name].extend(samples[cursor : cursor + take])
            cursor += take

    datasets: Dict[str, AudioWordDataset] = {}
    for split_name, samples in splits.items():
        datasets[split_name] = AudioWordDataset(samples=samples, class_names=words)
    return datasets


def audio_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    waveforms = [item["waveform"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    words = [item["word"] for item in batch]
    durations = [item.get("duration_ms") for item in batch]
    metadata = [item.get("metadata", {}) for item in batch]
    return {
        "waveforms": waveforms,
        "label": labels,
        "words": words,
        "duration_ms": durations,
        "metadata": metadata,
    }


def build_audio_dataloaders(
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    split_ratios: Dict[str, float] | None = None,
    target_sr: int = WAV2VEC_SR,
    shuffle_train: bool = True,
) -> Tuple[AudioWordDataset, AudioWordDataset, AudioWordDataset, Dict[str, DataLoader]]:
    datasets = build_audio_datasets(seed=seed, split_ratios=split_ratios, target_sr=target_sr)

    loaders: Dict[str, DataLoader] = {}
    for split_name, dataset in datasets.items():
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_train if split_name == "train" else False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=audio_collate_fn,
        )

    return datasets["train"], datasets["val"], datasets["test"], loaders


def load_audio_splits(
    seed: int,
    split_ratios: Dict[str, float] | None = None,
) -> Tuple[Dict[str, AudioWordDataset] | None, str | None]:
    try:
        datasets = build_audio_datasets(seed=seed, split_ratios=split_ratios)
        return datasets, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
