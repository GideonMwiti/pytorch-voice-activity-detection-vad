"""
Project 84: Voice Activity Detection (VAD)
===========================================
Enterprise-grade Voice Activity Detection (VAD) and speech segmentation in PyTorch
using deep neural Silero VAD (TorchScript JIT), Torchaudio functional filtering,
and energy-spectral entropy baselines for real-time and offline speech/silence classification.

Author: Gideon Mwiti
License: MIT
"""

import os
import sys
import time
import json
import glob
import math
import warnings
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "audio")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

SAMPLE_RATE = 16000
WINDOW_SIZE_SAMPLES = 512       # 32 ms at 16 kHz
FRAME_DURATION_MS = (WINDOW_SIZE_SAMPLES / SAMPLE_RATE) * 1000.0  # 32.0 ms
SILERO_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.jit")

# ---------------------------------------------------------------------------
# VAD Engine 1: Silero Neural VAD (TorchScript JIT)
# ---------------------------------------------------------------------------
class SileroVADEngine:
    """
    Production wrapper for deep neural Silero VAD (TorchScript JIT).
    Computes frame-level speech probabilities P(speech | x_t) in under 0.5 ms
    with stateful recurrent context across audio frames.
    """
    def __init__(self, model_path=SILERO_MODEL_PATH, default_threshold=0.50):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Silero VAD model not found at {model_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        self.default_threshold = default_threshold
        self.reset_states()

    def reset_states(self):
        """Reset internal recurrent hidden state vectors."""
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    @torch.no_grad()
    def predict_chunk(self, chunk: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
        """
        Evaluate a single chunk of exactly 512 audio samples (32 ms at 16 kHz).
        Returns speech probability in [0.0, 1.0].
        """
        if chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)
        prob = self.model(chunk, sr).item()
        return float(prob)

    @torch.no_grad()
    def predict_audio(self, audio: torch.Tensor, sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Evaluate full audio waveform sequentially, maintaining recurrent state context.
        Returns a 1D numpy array of probabilities, one per 512-sample chunk.
        """
        self.reset_states()
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        
        num_chunks = len(audio) // WINDOW_SIZE_SAMPLES
        probs = np.zeros(num_chunks, dtype=np.float32)
        
        for i in range(num_chunks):
            start = i * WINDOW_SIZE_SAMPLES
            end = start + WINDOW_SIZE_SAMPLES
            chunk = audio[start:end].unsqueeze(0)
            probs[i] = self.model(chunk, sr).item()
            
        return probs

    def get_speech_timestamps(
        self,
        audio: torch.Tensor,
        sr: int = SAMPLE_RATE,
        threshold: float = None,
        neg_threshold: float = None,
        min_speech_ms: float = 250.0,
        min_silence_ms: float = 100.0,
        speech_pad_ms: float = 30.0,
    ) -> list:
        """
        Extract continuous speech segment boundaries (start, end, duration)
        using dual-threshold hysteresis and minimum duration smoothing.
        """
        threshold = threshold or self.default_threshold
        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.05)

        probs = self.predict_audio(audio, sr)
        window_sec = WINDOW_SIZE_SAMPLES / sr
        min_speech_chunks = max(1, int(math.ceil(min_speech_ms / 1000.0 / window_sec)))
        min_silence_chunks = max(1, int(math.ceil(min_silence_ms / 1000.0 / window_sec)))
        pad_sec = speech_pad_ms / 1000.0
        total_duration = len(audio) / sr

        triggered = False
        temp_silence_count = 0
        current_start_chunk = 0
        speech_segments = []

        for i, p in enumerate(probs):
            if not triggered:
                if p >= threshold:
                    triggered = True
                    current_start_chunk = i
                    temp_silence_count = 0
            else:
                if p < neg_threshold:
                    temp_silence_count += 1
                    if temp_silence_count >= min_silence_chunks:
                        end_chunk = i - temp_silence_count + 1
                        duration_chunks = end_chunk - current_start_chunk
                        if duration_chunks >= min_speech_chunks:
                            start_time = max(0.0, current_start_chunk * window_sec - pad_sec)
                            end_time = min(total_duration, end_chunk * window_sec + pad_sec)
                            chunk_probs = probs[current_start_chunk:end_chunk]
                            mean_conf = float(np.mean(chunk_probs)) if len(chunk_probs) > 0 else float(p)
                            speech_segments.append({
                                "start": round(start_time, 3),
                                "end": round(end_time, 3),
                                "duration": round(end_time - start_time, 3),
                                "confidence": round(mean_conf, 4)
                            })
                        triggered = False
                        temp_silence_count = 0
                else:
                    temp_silence_count = 0

        # Final segment flush if triggered at boundary
        if triggered:
            end_chunk = len(probs)
            duration_chunks = end_chunk - current_start_chunk
            if duration_chunks >= min_speech_chunks:
                start_time = max(0.0, current_start_chunk * window_sec - pad_sec)
                end_time = total_duration
                chunk_probs = probs[current_start_chunk:end_chunk]
                mean_conf = float(np.mean(chunk_probs)) if len(chunk_probs) > 0 else 0.5
                speech_segments.append({
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "duration": round(end_time - start_time, 3),
                    "confidence": round(mean_conf, 4)
                })

        return speech_segments

# ---------------------------------------------------------------------------
# VAD Engine 2: Energy & Spectral Entropy Baseline
# ---------------------------------------------------------------------------
class EnergyEntropyVADEngine:
    """
    Mathematical feature baseline computing frame-level Root Mean Square (RMS) energy
    and Spectral Entropy across Short-Time Fourier Transform (STFT) frames.
    """
    def __init__(self, sr=SAMPLE_RATE, window_size=WINDOW_SIZE_SAMPLES):
        self.sr = sr
        self.window_size = window_size
        self.n_fft = 512
        self.hop_length = window_size

    def predict_audio(self, audio: torch.Tensor) -> np.ndarray:
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
            
        num_chunks = len(audio) // self.window_size
        probs = np.zeros(num_chunks, dtype=np.float32)
        
        # Pre-compute STFT
        stft = torch.stft(
            audio[:num_chunks * self.window_size],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=torch.hann_window(self.window_size),
            return_complex=True
        ) # [F, T]
        
        mag = torch.abs(stft) ** 2 # Power spectrum [257, T]
        eps = 1e-10
        total_power = mag.sum(dim=0) + eps
        norm_power = mag / total_power.unsqueeze(0)
        
        # Spectral entropy: -sum(p * log2(p)) / log2(F)
        entropy = -torch.sum(norm_power * torch.log2(norm_power + eps), dim=0)
        norm_entropy = entropy / np.log2(mag.shape[0]) # [0, 1]
        
        # RMS Energy
        frames = audio[:num_chunks * self.window_size].reshape(num_chunks, self.window_size)
        rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + eps)
        max_rms = rms.max()
        norm_rms = (rms / (max_rms + eps)).clamp(0.0, 1.0)
        
        # Combined heuristic: Speech has high RMS energy and moderate-to-low entropy
        # Non-speech / background noise typically has high entropy or very low energy
        min_len = min(len(norm_rms), len(norm_entropy), num_chunks)
        norm_rms = norm_rms[:min_len]
        norm_entropy = norm_entropy[:min_len]
        heuristic = (0.7 * norm_rms + 0.3 * (1.0 - norm_entropy)).cpu().numpy()
        min_h, max_h = heuristic.min(), heuristic.max()
        if max_h > min_h:
            probs = (heuristic - min_h) / (max_h - min_h)
        else:
            probs = heuristic
            
        final_probs = np.zeros(num_chunks, dtype=np.float32)
        final_probs[:min_len] = probs
        return final_probs

# ---------------------------------------------------------------------------
# VAD Engine 3: Torchaudio Functional VAD Wrapper
# ---------------------------------------------------------------------------
class TorchaudioVADEngine:
    """
    Wrapper around torchaudio.functional.vad using forward and backward passes
    for complete silence and background noise trimming.
    """
    def __init__(self, sr=SAMPLE_RATE, trigger_level=7.0):
        self.sr = sr
        self.trigger_level = trigger_level

    def trim_speech(self, audio: torch.Tensor) -> torch.Tensor:
        """Trim silence from both head and tail using forward-backward pass."""
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        # Forward pass trims leading silence
        trimmed_front = AF.vad(audio, sample_rate=self.sr, trigger_level=self.trigger_level)
        if trimmed_front.shape[-1] == 0:
            return trimmed_front
            
        # Reverse pass trims trailing silence
        reversed_audio = torch.flip(trimmed_front, dims=[-1])
        trimmed_back = AF.vad(reversed_audio, sample_rate=self.sr, trigger_level=self.trigger_level)
        return torch.flip(trimmed_back, dims=[-1])

# ---------------------------------------------------------------------------
# Multi-Turn Session Audio Synthesizer
# ---------------------------------------------------------------------------
def build_evaluation_session():
    """
    Constructs a realistic 20-second continuous multi-turn speech session
    interleaving real human speech utterances with natural background noises
    and pauses, recording precise ground-truth frame-level speech activity.
    """
    speech_files = sorted(glob.glob(os.path.join(DATA_DIR, "*speech.wav")))
    bg_files = sorted(glob.glob(os.path.join(DATA_DIR, "*background.wav")))
    
    if len(speech_files) == 0:
        raise RuntimeError("No speech audio clips found in data/audio/")

    session_chunks = []
    ground_truth_chunks = [] # 1 for speech, 0 for non-speech
    target_duration_s = 20.0
    current_time_s = 0.0
    turn_idx = 0

    print(f"Building realistic {target_duration_s}s multi-turn conversational audio session...")

    while current_time_s < target_duration_s:
        # 1. Non-speech pause / background noise (0.8s to 1.6s)
        pause_dur = 1.0 if turn_idx % 2 == 0 else 1.4
        num_pause_samples = int(pause_dur * SAMPLE_RATE)
        
        if len(bg_files) > 0 and turn_idx % 3 != 0:
            bg_path = bg_files[turn_idx % len(bg_files)]
            bg_wav, bg_sr = sf.read(bg_path)
            bg_wav = torch.tensor(bg_wav, dtype=torch.float32)
            if bg_wav.ndim > 1:
                bg_wav = bg_wav.mean(dim=-1)
            if bg_sr != SAMPLE_RATE:
                bg_wav = T.Resample(bg_sr, SAMPLE_RATE)(bg_wav.unsqueeze(0)).squeeze(0)
            # Repeat or slice to match pause length
            if len(bg_wav) < num_pause_samples:
                reps = (num_pause_samples // len(bg_wav)) + 1
                bg_wav = bg_wav.repeat(reps)[:num_pause_samples]
            else:
                bg_wav = bg_wav[:num_pause_samples]
            pause_chunk = bg_wav * 0.25 # Lower ambient volume
        else:
            # Clean silence with subtle micro-noise floor
            pause_chunk = torch.randn(num_pause_samples) * 0.005

        session_chunks.append(pause_chunk)
        num_pause_frames = len(pause_chunk) // WINDOW_SIZE_SAMPLES
        ground_truth_chunks.extend([0] * num_pause_frames)
        current_time_s += len(pause_chunk) / SAMPLE_RATE

        if current_time_s >= target_duration_s:
            break

        # 2. Genuine human speech turn (1.0s to 2.5s)
        sp_path = speech_files[turn_idx % len(speech_files)]
        sp_wav, sp_sr = sf.read(sp_path)
        sp_wav = torch.tensor(sp_wav, dtype=torch.float32)
        if sp_wav.ndim > 1:
            sp_wav = sp_wav.mean(dim=-1)
        if sp_sr != SAMPLE_RATE:
            sp_wav = T.Resample(sp_sr, SAMPLE_RATE)(sp_wav.unsqueeze(0)).squeeze(0)

        # Normalize speech utterance
        sp_wav = sp_wav / (torch.max(torch.abs(sp_wav)) + 1e-8) * 0.85
        
        session_chunks.append(sp_wav)
        num_speech_frames = len(sp_wav) // WINDOW_SIZE_SAMPLES
        ground_truth_chunks.extend([1] * num_speech_frames)
        current_time_s += len(sp_wav) / SAMPLE_RATE
        turn_idx += 1

    session_audio = torch.cat(session_chunks)
    total_frames = len(ground_truth_chunks)
    session_audio = session_audio[:total_frames * WINDOW_SIZE_SAMPLES]
    ground_truth = np.array(ground_truth_chunks, dtype=np.int32)

    return session_audio, ground_truth

# ---------------------------------------------------------------------------
# Benchmarks & Latency Evaluation
# ---------------------------------------------------------------------------
def run_latency_benchmark(silero_engine, iterations=200):
    """
    Profile frame-level inference latency on a 32 ms audio chunk (512 samples)
    across multiple warm-up and timed iterations.
    """
    dummy_chunk = torch.randn(1, WINDOW_SIZE_SAMPLES)
    
    # Warm-up
    for _ in range(20):
        _ = silero_engine.predict_chunk(dummy_chunk, SAMPLE_RATE)
        
    latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = silero_engine.predict_chunk(dummy_chunk, SAMPLE_RATE)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)
        
    latencies_ms = np.array(latencies_ms)
    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    max_ms = float(np.max(latencies_ms))
    rtf = float(mean_ms / FRAME_DURATION_MS)
    throughput = float(1000.0 / mean_ms)
    headroom_pct = float((1.0 - rtf) * 100.0)

    return {
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "p95_latency_ms": p95_ms,
        "max_latency_ms": max_ms,
        "step_budget_ms": FRAME_DURATION_MS,
        "real_time_factor_rtf": rtf,
        "throughput_frames_per_sec": throughput,
        "cpu_headroom_pct": headroom_pct,
        "distribution": latencies_ms.tolist()
    }

def evaluate_noise_robustness(silero_engine, session_audio, ground_truth):
    """
    Evaluate VAD F1-score across 5 Signal-to-Noise Ratio (SNR) levels:
    Clean (Original), +20 dB, +10 dB, 0 dB, and -5 dB with additive noise.
    """
    snr_levels = ["Clean (>30 dB)", "+20 dB", "+10 dB", "0 dB", "-5 dB"]
    snr_db_values = [None, 20.0, 10.0, 0.0, -5.0]
    results = []

    clean_power = torch.mean(session_audio ** 2)

    for label, snr_db in zip(snr_levels, snr_db_values):
        if snr_db is None:
            eval_audio = session_audio.clone()
        else:
            noise_power = clean_power / (10.0 ** (snr_db / 10.0))
            noise = torch.randn_like(session_audio) * torch.sqrt(noise_power)
            eval_audio = session_audio + noise

        probs = silero_engine.predict_audio(eval_audio, SAMPLE_RATE)
        preds = (probs >= 0.50).astype(int)
        
        f1 = float(f1_score(ground_truth, preds))
        acc = float(accuracy_score(ground_truth, preds))
        rec = float(recall_score(ground_truth, preds))
        prec = float(precision_score(ground_truth, preds, zero_division=0))
        
        results.append({
            "snr_level": label,
            "snr_db": snr_db,
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4)
        })

    return results

# ---------------------------------------------------------------------------
# 6-Panel Visualization Report
# ---------------------------------------------------------------------------
def generate_diagnostic_report(
    session_audio,
    ground_truth,
    silero_probs,
    energy_probs,
    speech_timestamps,
    latency_info,
    robustness_results,
    output_path=os.path.join(OUTPUTS_DIR, "vad_benchmark_report.png")
):
    """
    Generate an executive 6-panel diagnostic visual analytics report.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    time_sec = np.arange(len(ground_truth)) * (WINDOW_SIZE_SAMPLES / SAMPLE_RATE)
    audio_time = np.linspace(0, len(session_audio) / SAMPLE_RATE, len(session_audio))

    # --- Panel 1: Waveform & Detected Speech Segments ---
    ax1 = axes[0, 0]
    ax1.plot(audio_time, session_audio.numpy(), color="#718096", alpha=0.55, linewidth=0.7, label="Audio Waveform")
    ax1.step(time_sec, ground_truth * 0.8, color="#2B6CB0", linewidth=1.5, where="post", label="Ground Truth Speech")
    for seg in speech_timestamps:
        ax1.axvspan(seg["start"], seg["end"], color="#38A169", alpha=0.25)
    ax1.set_title("Panel 1: Speech Waveform & VAD Segment Detections", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.05, 1.05)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 2: Log-Mel Spectrogram & Frame Probabilities ---
    ax2 = axes[0, 1]
    mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=160, n_mels=64)
    mel_spec = mel_transform(session_audio.unsqueeze(0)).squeeze(0)
    log_mel = torch.log(mel_spec + 1e-6).numpy()
    im = ax2.imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        extent=[0, audio_time[-1], 0, 64],
        cmap="magma"
    )
    # Overlay probability curve on twin axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(time_sec, silero_probs, color="#00FFCC", linewidth=1.8, label="P(Speech) Silero")
    ax2_twin.axhline(0.50, color="#FF0055", linestyle="--", linewidth=1.2, label="Threshold (0.50)")
    ax2_twin.set_ylabel("Speech Probability", color="#00FFCC")
    ax2_twin.set_ylim(-0.05, 1.05)
    ax2.set_title("Panel 2: Log-Mel Spectrogram + Silero Probabilities", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Mel Frequency Bins")
    ax2_twin.legend(loc="lower right", fontsize=8)

    # --- Panel 3: ROC Curves (Silero vs Energy Baseline) ---
    ax3 = axes[0, 2]
    fpr_silero, tpr_silero, _ = roc_curve(ground_truth, silero_probs)
    auc_silero = auc(fpr_silero, tpr_silero)
    
    fpr_energy, tpr_energy, _ = roc_curve(ground_truth, energy_probs)
    auc_energy = auc(fpr_energy, tpr_energy)

    ax3.plot(fpr_silero, tpr_silero, color="#3182CE", linewidth=2.0, label=f"Silero VAD (AUC = {auc_silero:.4f})")
    ax3.plot(fpr_energy, tpr_energy, color="#DD6B20", linewidth=1.8, linestyle="-.", label=f"Energy-Entropy VAD (AUC = {auc_energy:.4f})")
    ax3.plot([0, 1], [0, 1], color="#A0AEC0", linestyle=":", label="Random Classifier")
    ax3.set_title("Panel 3: Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("False Positive Rate (FPR)")
    ax3.set_ylabel("True Positive Rate (TPR)")
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 4: Precision, Recall, and F1 vs Threshold ---
    ax4 = axes[1, 0]
    thresholds = np.linspace(0.05, 0.95, 30)
    prec_list, rec_list, f1_list = [], [], []
    for th in thresholds:
        bin_pred = (silero_probs >= th).astype(int)
        prec_list.append(precision_score(ground_truth, bin_pred, zero_division=0))
        rec_list.append(recall_score(ground_truth, bin_pred, zero_division=0))
        f1_list.append(f1_score(ground_truth, bin_pred, zero_division=0))

    ax4.plot(thresholds, prec_list, color="#48BB78", linewidth=1.8, label="Precision")
    ax4.plot(thresholds, rec_list, color="#E53E3E", linewidth=1.8, label="Recall")
    ax4.plot(thresholds, f1_list, color="#805AD5", linewidth=2.2, label="F1-Score")
    ax4.axvline(0.50, color="#718096", linestyle="--", alpha=0.7, label="Default θ = 0.50")
    ax4.set_title("Panel 4: Detection Metrics vs. Decision Threshold", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Decision Threshold (θ)")
    ax4.set_ylabel("Metric Value")
    ax4.set_ylim(0.0, 1.05)
    ax4.legend(loc="lower left", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 5: Latency Distribution vs Real-Time Budget ---
    ax5 = axes[1, 1]
    lats = latency_info["distribution"]
    ax5.hist(lats, bins=25, color="#319795", edgecolor="#234E52", alpha=0.75, density=True)
    ax5.axvline(latency_info["mean_latency_ms"], color="#D69E2E", linewidth=2.0, linestyle="-", label=f"Mean: {latency_info['mean_latency_ms']:.2f} ms")
    ax5.axvline(latency_info["p95_latency_ms"], color="#E53E3E", linewidth=1.5, linestyle="--", label=f"P95: {latency_info['p95_latency_ms']:.2f} ms")
    ax5.axvline(FRAME_DURATION_MS, color="#C53030", linewidth=2.0, linestyle=":", label=f"32ms Budget (1.0x RTF)")
    ax5.set_title(f"Panel 5: Real-Time Latency (RTF = {latency_info['real_time_factor_rtf']:.4f}x)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Latency per 32ms Frame (ms)")
    ax5.set_ylabel("Probability Density")
    ax5.set_xlim(0, max(max(lats) * 1.3, 10.0))
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 6: Noise Robustness (F1-Score vs SNR) ---
    ax6 = axes[1, 2]
    snr_names = [r["snr_level"] for r in robustness_results]
    f1_vals = [r["f1_score"] for r in robustness_results]
    colors = ["#2B6CB0", "#3182CE", "#4299E1", "#ED8936", "#E53E3E"]
    bars = ax6.bar(snr_names, f1_vals, color=colors, width=0.55, edgecolor="#1A202C")
    for bar in bars:
        h = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax6.set_title("Panel 6: Acoustic Noise Robustness (F1 vs SNR)", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Acoustic Noise Condition")
    ax6.set_ylabel("VAD F1-Score")
    ax6.set_ylim(0.0, 1.12)
    ax6.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(
        "Project 84: Voice Activity Detection (VAD) Performance & Latency Analytics",
        fontsize=14,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"6-Panel Visual Diagnostic Report saved to: {output_path}")

# ---------------------------------------------------------------------------
# Main Orchestration Routine
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" PROJECT 84: VOICE ACTIVITY DETECTION (VAD) BENCHMARK SUITE")
    print("=" * 65)
    print(f"Sample Rate         : {SAMPLE_RATE} Hz")
    print(f"Window Size         : {WINDOW_SIZE_SAMPLES} samples ({FRAME_DURATION_MS:.1f} ms)")
    print(f"Silero Model Source : {SILERO_MODEL_PATH}")

    # 1. Initialize Engines
    silero_engine = SileroVADEngine(SILERO_MODEL_PATH, default_threshold=0.50)
    energy_engine = EnergyEntropyVADEngine(sr=SAMPLE_RATE, window_size=WINDOW_SIZE_SAMPLES)
    torchaudio_engine = TorchaudioVADEngine(sr=SAMPLE_RATE, trigger_level=7.0)

    # 2. Synthesize Benchmark Multi-Turn Session
    session_audio, ground_truth = build_evaluation_session()
    session_wav_path = os.path.join(OUTPUTS_DIR, "sample_conversation.wav")
    sf.write(session_wav_path, session_audio.numpy(), SAMPLE_RATE)
    print(f"Saved session audio to {session_wav_path} ({len(session_audio)/SAMPLE_RATE:.2f}s)")

    # 3. Offline VAD Evaluation
    print("\n[Stage 1] Computing Offline VAD Probabilities & Speech Timestamps...")
    silero_probs = silero_engine.predict_audio(session_audio, SAMPLE_RATE)
    energy_probs = energy_engine.predict_audio(session_audio)
    
    silero_preds = (silero_probs >= 0.50).astype(int)
    speech_timestamps = silero_engine.get_speech_timestamps(
        session_audio,
        sr=SAMPLE_RATE,
        threshold=0.50,
        min_speech_ms=250.0,
        min_silence_ms=100.0,
        speech_pad_ms=30.0
    )
    
    print(f"Detected {len(speech_timestamps)} continuous speech segments:")
    for idx, seg in enumerate(speech_timestamps):
        print(f"  Segment #{idx+1:02d}: {seg['start']:5.2f}s -> {seg['end']:5.2f}s (Dur: {seg['duration']:4.2f}s, Conf: {seg['confidence']:.3f})")

    # 4. Save Extracted Speech Only
    speech_chunks = []
    for seg in speech_timestamps:
        start_idx = int(seg["start"] * SAMPLE_RATE)
        end_idx = int(seg["end"] * SAMPLE_RATE)
        speech_chunks.append(session_audio[start_idx:end_idx])
    
    if len(speech_chunks) > 0:
        speech_only_wav = torch.cat(speech_chunks)
        speech_only_path = os.path.join(OUTPUTS_DIR, "speech_only_silero.wav")
        sf.write(speech_only_path, speech_only_wav.numpy(), SAMPLE_RATE)
        print(f"Concatenated speech-only audio saved to: {speech_only_path} ({len(speech_only_wav)/SAMPLE_RATE:.2f}s)")

    # 5. Export Timestamps to JSON and RTTM
    json_path = os.path.join(OUTPUTS_DIR, "speech_timestamps.json")
    with open(json_path, "w") as f:
        json.dump(speech_timestamps, f, indent=2)
    print(f"Speech timestamps JSON exported to: {json_path}")

    rttm_path = os.path.join(OUTPUTS_DIR, "speech_segments.rttm")
    with open(rttm_path, "w") as f:
        for seg in speech_timestamps:
            f.write(f"SPEAKER sample_conversation 1 {seg['start']:.3f} {seg['duration']:.3f} <NA> <NA> SPEECH <NA> <NA>\n")
    print(f"NIST RTTM segment annotation exported to: {rttm_path}")

    # 6. Quantitative Accuracy Metrics
    acc = float(accuracy_score(ground_truth, silero_preds))
    prec = float(precision_score(ground_truth, silero_preds, zero_division=0))
    rec = float(recall_score(ground_truth, silero_preds, zero_division=0))
    f1 = float(f1_score(ground_truth, silero_preds, zero_division=0))
    fpr_silero, tpr_silero, _ = roc_curve(ground_truth, silero_probs)
    auc_val = float(auc(fpr_silero, tpr_silero))
    
    # Torchaudio filtering validation
    trimmed_audio = torchaudio_engine.trim_speech(session_audio)
    torchaudio_trimmed_dur = len(trimmed_audio.squeeze(0)) / SAMPLE_RATE

    print(f"\n[Stage 2] Quantitative Evaluation Summary:")
    print(f"  Frame Accuracy : {acc * 100:.2f}%")
    print(f"  Precision      : {prec * 100:.2f}%")
    print(f"  Recall         : {rec * 100:.2f}%")
    print(f"  F1-Score       : {f1 * 100:.2f}%")
    print(f"  ROC AUC        : {auc_val:.4f}")
    print(f"  Torchaudio VAD : Trimmed audio to {torchaudio_trimmed_dur:.2f}s")

    # 7. Latency Benchmarking
    print("\n[Stage 3] Running Real-Time Frame Latency Benchmarking...")
    latency_info = run_latency_benchmark(silero_engine, iterations=250)
    print(f"  Mean Latency   : {latency_info['mean_latency_ms']:.3f} ms / 32.0 ms frame")
    print(f"  Median Latency : {latency_info['median_latency_ms']:.3f} ms")
    print(f"  P95 Latency    : {latency_info['p95_latency_ms']:.3f} ms")
    print(f"  Max Latency    : {latency_info['max_latency_ms']:.3f} ms")
    print(f"  Real-Time Fact.: {latency_info['real_time_factor_rtf']:.5f}x ({1.0/latency_info['real_time_factor_rtf']:.1f}x speedup)")
    print(f"  Throughput     : {latency_info['throughput_frames_per_sec']:.1f} frames/sec")
    print(f"  CPU Headroom   : {latency_info['cpu_headroom_pct']:.2f}%")

    # 8. Noise Robustness Evaluation
    print("\n[Stage 4] Evaluating Acoustic Noise Robustness across SNR conditions...")
    robustness_results = evaluate_noise_robustness(silero_engine, session_audio, ground_truth)
    for r in robustness_results:
        print(f"  SNR: {r['snr_level']:<15} | F1: {r['f1_score']:.4f} | Acc: {r['accuracy']:.4f} | Rec: {r['recall']:.4f}")

    # 9. Visual Diagnostics Generation
    print("\n[Stage 5] Generating 6-Panel Diagnostic Analytics Report...")
    report_path = os.path.join(OUTPUTS_DIR, "vad_benchmark_report.png")
    generate_diagnostic_report(
        session_audio=session_audio,
        ground_truth=ground_truth,
        silero_probs=silero_probs,
        energy_probs=energy_probs,
        speech_timestamps=speech_timestamps,
        latency_info=latency_info,
        robustness_results=robustness_results,
        output_path=report_path
    )

    # 10. Export Master Metrics JSON
    metrics = {
        "project": "Project 84: Voice Activity Detection (VAD)",
        "model": {
            "name": "Silero VAD (TorchScript JIT Recurrent Model)",
            "sample_rate_hz": SAMPLE_RATE,
            "window_size_samples": WINDOW_SIZE_SAMPLES,
            "frame_duration_ms": FRAME_DURATION_MS
        },
        "performance_metrics": {
            "accuracy_pct": round(acc * 100.0, 2),
            "precision_pct": round(prec * 100.0, 2),
            "recall_pct": round(rec * 100.0, 2),
            "f1_score_pct": round(f1 * 100.0, 2),
            "roc_auc": round(auc_val, 4),
            "num_detected_segments": len(speech_timestamps)
        },
        "latency_benchmark": {
            "mean_latency_ms": round(latency_info["mean_latency_ms"], 4),
            "median_latency_ms": round(latency_info["median_latency_ms"], 4),
            "p95_latency_ms": round(latency_info["p95_latency_ms"], 4),
            "max_latency_ms": round(latency_info["max_latency_ms"], 4),
            "step_budget_ms": FRAME_DURATION_MS,
            "real_time_factor_rtf": round(latency_info["real_time_factor_rtf"], 6),
            "throughput_frames_per_sec": round(latency_info["throughput_frames_per_sec"], 2),
            "cpu_headroom_pct": round(latency_info["cpu_headroom_pct"], 2)
        },
        "robustness_snr_f1": robustness_results
    }

    metrics_path = os.path.join(OUTPUTS_DIR, "vad_benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Master benchmark metrics saved to: {metrics_path}")

    print("\n" + "=" * 65)
    print(" VAD PIPELINE EXECUTION & BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
