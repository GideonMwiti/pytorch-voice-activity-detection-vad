"""
Interactive Voice Activity Detection (VAD) CLI Tool
===================================================
Production command-line interface for offline speech segmentation and
real-time streaming VAD emulation using Silero Neural VAD, Torchaudio,
and Energy-Entropy detection.

Author: Gideon Mwiti
License: MIT
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import soundfile as sf
import torch

from main import (
    SileroVADEngine,
    EnergyEntropyVADEngine,
    TorchaudioVADEngine,
    SAMPLE_RATE,
    WINDOW_SIZE_SAMPLES,
    FRAME_DURATION_MS,
    SILERO_MODEL_PATH,
    OUTPUTS_DIR
)

def render_ascii_bar(prob: float, width: int = 25) -> str:
    """Render an ASCII confidence bar for terminal visualization."""
    filled = int(round(prob * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}|"

def run_offline_segmentation(args):
    """Segment an audio file and display speech timestamps table."""
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at '{args.audio}'")
        sys.exit(1)

    print("\n" + "=" * 68)
    print("       VOICE ACTIVITY DETECTION (VAD) - OFFLINE SEGMENTATION")
    print("=" * 68)
    print(f"Input Audio File : {args.audio}")
    print(f"Detection Engine : {args.engine.upper()}")
    print(f"Speech Threshold : {args.threshold:.2f}")

    wav, sr = sf.read(args.audio)
    wav_tensor = torch.tensor(wav, dtype=torch.float32)
    if wav_tensor.ndim > 1:
        wav_tensor = wav_tensor.mean(dim=-1)
    if sr != SAMPLE_RATE:
        import torchaudio.transforms as T
        wav_tensor = T.Resample(sr, SAMPLE_RATE)(wav_tensor.unsqueeze(0)).squeeze(0)

    total_duration = len(wav_tensor) / SAMPLE_RATE
    print(f"Audio Duration   : {total_duration:.2f} seconds ({len(wav_tensor):,} samples)")
    print("-" * 68)

    t0 = time.perf_counter()
    if args.engine == "silero":
        engine = SileroVADEngine(SILERO_MODEL_PATH, default_threshold=args.threshold)
        segments = engine.get_speech_timestamps(
            wav_tensor,
            sr=SAMPLE_RATE,
            threshold=args.threshold,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms
        )
    elif args.engine == "torchaudio":
        torchaudio_engine = TorchaudioVADEngine(sr=SAMPLE_RATE)
        trimmed = torchaudio_engine.trim_speech(wav_tensor)
        trimmed_dur = len(trimmed.squeeze(0)) / SAMPLE_RATE
        segments = [{
            "start": 0.0,
            "end": round(trimmed_dur, 3),
            "duration": round(trimmed_dur, 3),
            "confidence": 1.0
        }]
    else: # energy
        energy_engine = EnergyEntropyVADEngine(sr=SAMPLE_RATE)
        probs = energy_engine.predict_audio(wav_tensor)
        # Simple thresholding
        window_sec = WINDOW_SIZE_SAMPLES / SAMPLE_RATE
        speech_chunks = np.where(probs >= args.threshold)[0]
        segments = []
        if len(speech_chunks) > 0:
            diffs = np.diff(speech_chunks)
            splits = np.where(diffs > 1)[0] + 1
            grouped = np.split(speech_chunks, splits)
            for g in grouped:
                start_t = round(float(g[0] * window_sec), 3)
                end_t = round(float((g[-1] + 1) * window_sec), 3)
                segments.append({
                    "start": start_t,
                    "end": end_t,
                    "duration": round(end_t - start_t, 3),
                    "confidence": round(float(np.mean(probs[g])), 3)
                })

    proc_time_ms = (time.perf_counter() - t0) * 1000.0

    # Display segments
    print("\nDETECTED SPEECH INTERVALS:")
    print(f"{'Idx':<5} | {'Start (s)':<10} | {'End (s)':<10} | {'Duration (s)':<12} | {'Confidence':<10} | {'Visual Meter':<25}")
    print("-" * 85)

    total_speech_sec = 0.0
    for idx, seg in enumerate(segments):
        total_speech_sec += seg["duration"]
        meter = render_ascii_bar(seg["confidence"], width=20)
        print(f"#{idx+1:02d}  | {seg['start']:<10.3f} | {seg['end']:<10.3f} | {seg['duration']:<12.3f} | {seg['confidence']:<10.3f} | {meter}")

    speech_pct = (total_speech_sec / total_duration) * 100.0 if total_duration > 0 else 0.0
    silence_sec = max(0.0, total_duration - total_speech_sec)

    print("-" * 85)
    print("SESSION SUMMARY STATISTICS:")
    print(f"  • Total Segments Found : {len(segments)}")
    print(f"  • Total Speech Time    : {total_speech_sec:.2f} s ({speech_pct:.1f}% speech activity)")
    print(f"  • Total Silence Time   : {silence_sec:.2f} s ({100.0 - speech_pct:.1f}% silence)")
    print(f"  • Processing Time      : {proc_time_ms:.2f} ms (RTF: {proc_time_ms / (total_duration * 1000):.5f}x)")
    print("=" * 85)

    # Save outputs if requested
    if args.output_speech and len(segments) > 0:
        extracted = []
        for seg in segments:
            s_idx = int(seg["start"] * SAMPLE_RATE)
            e_idx = int(seg["end"] * SAMPLE_RATE)
            extracted.append(wav_tensor[s_idx:e_idx])
        speech_wav = torch.cat(extracted).numpy()
        sf.write(args.output_speech, speech_wav, SAMPLE_RATE)
        print(f"Speech-only audio exported to: {args.output_speech}")

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(segments, f, indent=2)
        print(f"JSON timestamps exported to: {args.export_json}")

    if args.export_rttm:
        base_name = os.path.splitext(os.path.basename(args.audio))[0]
        with open(args.export_rttm, "w") as f:
            for seg in segments:
                f.write(f"SPEAKER {base_name} 1 {seg['start']:.3f} {seg['duration']:.3f} <NA> <NA> SPEECH <NA> <NA>\n")
        print(f"RTTM annotations exported to: {args.export_rttm}")

def run_streaming_emulation(args):
    """Simulate real-time streaming audio frame-by-frame with live terminal output."""
    if not os.path.exists(args.stream):
        print(f"Error: Stream audio file not found at '{args.stream}'")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("       VOICE ACTIVITY DETECTION (VAD) - REAL-TIME STREAMING EMULATION")
    print("=" * 75)
    print(f"Stream Audio Source : {args.stream}")
    print(f"Frame Size          : {WINDOW_SIZE_SAMPLES} samples ({FRAME_DURATION_MS:.1f} ms)")
    print(f"Threshold           : {args.threshold:.2f}")
    print("Press Ctrl+C to stop simulation at any time.\n")
    print(f"{'Time':<8} | {'Chunk Latency':<14} | {'Prob':<6} | {'Confidence Meter':<25} | {'VAD Status'}")
    print("-" * 75)

    wav, sr = sf.read(args.stream)
    wav_tensor = torch.tensor(wav, dtype=torch.float32)
    if wav_tensor.ndim > 1:
        wav_tensor = wav_tensor.mean(dim=-1)
    if sr != SAMPLE_RATE:
        import torchaudio.transforms as T
        wav_tensor = T.Resample(sr, SAMPLE_RATE)(wav_tensor.unsqueeze(0)).squeeze(0)

    engine = SileroVADEngine(SILERO_MODEL_PATH, default_threshold=args.threshold)
    engine.reset_states()

    num_chunks = len(wav_tensor) // WINDOW_SIZE_SAMPLES
    is_speaking = False
    speech_start_time = 0.0

    try:
        for i in range(num_chunks):
            start = i * WINDOW_SIZE_SAMPLES
            end = start + WINDOW_SIZE_SAMPLES
            chunk = wav_tensor[start:end].unsqueeze(0)
            cur_time_s = (i * WINDOW_SIZE_SAMPLES) / SAMPLE_RATE

            t0 = time.perf_counter()
            prob = engine.predict_chunk(chunk, SAMPLE_RATE)
            chunk_lat_ms = (time.perf_counter() - t0) * 1000.0

            meter = render_ascii_bar(prob, width=20)
            state_label = "[SPEECH]" if prob >= args.threshold else "[SILENCE]"

            # Track transitions
            if prob >= args.threshold and not is_speaking:
                is_speaking = True
                speech_start_time = cur_time_s
                event_note = f"--> SPEECH START ({cur_time_s:.2f}s)"
            elif prob < (args.threshold - 0.15) and is_speaking:
                is_speaking = False
                dur = cur_time_s - speech_start_time
                event_note = f"<-- SPEECH END (Duration: {dur:.2f}s)"
            else:
                event_note = ""

            # Only print every 2 frames or upon transitions to keep terminal clean
            if i % 3 == 0 or event_note != "":
                print(f"{cur_time_s:6.2f}s  | {chunk_lat_ms:6.3f} ms / 32ms | {prob:5.3f} | {meter} | {state_label:<9} {event_note}")

            # Simulate 32ms real-time pacing (optional, default small sleep)
            if args.pace_real_time:
                time.sleep(FRAME_DURATION_MS / 1000.0)

    except (KeyboardInterrupt, BrokenPipeError):
        pass

    print("-" * 75)
    print("Real-time streaming simulation completed.")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="Voice Activity Detection (VAD) CLI Tool")
    parser.add_argument("--audio", type=str, help="Path to input audio file for offline segmentation")
    parser.add_argument("--stream", type=str, help="Path to audio file to simulate real-time frame streaming")
    parser.add_argument("--engine", type=str, choices=["silero", "torchaudio", "energy"], default="silero", help="Detection engine")
    parser.add_argument("--threshold", type=float, default=0.50, help="Speech detection probability threshold (default 0.50)")
    parser.add_argument("--min-speech-ms", type=float, default=250.0, help="Minimum speech duration in ms (default 250)")
    parser.add_argument("--min-silence-ms", type=float, default=100.0, help="Minimum silence duration in ms (default 100)")
    parser.add_argument("--output-speech", type=str, help="Output path to export concatenated speech audio")
    parser.add_argument("--export-rttm", type=str, help="Export segment annotations to NIST RTTM format")
    parser.add_argument("--export-json", type=str, help="Export segment annotations to JSON format")
    parser.add_argument("--pace-real-time", action="store_true", help="Pace streaming simulation to exact wall-clock time")

    args = parser.parse_args()

    if args.audio:
        run_offline_segmentation(args)
    elif args.stream:
        run_streaming_emulation(args)
    else:
        # Default fallback demo on generated conversation sample if present
        sample_path = os.path.join(OUTPUTS_DIR, "sample_conversation.wav")
        if os.path.exists(sample_path):
            args.audio = sample_path
            run_offline_segmentation(args)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
