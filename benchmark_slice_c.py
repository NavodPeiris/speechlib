"""
Benchmark: sequential model.transcribe() vs BatchedInferencePipeline(batch_size=16)

Uso:
    python benchmark_slice_c.py

Requiere: examples/obama_zach_16k.wav (16kHz, mono) como audio de referencia.
Mide solo el tiempo de transcripción, sin diarización ni preprocessing.
"""
import time
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline

AUDIO = "examples/obama_zach_16k.wav"
MODEL_SIZE = "base"
LANGUAGE = "en"

if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"
else:
    device = "cpu"
    compute_type = "float32"

print(f"Device: {device} ({compute_type})")
print(f"Audio:  {AUDIO}")
print(f"Model:  {MODEL_SIZE}")
print()

# Cargar modelo una sola vez para ambas corridas
print("Loading model...")
model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
print("Model loaded.\n")

# ── Corrida 1: secuencial (comportamiento anterior) ──────────────────────────
print("=== Run 1: sequential model.transcribe() ===")
t0 = time.perf_counter()
segments, info = model.transcribe(AUDIO, language=LANGUAGE, beam_size=5)
text_seq = "".join(s.text for s in segments)
t_seq = time.perf_counter() - t0
print(f"Time:  {t_seq:.2f}s")
print(f"Words: {len(text_seq.split())}")
print()

# ── Corrida 2: batched ────────────────────────────────────────────────────────
print("=== Run 2: BatchedInferencePipeline(batch_size=16) ===")
batched = BatchedInferencePipeline(model=model)
t0 = time.perf_counter()
segments, info = batched.transcribe(AUDIO, language=LANGUAGE, beam_size=5, batch_size=16)
text_bat = "".join(s.text for s in segments)
t_bat = time.perf_counter() - t0
print(f"Time:  {t_bat:.2f}s")
print(f"Words: {len(text_bat.split())}")
print()

# ── Resultado ─────────────────────────────────────────────────────────────────
speedup = t_seq / t_bat if t_bat > 0 else float("inf")
print("=" * 40)
print(f"Sequential:  {t_seq:.2f}s")
print(f"Batched:     {t_bat:.2f}s")
print(f"Speedup:     {speedup:.2f}x")
print("=" * 40)

# Verificar que el texto es comparable
words_seq = set(text_seq.lower().split())
words_bat = set(text_bat.lower().split())
overlap = len(words_seq & words_bat) / max(len(words_seq), 1)
print(f"Word overlap: {overlap:.0%}  (≥80% = output compatible)")
