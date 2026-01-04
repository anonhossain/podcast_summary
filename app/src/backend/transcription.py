# import os
# import json
# import math
# import torch
# import librosa
# import soundfile as sf
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from pydub import AudioSegment

# # ---------------- CONFIG ----------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# MODEL_DIR = os.path.join(BASE_DIR, "models", "whisper-large-v3-turbo")
# INPUT_AUDIO = os.path.join(BASE_DIR, "files", "videoplayback2.mp3")
# OUTPUT_DIR = os.path.join(BASE_DIR, "files", "output_segments")

# SEGMENT_LENGTH = 15  # seconds (you can set 10–15)
# SAMPLE_RATE = 16000
# # --------------------------------------

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---------------- DEVICE ----------------
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# # ---------------- LOAD MODEL ----------------
# print("Loading Whisper model...")

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     MODEL_DIR,
#     torch_dtype=torch_dtype,
#     low_cpu_mem_usage=True,
#     use_safetensors=True
# ).to(device)

# processor = AutoProcessor.from_pretrained(MODEL_DIR)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# # ---------------- LOAD AUDIO ----------------
# print("Loading audio...")
# audio = AudioSegment.from_file(INPUT_AUDIO)
# audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)

# duration_sec = len(audio) / 1000
# num_segments = math.ceil(duration_sec / SEGMENT_LENGTH)

# print(f"Total duration: {duration_sec:.2f}s")
# print(f"Total segments: {num_segments}")

# results = []

# # ---------------- SEGMENT + TRANSCRIBE ----------------
# for i in range(num_segments):
#     start_ms = i * SEGMENT_LENGTH * 1000
#     end_ms = min((i + 1) * SEGMENT_LENGTH * 1000, len(audio))

#     segment_audio = audio[start_ms:end_ms]

#     segment_path = os.path.join(
#         OUTPUT_DIR, f"segment_{i+1:02d}.wav"
#     )
#     segment_audio.export(segment_path, format="wav")

#     # Load as numpy for whisper
#     samples, sr = librosa.load(segment_path, sr=SAMPLE_RATE)

#     print(f"Transcribing segment {i+1}/{num_segments} "
#           f"({start_ms/1000:.2f}-{end_ms/1000:.2f})")

#     output = pipe(samples)

#     results.append({
#         "segment": i + 1,
#         "start_time": round(start_ms / 1000, 2),
#         "end_time": round(end_ms / 1000, 2),
#         "text": output["text"].strip()
#     })

# # ---------------- SAVE OUTPUTS ----------------
# txt_path = os.path.join(OUTPUT_DIR, "transcription_with_timestamps.txt")
# json_path = os.path.join(OUTPUT_DIR, "transcription_with_timestamps.json")

# with open(txt_path, "w", encoding="utf-8") as f:
#     for r in results:
#         f.write(
#             f"({r['start_time']:.2f}-{r['end_time']:.2f}): {r['text']}\n"
#         )

# with open(json_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)

# print("✅ Transcription completed")
# print(f"TXT saved to: {txt_path}")
# print(f"JSON saved to: {json_path}")


import os
import json
import math
import torch
import librosa
import shutil
from typing import List, Dict
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models", "whisper-large-v3-turbo")
FILES_DIR = os.path.join(BASE_DIR, "files")
OUTPUT_DIR = os.path.join(FILES_DIR, "output_segments")

SEGMENT_LENGTH = 15  # seconds
SAMPLE_RATE = 16000
# --------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- DEVICE ----------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------- LOAD MODEL (ONCE) ----------------
print("Loading Whisper model...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_DIR,
    dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(MODEL_DIR)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=torch_dtype,
    device=device,
)


# --------------------------------------------------
# Convert ANY audio/video → MP3
# --------------------------------------------------
def convert_to_mp3(input_path: str) -> str:
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)

    mp3_path = os.path.splitext(input_path)[0] + "_converted.mp3"
    audio.export(mp3_path, format="mp3")

    return mp3_path


# --------------------------------------------------
# Main transcription function (USED BY API)
# --------------------------------------------------
def transcribe_audio_segmented(
    audio_path: str,
    segment_length: int = SEGMENT_LENGTH
) -> List[Dict]:

    # Convert to mp3 first
    mp3_path = convert_to_mp3(audio_path)

    audio = AudioSegment.from_mp3(mp3_path)
    duration_sec = len(audio) / 1000
    num_segments = math.ceil(duration_sec / segment_length)

    results = []

    for i in range(num_segments):
        start_ms = i * segment_length * 1000
        end_ms = min((i + 1) * segment_length * 1000, len(audio))

        segment_audio = audio[start_ms:end_ms]

        segment_path = os.path.join(
            OUTPUT_DIR, f"segment_{i+1:02d}.wav"
        )
        segment_audio.export(segment_path, format="wav")

        samples, _ = librosa.load(segment_path, sr=SAMPLE_RATE)

        output = pipe(samples)

        results.append({
            "segment": i + 1,
            "start_time": round(start_ms / 1000, 2),
            "end_time": round(end_ms / 1000, 2),
            "text": output["text"].strip()
        })

    # Save outputs
    txt_path = os.path.join(OUTPUT_DIR, "transcription_with_timestamps.txt")
    json_path = os.path.join(OUTPUT_DIR, "transcription_with_timestamps.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                f"({r['start_time']:.2f}-{r['end_time']:.2f}): {r['text']}\n"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
