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


# import os
# import json
# import math
# import torch
# import librosa
# import shutil
# from typing import List, Dict
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from pydub import AudioSegment

# # ---------------- CONFIG ----------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# MODEL_DIR = os.path.join(BASE_DIR, "models", "whisper-large-v3-turbo")
# FILES_DIR = os.path.join(BASE_DIR, "files")
# OUTPUT_DIR = os.path.join(FILES_DIR, "output_segments")

# SEGMENT_LENGTH = 15  # seconds
# SAMPLE_RATE = 16000
# # --------------------------------------

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---------------- DEVICE ----------------
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# # ---------------- LOAD MODEL (ONCE) ----------------
# print("Loading Whisper model...")

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     MODEL_DIR,
#     dtype=torch_dtype,
#     low_cpu_mem_usage=True,
#     use_safetensors=True
# ).to(device)

# processor = AutoProcessor.from_pretrained(MODEL_DIR)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     dtype=torch_dtype,
#     device=device,
# )


# # --------------------------------------------------
# # Convert ANY audio/video → MP3
# # --------------------------------------------------
# def convert_to_mp3(input_path: str) -> str:
#     audio = AudioSegment.from_file(input_path)
#     audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)

#     mp3_path = os.path.splitext(input_path)[0] + "_converted.mp3"
#     audio.export(mp3_path, format="mp3")

#     return mp3_path


# # --------------------------------------------------
# # Main transcription function (USED BY API)
# # --------------------------------------------------
# def transcribe_audio_segmented(
#     audio_path: str,
#     segment_length: int = SEGMENT_LENGTH
# ) -> List[Dict]:

#     # Convert to mp3 first
#     mp3_path = convert_to_mp3(audio_path)

#     audio = AudioSegment.from_mp3(mp3_path)
#     duration_sec = len(audio) / 1000
#     num_segments = math.ceil(duration_sec / segment_length)

#     results = []

#     for i in range(num_segments):
#         start_ms = i * segment_length * 1000
#         end_ms = min((i + 1) * segment_length * 1000, len(audio))

#         segment_audio = audio[start_ms:end_ms]

#         segment_path = os.path.join(
#             OUTPUT_DIR, f"segment_{i+1:02d}.wav"
#         )
#         segment_audio.export(segment_path, format="wav")

#         samples, _ = librosa.load(segment_path, sr=SAMPLE_RATE)

#         output = pipe(samples)

#         results.append({
#             "segment": i + 1,
#             "start_time": round(start_ms / 1000, 2),
#             "end_time": round(end_ms / 1000, 2),
#             "text": output["text"].strip()
#         })

#     # Save outputs
#     txt_path = os.path.join(OUTPUT_DIR, "transcription_with_timestamps.txt")
#     json_path = os.path.join(OUTPUT_DIR, "transcription_with_timestamps.json")

#     with open(txt_path, "w", encoding="utf-8") as f:
#         for r in results:
#             f.write(
#                 f"({r['start_time']:.2f}-{r['end_time']:.2f}): {r['text']}\n"
#             )

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)

#     return results


###################################### Assembly AI Transcription Module ######################################

# Install the assemblyai package by executing the command "pip install assemblyai"

# import assemblyai as aai

# aai.settings.api_key = "your_api_key_here"

# audio_file = "app/files/Tucker.mp3"
# #audio_file = "https://assembly.ai/wildfires.mp3"

# config = aai.TranscriptionConfig(speech_models=["universal"])

# transcript = aai.Transcriber(config=config).transcribe(audio_file)

# if transcript.status == "error":
#   raise RuntimeError(f"Transcription failed: {transcript.error}")

# print(transcript.text)



# import assemblyai as aai

# aai.settings.api_key = "api_key_here"

# config = aai.TranscriptionConfig(
#   speaker_labels=True,  # This identifies different speakers
#   punctuate=True,       # Adds periods and commas
#   format_text=True      # Handles paragraphs and casing
# )

# transcript = aai.Transcriber().transcribe("app/files/Tucker.mp3", config)

# # Each "utterance" is a segment where one person spoke
# for utterance in transcript.utterances:
#     print(f"Speaker {utterance.speaker}: {utterance.text}")
#     print(f"Start: {utterance.start}ms, End: {utterance.end}ms")


# import assemblyai as aai

# aai.settings.api_key = "My API key"

# config = aai.TranscriptionConfig(
#     speaker_labels=True,
#     punctuate=True,
#     format_text=True
# )

# transcript = aai.Transcriber().transcribe("app/files/Tucker.mp3", config)

# # Extract sentences instead of utterances
# sentences = transcript.get_sentences()

# for sentence in sentences:
#     # Now you have total control over every sentence
#     print(f"Speaker {sentence.speaker}: {sentence.text}")
#     print(f"Time: {sentence.start}ms to {sentence.end}ms")
#     print("-" * 20)




import assemblyai as aai
from pydantic import BaseModel, Field
from typing import List
import json
import os

# 1. Define your schema
class FinalOutput(BaseModel):
    segment_id: int
    speaker: str = Field(..., description="A, B, C, etc")
    start: int = Field(..., description="start time of segment in ms")
    end: int = Field(..., description="end time of segment in ms")
    text: str = Field(..., description="text in that segment")

# 2. Setup AssemblyAI
aai.settings.api_key = "My API key"

config = aai.TranscriptionConfig(
    speaker_labels=True,
    punctuate=True,
    format_text=True
)

file_path = "app/files/The Money Expert.mp3"
transcript = aai.Transcriber().transcribe(file_path, config)

# 3. Process sentences into Pydantic models
sentences = transcript.get_sentences()
structured_data: List[FinalOutput] = []

for index, sentence in enumerate(sentences):
    segment = FinalOutput(
        segment_id=index,
        speaker=sentence.speaker,
        start=sentence.start,
        end=sentence.end,
        text=sentence.text
    )
    structured_data.append(segment)

# 4. Save to JSON
output_dir = "app/files/test2"
os.makedirs(output_dir, exist_ok=True) # Creates the folder if it doesn't exist

file_name = os.path.splitext(os.path.basename(file_path))[0]
save_path = os.path.join(output_dir, f"{file_name}.json")

# Use [model.dict() for model in list] or Pydantic's RootModel for the whole list
with open(save_path, "w", encoding="utf-8") as f:
    # Convert list of Pydantic objects to a list of dicts for JSON serialization
    json_output = [obj.model_dump() for obj in structured_data]
    json.dump(json_output, f, indent=4)

print(f"Transcription saved successfully to: {save_path}")