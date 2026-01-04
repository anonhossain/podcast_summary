from pydub import AudioSegment
import os
import json
import re
import shutil

# -------------------------------------------------------------
# TIMESTAMP PARSER
# -------------------------------------------------------------
def parse_timestamp(timestamp: str):
    """Convert [MM:SS-MM:SS] → (start_ms, end_ms)"""
    pattern = r"\[(\d{2}):(\d{2})-(\d{2}):(\d{2})\]"
    match = re.match(pattern, timestamp.strip())
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    start_min, start_sec, end_min, end_sec = map(int, match.groups())
    start_ms = (start_min * 60 + start_sec) * 1000
    end_ms = (end_min * 60 + end_sec) * 1000
    return start_ms, end_ms

# -------------------------------------------------------------
# WAV and Video CONVERTER
# -------------------------------------------------------------
class AudioConverter:
    def to_wav(self, src_path: str, dst_path: str):
        try:
            audio = AudioSegment.from_file(src_path)
            audio.export(dst_path, format="wav")
            print(f"Converted to WAV: {dst_path}")
        except Exception as e:
            raise RuntimeError(f"WAV Conversion Error: {e}")
        
    def mp4_to_mp3(self, src_path: str, dst_path: str):
        try:
            audio = AudioSegment.from_file(src_path, format="mp4")
            audio.export(dst_path, format="mp3", bitrate="192k")
            print(f"Converted MP4 to MP3: {dst_path}")
        except Exception as e:
            raise RuntimeError(f"MP3 Conversion Error: {e}")

# -------------------------------------------------------------
# MAIN CROPPER + JOINER
# -------------------------------------------------------------

class JsonAudioCropper:
    def __init__(self, audio_path: str, json_path: str, output_dir: str):
        self.audio_path = audio_path
        self.json_path = json_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.converter = AudioConverter()
        self.wav_path = None
        self.audio = None

        print(f"\nCropper initialized")
        print(f"Output folder → {self.output_dir}\n")

    # ---------------------------------------------------------
    def load_json(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON not found: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of segments")
        return data

    # ---------------------------------------------------------
    def prepare_wav(self):
        base = os.path.splitext(os.path.basename(self.audio_path))[0]
        self.wav_path = os.path.join(self.output_dir, f"{base}_temp_converted.wav")
        self.converter.to_wav(self.audio_path, self.wav_path)
        self.audio = AudioSegment.from_wav(self.wav_path)
        print("WAV loaded and ready for slicing.\n")

    # ---------------------------------------------------------
    def crop_segments(self):
        json_data = self.load_json()
        self.prepare_wav()
        print("Starting audio slicing...\n")

        for i, entry in enumerate(json_data, start=1):
            try:
                timestamp = entry.get("timestamp")
                if not timestamp:
                    print(f"Skipping segment {i}: no timestamp")
                    continue

                start_ms, end_ms = parse_timestamp(timestamp)
                end_ms = min(end_ms, len(self.audio))  # prevent overflow

                segment = self.audio[start_ms:end_ms]
                filename = f"segment_{i:02d}.wav"
                filepath = os.path.join(self.output_dir, filename)
                segment.export(filepath, format="wav")
                print(f"Saved {filename} | {timestamp}")

            except Exception as e:
                print(f"Error in segment {i}: {e}")

        print("\nAll segments cropped successfully!\n")

    # ---------------------------------------------------------
    def join_segments(self, final_output_path: str = None, add_silence_ms: int = 0):
        """
        Join all cropped segments into one final file.
        add_silence_ms: optional pause between segments (e.g. 1000 = 1 second)
        """
        if not self.audio:
            raise RuntimeError("Run crop_segments() first!")

        if final_output_path is None:
            base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
            final_output_path = os.path.join(self.output_dir, f"{base_name}_FINAL_EDITED.wav")

        print(f"Joining segments into final file...")

        combined = AudioSegment.empty()

        # Get and sort segment files correctly
        segment_files = sorted(
            [f for f in os.listdir(self.output_dir) if f.startswith("segment_") and f.endswith(".wav")],
            key=lambda x: int(re.findall(r"segment_(\d+)", x)[0])
        )

        for seg_file in segment_files:
            seg_path = os.path.join(self.output_dir, seg_file)
            segment = AudioSegment.from_wav(seg_path)
            combined += segment
            if add_silence_ms > 0 and seg_file != segment_files[-1]:
                combined += AudioSegment.silent(duration=add_silence_ms)
            print(f"   + {seg_file}")

        # Export final file (WAV + optional MP3)
        combined.export(final_output_path, format="wav")
        mp3_path = final_output_path.replace(".wav", ".mp3")
        combined.export(mp3_path, format="mp3", bitrate="192k")

        duration_min = len(combined) / 1000 / 60
        print(f"\nFINAL EDITED AUDIO READY!")
        print(f"   WAV → {final_output_path}")
        print(f"   MP3 → {mp3_path}")
        print(f"   Duration → {duration_min:.2f} minutes ({len(segment_files)} segments)")

        return final_output_path, mp3_path

    # ---------------------------------------------------------
    def cleanup_temp_files(self):
        """Optional: remove temporary WAV segments and converted file"""
        print("Cleaning up temporary files...")
        for f in os.listdir(self.output_dir):
            if f.startswith(("segment_", "_temp_converted")) and f.endswith(".wav"):
                os.remove(os.path.join(self.output_dir, f))
        print("Cleanup done.\n")

# -------------------------------------------------------------
# RUN SCRIPT
# -------------------------------------------------------------
if __name__ == "__main__":
    # UPDATE THESE PATHS TO YOUR FILES
    audio_file = r"C:\files\podcast\files\Tucker.mp3"
    json_file = r"C:\files\podcast\files\output.json"
    output_folder = r"C:\files\podcast\files\output_segments"

    # Create cropper instance
    cropper = JsonAudioCropper(audio_file, json_file, output_folder)

    # Step 1: Crop segments from JSON
    cropper.crop_segments()

    # Step 2: Join only the cropped parts → This is your final edited podcast!
    final_wav, final_mp3 = cropper.join_segments(
        add_silence_ms=800  # Optional: 0.8 sec pause between clips (remove if not wanted)
    )

    # Step 3: Optional cleanup (remove individual segments)
    # cropper.cleanup_temp_files()