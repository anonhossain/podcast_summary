import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict


# ============================================================
# Media Stitcher
# ============================================================

class MediaStitcher:
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    # --------------------------------------------------------
    # Detect if media is video
    # --------------------------------------------------------
    @staticmethod
    def is_video(file_path: str) -> bool:
        video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
        return Path(file_path).suffix.lower() in video_exts

    # --------------------------------------------------------
    # Load selected segments JSON
    # --------------------------------------------------------
    @staticmethod
    def load_segments(json_path: str) -> List[Dict]:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Segment JSON not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Segments JSON must be a list")

        # Ensure chronological order
        data.sort(key=lambda x: x["start"])
        return data

    # --------------------------------------------------------
    # Cut a single segment
    # --------------------------------------------------------
    def cut_segment(
        self,
        input_media: str,
        start_time: float,
        end_time: float,
        output_path: str,
        is_video: bool,
    ):
        duration = end_time - start_time
        if duration <= 0:
            raise ValueError("Invalid segment duration")

        cmd = [
            self.ffmpeg,
            "-y",
            "-ss", str(start_time),
            "-i", input_media,
            "-t", str(duration),
        ]

        # Fast stream copy
        if is_video:
            cmd += ["-c", "copy"]
        else:
            cmd += ["-vn", "-acodec", "copy"]

        cmd.append(output_path)

        subprocess.run(cmd, check=True)

    # --------------------------------------------------------
    # Stitch all segments
    # --------------------------------------------------------
    def stitch(
        self,
        input_media_path: str,
        segments_json_path: str,
        output_media_path: str,
    ):
        if not os.path.exists(input_media_path):
            raise FileNotFoundError(f"Media file not found: {input_media_path}")

        segments = self.load_segments(segments_json_path)
        if not segments:
            raise ValueError("No segments provided")

        is_video = self.is_video(input_media_path)
        ext = Path(output_media_path).suffix

        with tempfile.TemporaryDirectory() as tmpdir:
            clip_paths = []

            # -----------------------------
            # 1️⃣ Cut each segment
            # -----------------------------
            for idx, seg in enumerate(segments):
                clip_path = os.path.join(tmpdir, f"clip_{idx}{ext}")
                clip_paths.append(clip_path)

                self.cut_segment(
                    input_media=input_media_path,
                    start_time=seg["start"],
                    end_time=seg["end"],
                    output_path=clip_path,
                    is_video=is_video,
                )

            # -----------------------------
            # 2️⃣ Create concat file
            # -----------------------------
            concat_file = os.path.join(tmpdir, "concat.txt")
            with open(concat_file, "w", encoding="utf-8") as f:
                for clip in clip_paths:
                    f.write(f"file '{clip}'\n")

            # -----------------------------
            # 3️⃣ Concatenate
            # -----------------------------
            concat_cmd = [
                self.ffmpeg,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_media_path,
            ]

            subprocess.run(concat_cmd, check=True)

        return output_media_path

if __name__ == "__main__":
    stitcher = MediaStitcher()

    input_media = "app\\files\\videoplayback.mp4"
    segments_json = "app\\files\\test_output\\stitch.json"
    output_media = "app\\files\\output_stitched.mp4"

    print("\n🎬 Stitching media segments...\n")
    try:
        result_path = stitcher.stitch(
            input_media_path=input_media,
            segments_json_path=segments_json,
            output_media_path=output_media,
        )
        print(f"\n✅ Stitched media saved to: {result_path}\n")
    except Exception as e:
        print(f"\n❌ Error during stitching: {e}\n")