# import os
# import json
# import subprocess
# import tempfile
# from pathlib import Path
# from typing import List, Dict


# # ============================================================
# # Media Stitcher
# # ============================================================

# class MediaStitcher:
#     def __init__(self, ffmpeg_path: str = "ffmpeg"):
#         self.ffmpeg = ffmpeg_path

#     # --------------------------------------------------------
#     # Detect if media is video
#     # --------------------------------------------------------
#     @staticmethod
#     def is_video(file_path: str) -> bool:
#         video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
#         return Path(file_path).suffix.lower() in video_exts

#     # --------------------------------------------------------
#     # Load selected segments JSON
#     # --------------------------------------------------------
#     @staticmethod
#     def load_segments(json_path: str) -> List[Dict]:
#         if not os.path.exists(json_path):
#             raise FileNotFoundError(f"Segment JSON not found: {json_path}")

#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         if not isinstance(data, list):
#             raise ValueError("Segments JSON must be a list")

#         # Ensure chronological order
#         data.sort(key=lambda x: x["start"])
#         return data

#     # --------------------------------------------------------
#     # Cut a single segment
#     # --------------------------------------------------------
#     def cut_segment(
#         self,
#         input_media: str,
#         start_time: float,
#         end_time: float,
#         output_path: str,
#         is_video: bool,
#     ):
#         duration = end_time - start_time
#         if duration <= 0:
#             raise ValueError("Invalid segment duration")

#         cmd = [
#             self.ffmpeg,
#             "-y",
#             "-ss", str(start_time),
#             "-i", input_media,
#             "-t", str(duration),
#         ]

#         # Fast stream copy
#         if is_video:
#             cmd += ["-c", "copy"]
#         else:
#             cmd += ["-vn", "-acodec", "copy"]

#         cmd.append(output_path)

#         subprocess.run(cmd, check=True)

#     # --------------------------------------------------------
#     # Stitch all segments
#     # --------------------------------------------------------
#     def stitch(
#         self,
#         input_media_path: str,
#         segments_json_path: str,
#         output_media_path: str,
#     ):
#         if not os.path.exists(input_media_path):
#             raise FileNotFoundError(f"Media file not found: {input_media_path}")

#         segments = self.load_segments(segments_json_path)
#         if not segments:
#             raise ValueError("No segments provided")

#         is_video = self.is_video(input_media_path)
#         ext = Path(output_media_path).suffix

#         with tempfile.TemporaryDirectory() as tmpdir:
#             clip_paths = []

#             # -----------------------------
#             # 1️⃣ Cut each segment
#             # -----------------------------
#             for idx, seg in enumerate(segments):
#                 clip_path = os.path.join(tmpdir, f"clip_{idx}{ext}")
#                 clip_paths.append(clip_path)

#                 self.cut_segment(
#                     input_media=input_media_path,
#                     start_time=seg["start"],
#                     end_time=seg["end"],
#                     output_path=clip_path,
#                     is_video=is_video,
#                 )

#             # -----------------------------
#             # 2️⃣ Create concat file
#             # -----------------------------
#             concat_file = os.path.join(tmpdir, "concat.txt")
#             with open(concat_file, "w", encoding="utf-8") as f:
#                 for clip in clip_paths:
#                     f.write(f"file '{clip}'\n")

#             # -----------------------------
#             # 3️⃣ Concatenate
#             # -----------------------------
#             concat_cmd = [
#                 self.ffmpeg,
#                 "-y",
#                 "-f", "concat",
#                 "-safe", "0",
#                 "-i", concat_file,
#                 "-c", "copy",
#                 output_media_path,
#             ]

#             subprocess.run(concat_cmd, check=True)

#         return output_media_path

# if __name__ == "__main__":
#     stitcher = MediaStitcher()

#     input_media = "app\\files\\videoplayback.mp4"
#     segments_json = "app\\files\\test_output\\stitch.json"
#     output_media = "app\\files\\output_stitched.mp4"

#     print("\n🎬 Stitching media segments...\n")
#     try:
#         result_path = stitcher.stitch(
#             input_media_path=input_media,
#             segments_json_path=segments_json,
#             output_media_path=output_media,
#         )
#         print(f"\n✅ Stitched media saved to: {result_path}\n")
#     except Exception as e:
#         print(f"\n❌ Error during stitching: {e}\n")


import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict


class MediaStitcher:
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    @staticmethod
    def is_video(file_path: str) -> bool:
        video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"}
        audio_exts = {".mp3", ".wav", ".aac", ".m4a", ".ogg", ".flac", ".opus"}
        suffix = Path(file_path).suffix.lower()
        if suffix in video_exts:
            return True
        elif suffix in audio_exts:
            return False
        return True

    @staticmethod
    def load_segments(json_path: str) -> List[Dict]:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Segment JSON not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Segments JSON must be a list of objects")

        kept_segments = [seg for seg in data if seg.get("status") == 1]
        if not kept_segments:
            raise ValueError("No segments with status=1 found to keep")

        kept_segments.sort(key=lambda x: x["start_time"])

        for seg in kept_segments:
            start = seg["start_time"]
            end = seg["end_time"]
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValueError(f"Invalid time values: {seg}")
            if end <= start:
                raise ValueError(f"Invalid duration (end <= start): {seg}")

        return kept_segments

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
            "-ss", str(start_time),           # Fast seek before input
            "-i", input_media,
            "-t", str(duration),
            "-avoid_negative_ts", "make_zero",
            # Removed -fflags +genpts – it was causing extra padding in some cases
        ]

        if is_video:
            cmd += ["-c", "copy"]
        else:
            cmd += ["-vn", "-c:a", "copy"]

        cmd.append(output_path)

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stitch(
        self,
        input_media_path: str,
        segments_json_path: str,
        output_media_path: str = None,
    ) -> str:
        if not os.path.exists(input_media_path):
            raise FileNotFoundError(f"Media file not found: {input_media_path}")

        segments = self.load_segments(segments_json_path)
        if not segments:
            raise ValueError("No segments to process")

        is_video_flag = self.is_video(input_media_path)
        input_path = Path(input_media_path)
        input_ext = input_path.suffix.lower()

        if output_media_path is None:
            output_media_path = input_path.parent / f"{input_path.stem}_stitched{input_ext}"
        else:
            output_media_path = str(Path(output_media_path).with_suffix(input_ext))

        output_path = Path(output_media_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            clip_paths = []

            # Cut all segments
            for idx, seg in enumerate(segments):
                clip_path = tmp_dir / f"clip_{idx:04d}{input_ext}"
                clip_paths.append(str(clip_path))

                self.cut_segment(
                    input_media=input_media_path,
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    output_path=str(clip_path),
                    is_video=is_video_flag,
                )

            if len(clip_paths) == 1:
                subprocess.run([
                    self.ffmpeg, "-y", "-i", clip_paths[0], "-c", "copy", str(output_path)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return str(output_path)

            # ALWAYS use concat demuxer – reliable for many clips + no command line limit
            concat_file = tmp_dir / "concat_list.txt"
            with open(concat_file, "w", encoding="utf-8") as f:
                for clip in clip_paths:
                    escaped = str(Path(clip)).replace("\\", "/")
                    f.write(f"file '{escaped}'\n")

            cmd = [
                self.ffmpeg,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path)
            ]

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return str(output_path)


# ============================================================
# Usage
# ============================================================
if __name__ == "__main__":
    stitcher = MediaStitcher()

    input_media = "app\\files\\videoplayback.mp4"
    segments_json = "app\\files\\output_segments\\labeled_transcript_full.json"

    print("\n🎬 Stitching media segments (only status=1 kept)...\n")
    try:
        result = stitcher.stitch(
            input_media_path=input_media,
            segments_json_path=segments_json,
        )
        print(f"\n✅ Successfully stitched!\n   Saved to: {result}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")