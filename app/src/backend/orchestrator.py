import json
from pathlib import Path
from typing import List
from tempfile import TemporaryDirectory

from backend.transcription import transcribe_audio_segmented
from backend.summarizer import TranscriptClassifier, TranscriptSegment
from backend.media_stitcher import MediaStitcher


class PodcastOrchestrator:
    def __init__(
        self,
        llm_model: str = "llama3.1:8b",
        batch_size: int = 20,
    ):
        self.classifier = TranscriptClassifier(
            model_name=llm_model,
            temperature=0.0,
            batch_size=batch_size,
        )
        self.stitcher = MediaStitcher()

    def process(self, input_file_path: str, work_dir: Path) -> Path:
        """
        Entire pipeline runs inside work_dir.
        work_dir is TEMPORARY and auto-deleted.
        """

        input_file_path = Path(input_file_path)

        # -----------------------------
        # 1️⃣ TRANSCRIPTION
        # -----------------------------
        transcripts = transcribe_audio_segmented(str(input_file_path))

        segments: List[TranscriptSegment] = [
            TranscriptSegment(**s) for s in transcripts
        ]

        # -----------------------------
        # 2️⃣ CLASSIFICATION
        # -----------------------------
        labeled = self.classifier.classify_segments(segments)

        labeled_json = work_dir / "labeled.json"
        with open(labeled_json, "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in labeled], f, indent=2)

        # -----------------------------
        # 3️⃣ STITCH
        # -----------------------------
        output_path = work_dir / f"final{input_file_path.suffix}"

        final_file = self.stitcher.stitch(
            input_media_path=str(input_file_path),
            segments_json_path=str(labeled_json),
            output_media_path=str(output_path),
        )

        return Path(final_file)
