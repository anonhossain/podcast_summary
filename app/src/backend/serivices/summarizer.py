# from openai import OpenAI
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv
# import os
# from typing import List

# # Load environment variables
# load_dotenv()


# # --------------------------------------------------------------------
# # Pydantic Model for Summarizer Output
# # --------------------------------------------------------------------
# class SummarizedSegment(BaseModel):
#     timestamp: str = Field(..., description="Start and end time like [00:00-00:05]")
#     text: str = Field(..., description="Write exact text from that timestamp here")


# # --------------------------------------------------------------------
# # Read transcript from .txt file
# # --------------------------------------------------------------------
# def read_text_file(file_path: str) -> str:
#     """Read and return the content of a text file."""
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"❌ File not found: {file_path}")

#     with open(file_path, "r", encoding="utf-8") as file:
#         return file.read()


# # --------------------------------------------------------------------
# # Main Summarizer Function
# # --------------------------------------------------------------------
# def summarize_text(input_text: str) -> str:
#     """Summarize transcript text using GPT model while preserving wording."""
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     prompt = f"""
# You are a transcript summarizer.

# You will be given a transcript that contains timestamps and text.

# Your job:

# ✔ Read the entire transcript carefully.
# ✔ Select **ONLY the most important and meaningful parts**.
# ✔ Remove any repetitive, filler, or non-important sections.
# ✔ **DO NOT modify or rewrite ANY of the original words.**
# ✔ Keep the original wording exactly as it appears.
# ✔ Each selected segment must include a timestamp in the format [MM:SS-MM:SS].
# ✔ Output must be a clean JSON list, like:

# [
#   {{
#     "timestamp": "[00:00-00:05]",
#     "text": "Original text from transcript without change."
#   }},
#   {{
#     "timestamp": "[00:10-00:20]",
#     "text": "Another important excerpt exactly as in transcript."
#   }}
# ]

# THE MOST IMPORTANT REQUIREMENT:
# → Perform stronger summarization.
# → Keep only the top important segments.
# → All text in the output must appear EXACTLY in the original transcript.
# → Do not add, modify, or paraphrase any text.

# Now process the transcript below:

# --- TRANSCRIPT START ---
# {input_text}
# --- TRANSCRIPT END ---

# Return ONLY the final JSON list. No explanations.
# """

#     response = client.responses.create(
#         model="gpt-5",
#         input=prompt
#     )

#     return response.output_text.strip()


# # --------------------------------------------------------------------
# # Run Test
# # --------------------------------------------------------------------
# if __name__ == "__main__":
#     transcript_file = "C:\\files\\podcast\\files\\Tucker.txt"

#     print("\n📄 Reading transcript file...\n")
#     original_text = read_text_file(transcript_file)

#     print("🚀 Running Summarizer...\n")
#     summary_result = summarize_text(original_text)

#     print("\n📝 SUMMARY RESULT (JSON):\n")
#     print(summary_result if summary_result else "⚠️ No summary returned!")

#     # -------------------------------------------------------
#     # SAVE output.json in SAME FOLDER as Tucker.txt
#     # -------------------------------------------------------
#     if summary_result:
#         output_path = os.path.join(os.path.dirname(transcript_file), "output.json")

#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(summary_result)

#         print(f"\n💾 Saved summary to: {output_path}")
#     else:
#         print("\n⚠️ Nothing to save!")



# import os
# import json
# from typing import List
# from pydantic import BaseModel
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage


# # ============================================================
# # Pydantic Schema (Segment-level, video-safe)
# # ============================================================
# class SelectedSegment(BaseModel):
#     segment: int
#     start_time: float
#     end_time: float
#     text: str


# # ============================================================
# # Transcript Segment Summarizer
# # ============================================================
# class TranscriptSummarizer:
#     def __init__(
#         self,
#         model_name: str = "magistral:24b",
#         temperature: float = 0.2,
#     ):
#         self.llm = ChatOllama(
#             model=model_name,
#             temperature=temperature,
#         )

#     # --------------------------------------------------------
#     # Load transcript JSON (segments)
#     # --------------------------------------------------------
#     @staticmethod
#     def load_segments(json_path: str) -> List[dict]:
#         if not os.path.exists(json_path):
#             raise FileNotFoundError(f"File not found: {json_path}")

#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         if not isinstance(data, list):
#             raise ValueError("Transcript JSON must be a list of segments")

#         return data

#     # --------------------------------------------------------
#     # Segment-level summarization
#     # --------------------------------------------------------
#     def summarize(
#         self,
#         transcript_json_path: str,
#         summary_percentage: int,
#     ) -> List[SelectedSegment]:

#         segments = self.load_segments(transcript_json_path)

#         prompt = f"""
# You are a transcript SEGMENT selector.

# Each item is a FULL VIDEO SEGMENT.
# A segment must be kept or removed as a whole.

# TASK:
# - Select ONLY the most important {summary_percentage}% of the segments from all the segments.
# - If a segment is important → KEEP THE ENTIRE SEGMENT.
# - If a segment is not important → REMOVE THE ENTIRE SEGMENT.
# - Like this keep the top {summary_percentage}% of segments only.

# STRICT RULES:
# - DO NOT rewrite, trim, summarize, or edit text.
# - DO NOT merge segments.
# - DO NOT create new fields.
# - DO NOT add explanations.
# - DO NOT use markdown.
# - Output ONLY valid JSON.

# OUTPUT FORMAT (exactly this):
# [
#   {{
#     "segment": 1,
#     "start_time": 0.0,
#     "end_time": 15.0,
#     "text": "Exact original text"
#   }}
# ]

# Here are the segments:
# {json.dumps(segments, ensure_ascii=False, indent=2)}

# Return ONLY the JSON list.
# """

#         response = self.llm.invoke(
#             [HumanMessage(content=prompt)]
#         )

#         raw_output = response.content.strip()

#         # ---- HARD JSON EXTRACTION (defensive) ----
#         try:
#             json_start = raw_output.index("[")
#             json_end = raw_output.rindex("]") + 1
#             clean_json = raw_output[json_start:json_end]
#             parsed = json.loads(clean_json)
#         except Exception:
#             raise ValueError(
#                 "Model did not return valid JSON.\n\nRaw output:\n"
#                 + raw_output
#             )

#         # ---- Validate schema ----
#         return [SelectedSegment(**item) for item in parsed]


# # ============================================================
# # Example usage (safe to remove in prod)
# # ============================================================
# if __name__ == "__main__":
#     summarizer = TranscriptSummarizer()

#     result = summarizer.summarize(
#         transcript_json_path="app/files/output_segments/transcription_with_timestamps.json",
#         summary_percentage=40,
#     )

#     print(
#         json.dumps(
#             [r.model_dump() for r in result],
#             indent=2,
#             ensure_ascii=False,
#         )
#     )



# ==================================================== V3 ========================================================

import os
import json
from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# ============================================================
# Pydantic Schemas
# ============================================================

class TranscriptSegment(BaseModel):
    segment: int
    start_time: float
    end_time: float
    text: str


class LabeledSegment(BaseModel):
    segment: int
    start_time: float
    end_time: float
    status: int = Field(..., description="1 if meaningful, 0 if filler/non-substantive", ge=0, le=1)


class LabeledSegmentsList(BaseModel):
    labeled_segments: List[LabeledSegment]


# ============================================================
# Transcript Classifier - Forces Full Fields
# ============================================================

class TranscriptClassifier:
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        temperature: float = 0.0,
        batch_size: int = 20,
    ):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

        # Use structured output with full schema
        self.structured_llm = self.llm.with_structured_output(
            LabeledSegmentsList,
            method="json_mode"
        )

        self.batch_size = batch_size

    @staticmethod
    def load_segments(json_path: str) -> List[TranscriptSegment]:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON must be a list of segments.")

        segments = [TranscriptSegment(**item) for item in data]
        print(f"Loaded {len(segments)} segments from {json_path}")
        return segments

    def classify_batch(self, segments: List[TranscriptSegment]) -> List[LabeledSegment]:
        num_segments = len(segments)
        input_data = [seg.model_dump() for seg in segments]

        # Strong few-shot example + strict instructions
        prompt = f"""
You are a precise transcript classifier. Your job is to classify each segment and return ALL original fields plus the status.

CRITICAL RULES:
- Output EXACTLY {num_segments} items
- For each item, include: "segment", "start_time", "end_time", and "status"
- Copy "segment", "start_time", and "end_time" EXACTLY from the input (do not change them)
- Only decide the "status": 1 if the text has meaningful content (insights, advice, key ideas), 0 if filler/repetition/silence/non-substantive

EXAMPLE (for 3 segments):
Input segments:
[
  {{"segment": 1, "start_time": 0.0, "end_time": 15.0, "text": "Hello everyone..."}}
  {{"segment": 2, "start_time": 15.0, "end_time": 30.0, "text": "Today we're discussing AI..."}}
  {{"segment": 3, "start_time": 30.0, "end_time": 45.0, "text": "Um... yeah..."}}
]

Correct output:
{{
  "labeled_segments": [
    {{"segment": 1, "start_time": 0.0, "end_time": 15.0, "status": 0}},
    {{"segment": 2, "start_time": 15.0, "end_time": 30.0, "status": 1}},
    {{"segment": 3, "start_time": 30.0, "end_time": 45.0, "status": 0}}
  ]
}}

Now classify the following {num_segments} segments. Copy start_time and end_time exactly.

Segments:
{json.dumps(input_data, indent=2, ensure_ascii=False)}

Return the result in the exact structured format above.
"""

        try:
            response: LabeledSegmentsList = self.structured_llm.invoke([HumanMessage(content=prompt)])
            labeled = response.labeled_segments

            if len(labeled) != num_segments:
                raise ValueError(f"Expected {num_segments} segments, got {len(labeled)}")

            return labeled

        except Exception as e:
            print(f"Error classifying batch (segments {segments[0].segment}–{segments[-1].segment}): {e}")
            raise

    def classify_segments(self, segments: List[TranscriptSegment]) -> List[LabeledSegment]:
        results: List[LabeledSegment] = []
        total_batches = (len(segments) - 1) // self.batch_size + 1

        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches} (segments {batch[0].segment}–{batch[-1].segment})...")

            batch_results = self.classify_batch(batch)
            results.extend(batch_results)

        print(f"\nClassification complete: {len(results)} segments labeled.")
        return results


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    JSON_PATH = "app/files/output_segments/transcription_with_timestamps.json"

    classifier = TranscriptClassifier(
        model_name="llama3.1:8b",
        temperature=0.0,
        batch_size=20,
    )

    try:
        segments = classifier.load_segments(JSON_PATH)
        labeled_segments = classifier.classify_segments(segments)

        print("\n" + "="*80)
        print("FINAL RESULT: Labeled Segments (with full timestamps)")
        print("="*80)

        output_data = [s.model_dump() for s in labeled_segments]
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

        important_count = sum(1 for s in labeled_segments if s.status == 1)
        total_count = len(labeled_segments)
        print(f"\nSummary: {important_count}/{total_count} segments marked as important ({important_count/total_count:.1%})")

        # Save results
        output_path = "app/files/output_segments/labeled_transcript_full.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"\nCritical Error: {e}")
        raise