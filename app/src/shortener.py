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

# I will give you transcript text with timestamps.

# Your job:

# ✔ Read all text carefully.
# ✔ **DO NOT change any words from the original transcript.**
# ✔ The summarization should be in such a way that, only the most important segments with their timestamps are included. Donot change any words.
# ✔ Each segment must have a timestamp in the format [MM:SS-MM:SS].
# ✔ Output must be in clean JSON format like:

# [
#   {{
#     "timestamp": "[00:00-00:05]",
#     "text": "Hello everyone, welcome to the show."
#   }},
#   {{
#     "timestamp": "[00:11-00:17]",
#     "text": "Today we will discuss important topics."
#   }}
# ]

# Now process the transcript below:

# --- TRANSCRIPT START ---
# {input_text}
# --- TRANSCRIPT END ---

# Produce the cleaned narrative summary in JSON list format only.
# """

#     response = client.responses.create(
#         model="gpt-5",
#         input=prompt
#         #max_output_tokens=800,
#         #temperature=0.3
#     )

#     return response.output_text.strip()


# # --------------------------------------------------------------------
# # Run Test
# # --------------------------------------------------------------------
# if __name__ == "__main__":
#     # Change this path to your transcript file
#     transcript_file = "C:\\files\\podcast\\files\\Tucker.txt"

#     print("\n📄 Reading transcript file...\n")
#     original_text = read_text_file(transcript_file)

#     print("🚀 Running Summarizer...\n")
#     summary_result = summarize_text(original_text)

#     print("\n📝 SUMMARY RESULT (JSON):\n")
#     print(summary_result if summary_result else "⚠️ No summary returned!")


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

# I will give you transcript text with timestamps.

# Your job:

# ✔ Read all text carefully.
# ✔ **DO NOT change any words from the original transcript.**
# ✔ The summarization should be in such a way that, only the most important segments with their timestamps are included. Donot change any words.
# ✔ Each segment must have a timestamp in the format [MM:SS-MM:SS].
# ✔ Output must be in clean JSON format like:

# [
#   {{
#     "timestamp": "[00:00-00:05]",
#     "text": "Hello everyone, welcome to the show."
#   }},
#   {{
#     "timestamp": "[00:11-00:17]",
#     "text": "Today we will discuss important topics."
#   }}
# ]

# Now process the transcript below:

# --- TRANSCRIPT START ---
# {input_text}
# --- TRANSCRIPT END ---

# Produce the cleaned narrative summary in JSON list format only.
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
#     # Change this path to your transcript file
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


from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from typing import List

# Load environment variables
load_dotenv()


# --------------------------------------------------------------------
# Pydantic Model for Summarizer Output
# --------------------------------------------------------------------
class SummarizedSegment(BaseModel):
    timestamp: str = Field(..., description="Start and end time like [00:00-00:05]")
    text: str = Field(..., description="Write exact text from that timestamp here")


# --------------------------------------------------------------------
# Read transcript from .txt file
# --------------------------------------------------------------------
def read_text_file(file_path: str) -> str:
    """Read and return the content of a text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# --------------------------------------------------------------------
# Main Summarizer Function
# --------------------------------------------------------------------
def summarize_text(input_text: str) -> str:
    """Summarize transcript text using GPT model while preserving wording."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
You are a transcript summarizer.

You will be given a transcript that contains timestamps and text.

Your job:

✔ Read the entire transcript carefully.
✔ Select **ONLY the most important and meaningful parts**.
✔ Remove any repetitive, filler, or non-important sections.
✔ **DO NOT modify or rewrite ANY of the original words.**
✔ Keep the original wording exactly as it appears.
✔ Each selected segment must include a timestamp in the format [MM:SS-MM:SS].
✔ Output must be a clean JSON list, like:

[
  {{
    "timestamp": "[00:00-00:05]",
    "text": "Original text from transcript without change."
  }},
  {{
    "timestamp": "[00:10-00:20]",
    "text": "Another important excerpt exactly as in transcript."
  }}
]

THE MOST IMPORTANT REQUIREMENT:
→ Perform stronger summarization.
→ Keep only the top important segments.
→ All text in the output must appear EXACTLY in the original transcript.
→ Do not add, modify, or paraphrase any text.

Now process the transcript below:

--- TRANSCRIPT START ---
{input_text}
--- TRANSCRIPT END ---

Return ONLY the final JSON list. No explanations.
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return response.output_text.strip()


# --------------------------------------------------------------------
# Run Test
# --------------------------------------------------------------------
if __name__ == "__main__":
    transcript_file = "C:\\files\\podcast\\files\\Tucker.txt"

    print("\n📄 Reading transcript file...\n")
    original_text = read_text_file(transcript_file)

    print("🚀 Running Summarizer...\n")
    summary_result = summarize_text(original_text)

    print("\n📝 SUMMARY RESULT (JSON):\n")
    print(summary_result if summary_result else "⚠️ No summary returned!")

    # -------------------------------------------------------
    # SAVE output.json in SAME FOLDER as Tucker.txt
    # -------------------------------------------------------
    if summary_result:
        output_path = os.path.join(os.path.dirname(transcript_file), "output.json")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_result)

        print(f"\n💾 Saved summary to: {output_path}")
    else:
        print("\n⚠️ Nothing to save!")
