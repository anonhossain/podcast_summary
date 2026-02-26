################################# FIXED VERSION — STOPWORDS FORMATTING ERROR SOLVED ##########################################

import os
import json
import assemblyai as aai
import re
import math
from pydantic import BaseModel
from typing import List, Dict, Any
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pathlib import Path


class FinalOutput(BaseModel):
    segment_id: int
    speaker: str
    start: int
    end: int
    text: str


# ── Exact stopwords (moved outside template for safety) ──────────────────────
STOPWORDS = {
    "a","an","the","and","but","or","for","nor","on","at","to","by","in","of","with",
    "is","was","are","were","be","been","being","have","has","had","do","does","did",
    "will","would","shall","should","may","might","can","could","i","you","he","she",
    "it","we","they","me","him","her","us","them","my","your","his","its","our","their",
    "what","which","who","that","this","these","those","not","no","so","if","then",
    "because","as","when","where","how","all","just","about","up","out","also","very",
    "even","more","other","some","there","from","well","yeah","oh","got","get","go",
    "like","right","okay","really","know","think","want","people","going","said","say",
    "actually","basically","literally","mean","bit","lot","good",
}

# Pre-format as a readable string for the prompt
STOPWORDS_FORMATTED = ",\n".join(f'"{w}"' for w in sorted(STOPWORDS))


# ====================== EXACT NARRATIVE FLOW SCORER (official) ======================
def meaningful_tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def tfidf_cosine_sim(text_a: str, text_b: str) -> float:
    tok_a = meaningful_tokens(text_a)
    tok_b = meaningful_tokens(text_b)
    if not tok_a or not tok_b:
        return 0.0

    def tf(tokens):
        c = Counter(tokens)
        n = len(tokens)
        return {w: v / n for w, v in c.items()}

    N = 2
    df = Counter()
    for tokens in [tok_a, tok_b]:
        for w in set(tokens):
            df[w] += 1
    idf = {w: math.log((N + 1) / (v + 1)) + 1 for w, v in df.items()}

    def vec(tokens):
        tf_s = tf(tokens)
        return {w: tf_s[w] * idf.get(w, 1.0) for w in tf_s}

    v_a, v_b = vec(tok_a), vec(tok_b)

    shared = set(v_a) & set(v_b)
    if not shared:
        return 0.0
    dot = sum(v_a[w] * v_b[w] for w in shared)
    mag_a = math.sqrt(sum(x ** 2 for x in v_a.values()))
    mag_b = math.sqrt(sum(x ** 2 for x in v_b.values()))
    return dot / (mag_a * mag_b) if (mag_a and mag_b) else 0.0


def evaluate_narrative_flow(edited_json_path: str, window: int = 2, threshold: float = 0.32) -> Dict[str, Any]:
    with open(edited_json_path, encoding="utf-8") as f:
        segments = sorted(json.load(f), key=lambda s: s["start"])

    sims = []
    flagged = []
    for i in range(len(segments) - window):
        win_a = " ".join(s["text"] for s in segments[i:i + window])
        win_b = " ".join(s["text"] for s in segments[i + 1:i + 1 + window])
        sim = tfidf_cosine_sim(win_a, win_b)
        sims.append(sim)
        if sim < threshold:
            flagged.append({
                "pair_index": i,
                "seg_ids": [segments[i]["segment_id"], segments[i + window - 1]["segment_id"]],
                "similarity": round(sim, 4),
                "win_a_preview": win_a[:120] + ("..." if len(win_a) > 120 else ""),
                "win_b_preview": win_b[:120] + ("..." if len(win_b) > 120 else ""),
            })

    total = len(sims)
    if total == 0:
        return {"score": 5, "total_pairs": 0, "pairs_below_threshold": 0, "failure_ratio": 0.0,
                "avg_similarity": 1.0, "flagged_pairs": [], "passes": True}

    below = sum(1 for s in sims if s < threshold)
    near_zero = sum(1 for s in sims if s < 0.10)
    ratio = below / total
    avg_sim = sum(sims) / total

    if near_zero > total * 0.5:
        score = 1
    elif ratio > 0.50:
        score = 2
    elif ratio > 0.30:
        score = 3
    elif ratio > 0.10:
        score = 4
    else:
        score = 5

    return {
        "score": score,
        "total_pairs": total,
        "pairs_below_threshold": below,
        "failure_ratio": round(ratio, 4),
        "avg_similarity": round(avg_sim, 4),
        "flagged_pairs": flagged,
        "passes": score >= 4
    }
# ====================== END SCORER ======================


# ====================== UPGRADED SYSTEM PROMPT (fixed formatting) ======================
SYSTEM_PROMPT_TEMPLATE = """### ROLE: High-Precision Narrative Architect for Podcast Supercut
You are an expert editor tasked with creating a condensed, coherent supercut from podcast transcript segments. Your selections must form a natural, flowing narrative that feels like a seamless edit of the original audio, not a rewrite. Prioritize verbatim fidelity, smooth transitions, and overall coherence to meet strict evaluation criteria (all must score 4/5 or higher).

### CRITICAL RULES (5/5 SCORE TARGET):
1. **BOUNDARY INTEGRITY:** You are FORBIDDEN from ending a selection on a segment that does not conclude a sentence. If a thought spans multiple IDs, you MUST take the entire block.
2. **VERBATIM FIDELITY:** Do not rewrite or paraphrase.

### NARRATIVE FLOW OPTIMIZATION (OFFICIAL TEST GUIDANCE — MUST FOLLOW)
STOPWORDS (exact list used by the test — avoid segments that consist only of these after tokenization):
{{{STOPWORDS_FORMATTED}}}

def meaningful_tokens(text):
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

The Narrative Flow test creates overlapping 2-segment windows, computes TF-IDF vectors on each pair (corpus size = 2), and measures cosine similarity. Any window pair < 0.32 is a bad jump. Goal: ≤10% bad pairs = score 5; ≤30% = score 4. Above 30% = FAIL.

### WORKED EXAMPLES (STUDY THESE — NEVER REPEAT BAD PATTERNS)

Worked Example 1 — Back-Channel Segment (FAIL)
Seg 59: "1.3 million in 19 hours."
Seg 60: "Okay."
Seg 68: "You know who is not one of my guests tonight?"
After tokenisation Window A = ["million","hours"], Window B = ["one","guests","tonight"] → no shared tokens → similarity = 0.000 → FAIL

Worked Example 2 — Duplicate Back-Channels (FAIL)
Seg 331: "Right."
Seg 332: "Right."
After tokenisation both windows become empty → similarity = 0.000 → FAIL

Worked Example 3 — Coherent Thematic Transition (PASS)
Seg 5: "This is a song about grief."
Seg 6: "...piercing sadness that borders on and perhaps even touches despair."
Seg 9: "The song we've just heard called Camden begins with the line..."
Many shared tokens across windows → similarity = 0.548 → PASS

Never produce patterns like Example 1 or 2. Always aim for patterns like Example 3 (shared content words across the bridge segment).

When choosing the next segment, mentally compute the meaningful token overlap with the previous selected segment. If overlap is low, prefer a bridge segment that re-uses at least 3–4 content words (names, dates, key nouns, repeated concepts).

You may go up to ±8% on duration if it removes a flagged jump.

### OBJECTIVE:
- Select a contiguous or near-contiguous sequence of `segment_id`s (in original order, no reordering).
- Target total duration: Exactly {length} ±8% of the original (approx. {target_duration_seconds:.1f}s out of {original_duration_seconds:.1f}s).
- Preserve core narrative arc while compressing filler and redundancies.
- Ensure output passes all evaluation categories at 4/5 or higher.

### INPUT DATA:
- Transcript segments: {original_text}
- Original duration: {original_duration_seconds:.1f}s

### SELECTION RULES:
1. Editorial Fidelity — verbatim only
2. Editorial Smoothness — natural joins
3. Narrative Flow — logical progression, aim >0.32 similarity
4. Segment Boundaries — full thoughts only
5. Ending Completeness — feels finished
6. Length — within ±8% of target

### STEP-BY-STEP:
1. Analyze arc
2. Prioritize segments
3. Iterate for length
4. Self-evaluate vs criteria + examples
5. Finalize

### OUTPUT FORMAT:
Return ONLY JSON:
{{
  "selected_ids": [0, 1, 2, 5, 10]
}}
"""


def create_podcast_supercut(
    audio_path: str,
    target_percent: float,
    output_dir: str = "app/files/supercut_output",
    temp_dir: str = r"app/files/temp",
    gap_ms: int = 200,
    fade_ms: int = 50,
    head_padding_ms: int = 50,
    tail_padding_ms: int = 150,
) -> dict:
    if not (0 < target_percent <= 100):
        raise ValueError("target_percent must be between 0 and 100 (exclusive on 0)")

    load_dotenv()
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    openai_client = OpenAI()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    input_ext = Path(audio_path).suffix.lower().lstrip(".")

    # Working MP3
    working_audio_path = os.path.join(temp_dir, f"{base_name}_working.mp3")
    AudioSegment.from_file(audio_path).export(working_audio_path, format="mp3", bitrate="192k")

    # Transcription
    config = aai.TranscriptionConfig(speaker_labels=True, punctuate=True, format_text=True)
    transcript = aai.Transcriber().transcribe(working_audio_path, config)

    structured_segments = []
    for idx, sentence in enumerate(transcript.get_sentences()):
        seg = FinalOutput(
            segment_id=idx,
            speaker=sentence.speaker or "",
            start=sentence.start,
            end=sentence.end,
            text=sentence.text
        ).model_dump()
        structured_segments.append(seg)

    # Client restrictions
    if input_ext not in {"mp3", "wav"}:
        return {"status": "failed", "reason": "invalid_format"}
    total_ms = structured_segments[-1]["end"] - structured_segments[0]["start"]
    if total_ms > 1_800_000:
        return {"status": "failed", "reason": "file_too_long"}

    full_json_path = os.path.join(output_dir, f"{base_name}_full.json")
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_segments, f, indent=2, ensure_ascii=False)

    # LLM data
    simplified_input = [{"segment_id": s["segment_id"], "text": s["text"]} for s in structured_segments]
    original_duration_sec = (structured_segments[-1]["end"] - structured_segments[0]["start"]) / 1000
    target_duration_sec = original_duration_sec * (target_percent / 100)

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        STOPWORDS_FORMATTED=STOPWORDS_FORMATTED,
        length=target_percent,
        original_duration_seconds=original_duration_sec,
        target_duration_seconds=target_duration_sec,
        original_text=json.dumps(simplified_input, indent=1)
    )

    # Self-correction loop
    selected_ids = []
    fixed_segments = []
    max_attempts = 3

    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")

        if attempt > 0:
            temp_path = os.path.join(temp_dir, f"{base_name}_temp.json")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(fixed_segments, f, indent=2, ensure_ascii=False)

            score_result = evaluate_narrative_flow(temp_path)
            os.remove(temp_path)

            if score_result["passes"]:
                print(f"✓ Passed with score {score_result['score']}")
                break

            flagged_text = "\n".join(
                f"Pair {p['pair_index']}: segs {p['seg_ids']} sim={p['similarity']} | A: {p['win_a_preview']} | B: {p['win_b_preview']}"
                for p in score_result["flagged_pairs"][:5]
            )

            user_msg = f"""Previous score: {score_result['score']}/5 (failure {score_result['failure_ratio']*100:.1f}%, {score_result['pairs_below_threshold']}/{score_result['total_pairs']} bad)

Flagged:
{flagged_text}

Fix by adding/swap bridge segments to get all windows ≥0.32. Keep target duration ±8%.
Return only JSON with "selected_ids"."""

        else:
            user_msg = "Analyze the transcript and return selected_ids."

        response = openai_client.chat.completions.create(
            model="gpt-5.2",
            reasoning_effort="medium",
            verbosity="low",
            timeout=1600.0,
            
              # ← changed to a more stable model — adjust if needed
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"}
        )

        try:
            ai_output = json.loads(response.choices[0].message.content)
            selected_ids = ai_output.get("selected_ids", [])
        except:
            selected_ids = selected_ids  # keep previous on error

        # Build segments
        ai_selected = [s for s in structured_segments if s["segment_id"] in selected_ids]

        # Fix incomplete sentences
        fixed = []
        processed = set()
        for seg in ai_selected:
            cid = seg["segment_id"]
            if cid in processed:
                continue
            fixed.append(seg)
            processed.add(cid)
            check = seg
            while check["text"].strip() and check["text"].strip()[-1] not in ".!?":
                nid = check["segment_id"] + 1
                if nid >= len(structured_segments):
                    break
                nseg = structured_segments[nid]
                if nseg["segment_id"] not in processed:
                    fixed.append(nseg)
                    processed.add(nseg["segment_id"])
                check = nseg

        fixed.sort(key=lambda x: x["segment_id"])
        fixed_segments = fixed

    # Save edited
    edited_json_path = os.path.join(output_dir, f"{base_name}_edited.json")
    with open(edited_json_path, "w", encoding="utf-8") as f:
        json.dump(fixed_segments, f, indent=2, ensure_ascii=False)

    # Audio stitching (your original logic)
    audio = AudioSegment.from_file(working_audio_path)
    combined = AudioSegment.empty()
    cut_silence = AudioSegment.silent(duration=75, frame_rate=audio.frame_rate)

    for i, seg in enumerate(fixed_segments):
        prev_cont = i > 0 and fixed_segments[i-1]["segment_id"] == seg["segment_id"] - 1
        next_cont = i < len(fixed_segments)-1 and fixed_segments[i+1]["segment_id"] == seg["segment_id"] + 1

        start = max(0, seg["start"] - (head_padding_ms if not prev_cont else 0))
        end = seg["end"] + (tail_padding_ms if not next_cont else 0)

        clip = audio[start:end]

        if not prev_cont:
            clip = clip.fade_in(fade_ms)
        if not next_cont:
            clip = clip.fade_out(fade_ms)

        combined += clip

        if i < len(fixed_segments)-1 and not next_cont:
            combined += cut_silence

    final_filename = f"{base_name}_cropped.{input_ext}"
    final_audio_path = os.path.join(output_dir, final_filename)
    combined.export(final_audio_path, format=input_ext, bitrate="192k")

    if os.path.exists(working_audio_path):
        os.remove(working_audio_path)

    final_score = evaluate_narrative_flow(edited_json_path)

    return {
        "status": "success",
        "original_duration_seconds": round(original_duration_sec, 1),
        "target_duration_seconds": round(target_duration_sec, 1),
        "final_audio_path": final_audio_path,
        "edited_json_path": edited_json_path,
        "full_transcript_json": full_json_path,
        "number_of_segments_original": len(structured_segments),
        "number_of_segments_final": len(fixed_segments),
        "final_narrative_score": final_score["score"],
        "passes_narrative": final_score["passes"]
    }


if __name__ == "__main__":
    try:
        val = input("Enter percentage to keep (e.g. 20): ")
        percent = float(val)

        result = create_podcast_supercut(
            audio_path=r"app\files\demo\Colbert ATTACKS CBS Over Pulled Interview With James Talarico!.mp3",
            target_percent=percent,
            output_dir=r"app/files/test",
            gap_ms=200
        )

        print("\n" + "="*60)
        print("SUPERCUT CREATED SUCCESSFULLY")
        print(json.dumps(result, indent=2))
        print("="*60)

    except Exception as e:
        print(f"Pipeline failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

# ################################# version 3 ##########################################

# import os
# import json
# import assemblyai as aai
# from pydantic import BaseModel
# from typing import List, Optional
# from dotenv import load_dotenv
# from openai import OpenAI
# from pydub import AudioSegment
# from pathlib import Path


# class FinalOutput(BaseModel):
#     segment_id: int
#     speaker: str
#     start: int
#     end: int
#     text: str


# SYSTEM_PROMPT_TEMPLATE = """### ROLE: High-Precision Narrative Architect for Podcast Supercut
# You are an expert editor tasked with creating a condensed, coherent supercut from podcast transcript segments. Your selections must form a natural, flowing narrative that feels like a seamless edit of the original audio, not a rewrite. Prioritize verbatim fidelity, smooth transitions, and overall coherence to meet strict evaluation criteria (all must score 4/5 or higher).

# ### CRITICAL RULES (5/5 SCORE TARGET):
# 1. **BOUNDARY INTEGRITY:** You are FORBIDDEN from ending a selection on a segment that does not conclude a sentence. If a thought spans multiple IDs, you MUST take the entire block.
# 2. **VERBATIM FIDELITY:** Do not rewrite or paraphrase.

# ### OBJECTIVE:
# - Select a contiguous or near-contiguous sequence of `segment_id`s (in original order, no reordering) to create a shortened version.
# - Target total duration: Exactly {length} ±5% of the original (approx. {target_duration_seconds:.1f}s out of {original_duration_seconds:.1f}s).
# - The supercut must preserve the core narrative arc (e.g., introduction of topic, key examples, contrasts, conclusion) while compressing by removing filler, redundancies, and less essential details.
# - Ensure the output passes all evaluation categories (detailed below) at 4/5 or higher. If a selection would fail any, adjust until it passes.

# ### INPUT DATA:
# - The transcript is provided as a list of segments: Each has `segment_id` (sequential integer) and `text` (verbatim sentence or clause from the audio).
# - Original total duration: {original_duration_seconds:.1f}s.
# - Simplified segments for selection: {original_text}

# ### SELECTION RULES (MUST FOLLOW TO PASS CRITERIA):
# 1. **Editorial Fidelity (Transcript-Level)**: 
#    - Select only existing segments verbatim—no paraphrasing, adding words, or modifying text.
#    - All selected text must appear exactly as in the input (perfect alignment).
#    - Aim for 5/5: Perfect verbatim; minor omissions only if they don't impact meaning.

# 2. **Editorial Smoothness**:
#    - Ensure joins between selected segments read naturally (no abrupt grammar breaks).
#    - Reduce filler (e.g., "um", repetitions) by skipping segments, but keep flow syntactically sound.
#    - Aim for 5/5: Clean, natural flow; at worst, minor roughness (4/5).

# 3. **Narrative Flow & Continuity**:
#    - Maintain logical progression: Selected segments must form a coherent story, not a patchwork.
#    - Check semantic similarity: Adjacent segments should be thematically linked (e.g., avoid jumps from one topic to unrelated; simulate cosine similarity >0.32 by ensuring related ideas).
#    - If cosine-like similarity <0.32 (e.g., abrupt topic shift), it's an automatic failure—adjust by including bridge segments.
#    - Avoid repetitions; justify any timestamp jumps with continuity.
#    - Aim for 5/5: Logical throughout; allow one minor issue (4/5).

# 4. **Segment Boundaries**:
#    - Cut only at natural, complete thoughts: Start/end selections at full sentence boundaries.
#    - No truncated clauses (e.g., avoid mid-sentence ends like "During.").
#    - Aim for 5/5: Clean starts/ends; minor edges ok (4/5).

# 5. **Ending Completeness**:
#    - The supercut must feel finished: End on a full sentence that resolves or concludes the main idea.
#    - No dangling concepts or abrupt cut-offs.
#    - Aim for 5/5: Clear, intentional ending; slightly abrupt but complete (4/5).

# 6. **Length & Information Density**:
#    - Total selected duration must be {length} ±5% of original (prioritize meaningful content over filler).
#    - Calculate precisely: Sum (end - start)/1000 for selected segments.
#    - If outside range, adjust by adding/removing low-priority segments.
#    - Focus on dense, key info (e.g., core arguments, examples, conclusions).
#    - Aim for 5/5: Within range, clear & dense; slightly off ok (4/5).

# ### STEP-BY-STEP SELECTION PROCESS:
# 1. **Analyze Narrative Arc**: Read the full input. Identify core elements (e.g., intro, historical examples, contrasts, irony/conclusion in this Tucker Carlson-style podcast).
# 2. **Prioritize Segments**: 
#    - Must-keep: Key thesis statements, quotes, historical facts, ironic twists.
#    - Optional/skip: Filler, tangents, promotions (e.g., YouTube plugs).
#    - Preserve order: Select IDs in ascending sequence (gaps ok if flow holds).
# 3. **Iterate for Length**: Start with essential segments; add/remove to hit {target_duration_seconds:.1f}s (±10% tolerance).
# 4. **Self-Evaluate**: Mentally score your selection against the 6 criteria. If any <4/5, revise (e.g., add bridges for continuity, extend ending).
# 5. **Finalize**: Ensure overall consistent 4+/5; regenerate mentally if needed.

# ### OUTPUT FORMAT:
# Return ONLY a JSON object with the list of chosen IDs (in order selected).
# Example:
# {{
#   "selected_ids": [0, 1, 2, 5, 10]
# }}
# No explanations or additional text—strict JSON only.
# """

# # 2. Conversion Utility Function
# def to_mp3(src_path: str, dst_path: str):
#     try:
#         audio = AudioSegment.from_file(src_path)
#         audio.export(dst_path, format="mp3", bitrate="192k")
#         print(f"Converted to MP3: {dst_path}")
#     except Exception as e:
#         raise RuntimeError(f"MP3 Conversion Error: {e}")


# def create_podcast_supercut(
#     audio_path: str,
#     target_percent: float,
#     output_dir: str = "app/files/supercut_output",
#     # 1. Added temp_dir parameter
#     temp_dir: str = r"app/files/temp",
#     gap_ms: int = 200,
#     fade_ms: int = 50,
#     head_padding_ms: int = 50,
#     tail_padding_ms: int = 150,
#     output_filename: str = "cropped_audio" # Extension handled dynamically now
# ) -> dict:
#     """
#     Single entry-point function that performs the complete podcast supercut pipeline.
#     """
#     if not (0 < target_percent <= 100):
#         raise ValueError("target_percent must be between 0 and 100 (exclusive on 0)")

#     # ── Setup ────────────────────────────────────────────────────────────────
#     load_dotenv()
#     aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
#     openai_client = OpenAI()

#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(temp_dir, exist_ok=True)

#     # Determine file info and formats
#     base_name = os.path.splitext(os.path.basename(audio_path))[0]
#     audio_path_obj = Path(audio_path).resolve() # used for restriction checks
#     input_ext = audio_path_obj.suffix.lower() # used for restriction checks
#     input_extension = os.path.splitext(audio_path)[1].lower().replace(".", "")
    
#     # 2. Convert to MP3 and save in temp folder
#     working_audio_path = os.path.join(temp_dir, f"{base_name}_working.mp3")
#     to_mp3(audio_path, working_audio_path)

#     # ── 1. Transcription (Using the temp MP3) ────────────────────────────────
#     config = aai.TranscriptionConfig(speaker_labels=True, punctuate=True, format_text=True)
#     transcript = aai.Transcriber().transcribe(working_audio_path, config)

#     sentences = transcript.get_sentences()
#     structured_segments: List[dict] = []

#     for idx, sentence in enumerate(sentences):
#         seg = FinalOutput(
#             segment_id=idx,
#             speaker=sentence.speaker or "",
#             start=sentence.start,
#             end=sentence.end,
#             text=sentence.text
#         )
#         structured_segments.append(seg.model_dump())

#     # ##########################################################################
#     # CLIENT RESTRICTION BLOCK
#     # ##########################################################################
#     # 1. Check Format: Only allow mp3 or wav
#     allowed_formats = {".mp3", ".wav"}
#     if input_ext not in allowed_formats:
#         print(f"Error: Format {input_ext} is not supported. Please use MP3 or WAV.")
#         return {"status": "failed", "reason": "invalid_format"}

#     # 2. Check Length: Max 30 minutes (1800000 ms)
#     total_duration_ms = structured_segments[-1]["end"] - structured_segments[0]["start"]
#     if total_duration_ms > 1800000:
#         print("Error: The file is too long. Maximum allowed length is 30 minutes.")
#         return {"status": "failed", "reason": "file_too_long"}
#     # ##########################################################################

#     # 4. Save full transcription to output_dir
#     full_json_path = os.path.join(output_dir, f"{base_name}_full.json")
#     with open(full_json_path, "w", encoding="utf-8") as f:
#         json.dump(structured_segments, f, indent=2, ensure_ascii=False)

#     # ── 2. Prepare prompt & ask LLM for selection ────────────────────────────
#     simplified_input = [{"segment_id": s["segment_id"], "text": s["text"]} for s in structured_segments]

#     first_start = structured_segments[0].get("start", 0)
#     last_end = structured_segments[-1].get("end", 0)
#     original_duration_sec = (last_end - first_start) / 1000.0
#     target_duration_sec = original_duration_sec * (target_percent / 100)

#     prompt = SYSTEM_PROMPT_TEMPLATE.format(
#         length=target_percent,
#         original_duration_seconds=original_duration_sec,
#         target_duration_seconds=target_duration_sec,
#         original_text=json.dumps(simplified_input, indent=1)
#     )

#     response = openai_client.chat.completions.create(
#         model="gpt-5.2",
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": "Analyze and return selected_ids."}
#         ],
#         response_format={"type": "json_object"},
#         reasoning_effort="medium",
#         verbosity="low"
#     )

#     ai_output = json.loads(response.choices[0].message.content)
#     selected_ids = ai_output.get("selected_ids", [])

#     ai_selected_segments = [s for s in structured_segments if s["segment_id"] in selected_ids]

#     # ── 3. Fix fragmented sentences ─────────────────────────────────────────
#     fixed_segments = []
#     processed = set()

#     for seg in ai_selected_segments:
#         curr_id = seg["segment_id"]
#         if curr_id in processed:
#             continue

#         fixed_segments.append(seg)
#         processed.add(curr_id)

#         check = seg
#         while check["text"].strip() and check["text"].strip()[-1] not in [".", "!", "?"]:
#             next_id = check["segment_id"] + 1
#             if next_id >= len(structured_segments):
#                 break
#             next_seg = structured_segments[next_id]
#             if next_seg["segment_id"] not in processed:
#                 fixed_segments.append(next_seg)
#                 processed.add(next_seg["segment_id"])
#             check = next_seg

#     fixed_segments.sort(key=lambda x: x["segment_id"])

#     # 4. Save AI shorten json to output_dir
#     edited_json_path = os.path.join(output_dir, f"{base_name}_edited.json")
#     with open(edited_json_path, "w", encoding="utf-8") as f:
#         json.dump(fixed_segments, f, indent=2, ensure_ascii=False)

#     # ── 4. Create final audio (Using temp MP3 for cropping) ──────────────────
#     # 3. Using the file from temp to crop
#     audio = AudioSegment.from_file(working_audio_path)
#     combined = AudioSegment.empty()

#     # Define your custom cut-point silence (e.g., 75ms)
#     cut_silence_duration = 75 
#     cut_silence = AudioSegment.silent(duration=cut_silence_duration, frame_rate=audio.frame_rate)

#     for i, seg in enumerate(fixed_segments):
#         # Determine if this segment is part of a continuous block or a new cut
#         is_prev_contiguous = (i > 0 and fixed_segments[i-1]["segment_id"] == seg["segment_id"] - 1)
#         is_next_contiguous = (i < len(fixed_segments) - 1 and fixed_segments[i+1]["segment_id"] == seg["segment_id"] + 1)

#         # 1. Calculate Start/End with padding ONLY at the boundaries of a cut
#         # This prevents the "double-word" overlap
#         start_point = seg["start"] - (head_padding_ms if not is_prev_contiguous else 0)
#         end_point = seg["end"] + (tail_padding_ms if not is_next_contiguous else 0)
        
#         # Ensure we don't go below 0
#         start_point = max(0, start_point)
        
#         clip = audio[start_point:end_point]

#         # 2. Apply fades only at the start and end of a full narrative "chunk"
#         if not is_prev_contiguous:
#             clip = clip.fade_in(fade_ms)
#         if not is_next_contiguous:
#             clip = clip.fade_out(fade_ms)

#         # 3. Append the audio
#         combined += clip

#         # 4. If the NEXT segment is a "cut" (not sequential), add your silent gap
#         if i < len(fixed_segments) - 1 and not is_next_contiguous:
#             combined += cut_silence

#     # ── 5. Export ──────────────────────────────────────────────────────────────
#     final_filename = f"{base_name}_cropped.{input_extension}"
#     final_audio_path = os.path.join(output_dir, final_filename)
#     combined.export(final_audio_path, format=input_extension, bitrate="192k")

#     # 4. Clean up: Delete the temporary working MP3
#     if os.path.exists(working_audio_path):
#         os.remove(working_audio_path)

#     return {
#         "status": "success",
#         "original_duration_seconds": round(original_duration_sec, 1),
#         "target_duration_seconds": round(target_duration_sec, 1),
#         "final_audio_path": final_audio_path,
#         "edited_json_path": edited_json_path,
#         "full_transcript_json": full_json_path,
#         "number_of_segments_original": len(structured_segments),
#         "number_of_segments_final": len(fixed_segments),
#     }

# if __name__ == "__main__":
#     try:
#         val = input("Enter percentage to keep (e.g. 20): ")
#         percent = float(val)

#         result = create_podcast_supercut(
#             audio_path=r"app\files\demo\Canadian Mother Outraged Over Doctor Euthanizing MENTALLY ILL Son!.mp3",
#             target_percent=percent,
#             output_dir=r"app/files/test",
#             gap_ms=200
#         )

#         print("\n" + "="*50)
#         print("SUPERCUT CREATED SUCCESSFULLY")
#         print(json.dumps(result, indent=2))
#         print("="*50)

#     except Exception as e:
#         print(f"Pipeline failed: {e}")





############################## video output is not working version 4 ##########################################

# import os
# import json
# import subprocess
# import assemblyai as aai
# from pydantic import BaseModel
# from typing import List
# from pathlib import Path
# from dotenv import load_dotenv
# from openai import OpenAI
# from pydub import AudioSegment

# class FinalOutput(BaseModel):
#     segment_id: int
#     speaker: str
#     start: int
#     end: int
#     text: str

# SYSTEM_PROMPT_TEMPLATE = """### ROLE: High-Precision Narrative Architect for Podcast Supercut
# You are an expert editor tasked with creating a condensed, coherent supercut from podcast transcript segments. Your selections must form a natural, flowing narrative that feels like a seamless edit of the original audio, not a rewrite. Prioritize verbatim fidelity, smooth transitions, and overall coherence to meet strict evaluation criteria (all must score 4/5 or higher).

# ### CRITICAL RULES (5/5 SCORE TARGET):
# 1. **BOUNDARY INTEGRITY:** You are FORBIDDEN from ending a selection on a segment that does not conclude a sentence. If a thought spans multiple IDs, you MUST take the entire block.
# 2. **VERBATIM FIDELITY:** Do not rewrite or paraphrase.

# ### OBJECTIVE:
# - Select a contiguous or near-contiguous sequence of `segment_id`s (in original order, no reordering) to create a shortened version.
# - Target total duration: Exactly {length} ±5% of the original (approx. {target_duration_seconds:.1f}s out of {original_duration_seconds:.1f}s).
# - The supercut must preserve the core narrative arc (e.g., introduction of topic, key examples, contrasts, conclusion) while compressing by removing filler, redundancies, and less essential details.
# - Ensure the output passes all evaluation categories (detailed below) at 4/5 or higher. If a selection would fail any, adjust until it passes.

# ### INPUT DATA:
# - The transcript is provided as a list of segments: Each has `segment_id` (sequential integer) and `text` (verbatim sentence or clause from the audio).
# - Original total duration: {original_duration_seconds:.1f}s.
# - Simplified segments for selection: {original_text}

# ### SELECTION RULES (MUST FOLLOW TO PASS CRITERIA):
# 1. **Editorial Fidelity (Transcript-Level)**: 
#    - Select only existing segments verbatim—no paraphrasing, adding words, or modifying text.
#    - All selected text must appear exactly as in the input (perfect alignment).
#    - Aim for 5/5: Perfect verbatim; minor omissions only if they don't impact meaning.

# 2. **Editorial Smoothness**:
#    - Ensure joins between selected segments read naturally (no abrupt grammar breaks).
#    - Reduce filler (e.g., "um", repetitions) by skipping segments, but keep flow syntactically sound.
#    - Aim for 5/5: Clean, natural flow; at worst, minor roughness (4/5).

# 3. **Narrative Flow & Continuity**:
#    - Maintain logical progression: Selected segments must form a coherent story, not a patchwork.
#    - Check semantic similarity: Adjacent segments should be thematically linked (e.g., avoid jumps from one topic to unrelated; simulate cosine similarity >0.32 by ensuring related ideas).
#    - If cosine-like similarity <0.32 (e.g., abrupt topic shift), it's an automatic failure—adjust by including bridge segments.
#    - Avoid repetitions; justify any timestamp jumps with continuity.
#    - Aim for 5/5: Logical throughout; allow one minor issue (4/5).

# 4. **Segment Boundaries**:
#    - Cut only at natural, complete thoughts: Start/end selections at full sentence boundaries.
#    - No truncated clauses (e.g., avoid mid-sentence ends like "During.").
#    - Aim for 5/5: Clean starts/ends; minor edges ok (4/5).

# 5. **Ending Completeness**:
#    - The supercut must feel finished: End on a full sentence that resolves or concludes the main idea.
#    - No dangling concepts or abrupt cut-offs.
#    - Aim for 5/5: Clear, intentional ending; slightly abrupt but complete (4/5).

# 6. **Length & Information Density**:
#    - Total selected duration must be {length} ±5% of original (prioritize meaningful content over filler).
#    - Calculate precisely: Sum (end - start)/1000 for selected segments.
#    - If outside range, adjust by adding/removing low-priority segments.
#    - Focus on dense, key info (e.g., core arguments, examples, conclusions).
#    - Aim for 5/5: Within range, clear & dense; slightly off ok (4/5).

# ### STEP-BY-STEP SELECTION PROCESS:
# 1. **Analyze Narrative Arc**: Read the full input. Identify core elements (e.g., intro, historical examples, contrasts, irony/conclusion in this Tucker Carlson-style podcast).
# 2. **Prioritize Segments**: 
#    - Must-keep: Key thesis statements, quotes, historical facts, ironic twists.
#    - Optional/skip: Filler, tangents, promotions (e.g., YouTube plugs).
#    - Preserve order: Select IDs in ascending sequence (gaps ok if flow holds).
# 3. **Iterate for Length**: Start with essential segments; add/remove to hit {target_duration_seconds:.1f}s (±10% tolerance).
# 4. **Self-Evaluate**: Mentally score your selection against the 6 criteria. If any <4/5, revise (e.g., add bridges for continuity, extend ending).
# 5. **Finalize**: Ensure overall consistent 4+/5; regenerate mentally if needed.

# ### OUTPUT FORMAT:
# Return ONLY a JSON object with the list of chosen IDs (in order selected).
# Example:
# {{
#   "selected_ids": [0, 1, 2, 5, 10]
# }}
# No explanations or additional text—strict JSON only.
# """

# # ── 1. Utilities ─────────────────────────────────────────────────────────────
# def to_mp3(src_path: str, dst_path: str):
#     try:
#         audio = AudioSegment.from_file(src_path)
#         audio.export(dst_path, format="mp3", bitrate="192k")
#         print(f"Converted to MP3: {dst_path}")
#     except Exception as e:
#         raise RuntimeError(f"MP3 Conversion Error: {e}")

# def is_video(file_path: str) -> bool:
#     video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"}
#     return Path(file_path).suffix.lower() in video_exts

# def create_podcast_supercut(
#     audio_path: str,
#     target_percent: float,
#     output_dir: str,
#     temp_dir: str,
#     gap_ms: int = 200,
#     head_padding_ms: int = 30, 
#     tail_padding_ms: int = 50,
#     output_filename: str = "cropped_audio"
# ) -> dict:
    
#     if not (0 < target_percent <= 100):
#         raise ValueError("target_percent must be between 0 and 100 (exclusive on 0)")

#     # ── Setup ────────────────────────────────────────────────────────────────
#     load_dotenv()
#     aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
#     openai_client = OpenAI()

#     # Normalize all paths to absolute paths immediately
#     audio_path_obj = Path(audio_path).resolve()
#     output_dir_path = Path(output_dir).resolve()
#     temp_dir_path = Path(temp_dir).resolve()
    
#     output_dir_path.mkdir(parents=True, exist_ok=True)
#     temp_dir_path.mkdir(parents=True, exist_ok=True)

#     base_name = audio_path_obj.stem
#     input_ext = audio_path_obj.suffix.lower()
#     is_video_flag = is_video(str(audio_path_obj))
    
#     working_audio_path = temp_dir_path / f"{base_name}_working.mp3"
#     to_mp3(str(audio_path_obj), str(working_audio_path))

#     # ── 2. Transcription ─────────────────────────────────────────────────────
#     config = aai.TranscriptionConfig(speaker_labels=True, punctuate=True, format_text=True)
#     transcript = aai.Transcriber().transcribe(str(working_audio_path), config)

#     sentences = transcript.get_sentences()
#     structured_segments = [
#         FinalOutput(
#             segment_id=idx,
#             speaker=s.speaker or "",
#             start=s.start,
#             end=s.end,
#             text=s.text
#         ).model_dump() for idx, s in enumerate(sentences)
#     ]

#     full_json_path = output_dir_path / f"{base_name}_full.json"
#     with open(full_json_path, "w", encoding="utf-8") as f:
#         json.dump(structured_segments, f, indent=2, ensure_ascii=False)

#     # ── 3. LLM Selection ─────────────────────────────────────────────────────
#     simplified_input = [{"segment_id": s["segment_id"], "text": s["text"]} for s in structured_segments]
#     original_duration_sec = (structured_segments[-1]["end"] - structured_segments[0]["start"]) / 1000.0
#     target_duration_sec = original_duration_sec * (target_percent / 100)

#     prompt = SYSTEM_PROMPT_TEMPLATE.format(
#         length=target_percent,
#         original_duration_seconds=original_duration_sec,
#         target_duration_seconds=target_duration_sec,
#         original_text=json.dumps(simplified_input, indent=1)
#     )

#     response = openai_client.chat.completions.create(
#         model="gpt-4o", 
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": "Analyze and return selected_ids."}
#         ],
#         response_format={"type": "json_object"}
#     )

#     ai_output = json.loads(response.choices[0].message.content)
#     selected_ids = ai_output.get("selected_ids", [])
#     ai_selected_segments = [s for s in structured_segments if s["segment_id"] in selected_ids]

#     # ── 4. Fix Fragmented Sentences ─────────────────────────────────────────
#     fixed_segments = []
#     processed = set()
#     for seg in ai_selected_segments:
#         if seg["segment_id"] in processed: continue
#         fixed_segments.append(seg)
#         processed.add(seg["segment_id"])
        
#         check = seg
#         while check["text"].strip() and check["text"].strip()[-1] not in [".", "!", "?"]:
#             next_id = check["segment_id"] + 1
#             if next_id >= len(structured_segments): break
#             next_seg = structured_segments[next_id]
#             if next_seg["segment_id"] not in processed:
#                 fixed_segments.append(next_seg)
#                 processed.add(next_seg["segment_id"])
#             check = next_seg

#     fixed_segments.sort(key=lambda x: x["segment_id"])
#     edited_json_path = output_dir_path / f"{base_name}_edited.json"
#     with open(edited_json_path, "w", encoding="utf-8") as f:
#         json.dump(fixed_segments, f, indent=2, ensure_ascii=False)

#     # ── 5. Create Final Media (Improved FFmpeg Logic) ────────────────────────
#     final_output_path = output_dir_path / f"{output_filename}{input_ext}"
#     clips_subdir = temp_dir_path / "clips_cache"
#     clips_subdir.mkdir(parents=True, exist_ok=True)
    
#     clip_filenames = []
#     try:
#         # STEP 1: Cut all segments
#         for idx, seg in enumerate(fixed_segments):
#             clip_name = f"clip_{idx:04d}{input_ext}"
#             clip_path = clips_subdir / clip_name
#             clip_filenames.append(clip_name)
            
#             start_s = max(0, (seg["start"] - head_padding_ms) / 1000.0)
#             duration_s = ((seg["end"] + tail_padding_ms) - (seg["start"] - head_padding_ms)) / 1000.0
            
#             cmd = [
#                 "ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(audio_path_obj),
#                 "-t", f"{duration_s:.3f}", "-avoid_negative_ts", "make_zero"
#             ]
#             cmd += ["-c", "copy"] if is_video_flag else ["-vn", "-c:a", "copy"]
#             cmd.append(str(clip_path))
            
#             subprocess.run(cmd, check=True, capture_output=True)

#         # STEP 2: Stitch segments using relative paths (fixes Windows 4294967294 error)
#         if clip_filenames:
#             concat_file = clips_subdir / "concat_list.txt"
#             with open(concat_file, "w", encoding="utf-8") as f:
#                 for name in clip_filenames:
#                     f.write(f"file '{name}'\n")

#             # Run FFmpeg from WITHIN the clips directory so paths are simple
#             stitch_cmd = [
#                 "ffmpeg", "-y", "-f", "concat", "-safe", "0",
#                 "-i", "concat_list.txt", "-c", "copy", str(final_output_path)
#             ]
#             subprocess.run(stitch_cmd, check=True, capture_output=True, cwd=str(clips_subdir))
            
#     finally:
#         # Cleanup
#         for name in clip_filenames:
#             cp = clips_subdir / name
#             if cp.exists(): cp.unlink()
#         if (clips_subdir / "concat_list.txt").exists(): (clips_subdir / "concat_list.txt").unlink()
#         if working_audio_path.exists(): working_audio_path.unlink()

#     return {
#         "status": "success",
#         "final_media": str(final_output_path),
#         "edited_json": str(edited_json_path),
#         "full_transcript": str(full_json_path),
#     }

# # ── MAIN SECTION ────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     MAIN_OUTPUT_DIR = r"app/files/test3"
#     MAIN_TEMP_DIR = r"app/files/temp"
#     INPUT_FILE = r"app/files/videoplayback.mp4" 

#     try:
#         val = input("Enter percentage to keep (e.g. 20): ")
#         percent = float(val)

#         result = create_podcast_supercut(
#             audio_path=INPUT_FILE,
#             target_percent=percent,
#             output_dir=MAIN_OUTPUT_DIR,
#             temp_dir=MAIN_TEMP_DIR
#         )

#         print("\n" + "="*50)
#         print("PROCESS COMPLETED SUCCESSFULLY")
#         print(json.dumps(result, indent=2))
#         print("="*50)

#     except Exception as e:
#         # If capture_output was used, print the stderr for better debugging
#         if hasattr(e, 'stderr') and e.stderr:
#             print(f"FFmpeg Error: {e.stderr.decode()}")
#         print(f"Pipeline failed: {e}")