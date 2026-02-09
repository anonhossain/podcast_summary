############################### This version works perfectly perfect ###############################

# import os
# import json
# import assemblyai as aai
# from pydantic import BaseModel, Field
# from typing import List
# from dotenv import load_dotenv
# from openai import OpenAI
# from pydub import AudioSegment

# # --- 1. SETUP & CONFIGURATION ---
# load_dotenv()
# aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
# client = OpenAI() # Uses OPENAI_API_KEY from .env

# # Paths
# INPUT_AUDIO = r"app/files/The Money Expert.mp3"
# BASE_DIR = r"app/files/test2"
# os.makedirs(BASE_DIR, exist_ok=True)

# # Schema for AssemblyAI processing
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

# # --- 2. STEP 1: TRANSCRIPTION ---
# def run_transcription(audio_path):
#     print(f"--- Starting Transcription for {audio_path} ---")
#     config = aai.TranscriptionConfig(speaker_labels=True, punctuate=True, format_text=True)
#     transcript = aai.Transcriber().transcribe(audio_path, config)
    
#     sentences = transcript.get_sentences()
#     structured_data = []
    
#     for index, sentence in enumerate(sentences):
#         segment = FinalOutput(
#             segment_id=index,
#             speaker=sentence.speaker,
#             start=sentence.start,
#             end=sentence.end,
#             text=sentence.text
#         )
#         structured_data.append(segment.model_dump())
    
#     file_basename = os.path.splitext(os.path.basename(audio_path))[0]
#     json_path = os.path.join(BASE_DIR, f"{file_basename}.json")
    
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(structured_data, f, indent=4)
    
#     print(f"Transcription complete: {json_path}")
#     return json_path, structured_data

# # --- 3. STEP 2: AI SELECTION ---
# def get_ai_selection(json_data, target_percent):
#     print(f"--- Asking AI to select {target_percent}% of content ---")
    
#     # Prep simplified data for AI tokens
#     simplified_input = [{"segment_id": s["segment_id"], "text": s["text"]} for s in json_data]
    
#     first_start = json_data[0].get("start", 0)
#     last_end = json_data[-1].get("end", 0)
#     total_dur = (last_end - first_start) / 1000
#     target_dur = total_dur * (target_percent / 100)

#     formatted_prompt = SYSTEM_PROMPT_TEMPLATE.format(
#         length=target_percent,
#         original_duration_seconds=total_dur,
#         target_duration_seconds=target_dur,
#         original_text=json.dumps(simplified_input, indent=1)
#     )

#     response = client.chat.completions.create(
#         model="gpt-5.2",
#         messages=[
#             {"role": "system", "content": formatted_prompt},
#             {"role": "user", "content": "Analyze and return selected_ids."}
#         ],
#         response_format={ "type": "json_object" },
#         reasoning_effort="medium",
#         verbosity="low"
#     )

#     ai_output = json.loads(response.choices[0].message.content)
#     selected_ids = ai_output.get("selected_ids", [])
    
#     # Filter original data by selected IDs
#     final_selection = [s for s in json_data if s["segment_id"] in selected_ids]
#     return final_selection

# def fix_fragmented_sentences(final_selection, all_segments):
#     """
#     Force-includes all segments until a terminal punctuation is found.
#     This fixes 'Editorial Smoothness' by ensuring thoughts are completed.
#     """
#     fixed_list = []
#     selected_ids = {s['segment_id'] for s in final_selection}
    
#     # Track which IDs we've already handled to avoid duplicates
#     processed_ids = set()

#     for seg in final_selection:
#         curr_id = seg['segment_id']
#         if curr_id in processed_ids:
#             continue
            
#         # Add the AI-selected segment
#         fixed_list.append(seg)
#         processed_ids.add(curr_id)

#         # RECURSIVE CHECK: If it doesn't end in . ! or ?, grab the next one
#         check_seg = seg
#         while check_seg['text'].strip()[-1] not in ['.', '!', '?']:
#             next_id = check_seg['segment_id'] + 1
#             if next_id < len(all_segments):
#                 check_seg = all_segments[next_id]
#                 if check_seg['segment_id'] not in processed_ids:
#                     print(f"Repairing Boundary: Adding ID {next_id} to finish sentence.")
#                     fixed_list.append(check_seg)
#                     processed_ids.add(check_seg['segment_id'])
#             else:
#                 break
                
#     # Always sort to maintain original audio order
#     return sorted(fixed_list, key=lambda x: x['segment_id'])

# def create_audio_crop(original_audio_path, selected_segments, output_path, gap_ms=150):
#     audio = AudioSegment.from_file(original_audio_path)
#     combined_audio = AudioSegment.empty()
    
#     # Use a slightly shorter gap for better narrative flow
#     silence = AudioSegment.silent(duration=gap_ms, frame_rate=audio.frame_rate)
    
#     for i, seg in enumerate(selected_segments):
#         # ADD 150ms TAIL: Prevents clipping the 'decay' of the last word
#         # ADD 50ms HEAD: Prevents 'clipping' the start of the first word
#         start_time = max(0, seg['start'] - 50)
#         end_time = seg['end'] + 150
        
#         clip = audio[start_time:end_time]
        
#         # Apply a micro-fade (50ms) to eliminate 'digital pops' at join points
#         clip = clip.fade_in(50).fade_out(50)
        
#         combined_audio += clip
        
#         # Check if the NEXT segment is sequential. 
#         # Only add silence if there is a 'jump' in the narrative.
#         if i < len(selected_segments) - 1:
#             next_seg = selected_segments[i+1]
#             if next_seg['segment_id'] > seg['segment_id'] + 1:
#                 combined_audio += silence

#     combined_audio.export(output_path, format="mp3", bitrate="192k")


# # --- 5. MAIN EXECUTION ---
# if __name__ == "__main__":
#     try:
#         val = input("Enter percentage to keep (e.g. 20): ")
#         target_percent = float(val)
        
#         if not (0 < target_percent <= 100):
#             print("Error: Percent must be between 1 and 100")
#         else:
#             # A. Transcribe original audio
#             json_file_path, structured_json = run_transcription(INPUT_AUDIO)
            
#             # B. AI makes initial selection based on narrative
#             ai_selected_segments = get_ai_selection(structured_json, target_percent)
            
#             # C. THE FIXER: Process AI selection to prevent fragmented sentences
#             # We pass both the selection AND the full master list (structured_json)
#             final_segments = fix_fragmented_sentences(ai_selected_segments, structured_json)
            
#             # D. Save Edited JSON (now containing fixed fragments)
#             edited_json_path = os.path.join(BASE_DIR, f"edited_{os.path.basename(json_file_path)}")
#             with open(edited_json_path, 'w', encoding='utf-8') as f:
#                 json.dump(final_segments, f, indent=4)
            
#             # E. Final Audio Crop (using our fixed segments and adding gaps)
#             final_audio_name = os.path.join(BASE_DIR, "cropped_audio.mp3")
#             create_audio_crop(INPUT_AUDIO, final_segments, final_audio_name, gap_ms=200)
            
#             print("\n" + "="*30)
#             print("PIPELINE COMPLETE - Fragmentation Fixed!")
#             print(f"Final Audio: {final_audio_name}")
#             print("="*30)

#     except Exception as e:
#         print(f"Pipeline Failed: {e}")





########################################## Version 2 ###########################################



import os
import json
import assemblyai as aai
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment


class FinalOutput(BaseModel):
    segment_id: int
    speaker: str
    start: int
    end: int
    text: str


SYSTEM_PROMPT_TEMPLATE = """### ROLE: High-Precision Narrative Architect for Podcast Supercut
You are an expert editor tasked with creating a condensed, coherent supercut from podcast transcript segments. Your selections must form a natural, flowing narrative that feels like a seamless edit of the original audio, not a rewrite. Prioritize verbatim fidelity, smooth transitions, and overall coherence to meet strict evaluation criteria (all must score 4/5 or higher).

### CRITICAL RULES (5/5 SCORE TARGET):
1. **BOUNDARY INTEGRITY:** You are FORBIDDEN from ending a selection on a segment that does not conclude a sentence. If a thought spans multiple IDs, you MUST take the entire block.
2. **VERBATIM FIDELITY:** Do not rewrite or paraphrase.

### OBJECTIVE:
- Select a contiguous or near-contiguous sequence of `segment_id`s (in original order, no reordering) to create a shortened version.
- Target total duration: Exactly {length} ±5% of the original (approx. {target_duration_seconds:.1f}s out of {original_duration_seconds:.1f}s).
- The supercut must preserve the core narrative arc (e.g., introduction of topic, key examples, contrasts, conclusion) while compressing by removing filler, redundancies, and less essential details.
- Ensure the output passes all evaluation categories (detailed below) at 4/5 or higher. If a selection would fail any, adjust until it passes.

### INPUT DATA:
- The transcript is provided as a list of segments: Each has `segment_id` (sequential integer) and `text` (verbatim sentence or clause from the audio).
- Original total duration: {original_duration_seconds:.1f}s.
- Simplified segments for selection: {original_text}

### SELECTION RULES (MUST FOLLOW TO PASS CRITERIA):
1. **Editorial Fidelity (Transcript-Level)**: 
   - Select only existing segments verbatim—no paraphrasing, adding words, or modifying text.
   - All selected text must appear exactly as in the input (perfect alignment).
   - Aim for 5/5: Perfect verbatim; minor omissions only if they don't impact meaning.

2. **Editorial Smoothness**:
   - Ensure joins between selected segments read naturally (no abrupt grammar breaks).
   - Reduce filler (e.g., "um", repetitions) by skipping segments, but keep flow syntactically sound.
   - Aim for 5/5: Clean, natural flow; at worst, minor roughness (4/5).

3. **Narrative Flow & Continuity**:
   - Maintain logical progression: Selected segments must form a coherent story, not a patchwork.
   - Check semantic similarity: Adjacent segments should be thematically linked (e.g., avoid jumps from one topic to unrelated; simulate cosine similarity >0.32 by ensuring related ideas).
   - If cosine-like similarity <0.32 (e.g., abrupt topic shift), it's an automatic failure—adjust by including bridge segments.
   - Avoid repetitions; justify any timestamp jumps with continuity.
   - Aim for 5/5: Logical throughout; allow one minor issue (4/5).

4. **Segment Boundaries**:
   - Cut only at natural, complete thoughts: Start/end selections at full sentence boundaries.
   - No truncated clauses (e.g., avoid mid-sentence ends like "During.").
   - Aim for 5/5: Clean starts/ends; minor edges ok (4/5).

5. **Ending Completeness**:
   - The supercut must feel finished: End on a full sentence that resolves or concludes the main idea.
   - No dangling concepts or abrupt cut-offs.
   - Aim for 5/5: Clear, intentional ending; slightly abrupt but complete (4/5).

6. **Length & Information Density**:
   - Total selected duration must be {length} ±5% of original (prioritize meaningful content over filler).
   - Calculate precisely: Sum (end - start)/1000 for selected segments.
   - If outside range, adjust by adding/removing low-priority segments.
   - Focus on dense, key info (e.g., core arguments, examples, conclusions).
   - Aim for 5/5: Within range, clear & dense; slightly off ok (4/5).

### STEP-BY-STEP SELECTION PROCESS:
1. **Analyze Narrative Arc**: Read the full input. Identify core elements (e.g., intro, historical examples, contrasts, irony/conclusion in this Tucker Carlson-style podcast).
2. **Prioritize Segments**: 
   - Must-keep: Key thesis statements, quotes, historical facts, ironic twists.
   - Optional/skip: Filler, tangents, promotions (e.g., YouTube plugs).
   - Preserve order: Select IDs in ascending sequence (gaps ok if flow holds).
3. **Iterate for Length**: Start with essential segments; add/remove to hit {target_duration_seconds:.1f}s (±10% tolerance).
4. **Self-Evaluate**: Mentally score your selection against the 6 criteria. If any <4/5, revise (e.g., add bridges for continuity, extend ending).
5. **Finalize**: Ensure overall consistent 4+/5; regenerate mentally if needed.

### OUTPUT FORMAT:
Return ONLY a JSON object with the list of chosen IDs (in order selected).
Example:
{{
  "selected_ids": [0, 1, 2, 5, 10]
}}
No explanations or additional text—strict JSON only.
"""


def create_podcast_supercut(
    audio_path: str,
    target_percent: float,
    output_dir: str = "app/files/supercut_output",
    gap_ms: int = 200,
    fade_ms: int = 50,
    head_padding_ms: int = 50,
    tail_padding_ms: int = 150,
    output_filename: str = "cropped_audio.mp3"
) -> dict:
    """
    Single entry-point function that performs the complete podcast supercut pipeline:
    1. Transcribes the audio (AssemblyAI)
    2. Uses LLM to select best segments
    3. Fixes fragmented sentences
    4. Creates the final cropped audio file

    Returns a dictionary with paths and basic info.
    Raises exceptions on failure — backend should catch them.
    """
    if not (0 < target_percent <= 100):
        raise ValueError("target_percent must be between 0 and 100 (exclusive on 0)")

    # ── Setup ────────────────────────────────────────────────────────────────
    load_dotenv()
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    openai_client = OpenAI()  # reads OPENAI_API_KEY from env

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Transcription ─────────────────────────────────────────────────────
    config = aai.TranscriptionConfig(speaker_labels=True, punctuate=True, format_text=True)
    transcript = aai.Transcriber().transcribe(audio_path, config)

    sentences = transcript.get_sentences()
    structured_segments: List[dict] = []

    for idx, sentence in enumerate(sentences):
        seg = FinalOutput(
            segment_id=idx,
            speaker=sentence.speaker or "",
            start=sentence.start,
            end=sentence.end,
            text=sentence.text
        )
        structured_segments.append(seg.model_dump())

    # Save full transcription (optional but useful for debugging)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    full_json_path = os.path.join(output_dir, f"{base_name}_full.json")
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_segments, f, indent=2, ensure_ascii=False)

    # ── 2. Prepare prompt & ask LLM for selection ────────────────────────────
    simplified_input = [{"segment_id": s["segment_id"], "text": s["text"]} for s in structured_segments]

    first_start = structured_segments[0].get("start", 0)
    last_end = structured_segments[-1].get("end", 0)
    original_duration_sec = (last_end - first_start) / 1000.0
    target_duration_sec = original_duration_sec * (target_percent / 100)

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        length=target_percent,
        original_duration_seconds=original_duration_sec,
        target_duration_seconds=target_duration_sec,
        original_text=json.dumps(simplified_input, indent=1)
    )

    response = openai_client.chat.completions.create(
        model="gpt-5.2",  # ← keep exactly your original model name
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze and return selected_ids."}
        ],
        response_format={"type": "json_object"},
        reasoning_effort="medium",
        verbosity="low"
    )

    ai_output = json.loads(response.choices[0].message.content)
    selected_ids = ai_output.get("selected_ids", [])

    ai_selected_segments = [s for s in structured_segments if s["segment_id"] in selected_ids]

    # ── 3. Fix fragmented sentences ─────────────────────────────────────────
    fixed_segments = []
    processed = set()

    for seg in ai_selected_segments:
        curr_id = seg["segment_id"]
        if curr_id in processed:
            continue

        fixed_segments.append(seg)
        processed.add(curr_id)

        check = seg
        while check["text"].strip() and check["text"].strip()[-1] not in [".", "!", "?"]:
            next_id = check["segment_id"] + 1
            if next_id >= len(structured_segments):
                break
            next_seg = structured_segments[next_id]
            if next_seg["segment_id"] not in processed:
                print(f"Repairing boundary: adding segment {next_id}")
                fixed_segments.append(next_seg)
                processed.add(next_seg["segment_id"])
            check = next_seg

    # Maintain original order
    fixed_segments.sort(key=lambda x: x["segment_id"])

    # Save edited selection
    edited_json_path = os.path.join(output_dir, f"{base_name}_edited.json")
    with open(edited_json_path, "w", encoding="utf-8") as f:
        json.dump(fixed_segments, f, indent=2, ensure_ascii=False)

    # ── 4. Create final audio ───────────────────────────────────────────────
    audio = AudioSegment.from_file(audio_path)
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=gap_ms, frame_rate=audio.frame_rate)

    for i, seg in enumerate(fixed_segments):
        start = max(0, seg["start"] - head_padding_ms)
        end = seg["end"] + tail_padding_ms
        clip = audio[start:end]

        # Apply micro-fades to prevent pops
        clip = clip.fade_in(fade_ms).fade_out(fade_ms)

        combined += clip

        # Add silence only when there's a real jump
        if i < len(fixed_segments) - 1:
            next_seg = fixed_segments[i + 1]
            if next_seg["segment_id"] > seg["segment_id"] + 1:
                combined += silence

    final_audio_path = os.path.join(output_dir, output_filename)
    combined.export(final_audio_path, format="mp3", bitrate="192k")

    # ── Return result info ──────────────────────────────────────────────────
    return {
        "status": "success",
        "original_duration_seconds": round(original_duration_sec, 1),
        "target_duration_seconds": round(target_duration_sec, 1),
        "final_audio_path": final_audio_path,
        "edited_json_path": edited_json_path,
        "full_transcript_json": full_json_path,
        "number_of_segments_original": len(structured_segments),
        "number_of_segments_final": len(fixed_segments),
    }


# ── Only used when running this file directly (for testing) ────────────────
if __name__ == "__main__":
    try:
        val = input("Enter percentage to keep (e.g. 20): ")
        percent = float(val)

        result = create_podcast_supercut(
            audio_path=r"app/files/Tucker.mp3",
            target_percent=percent,
            output_dir=r"app/files/test3",
            gap_ms=200
        )

        print("\n" + "="*50)
        print("SUPERCUT CREATED SUCCESSFULLY")
        print(json.dumps(result, indent=2))
        print("="*50)

    except Exception as e:
        print(f"Pipeline failed: {e}")