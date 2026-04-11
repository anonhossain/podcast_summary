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
            timeout=1200.0,
            
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
            audio_path=r"C:\files\podcast\app\files\demo\Colbert ATTACKS CBS Over Pulled Interview With James Talarico!.mp3",
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