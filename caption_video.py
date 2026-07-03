import os
import sys
import csv
import cv2
import torch
import subprocess
import tempfile
from datetime import datetime
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# -----------------------------
# Configuration
# -----------------------------
output_dir = "output"
device = "cuda" if torch.cuda.is_available() else "cpu"
keyword_len = 3

SYSTEM_CONTENT = (
    "You are an expert long-video summarization assistant."
    "Your task is to analyze long-form egocentric (first-person) video content"
    "Be concise and follow the output format exactly."
)

import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# -----------------------------
# Load Models
# -----------------------------
# print("Loading Qwen3-VL...")
vlm_model_id = "Qwen/Qwen3-VL-2B-Instruct"
vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
    vlm_model_id,
    dtype=torch.float16,
    device_map="auto"
)

# -----------------------------
# FFmpeg clip creation
# -----------------------------
def create_video_clip_ffmpeg(video_path, start_frame, end_frame, fps):
    start_time = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps

    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start_time), "-t", str(duration), "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "28",
        tmp_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_path

# -----------------------------
# Caption + Keywords (VIDEO)
# -----------------------------
def caption_segment(video_path, start_idx, end_idx, fps):
    clip_path = create_video_clip_ffmpeg(video_path, start_idx, end_idx, fps)
    
    if not os.path.exists(clip_path):
        return "", ""

    try:
        messages = [
            {"role": "system", "content": SYSTEM_CONTENT},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{clip_path}"},
                    {
                        "type": "text",
                        "text": (
                            f"You are individual C (the camera wearer), with others represented as O."
                            f"Describe this video segment in exactly three sentences for caption.\n"
                            f"All narration must be from C's first-person perspective. (e.g., C walks into the house.)"
                            f"Do NOT switch to third-person narration."
                            "Output Format:\n"
                            "Caption: <text>\n"
                            "Return EXACTLY this format."
                        )
                    }
                ],
            }
        ]

        text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        inputs = vlm_processor(
            text=text, images=images, videos=videos, return_tensors="pt", **video_kwargs
        ).to(vlm_model.device)

        generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
        output_ids = generated_ids[0][inputs["input_ids"].shape[-1]:]
        result = vlm_processor.decode(output_ids, skip_special_tokens=True)
        
        # for line in result.split("\n"):
        #     if line.lower().startswith("caption:"):
        #         caption = line.split(":", 1)[1].strip()
        #     elif line.lower().startswith("keywords:"):
        #         raw_keywords = line.split(":", 1)[1]
        #         keywords_list = [k.strip().lower() for k in raw_keywords.split(",") if k.strip()]
        #         keywords = ",".join(keywords_list[:keyword_len])

        return result
    finally:
        if os.path.exists(clip_path):
            os.remove(clip_path)

# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python caption_video.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    base = os.path.basename(video_path).split(".")[0]

    # Get FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Locate base tracking layout inside output/metadata
    metadata_dir = os.path.join(output_dir, 'metadata')
    csv_in_path = f"{metadata_dir}/{base}.csv"

    if not os.path.exists(csv_in_path):
        raise FileNotFoundError(f"Missing layout CSV data: {csv_in_path}. Run segment_video.py first.")

    rows = []
    with open(csv_in_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)

    # Deduplicate segments to prevent running heavy model queries on redundant items
    unique_segments = sorted(list(set((int(r[1]), int(r[2])) for r in rows)))
    
    # print("Generating captions for video blocks...")
    segment_captions = {}

    start = datetime.now()

    for start_idx, end_idx in unique_segments:
        caption = caption_segment(video_path, start_idx, end_idx, fps)
        segment_captions[(start_idx, end_idx)] = caption

    print((datetime.now() - start).total_seconds())
    # Save complete populated columns data to the destination final CSV
    # md_out_path = f"{metadata_dir}/{base}.csv"

    # with open(md_out_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["path", "start", "end", "index", "caption"])
    #     for r in rows:
    #         path, start, end, index = r[:4]
    #         seg_key = (int(start), int(end))
    #         caption = segment_captions.get(seg_key, "")
    #         writer.writerow([path, start, end, index, caption])

    # print(f"Captions successfully added! Consolidated metadata saved to: {md_out_path}")

if __name__ == "__main__":
    # start = datetime.now()
    main()
    # print(f"Duration: {datetime.now() - start}")