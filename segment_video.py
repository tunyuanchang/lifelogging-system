import os
import sys
import math
import csv
import cv2
import numpy as np
import torch
import open_clip
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
output_dir = "output"
model_name = "ViT-B-32"
pretrained = "laion2b_s34b_b79k"
device = "cuda" if torch.cuda.is_available() else "cpu"

frame_skip = 1

import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Load Models
# -----------------------------
# print("Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)
clip_model.to(device).eval()

# -----------------------------
# Video Segmentation (CLIP)
# -----------------------------
def segment_video(video_path, sim_percentile=0.95, min_duration_sec=5.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        raise RuntimeError("Could not read FPS")

    min_frames = max(1, int(min_duration_sec * fps))

    cos_sims = []
    frame_indices = []
    frame_paths = []
    embeddings = []

    frame_idx = 0
    base = os.path.basename(video_path).split(".")[0]
    frame_dir = os.path.join(output_dir, "frames", base)
    os.makedirs(frame_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_indices.append(frame_idx)

            img = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_model.encode_image(img)
                emb /= emb.norm(dim=-1, keepdim=True)

            embeddings.append(emb.cpu().numpy())
            
            img_name = f"img_{frame_idx:06d}.jpg"
            img_path = os.path.join(frame_dir, img_name)
            if not os.path.isfile(img_path): 
                Image.fromarray(frame_rgb).save(img_path, format="JPEG", quality=100)
            frame_paths.append(img_path)

        frame_idx += 1

    cap.release()

    embeddings = np.vstack(embeddings)
    embeddings_tensor = torch.from_numpy(embeddings).float()

    sims = F.cosine_similarity(embeddings_tensor[:-1], embeddings_tensor[1:], dim=1)
    cos_sims = sims.cpu().numpy()
    frame_indices = np.array(frame_indices)

    if len(cos_sims) == 0:
        return [], frame_indices, frame_paths, embeddings, fps

    # Scene Detection
    n = len(cos_sims)
    k = max(0, min(n - 1, math.ceil(n * (1 - sim_percentile))))
    boundary = np.partition(cos_sims, k)[k]
    raw_cuts = np.where(cos_sims <= boundary)[0] + 1

    segments = []
    start_ptr = 0
    for cut_idx in raw_cuts:
        seg_len = frame_indices[cut_idx] - frame_indices[start_ptr]
        segments.append([frame_indices[start_ptr], frame_indices[cut_idx] - 1, seg_len])
        start_ptr = cut_idx

    seg_len = frame_indices[-1] - frame_indices[start_ptr] + 1
    segments.append([frame_indices[start_ptr], frame_indices[-1], seg_len])

    # Merge short segments
    merged = True
    while merged:
        merged = False
        for i, seg in enumerate(segments):
            if seg[2] >= min_frames:
                continue

            left = segments[i - 1] if i > 0 else None
            right = segments[i + 1] if i < len(segments) - 1 else None

            options = []
            if left:
                options.append(("left", cos_sims[left[1]]))
            if right:
                options.append(("right", cos_sims[right[0] - 1]))

            if not options:
                continue

            best = max(options, key=lambda x: x[1])

            if best[0] == "left":
                left_seg = segments[i - 1]
                new_seg = [left_seg[0], seg[1], left_seg[2] + seg[2]]
                segments[i - 1] = new_seg
                segments.pop(i)
            else:
                right_seg = segments[i + 1]
                new_seg = [seg[0], right_seg[1], seg[2] + right_seg[2]]
                segments[i] = new_seg
                segments.pop(i + 1)

            merged = True
            break

    return segments, frame_indices, frame_paths, embeddings, fps

# -----------------------------
# Save Segmentation Output Data
# -----------------------------
def save_outputs(segments, frame_indices, frame_paths, embeddings, video_path):
    base = os.path.basename(video_path).split(".")[0]
    
    # Create required directory targets
    metadata_dir = os.path.join(output_dir, 'metadata')
    embedding_dir = os.path.join(output_dir, 'embedding')
    timestamp_dir = os.path.join(output_dir, 'timestamp')
    
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(timestamp_dir, exist_ok=True)
    
    # 1. Save structural frame tracking layout as CSV inside metadata_dir
    csv_path = f"{metadata_dir}/{base}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "start", "end", "index"])
        
        for seg in segments:
            start_idx, end_idx, _ = seg
            for i, frame_idx in enumerate(frame_indices):
                if start_idx <= frame_idx <= end_idx:
                    writer.writerow([frame_paths[i], start_idx, end_idx, i])

    # 2. Save pure segment block metadata boundaries to a text file inside timestamp_dir
    ts_path = f"{timestamp_dir}/{base}.txt"
    with open(ts_path, "w") as f:
        for seg in segments:
            f.write(f"{seg[0]} {seg[1]} {seg[2]}\n")

    # 3. Save raw frame clip array embeddings matrix data inside embedding_dir
    emb_path = f"{embedding_dir}/{base}.npy"
    np.save(emb_path, embeddings)

    # print(f"Segmentation finished successfully.")
    # print(f" -> Base CSV structure saved to: {csv_path}")
    # print(f" -> Segment Timestamps saved to: {ts_path}")
    # print(f" -> Frame Embeddings saved to: {emb_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python segment_video.py <video_path> [sim_percentile] [min_duration]")
        sys.exit(1)

    video_path = sys.argv[1]
    sim_percentile = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
    min_duration = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    start = datetime.now()
    # print("Segmenting video...")
    segments, frame_indices, frame_paths, embeddings, fps = segment_video(
        video_path, sim_percentile, min_duration
    )
    print((datetime.now() - start).total_seconds())
   # save_outputs(segments, frame_indices, frame_paths, embeddings, video_path)

if __name__ == "__main__":
    # start = datetime.now()
    main()
    # print(f"Duration: {datetime.now() - start}")