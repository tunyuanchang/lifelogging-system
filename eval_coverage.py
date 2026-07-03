import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import sys


root_path = os.path.join("output", "embedding")


# -----------------------------
# Normalize
# -----------------------------
def normalize(x):
    x = torch.from_numpy(x).float()
    return F.normalize(x, dim=1)


# -----------------------------
# Coverage score
# -----------------------------
def coverage_score(A, B, chunk_size=4096):
    A = normalize(A)
    B = normalize(B)

    outs = []

    with torch.no_grad():
        for i in range(0, A.shape[0], chunk_size):
            A_chunk = A[i:i + chunk_size]
            sim = A_chunk @ B.T
            outs.append(sim.max(dim=1).values)

    return torch.cat(outs).mean().item()


# -----------------------------
# Build video -> indices
# -----------------------------
def build_video_index(df):
    idx_map = {}

    for i, vid in enumerate(df["video_id"].tolist()):
        idx_map.setdefault(vid, []).append(i)

    return idx_map


# -----------------------------
# Get all full video IDs from folder
# -----------------------------
def get_all_video_ids(folder_path):
    return sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(folder_path)
        if f.endswith(".npy")
    ])


# -----------------------------
# Load one video only
# -----------------------------
def load_video(video_id):
    path = os.path.join(root_path, f"{video_id}.npy")
    return np.load(path)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config>")
        sys.exit(1)

    config = sys.argv[1]

    # selected_emb_path = os.path.join("merged", f"{config}.npy")
    selected_csv_path = os.path.join("merged", f"{config}.csv")

    # selected_emb = np.load(selected_emb_path)
    selected_df = pd.read_csv(selected_csv_path)

    video_to_idx = build_video_index(selected_df)
    # print(video_to_idx)

    full_video_ids = get_all_video_ids(root_path)

    parts = config.split("_")
    parts += [""] * 3
    config = ",".join(parts[:3])

    # print(f"Loaded {config}")
    scores = []
    for vid in full_video_ids:
        if int(vid) not in video_to_idx:
            # print(f"{vid}: 0.0000")
            scores.append(0.0)
            continue

        full_v = load_video(vid)
        selected_indices = selected_df[selected_df["video_id"] == int(vid)]["index"].to_numpy()
        selected_v = full_v[selected_indices]

        score = coverage_score(full_v, selected_v)
        scores.append(score)

        # print(f"{vid}: {score:.4f}")

        del full_v

    print(f"{config},{sum(scores)/len(scores):.4f}")