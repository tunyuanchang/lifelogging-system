import numpy as np
import pandas as pd
import spacy
import argparse
import open_clip
import torch
import torch.nn.functional as F
import os
from datetime import datetime
from collections import defaultdict

# -----------------------------
# MODEL SETUP
# -----------------------------
model_name = "ViT-B-32"
pretrained = "laion2b_s34b_b79k"
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, _ = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)
clip_model.to(device).eval()

tokenizer = open_clip.get_tokenizer(model_name)
sentence_model = spacy.load("en_core_web_sm")

# -----------------------------
# CONFIG
# -----------------------------
ROOT_PATH = "output"
IDX_DIR = "/mnt/ssd_nvme0/tunyuan/egoschema_500/"

fps_list = np.array([30.0, 15.0, 10.0, 6.0, 3.0, 2.0, 0.5, 0.25, 0.125])

fps_to_stride = {
    30.0: 1, 15.0: 2, 10.0: 3, 6.0: 5,
    3.0: 10, 2.0: 15, 0.5: 60, 0.25: 120, 0.125: 240
}

log_a = 0.0299
log_b = 0.9076
FPS_SCALE = log_a * np.log(fps_list) + log_b

FRAME_COST = 0.12

# -----------------------------
# IO
# -----------------------------
def load_index_list(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_segments(ts_file):
    data = np.loadtxt(ts_file)
    return data[:, 0].astype(int), data[:, 1].astype(int)


def load_embeddings(emb_file):
    return np.load(emb_file)


def load_metadata(starts, md_file):
    data = pd.read_csv(md_file)
    captions = data.loc[data["index"].isin(starts), "caption"].tolist()

    caption_list = []
    for cap in captions:
        doc = sentence_model(cap)
        caption_list.append([sent.text.strip() for sent in doc.sents])

    return data, caption_list


# -----------------------------
# UTILITIES
# -----------------------------
def compute_utility_visual(starts, ends, embeddings, method="sim"):
    embeds = torch.from_numpy(embeddings).float()

    if method == "sim":
        sims = F.cosine_similarity(embeds[:-1], embeds[1:], dim=1).cpu().numpy()
        return np.array([
            1.0 - sims[s:e].mean()
            for s, e in zip(starts, ends)
        ])

    elif method == "var":
        return np.array([
            ((embeds[s:e+1] - embeds[s:e+1].mean(dim=0)) ** 2)
            .sum(dim=1).mean().item()
            for s, e in zip(starts, ends)
        ])

    else:
        raise ValueError("Unknown method")


def compute_utility_textual(starts, ends, embeddings, caption_list):
    embeds = torch.from_numpy(embeddings).float().to(device)
    utilities = []

    for i, cap_sentences in enumerate(caption_list):
        cap = " ".join(cap_sentences)
        tokens = tokenizer(cap).to(device)

        with torch.no_grad():
            t_emb = clip_model.encode_text(tokens)
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)

        v_emb = embeds[starts[i]:ends[i]+1]
        sim = v_emb @ t_emb.T
        utilities.append(sim.max(dim=0).values.mean().item())

    return np.array(utilities)


# -----------------------------
# BUILD ITEMS
# -----------------------------
def build_visual_items(video_id, starts, ends, base_util):
    items = []

    for seg_id, (s, e) in enumerate(zip(starts, ends)):
        seg_len = e - s + 1

        for fps, scale in zip(fps_list, FPS_SCALE):
            stride = fps_to_stride[fps]
            sampled_len = ((seg_len - 1) // stride) + 1

            storage = sampled_len * FRAME_COST
            util = base_util[seg_id] * scale

            items.append({
                "video_id": video_id,
                "seg_id": seg_id,
                "start": s,
                "end": e,
                "modality": "video",
                "quality": fps,
                "stride": stride,
                "util": util,
                "storage": storage,
                "ratio": util / storage
            })

    return items


def build_textual_items(video_id, starts, ends, base_util, caption_list):
    items = []

    for seg_id in range(len(starts)):
        cap_len = len(caption_list[seg_id])
        storage = 0.01 * cap_len
        util = base_util[seg_id]

        items.append({
            "video_id": video_id,
            "seg_id": seg_id,
            "start": starts[seg_id],
            "end": ends[seg_id],
            "modality": "text",
            "quality": 100,
            "stride": 1,
            "util": util,
            "storage": storage,
            "ratio": util / (storage + 1e-8)
        })

    return items


# -----------------------------
# GREEDY (re-run each iteration)
# -----------------------------
def compute_max_storage(items):
    max_storage = {}
    seen = set()

    for it in items:
        key = (it["video_id"], it["seg_id"], it["modality"])

        max_storage[key] = max(
            max_storage.get(key, 0),
            it["storage"]
        )

    return sum(max_storage.values())

def greedy_select(items, budget):
    items = sorted(items, key=lambda x: x["ratio"], reverse=True)

    selected = []
    used = 0.0
    chosen_keys = {}

    for it in items:
        key = (it["video_id"], it["seg_id"], it["modality"])

        if key in chosen_keys:
            idx = chosen_keys[key]
            old = selected[idx]

            new_used = used - old["storage"] + it["storage"]

            if it["util"] > old["util"] and new_used <= budget:
                # replace
                selected[idx] = it
                used = new_used

            continue

        if used + it["storage"] > budget:
            continue

        selected.append(it)
        chosen_keys[key] = len(selected) - 1
        used += it["storage"]

        if used >= budget:
            break

    return sorted(selected, key=lambda x: (it["video_id"], it["seg_id"])), used

def remove_items(all_items, selected):
    # keep only selected segment keys
    selected_keys = set(
        (it["video_id"], it["seg_id"], it["modality"])
        for it in selected
    )

    # remain_vid = set(it["video_id"] for it in selected)
    # print(len(remain_vid))
    
    # map selected storage
    selected_map = {
        (it["video_id"], it["seg_id"], it["modality"]): it["storage"]
        for it in selected
    }

    remaining = []

    for it in all_items:
        key = (it["video_id"], it["seg_id"], it["modality"])

        if key not in selected_keys:
            continue

        # keep only lower-quality versions (optional)
        if it["storage"] <= selected_map[key]:
            remaining.append(it)

    return remaining

# -----------------------------
# EXPORT (SINGLE MERGED FILE)
# -----------------------------
def export_global_data(selected, metadata_map, video_rank):
    rows = []
    # embeds = []
    seen = set()

    for item in selected:
        vid = item["video_id"]

        df = metadata_map[vid]
        # emb = embeddings_map[vid]

        start = item["start"]
        end = item["end"]
        stride = item["stride"]

        for r in range(start, end + 1, stride):
            key = (vid, r)

            if key in seen:
                continue
            seen.add(key)

            row = df.iloc[r].copy()
            row["video_id"] = vid

            rows.append(row)
            # embeds.append(emb[r])

    final_df = pd.DataFrame(rows).reset_index(drop=True)
    # final_embeds = np.array([]) if len(embeds) == 0 else np.stack(embeds)
    
    if len(final_df) == 0:
        return final_df
    
    final_df["video_rank"] = final_df["video_id"].map(video_rank)

    sort_idx = final_df.sort_values(
        ["video_rank", "index"]
    ).index.to_numpy()

    final_df = final_df.iloc[sort_idx].reset_index(drop=True)
    # final_embeds = final_embeds[sort_idx]

    return final_df

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=-1)
    parser.add_argument("-c", "--storage", type=int, default=10000)
    parser.add_argument("-m", "--method", type=str, default="sim")

    args = parser.parse_args()

    IDX = args.idx
    STORAGE = args.storage
    METHOD = args.method

    if IDX == -1:
        LIST = os.path.join(IDX_DIR, 'subset_500.txt')
        FILENAME = f"{STORAGE}_{METHOD}_gd"
    else:
        LIST = os.path.join(IDX_DIR, f"subset_{args.idx}.txt")
        FILENAME = f"{STORAGE}_{METHOD}_gd_{IDX}"

    idx_list = load_index_list(LIST)

    all_items = []
    all_metadata = {}
    # all_embeddings = {}
    vid_rank = {}

    selected = []
    used_storage = 0.0
    
    # print(f"Processing {len(idx_list)} videos...")

    for step, video_id in enumerate(idx_list):
        try:
            ts_file = os.path.join(ROOT_PATH, "timestamp", f"{video_id}.txt")
            emb_file = os.path.join(ROOT_PATH, "embedding", f"{video_id}.npy")
            md_file = os.path.join(ROOT_PATH, "metadata", f"{video_id}.csv")

            starts, ends = load_segments(ts_file)
            embeddings = load_embeddings(emb_file)
            metadata_df, caption_list = load_metadata(starts, md_file)

            base_visual = compute_utility_visual(starts, ends, embeddings, METHOD)
            # base_text = compute_utility_textual(starts, ends, embeddings, caption_list)

            visual_items = build_visual_items(video_id, starts, ends, base_visual)
            # textual_items = build_textual_items(video_id, starts, ends, base_text, caption_list)

            all_metadata[video_id] = metadata_df
            # all_embeddings[video_id] = embeddings

            # add items
            all_items.extend(visual_items)

            max_storage = compute_max_storage(all_items)

            # greedy
            start = datetime.now()
            if max_storage > STORAGE:
                # print(f"{max_storage} / {STORAGE}")
                selected, used_storage = greedy_select(all_items, STORAGE)
                all_items = remove_items(all_items, selected)
            else:
                used_storage = max_storage
            print((datetime.now() - start).total_seconds())

            # print(f"[{step+1}/{len(idx_list)}] video_id={video_id}")
            # print(f"  total items: {len(all_items)}")
            # print(f"  used storage: {used_storage:.2f}/{STORAGE}")

            vid_rank[video_id] = step

        except Exception as e:
            print(f"[ERROR] {video_id}: {e}")

    # print("Exporting merged output...")

    # final_df = export_global_data(
    #     selected,
    #     all_metadata,
    #     vid_rank
    # )

    # out_md_path = os.path.join("merged", f"{FILENAME}.csv")
    # final_df.to_csv(out_md_path, index=False)

    # out_emb_path = os.path.join("merged", f"{FILENAME}.npy")
    # np.save(out_emb_path, final_embeds)

    storage = compute_max_storage(selected)
    
    # print(f"Total selected frames: {len(final_df)}")
    # print(f"Storage used: {storage:.2f} / {STORAGE}")
    # print(f"Duration: {datetime.now() - start_time}")