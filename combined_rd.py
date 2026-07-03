import numpy as np
import pandas as pd
import argparse
import open_clip
import random
import torch
import os
import re
from PIL import Image
from datetime import datetime
import json

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()

# -----------------------------
# MODEL SETUP (CLIP — visual utility only)
# -----------------------------
clip_model_name = "ViT-B-32"
pretrained = "laion2b_s34b_b79k"
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained)
clip_model.to(device).eval()

# -----------------------------
# CONFIG
# -----------------------------
ROOT_PATH = "output"
IDX_DIR = "/mnt/ssd_nvme0/tunyuan/egoschema_500/"
QA_DIR = "/mnt/ssd_nvme0/tunyuan/egoschema_500/"

fps_list = np.array([30.0, 15.0, 10.0, 6.0, 3.0, 2.0, 0.5, 0.25, 0.125])
fps_to_stride = {
    30.0: 1, 15.0: 2, 10.0: 3, 6.0: 5,
    3.0: 10, 2.0: 15, 0.5: 60, 0.25: 120, 0.125: 240
}

log_a = 0.0299
log_b = 0.9076
FPS_SCALE = log_a * np.log(fps_list) + log_b
FRAME_COST = 0.12
MAX_LENGTH = 400

TRIALS = 1
RESULT_FILE = 'result_exp_comb.txt'

# -----------------------------
# QA PROMPTS
# -----------------------------
SYSTEM_PROMPT = (
    "You are a long-video question answering assistant. "
    "Be concise and follow the output format exactly."
)

USER_TEMPLATE = (
    "You are individual C, with others represented as O."
    "In your responses to questions about past events, it is vital to provide not only the key"
    "You are presented with textual descriptions and visual keyframes of a video clip."
    "Your task is to answer a question related to this video, choosing the correct option out of five possible answers."
    "It is crucial that you imagine the full visual scene as vividly as possible to enhance the accuracy of your response."
    "Please provide a concise one-sentence explanation for your chosen answer."
    "If you are uncertain about the correct option, select the one that seems closest to being correct."
    "Output format (MUST FOLLOW EXACTLY):\n"
    "<answer>A</answer>\n\n"
    "<reasoning>brief reasoning</reasoning>\n"
    "Only A, B, C, D, or E are allowed inside <answer>.\n"
)

ANSWER_FIX_TEMPLATE = (
    "Output ONLY the final answer using the exact format:\n"
    "<answer>A</answer>\n"
    "where A is one of A, B, C, D, or E."
)

SYSTEM_CONTENT = [{"type": "text", "text": SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE + "\n" + ANSWER_FIX_TEMPLATE}]

# ==============================
# IO
# ==============================
def load_index_list(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    video_rank = {vid: i for i, vid in enumerate(lines)}
    return lines, video_rank


def load_segments(ts_file):
    data = np.loadtxt(ts_file)
    return data[:, 0].astype(int), data[:, 1].astype(int)


def load_embeddings(emb_file):
    return np.load(emb_file)


def load_metadata(md_file):
    data = pd.read_csv(md_file)
    return data


# ==============================
# BUILD ITEMS (visual only)
# ==============================
def build_visual_items(video_id, starts, ends, base_util=None):
    items = []
    for seg_id, (s, e) in enumerate(zip(starts, ends)):
        seg_len = e - s + 1
        for fps, scale in zip(fps_list, FPS_SCALE):
            stride = fps_to_stride[fps]
            sampled_len = ((seg_len - 1) // stride) + 1
            storage = sampled_len * FRAME_COST
            util = 0.0
            items.append({
                "video_id": video_id, "seg_id": seg_id,
                "start": s, "end": e,
                "modality": "video", "quality": fps, "stride": stride,
                "util": util, "storage": storage, "ratio": util / storage
            })
    return items


# ==============================
# GREEDY
# ==============================
def compute_max_storage(items):
    max_storage = {}
    for it in items:
        key = (it["video_id"], it["seg_id"], it["modality"])
        max_storage[key] = max(max_storage.get(key, 0), it["storage"])
    return sum(max_storage.values())


def random_select(items, budget):
    items = random.sample(items, len(items))

    selected = []
    used = 0.0
    chosen_keys = {}

    for it in items:
        key = (it["video_id"], it["seg_id"], it["modality"])

        if key in chosen_keys:
            continue

        if used + it["storage"] > budget:
            continue

        rand = random.random()
        
        if rand > 0.5:
            selected.append(it)
            chosen_keys[key] = len(selected) - 1
            used += it["storage"]

        if used >= budget:
            break

    return sorted(selected, key=lambda x: (it["video_id"], it["seg_id"])), used

def remove_items(all_items, selected = None):
    if not selected:
        best_per_key = {}

        for it in all_items:
            key = (it["video_id"], it["seg_id"], it["modality"])

            if key not in best_per_key or it["util"] > best_per_key[key]["util"]:
                best_per_key[key] = it

        return list(best_per_key.values())
    
    remaining = []

    selected_keys = set((it["video_id"], it["seg_id"], it["modality"]) for it in selected)
    selected_map = {(it["video_id"], it["seg_id"], it["modality"]): it["storage"] for it in selected}
    
    for it in all_items:
        key = (it["video_id"], it["seg_id"], it["modality"])
        if key not in selected_keys:
            continue
        if it["storage"] <= selected_map[key]:
            remaining.append(it)
    return remaining


# ==============================
# EXPORT
# ==============================
def export_global_data(selected, metadata_map, video_rank):
    rows = []
    seen = set()
    for item in selected:
        vid = item["video_id"]
        df = metadata_map[vid]
        start, end, stride = item["start"], item["end"], item["stride"]
        for r in range(start, end + 1, stride):
            key = (vid, r)
            if key in seen:
                continue
            seen.add(key)
            row = df.iloc[r].copy()
            row["video_id"] = vid
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    final_df = pd.DataFrame(rows).reset_index(drop=True)
    final_df["video_rank"] = final_df["video_id"].map(video_rank)

    return final_df.sort_values(["video_rank", "index"]).reset_index(drop=True)


# ==============================
# QA HELPERS
# ==============================
def load_images(rows):
    images, captions = [], []
    for _, r in rows.iterrows():
        path = r["path"]
        if not os.path.exists(path):
            print(f"Error loading {path}: not exist.")
            continue
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            captions.append(f"{r['caption']}\n")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    assert len(images) == len(captions), "Images and captions count mismatch"
    captions = list(set(captions))
    return images, captions


def run_qwen(model, processor, content):
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": content}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def extract_answer(text):
    match = re.search(r"<answer>([A-E])</answer>", text)
    return match.group(1) if match else None


def fix_answer(model, processor, bad_output):
    prompt = ANSWER_FIX_TEMPLATE + "\n\n" + bad_output
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    fixed = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    match = re.search(r"<answer>([A-E])</answer>", fixed)
    return match.group(1) if match else "N/A"


def run_qa_on_snapshot(snapshot_df, qa_df, vid_rank, model, processor, snapshot_tag, filename_base):
    results = []

    snapshot_by_vid = (
        {int(vid): grp for vid, grp in snapshot_df.groupby("video_id")}
        if not snapshot_df.empty else {}
    )

    for _, row in qa_df.iterrows():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        question_id = row["question_id"]
        question = row["question"]
        options_str = row["options"]

        rows = None
        q_rank = vid_rank.get(f"{question_id:04d}", -1)
        closest_vids = sorted(
            (vid for vid in vid_rank if int(vid) in snapshot_by_vid),
            key=lambda vid: (
                abs(vid_rank[vid] - q_rank) if q_rank != -1 else -vid_rank[vid]
            )
        )
        for vid in closest_vids:
            rows = snapshot_by_vid[int(vid)]
            if not rows.empty:
                break

        if len(rows) > MAX_LENGTH:
            idx = np.unique(np.linspace(0, len(rows) - 1, MAX_LENGTH).astype(int))
            rows = rows.iloc[idx]

        images, captions = load_images(rows)

        content = []
        if len(images) == 1:
            content.append({"type": "image", "image": images[0]})
        elif len(images) > 1:
            content.append({"type": "video", "video": images})
        for cap in captions:
            content.append({"type": "text", "text": cap})
        content.append({"type": "text", "text": f"Question:\n{question}\n\nOptions:\n{options_str}"})

        for trial in range(TRIALS):
            raw_output = run_qwen(model, processor, content)
            answer = extract_answer(raw_output)
            if answer is None:
                answer = fix_answer(model, processor, raw_output)
            correct = (answer == row["answer"])
            # print(question_id, correct)
            results.append({
                "snapshot": snapshot_tag,
                "question_id": question_id,
                # "tries": trial,
                "answer": answer,
                "gt": row["answer"],
                "correct": correct
            })

    if not results:
        return None

    # os.makedirs("results", exist_ok=True)
    out_df = pd.DataFrame(results)
    # out_path = os.path.join("results", f"results_{filename_base}_snap{snapshot_tag}.csv")
    # out_df.to_csv(out_path, index=False)
    # acc = out_df['correct'].mean()
    # print(f"  [QA snapshot={snapshot_tag}] Accuracy: {acc:.4f}")
    return out_df


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=-1)
    parser.add_argument("-c", "--storage", type=int, default=10000)
    parser.add_argument("-m", "--method", type=str, default="rd")
    parser.add_argument("-a", "--algorithm", type=str, default="")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-n", "--network", type=bool, default=False)
    args = parser.parse_args()

    IDX = args.idx
    STORAGE = args.storage
    METHOD = args.method
    ALGORITHM = args.algorithm
    NW = args.network
    np.random.seed(args.seed)

    if METHOD not in ["sim", "var", "rd", "fifo"]:
        exit(1)

    if METHOD in ["rd", "fifo"]:
        ALGORITHM = ""

    if IDX == -1:
        LIST_PATH = os.path.join(IDX_DIR, 'subset_500.txt')
        QA_PATH = os.path.join(QA_DIR, '500_qa.csv')
    else:
        LIST_PATH = os.path.join(IDX_DIR, f"subset_{IDX}.txt")
        QA_PATH = os.path.join(QA_DIR, f"qa_{IDX}.csv")

    name_list = [str(STORAGE), METHOD]
    if ALGORITHM:
        name_list.append(ALGORITHM)
    if IDX != -1:
        name_list.append(str(IDX))
    if NW:
        QA_SNAPSHOT_PATH = "idx_by_snapshot_nw.json"
        name_list.append("nw")
    else:
        QA_SNAPSHOT_PATH = "idx_by_snapshot.json"
    FILENAME = "_".join(name_list)
    print(FILENAME)

    idx_list, vid_rank = load_index_list(LIST_PATH)
    N = len(idx_list)

    all_items = []
    all_metadata = {}
    selected = []
    used_storage = 0.0

    qa_df = pd.read_csv(QA_PATH)

    video_lookup = qa_df.set_index("question_id", drop=False)
    
    with open(QA_SNAPSHOT_PATH, "r") as f:
        snap_to_videos = json.load(f)

    qa_by_snapshot = {
        int(snap): video_lookup.loc[video_ids].reset_index(drop=True)
        for snap, video_ids in snap_to_videos.items()
    }
    snapshot_list = sorted(qa_by_snapshot.keys())
    # print(f"Processing {N} videos. Questions spread across {len(snapshot_list)} unique steps.")

    # print("Loading Qwen model...")
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.float16, device_map="auto"
    )
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    all_qa_results = []

    for step, video_id in enumerate(idx_list):
        step_num = step + 1
        replacement = False
        try:
            ts_file = os.path.join(ROOT_PATH, "timestamp", f"{video_id}.txt")
            emb_file = os.path.join(ROOT_PATH, "embedding", f"{video_id}.npy")
            md_file = os.path.join(ROOT_PATH, "metadata", f"{video_id}.csv")

            starts, ends = load_segments(ts_file)
            embeddings = load_embeddings(emb_file)
            metadata_df = load_metadata(md_file)

            # base_visual = compute_utility_visual(starts, ends, embeddings, METHOD)
            visual_items = build_visual_items(video_id, starts, ends)

            all_metadata[video_id] = metadata_df
            vid_rank[video_id] = step

            all_items.extend(visual_items)

            max_storage = compute_max_storage(all_items)
            if max_storage > STORAGE:
                replacement = True
                selected, used_storage = random_select(all_items, STORAGE)
                all_items = remove_items(all_items, selected)
            else:
                used_storage = max_storage
                # selected = remove_items(all_items)

            # print(f"[{step_num}/{N}] video_id={video_id} | storage={used_storage:.1f}/{STORAGE}")

        except Exception as e:
            print(f"[ERROR] {video_id}: {e}")

        # ---- QA SNAPSHOT ----
        if step_num in snapshot_list:
            snap_qa = qa_by_snapshot.get(step_num, pd.DataFrame())

            if snap_qa.empty:
                continue
            
            if replacement == False:
                selected = remove_items(all_items)
            # print(f">>> Step {step_num}/{N} ({len(snap_qa)})")
            snapshot_df = export_global_data(selected, all_metadata, vid_rank)

            method_label = "div" if METHOD == "sim" else METHOD
            result_df = run_qa_on_snapshot(
                snapshot_df, snap_qa, vid_rank,
                qwen_model, qwen_processor,
                snapshot_tag=step_num,
                filename_base=FILENAME
            )
            if result_df is not None:
                all_qa_results.append(result_df)
                # with open(RESULT_FILE, 'a') as f:
                #     f.write(f"{STORAGE},{method_label},{ALGORITHM},snap={step_num}/{N},{acc:.4f}\n")
            # print()

    # ---- FINAL EXPORT (RP output) ----
    # print("Exporting final merged RP output...")
    # os.makedirs("merged", exist_ok=True)

    # final_df = export_global_data(selected, all_metadata, vid_rank)
    # final_df.to_csv(os.path.join("merged", f"{FILENAME}.csv"), index=False)

    # storage = compute_max_storage(selected)
    # print(f"Total selected frames: {len(final_df)}")
    # print(f"Storage used: {storage:.2f} / {STORAGE}")

    # ---- AGGREGATE QA RESULTS ----
    if all_qa_results:
        combined = pd.concat(all_qa_results, ignore_index=True)
        # combined.to_csv(os.path.join("results", f"results_{FILENAME}_comb.csv"), index=False)
        print(f"\nAll QA results saved to results_{FILENAME}_comb.csv")
        # print("\nAccuracy by snapshot:")
        # for snap, grp in combined.groupby("snapshot"):
        #     print(f"  step={snap}: {grp['correct'].mean():.4f}")
        avg_acc = combined['correct'].mean()
        # print(f"Overall average accuracy: {avg_acc:.4f}")
        with open(RESULT_FILE, 'a') as f:
            f.write(f"{STORAGE},{method_label},{ALGORITHM},{avg_acc:.4f}\n")

    print(f"\nTotal duration: {datetime.now() - start_time}")