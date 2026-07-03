import sys
import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import warnings
warnings.filterwarnings('ignore') # Hide all warnings

from transformers import logging
logging.set_verbosity_error() # Only show errors, ignore info/warnings

# =========================
# CONFIG
# =========================
QA_DIR = "/mnt/ssd_nvme0/tunyuan/egoschema_500/"
IDX_DIR = "/mnt/ssd_nvme0/tunyuan/egoschema_500/"
RESULT_FILE = 'result_exp.txt'
ROOT_PATH = 'merged'
TRIALS = 1

# Load OpenCLIP
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)
clip_model.eval()

# =========================
# PROMPTS
# =========================
SYSTEM_PROMPT = (
    "You are a long-video question answering assistant. "
    "Be concise and follow the output format exactly."
)

USER_TEMPLATE = (
    # "You will optionally be given visual context extracted from a video, "
    # "and/or text context generated from the video (such as caption).\n\n"
    # "Task: Answer the multiple-choice question.\n\n"
    # "Only use the provided visual and/or text context to answer.\n\n"
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

SYSTEM_CONTENT = [{
    "type": "text",
    "text": SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE + "\n" + ANSWER_FIX_TEMPLATE
}]

# =========================
# LOAD DATA
# =========================
def load_index_list(file_path):
    with open(file_path, "r") as f:
        index_list = [line.strip() for line in f if line.strip()]
        video_rank = {vid: i for i, vid in enumerate(index_list)}
        return index_list, video_rank

def load_file(filename):
    mf = os.path.join(ROOT_PATH, f"{filename}.csv")

    try:
        df = pd.read_csv(mf)

    except Exception as e:
        df = pd.DataFrame()

    return df

# =========================
# NORMALIZE
# =========================
def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# =========================
# LOAD IMAGES
# =========================
def load_images(rows):
    images, captions = [], []

    for _, r in rows.iterrows():
        path = r["path"]

        if not os.path.exists(path):
            print(f"Error loading {path}: not exist.")
            continue

        try:
            img = Image.open(path).convert("RGB")
            caption = (f"{r['caption']}\n")

            images.append(img)
            captions.append(caption)

        except Exception as e:
            print(f"Error loading {path}: {e}")

    assert len(images) == len(captions), "Images and captions count mismatch"
    # print(len(images))
    captions = list(set(captions))
    return images, captions


# =========================
# MODEL RUN
# =========================
def run_qwen(model, processor, content):
    """
    Uses chat-style message input for multimodal Qwen inference.
    """

    messages = [
        {"role": "system", "content":SYSTEM_CONTENT},
        {"role": "user", "content": content}
    ]

    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    # Trim prefix (prompt) from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text


# =========================
# EXTRACT OUTPUT
# =========================
def extract_structured(text):
    answer_match = re.search(r"<answer>([A-E])</answer>", text)
    answer = answer_match.group(1) if answer_match else None

    return answer


# =========================
# FIX ANSWER
# =========================
def fix_answer(model, processor, bad_output):
    prompt = ANSWER_FIX_TEMPLATE + "\n\n" + bad_output

    inputs = processor(
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False
        )

    fixed = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    match = re.search(r"<answer>([A-E])</answer>", fixed)
    return match.group(1) if match else "N/A"


# =========================
# Query EMBEDDING
# =========================
def embed_query(text):
    """
    Encode a text query into a normalized vector using OpenCLIP.
    """
    with torch.no_grad():
        tokens = open_clip.tokenize([text]).to(device)  # tokenize batch of 1
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # normalize
    return text_features.cpu().numpy()[0]  # return as 1D numpy vector


# =========================
# MAIN
# =========================
def main():

    # print("Loading data...")
    idx_list, vid_rank = load_index_list(LIST)
    metadata = load_file(FILENAME)

    # print("Loading QA...")
    qa_df = pd.read_csv(QA_PATH)

    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    # print("Loading model...")
    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_name)

    results = []

    # print("Running inference...")
    # start_time = datetime.now()
    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), disable=True):
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        question_id = row["question_id"]
        question = row["question"]
        options_str = row["options"]

        # q_emb = embed_query(question)
        rows = metadata[metadata["video_id"] == question_id]

        if rows.empty:
            # print(f"no video {question_id}")
            rank = vid_rank.get(f"{question_id:04d}")
            next_vids = sorted(
                (vid for vid, r in vid_rank.items() if r > rank),
                key=lambda vid: vid_rank[vid]
            )
            for vid in next_vids:
                rows = metadata[metadata["video_id"] == int(vid)]
                if not rows.empty:
                    break
        
        if len(rows) > 400:
            retrieved_idx = np.linspace(0, len(rows) - 1, 400)
            retrieved_idx = np.unique(retrieved_idx.astype(int))
            rows = rows.iloc[retrieved_idx]


        images, captions = load_images(rows)

        # Build the message list
        content = []

        # images and captions
        # for img, cap in zip(images, captions):
        #     content.append({"type": "image", "image": img})
        #     content.append({"type": "text", "text": cap})
        if len(images) == 1:
            content.append({"type": "image", "image": images[0]})
        elif len(images) != 0:
            content.append({"type": "video", "video": images})

        for cap in captions:
            content.append({"type": "text", "text": cap})
                
        # question an options
        content.append({
            "type": "text",
            "text": f"Question:\n{question}\n\nOptions:\n{options_str}"
        })

        for trial in range(TRIALS):
            # Model
            raw_output = run_qwen(
                model,
                processor,
                content
            )

            answer = extract_structured(raw_output)

            # Fix if needed
            if answer is None:
                answer = fix_answer(model, processor, raw_output)

            correct = (answer == row["answer"])
            # print(question_id, trial, answer, correct)
            
            results.append({
                "question_id": row["question_id"],
                "tries": trial,
                "answer": answer,
                "gt": row["answer"],
                "correct": correct
            })

        # break

    # end_time = datetime.now()

    OUTPUT_PATH = f"results_{FILENAME}.csv"
    df = pd.DataFrame(results)
    df.to_csv(os.path.join("results", OUTPUT_PATH), index=False)

    correct_rate = df['correct'].mean()

    with open(RESULT_FILE, 'a') as f:
        # if METHOD == "sim": METHOD = "div"
        f.write(f"{STORAGE},{METHOD},{ALGORITHM},{correct_rate:.4f}\n")

    print(f"Saved to {OUTPUT_PATH}")
    # print(str(end_time - start_time))


# =========================
# RUN
# =========================
if __name__ == "__main__":
    global STORAGE
    global METHOD
    global ALGORITHM
    global FILENAME
    global LIST

    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "-l", "--list", 
    #     type=str, 
    #     required=True
    # )
    
    # parser.add_argument(
    #     "-f", "--filename",
    #     type=str,
    #     required=True
    # )

    parser.add_argument(
        "-a", "--algorithm",
        type=str,
        default="gd"
    )

    parser.add_argument(
        "-c", "--storage",
        type=int,
        default=10000
    )

    parser.add_argument(
        "-m", "--method",
        type=str,
        default="sim"
    )

    parser.add_argument(
        "-i", "--idx",
        type=int,
        default=-1
    )

    args = parser.parse_args()

    STORAGE = args.storage
    METHOD = args.method
    ALGORITHM = args.algorithm
    IDX = args.idx
    # FILENAME = args.filename
    # LIST = args.list

    FILENAME = f"{STORAGE}_{METHOD}_{ALGORITHM}"

    if METHOD not in ["sim", "var"]:
        ALGORITHM = ""
        FILENAME = f"{STORAGE}_{METHOD}"

    if IDX == -1:
        QA_PATH = os.path.join(QA_DIR, '500_qa.csv')
        LIST = os.path.join(IDX_DIR, '500_subset.txt')
    else:
        QA_PATH = os.path.join(QA_DIR, f'qa_{args.idx}.csv')
        FILENAME = f"{FILENAME}_{args.idx}"
        LIST = os.path.join(IDX_DIR, f"subset_{args.idx}.txt")

    if METHOD not in ["sim", "var", "rd", "fifo"]:
        exit(1)

    main()