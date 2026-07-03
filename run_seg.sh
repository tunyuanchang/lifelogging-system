VIDEO_FOLDER="dataset"
INPUT_FILE="subset_idx_500.txt"

mapfile -t idx_list < "$INPUT_FILE"

for idx in "${idx_list[@]}"; do
    f="$VIDEO_FOLDER/$idx.mp4"
    # echo "---"
    # echo "Processing $f ..."
    python3 segment_video.py "$f"
done

# echo "Done"