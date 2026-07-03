VIDEO_FOLDER="dataset"
INPUT_FILE="500_subset.txt"

mapfile -t idx_list < "$INPUT_FILE"

for idx in "${idx_list[@]}"; do
    f="$VIDEO_FOLDER/$idx.mp4"
    # echo "---"
    # echo "Processing $f ..."
    python3 segment_video.py "$f"
done

# echo "Done"
