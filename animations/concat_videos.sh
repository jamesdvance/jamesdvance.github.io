#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_directory>"
    echo "Example: $0 /path/to/video/directory"
    exit 1
fi

VIDEO_DIR="$1"

if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: Directory '$VIDEO_DIR' does not exist"
    exit 1
fi


VIDEO_FILES=$(ls -1 *.mp4 *.avi *.mov *.mkv *.webm 2>/dev/null | sort)

if [ -z "$VIDEO_FILES" ]; then
    echo "No video files found in $VIDEO_DIR"
    exit 1
fi

echo "Found video files:"
echo "$VIDEO_FILES"

TEMP_LIST=$(mktemp)
for file in $VIDEO_FILES; do
    echo "file '$file'" >> "$TEMP_LIST"
done

OUTPUT_FILE="combined_$(basename "$VIDEO_DIR").mp4"

echo "Combining videos into: $OUTPUT_FILE"

ffmpeg -f concat -safe 0 -i "$TEMP_LIST" -c copy "$OUTPUT_FILE"

rm "$TEMP_LIST"

if [ $? -eq 0 ]; then
    echo "Successfully created: $OUTPUT_FILE"
else
    echo "Error: Failed to combine videos"
    exit 1
fi