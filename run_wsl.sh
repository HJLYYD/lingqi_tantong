#!/bin/bash

PROJECT_DIR=$(find /mnt/d/shool -type d -name "lingqi_tantong_c" 2>/dev/null | head -1)

if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: Cannot find lingqi_tantong_c directory"
    exit 1
fi

echo "Found project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

VIDEO_FILE=$(find /mnt/d/shool -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" 2>/dev/null | grep -i "室内" | head -1)

if [ -z "$VIDEO_FILE" ]; then
    echo "WARNING: Cannot find input video file"
    VIDEO_FILE="./test.mp4"
fi

echo "Using video file: $VIDEO_FILE"

echo ""
echo "=== Building project ==="
python3 build.py build 2>&1

if [ ! -f "./lingqi_tantong" ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "=== Running program ==="
./lingqi_tantong --video_path "$VIDEO_FILE" --max_frames 50 2>&1 | head -100