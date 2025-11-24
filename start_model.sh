#!/bin/bash

# 1. Get the absolute path of the current directory
CURRENT_DIR="$(pwd)"

# 2. Detect the processor architecture
ARCH=$(uname -m)

echo "Detected architecture: $ARCH"

# 3. Select the correct image based on the chip
if [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    # Mac M1/M2/M3 (Apple Silicon)
    echo "✅ Apple Silicon detected. Using optimized ARM64 image."
    IMAGE="emacski/tensorflow-serving:latest-linux_arm64"
else
    # Intel/AMD Windows or Intel Mac
    echo "✅ Intel/AMD architecture detected. Using standard TensorFlow image."
    IMAGE="tensorflow/serving"
fi

# 4. Run Docker
# We use MSYS_NO_PATHCONV=1 to stop Git Bash from messing up paths on Windows
MSYS_NO_PATHCONV=1 docker run -t --rm -p 8501:8501 \
    -v "$CURRENT_DIR/model/square_pirates_edge:/models/square_pirates_edge" \
    -e MODEL_NAME=square_pirates_edge \
    "$IMAGE"