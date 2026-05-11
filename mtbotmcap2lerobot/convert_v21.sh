#!/bin/bash
# Convert MTBot MCAP recordings to LeRobot v2.1 format.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

SRC_DIR="${SRC_DIR:-/home/mtbot/blackbox/robot_arm_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/lrobot/mtbot_mcap}"
REPO_ID="${REPO_ID:-luobai/mtbot_mcap}"
TASK_TEXT="${TASK_TEXT:-robot arm demonstration}"
FPS="${FPS:-30}"
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
ROBOT_TYPE="${ROBOT_TYPE:-mtbot fairino arm}"

if [ ! -d ".venv_lerobot_v21" ]; then
    echo "Creating virtual environment with ROS system packages visible..."
    uv venv .venv_lerobot_v21 --python 3.11 --system-site-packages
fi

echo "Installing LeRobot v2.1 and converter dependencies..."
export GIT_LFS_SKIP_SMUDGE=1
uv pip install --python .venv_lerobot_v21/bin/python \
    "lerobot @ git+https://github.com/huggingface/lerobot.git@d602e8169cbad9e93a4a3b3ee1dd8b332af7ebf8"
uv pip install --python .venv_lerobot_v21/bin/python opencv-python pyyaml numpy

if [ -d "${OUTPUT_DIR}" ]; then
    echo "Removing existing output directory: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
fi

echo "Starting MCAP to LeRobot conversion..."
.venv_lerobot_v21/bin/python ./mtbotmcap2lerobot/dataset_convert.py \
    --src-dir "${SRC_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --fps "${FPS}" \
    --use-videos \
    --robot-type "${ROBOT_TYPE}" \
    --repo_id "${REPO_ID}" \
    --image-writer-threads 4 \
    --image-writer-process 2 \
    --image-width "${IMAGE_WIDTH}" \
    --image-height "${IMAGE_HEIGHT}" \
    --task-text "${TASK_TEXT}" \
    --tags "mtbot" "fairino" "manipulation" "mcap"

echo "Conversion completed: ${OUTPUT_DIR}"
