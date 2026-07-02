#!/bin/bash
# Convert MTBot MCAP recordings to LeRobot v2.1 format.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-/home/mtbot/project/mtbot_mobile_manipulator/install/setup.bash}"

SRC_DIR="${SRC_DIR:-/home/mtbot/blackbox/robot_arm_data/test111}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/lrobot/test111}"
REPO_ID="${REPO_ID:-luobai/test111}"
TASK_TEXT="${TASK_TEXT:-move the bag to the shelf}"
FPS="${FPS:-30}"
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
ROBOT_TYPE="${ROBOT_TYPE:-mtbot fairino arm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [ -x ".venv_lerobot_v21/bin/python" ]; then
    VENV_PYTHON_VERSION="$(
        .venv_lerobot_v21/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )"
    if [ "${VENV_PYTHON_VERSION}" != "${PYTHON_VERSION}" ]; then
        echo "Removing Python ${VENV_PYTHON_VERSION} venv; ROS Humble Python packages need Python ${PYTHON_VERSION}."
        rm -rf .venv_lerobot_v21
    fi
fi

if [ ! -d ".venv_lerobot_v21" ]; then
    echo "Creating Python ${PYTHON_VERSION} virtual environment with ROS system packages visible..."
    uv venv .venv_lerobot_v21 --python "${PYTHON_VERSION}" --system-site-packages
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

export ROS_SETUP WORKSPACE_SETUP
export SRC_DIR OUTPUT_DIR REPO_ID TASK_TEXT FPS IMAGE_WIDTH IMAGE_HEIGHT ROBOT_TYPE

# Activate venv and source ROS environment for Python subprocess
bash -c '
    source .venv_lerobot_v21/bin/activate
    if [ -f "${ROS_SETUP}" ]; then
        source "${ROS_SETUP}"
    else
        echo "ROS setup not found: ${ROS_SETUP}" >&2
        exit 1
    fi
    if [ -f "${WORKSPACE_SETUP}" ]; then
        source "${WORKSPACE_SETUP}"
    fi
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
'

echo "Conversion completed: ${OUTPUT_DIR}"
