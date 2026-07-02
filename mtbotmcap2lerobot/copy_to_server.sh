#!/bin/bash
# Copy dataset to remote server via SCP

set -e  # Exit on error

LOCAL_PATH="./data/lrobot1/move_to_shelf_2"
REMOTE_USER="axgu"
# REMOTE_HOST="10.16.46.75"
REMOTE_HOST="10.16.118.8"
REMOTE_PATH="/data2/axgu/.cache/huggingface/lerobot/luobai/move_to_shelf_2"  # Adjust as needed
# REMOTE_PATH="/media/axgu/Data/hf_cache/huggingface/lerobot/luobai/move_to_shelf_2"  # Adjust as needed

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_xixi}"

echo "📦 Copying dataset to remote server..."
echo "   Local:  $LOCAL_PATH"
echo "   Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "   SSH key: ${SSH_KEY}"
echo ""

# Check if local path exists
if [ ! -d "$LOCAL_PATH" ]; then
    echo "❌ Error: Local dataset not found at $LOCAL_PATH"
    exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
    echo "❌ Error: SSH key not found at $SSH_KEY"
    exit 1
fi

# Use SCP to copy recursivel/
# -r: recursive copy
# -p: preserve modification times and modes
# -C: enable compression
echo "🚀 Starting transfer (this may take a while)..."

ssh -i "$SSH_KEY" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '$(dirname "$REMOTE_PATH")'"
scp -i "$SSH_KEY" -r -p -C "$LOCAL_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dataset successfully copied to remote server!"
    echo "   Remote path: ${REMOTE_PATH}"
else
    echo ""
    echo "❌ Transfer failed"
    echo "   Please check:"
    echo "   - SSH connection to ${REMOTE_HOST}"
    echo "   - Remote directory permissions"
    echo "   - Network connectivity"
    exit 1
fi
