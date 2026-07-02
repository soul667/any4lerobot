# MTBot MCAP to LeRobot Converter

Convert MTBot ROS2 MCAP recordings into a trainable LeRobot dataset.

## Input

The recorder saves MCAP data under:

```text
/home/mtbot/blackbox/robot_arm_data/<group>/<sequence_timestamp>/*.mcap
```

The converter expects these topics:

- `/slave/nonrt_state_data`
- `/camera/d405/color/image_raw`
- `/camera/d405/aligned_depth_to_color/image_raw` if `--include-depth` is used

The robot state is exported as 7 motors:

- `joint_1` to `joint_6`, converted from degrees to radians
- `gripper`, using the raw `gripper_position`

## Quick Start

Run from `any4lerobot`:

```bash
chmod +x ./mtbotmcap2lerobot/convert_latest.sh
./mtbotmcap2lerobot/convert_latest.sh
```

Conda alternative:

```bash
chmod +x ./mtbotmcap2lerobot/convert_latest_conda.sh
./mtbotmcap2lerobot/convert_latest_conda.sh
```

The conda script opens an interactive prompt for the fields you usually need to change:

- MCAP source directory
- final LeRobot v3.0 output directory
- dataset repo id
- task text
- FPS and image size
- robot type
- conda environment name
- whether to store camera observations as videos

For non-interactive runs, keep using environment variables and add `--yes`:

```bash
./mtbotmcap2lerobot/convert_latest_conda.py --yes \
    --src-dir /home/mtbot/blackbox/robot_arm_data/test111 \
    --output-dir ./data/lrobot/test111_latest \
    --repo-id luobai/test111 \
    --task-text "move the bag to the shelf"
```

The conda path is now a single Python entrypoint and writes the current LeRobot
dataset format directly through the official `LeRobotDataset.create()`,
`add_frame()`, `save_episode()`, and `finalize()` API. It no longer creates a
temporary v2.1 dataset or hand-converts metadata.

Prepare the environments first if you want to check dependencies without
converting data:

```bash
python3 ./mtbotmcap2lerobot/convert_latest_conda.py --yes --setup-only
```

The script intentionally uses two conda environments behind the single entrypoint:

- `lerobot-mtbot-py310`: ROS Humble / MCAP extraction, because ROS Humble Python packages are Python 3.10.
- `lerobot-mtbot-py312`: current `lerobot>=0.5.1` dataset writer, because that package requires Python >=3.12.

You can also call the Python file directly from outside:

```bash
python3 ./mtbotmcap2lerobot/convert_latest_conda.py --yes \
    --src-dir /home/mtbot/blackbox/robot_arm_data/test111 \
    --output-dir ./data/lrobot/test111_latest \
    --repo-id luobai/test111 \
    --task-text "move the bag to the shelf"
```

Environment overrides still work:

```bash
SRC_DIR=/home/mtbot/blackbox/robot_arm_data \
OUTPUT_DIR=./data/lrobot/my_mtbot_dataset \
REPO_ID=luobai/my_mtbot_dataset \
TASK_TEXT="pick and place object" \
./mtbotmcap2lerobot/convert_latest_conda.py --yes
```

`convert_latest.sh` and `convert_latest_conda.sh` are now thin wrappers around `convert_latest_conda.py`.

`convert_latest_conda.sh` is a thin wrapper around `convert_latest_conda.py`. The Python entrypoint creates or reuses `CONDA_ENV_NAME=lerobot-mtbot-py310` for ROS extraction and `WRITER_CONDA_ENV_NAME=lerobot-mtbot-py312` for the current LeRobot writer. Override `CONDA_EXE` if your conda binary is not `/home/mtbot/miniconda3/condabin/conda`; pass `--yes` to skip prompts.

## Legacy direct converter

`dataset_convert.py`, `convert_v21.sh`, and `local_v21_to_v30.py` are retained for older v2.1 debugging only. Prefer `convert_latest_conda.py` for training with current LeRobot.

```bash
python ./mtbotmcap2lerobot/dataset_convert.py \
    --src-dir /home/mtbot/blackbox/robot_arm_data \
    --output-dir ./data/lrobot/mtbot_mcap \
    --fps 30 \
    --use-videos \
    --robot-type "mtbot fairino arm" \
    --repo_id "luobai/mtbot_mcap" \
    --image-width 640 \
    --image-height 480 \
    --task-text "robot arm demonstration"
```

Add `--include-depth` to also write `observation.images.depth` as an 8-bit normalized depth image.

To upgrade an already-created local v2.1 dataset without reading MCAP again:

```bash
.venv_lerobot_v21/bin/python ./mtbotmcap2lerobot/local_v21_to_v30.py \
    --src-dir ./data/lrobot/mtbot_mcap_v21 \
    --output-dir ./data/lrobot/mtbot_mcap \
    --overwrite
```

## Notes

- Each MCAP bag directory becomes one LeRobot episode.
- Frames are aligned on joint-state timestamps.
- RGB and depth frames are selected by nearest timestamp.
- `action` is currently the next 7D state, while `observation.state` is the current 7D state.
- `convert_latest.sh` reuses `convert_v21.sh`, which creates `.venv_lerobot_v21` with `--system-site-packages` so ROS Python packages such as `rosbag2_py` remain visible after sourcing the ROS environment.
- `convert_latest_conda.sh` sources ROS after activating conda so ROS Python packages such as `rosbag2_py` are visible inside the conda process.
- ROS Humble Python packages are built for Python 3.10 on Ubuntu, so the converter defaults to a Python 3.10 venv. Override `PYTHON_VERSION` only if your ROS install uses a different Python minor version.

## Task Mapping

Use `--task-file ./mtbotmcap2lerobot/task_config.yml` to map episode directory names to task text with regex patterns.
