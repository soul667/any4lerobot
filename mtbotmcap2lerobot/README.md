# MTBot MCAP to LeRobot Converter

Convert MTBot ROS2 MCAP recordings into a LeRobot v2.1 dataset.

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
chmod +x ./mtbotmcap2lerobot/convert_v21.sh
./mtbotmcap2lerobot/convert_v21.sh
```

Useful environment overrides:

```bash
SRC_DIR=/home/mtbot/blackbox/robot_arm_data \
OUTPUT_DIR=./data/lrobot/my_mtbot_dataset \
REPO_ID=luobai/my_mtbot_dataset \
TASK_TEXT="pick and place object" \
./mtbotmcap2lerobot/convert_v21.sh
```

## Direct Python Usage

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

## Notes

- Each MCAP bag directory becomes one LeRobot episode.
- Frames are aligned on joint-state timestamps.
- RGB and depth frames are selected by nearest timestamp.
- `action` is currently the next 7D state, while `observation.state` is the current 7D state.
- The shell script creates `.venv_lerobot_v21` with `--system-site-packages` so ROS Python packages such as `rosbag2_py` remain visible after sourcing the ROS environment.

## Task Mapping

Use `--task-file ./mtbotmcap2lerobot/task_config.yml` to map episode directory names to task text with regex patterns.
