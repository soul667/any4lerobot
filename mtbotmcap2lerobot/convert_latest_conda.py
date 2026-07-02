#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JOINT_TOPIC = "/slave/nonrt_state_data"
COLOR_TOPIC = "/camera/d405/color/image_raw"
DEPTH_TOPIC = "/camera/d405/aligned_depth_to_color/image_raw"

DEFAULT_CONDA_EXE = "/home/mtbot/miniconda3/condabin/conda"
DEFAULT_CONDA_ENV_NAME = "lerobot-mtbot-py310"
DEFAULT_WRITER_CONDA_ENV_NAME = "lerobot-mtbot-py312"
DEFAULT_PYTHON_VERSION = "3.10"
DEFAULT_WRITER_PYTHON_VERSION = "3.12"
DEFAULT_ROS_SETUP = "/opt/ros/humble/setup.bash"
DEFAULT_WORKSPACE_SETUP = "/home/mtbot/project/mtbot_mobile_manipulator/install/setup.bash"
DEFAULT_SRC_DIR = "/home/mtbot/blackbox/robot_arm_data/move_to_shelf"
DEFAULT_OUTPUT_DIR = "./data/lrobot1/move_to_shelf_1"
DEFAULT_REPO_ID = "luobai/move_to_shelf_1"
DEFAULT_TASK_TEXT = "move the bag to the shelf"
DEFAULT_FPS = 30
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_ROBOT_TYPE = "mtbot fairino arm"

cv2 = None
np = None


@dataclass
class McapEpisode:
    bag_uri: Path
    file_name: str
    states: list[np.ndarray]
    main_images: list[np.ndarray]
    depth_images: list[np.ndarray] | None = None


@dataclass
class ConverterConfig:
    src_dir: Path
    output_dir: Path
    repo_id: str
    task_text: str
    fps: int
    image_width: int
    image_height: int
    robot_type: str
    include_depth: bool
    use_videos: bool
    image_writer_processes: int
    image_writer_threads: int
    push_to_hub: bool
    log_level: str


@dataclass
class EnvConfig:
    conda_exe: str
    ros_env_name: str
    ros_python_version: str
    writer_env_name: str
    writer_python_version: str
    ros_setup: str
    workspace_setup: str


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def ask_value(label: str, default: str, interactive: bool) -> str:
    if not interactive:
        return default
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def ask_bool(label: str, default: bool, interactive: bool) -> bool:
    if not interactive:
        return default
    suffix = "Y/n" if default else "y/N"
    value = input(f"{label} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MTBot ROS2 MCAP recordings directly to the current LeRobot dataset format."
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip interactive prompts")
    parser.add_argument("--setup-only", action="store_true", help="Create/check the conda environment and install dependencies, then exit")
    parser.add_argument("--inside-env", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=["extract", "write"], help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--conda-exe", default=env_default("CONDA_EXE", DEFAULT_CONDA_EXE))
    parser.add_argument("--conda-env-name", default=env_default("CONDA_ENV_NAME", DEFAULT_CONDA_ENV_NAME), help="ROS/MCAP extraction conda environment name")
    parser.add_argument("--writer-conda-env-name", default=env_default("WRITER_CONDA_ENV_NAME", DEFAULT_WRITER_CONDA_ENV_NAME), help="LeRobot writer conda environment name")
    parser.add_argument("--python-version", default=env_default("PYTHON_VERSION", DEFAULT_PYTHON_VERSION))
    parser.add_argument("--writer-python-version", default=env_default("WRITER_PYTHON_VERSION", DEFAULT_WRITER_PYTHON_VERSION))
    parser.add_argument("--ros-setup", default=env_default("ROS_SETUP", DEFAULT_ROS_SETUP))
    parser.add_argument("--workspace-setup", default=env_default("WORKSPACE_SETUP", DEFAULT_WORKSPACE_SETUP))
    parser.add_argument("--src-dir", type=Path, default=Path(env_default("SRC_DIR", DEFAULT_SRC_DIR)))
    parser.add_argument("--output-dir", type=Path, default=Path(env_default("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--repo-id", "--repo_id", dest="repo_id", default=env_default("REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--task-text", default=env_default("TASK_TEXT", DEFAULT_TASK_TEXT))
    parser.add_argument("--fps", type=int, default=int(env_default("FPS", str(DEFAULT_FPS))))
    parser.add_argument("--image-width", type=int, default=int(env_default("IMAGE_WIDTH", str(DEFAULT_IMAGE_WIDTH))))
    parser.add_argument("--image-height", type=int, default=int(env_default("IMAGE_HEIGHT", str(DEFAULT_IMAGE_HEIGHT))))
    parser.add_argument("--robot-type", default=env_default("ROBOT_TYPE", DEFAULT_ROBOT_TYPE))
    parser.add_argument("--include-depth", action="store_true", default=parse_bool(os.environ.get("INCLUDE_DEPTH")))
    parser.add_argument("--no-videos", action="store_true", default=parse_bool(os.environ.get("NO_VIDEOS")))
    parser.add_argument("--image-writer-processes", type=int, default=int(env_default("IMAGE_WRITER_PROCESSES", "2")))
    parser.add_argument("--image-writer-threads", type=int, default=int(env_default("IMAGE_WRITER_THREADS", "4")))
    parser.add_argument("--push-to-hub", action="store_true", default=parse_bool(os.environ.get("PUSH_TO_HUB")))
    parser.add_argument("--log-level", default=env_default("LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def apply_interactive_prompts(args: argparse.Namespace) -> argparse.Namespace:
    interactive = not args.yes and sys.stdin.isatty() and not args.inside_env
    if not interactive:
        return args

    print("MTBot MCAP -> current LeRobot conda converter")
    print()
    args.src_dir = Path(ask_value("MCAP source directory", str(args.src_dir), interactive))
    args.output_dir = Path(ask_value("LeRobot output directory", str(args.output_dir), interactive))
    args.repo_id = ask_value("Dataset repo id", args.repo_id, interactive)
    args.task_text = ask_value("Task text for episodes", args.task_text, interactive)
    args.fps = int(ask_value("Dataset FPS", str(args.fps), interactive))
    args.image_width = int(ask_value("Image width", str(args.image_width), interactive))
    args.image_height = int(ask_value("Image height", str(args.image_height), interactive))
    args.robot_type = ask_value("Robot type", args.robot_type, interactive)
    args.conda_env_name = ask_value("Conda environment name", args.conda_env_name, interactive)
    args.writer_conda_env_name = ask_value("LeRobot writer conda environment name", args.writer_conda_env_name, interactive)
    args.include_depth = ask_bool("Include depth image", bool(args.include_depth), interactive)
    args.no_videos = not ask_bool("Store camera observations as videos", not bool(args.no_videos), interactive)
    print()
    print("Summary")
    print(f"  source: {args.src_dir}")
    print(f"  output: {args.output_dir}")
    print(f"  repo_id: {args.repo_id}")
    print(f"  task: {args.task_text}")
    print(f"  fps: {args.fps}")
    print(f"  image: {args.image_width}x{args.image_height}")
    print(f"  robot: {args.robot_type}")
    print(f"  conda env: {args.conda_env_name}")
    print(f"  writer env: {args.writer_conda_env_name}")
    print(f"  videos: {not args.no_videos}")
    print()
    confirm = input("Start conversion? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Cancelled.")
        raise SystemExit(0)
    return args


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    logging.debug("Running: %s", " ".join(command))
    return subprocess.run(command, check=check, text=True)


def conda_env_exists(conda_exe: str, env_name: str) -> bool:
    result = subprocess.run([conda_exe, "env", "list"], check=True, text=True, stdout=subprocess.PIPE)
    for line in result.stdout.splitlines():
        columns = line.split()
        if columns and columns[0] == env_name:
            return True
    return False


def conda_env_python(conda_exe: str, env_name: str) -> Path:
    result = subprocess.run([conda_exe, "info", "--envs"], check=True, text=True, stdout=subprocess.PIPE)
    for line in result.stdout.splitlines():
        columns = line.split()
        if columns and columns[0] == env_name:
            return Path(columns[-1]) / "bin" / "python"
    raise RuntimeError(f"Could not resolve conda env prefix for {env_name!r}")


def ensure_base_conda(conda_exe_value: str) -> Path:
    conda_exe = Path(conda_exe_value).expanduser()
    if not conda_exe.exists() or not os.access(conda_exe, os.X_OK):
        raise FileNotFoundError(f"conda not found or not executable: {conda_exe}")
    return conda_exe


def ensure_env(conda_exe: Path, env_name: str, python_version: str) -> Path:
    if not conda_env_exists(str(conda_exe), env_name):
        print(f"Creating conda environment {env_name!r}...")
        run([str(conda_exe), "create", "-y", "-n", env_name, f"python={python_version}", "pip"])
    return conda_env_python(str(conda_exe), env_name)


def ensure_ros_environment(args: argparse.Namespace) -> Path:
    conda_exe = ensure_base_conda(args.conda_exe)
    python = ensure_env(conda_exe, args.conda_env_name, args.python_version)
    probe = subprocess.run([str(python), "-c", "import cv2, numpy"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode != 0:
        print(f"Installing ROS extraction dependencies in {args.conda_env_name!r}...")
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "opencv-python", "numpy"], check=True)
    return python


def ensure_writer_environment(args: argparse.Namespace) -> Path:
    conda_exe = ensure_base_conda(args.conda_exe)
    python = ensure_env(conda_exe, args.writer_conda_env_name, args.writer_python_version)
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import cv2, numpy; from importlib.metadata import version; "
            "from packaging.version import Version; "
            "raise SystemExit(0 if Version(version('lerobot')) >= Version('0.5.1') else 1)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        print(f"Installing/upgrading current LeRobot writer dependencies in {args.writer_conda_env_name!r}...")
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "opencv-python", "pyyaml", "numpy", "packaging", "lerobot>=0.5.1"],
            check=True,
            env=env,
        )
    return python


def ensure_conda_environments(args: argparse.Namespace) -> tuple[Path, Path]:
    ros_python = ensure_ros_environment(args)
    writer_python = ensure_writer_environment(args)
    return ros_python, writer_python


def reexec_inside_ros_conda(args: argparse.Namespace) -> None:
    ensure_conda_environments(args)
    script = Path(__file__).resolve()
    conda_sh = Path(args.conda_exe).expanduser().resolve().parent.parent / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.exists():
        raise FileNotFoundError(f"conda activation script not found: {conda_sh}")
    if not Path(args.ros_setup).exists():
        raise FileNotFoundError(f"ROS setup not found: {args.ros_setup}")

    inner_args = [
        "python",
        str(script),
        "--inside-env",
        "--yes",
        "--src-dir",
        str(args.src_dir),
        "--output-dir",
        str(args.output_dir),
        "--repo-id",
        args.repo_id,
        "--task-text",
        args.task_text,
        "--fps",
        str(args.fps),
        "--image-width",
        str(args.image_width),
        "--image-height",
        str(args.image_height),
        "--robot-type",
        args.robot_type,
        "--image-writer-processes",
        str(args.image_writer_processes),
        "--image-writer-threads",
        str(args.image_writer_threads),
        "--log-level",
        args.log_level,
    ]
    if args.include_depth:
        inner_args.append("--include-depth")
    if args.no_videos:
        inner_args.append("--no-videos")
    if args.push_to_hub:
        inner_args.append("--push-to-hub")

    quoted = " ".join(shlex_quote(part) for part in inner_args)
    workspace_source = f"source {shlex_quote(args.workspace_setup)} && " if Path(args.workspace_setup).exists() else ""
    command = (
        f"source {shlex_quote(str(conda_sh))} && "
        f"conda activate {shlex_quote(args.conda_env_name)} && "
        f"source {shlex_quote(args.ros_setup)} && "
        f"{workspace_source}exec {quoted}"
    )
    raise SystemExit(subprocess.run(["bash", "-lc", command]).returncode)


def conda_sh_path(args: argparse.Namespace) -> Path:
    conda_sh = Path(args.conda_exe).expanduser().resolve().parent.parent / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.exists():
        raise FileNotFoundError(f"conda activation script not found: {conda_sh}")
    return conda_sh


def command_args_for_stage(args: argparse.Namespace, stage: str, cache_dir: Path) -> list[str]:
    command = [
        "python",
        str(Path(__file__).resolve()),
        "--inside-env",
        "--stage",
        stage,
        "--cache-dir",
        str(cache_dir),
        "--yes",
        "--src-dir",
        str(args.src_dir),
        "--output-dir",
        str(args.output_dir),
        "--repo-id",
        args.repo_id,
        "--task-text",
        args.task_text,
        "--fps",
        str(args.fps),
        "--image-width",
        str(args.image_width),
        "--image-height",
        str(args.image_height),
        "--robot-type",
        args.robot_type,
        "--image-writer-processes",
        str(args.image_writer_processes),
        "--image-writer-threads",
        str(args.image_writer_threads),
        "--log-level",
        args.log_level,
    ]
    if args.include_depth:
        command.append("--include-depth")
    if args.no_videos:
        command.append("--no-videos")
    if args.push_to_hub:
        command.append("--push-to-hub")
    return command


def run_stage_in_conda(args: argparse.Namespace, env_name: str, stage: str, cache_dir: Path, source_ros: bool) -> None:
    quoted = " ".join(shlex_quote(part) for part in command_args_for_stage(args, stage, cache_dir))
    parts = [
        f"source {shlex_quote(str(conda_sh_path(args)))}",
        f"conda activate {shlex_quote(env_name)}",
    ]
    if source_ros:
        if not Path(args.ros_setup).exists():
            raise FileNotFoundError(f"ROS setup not found: {args.ros_setup}")
        parts.append(f"source {shlex_quote(args.ros_setup)}")
        if Path(args.workspace_setup).exists():
            parts.append(f"source {shlex_quote(args.workspace_setup)}")
    parts.append(f"exec {quoted}")
    subprocess.run(["bash", "-lc", " && ".join(parts)], check=True)


def orchestrate_two_env_conversion(args: argparse.Namespace) -> None:
    ensure_conda_environments(args)
    with tempfile.TemporaryDirectory(prefix="mtbot_lerobot_cache_") as tmp:
        cache_dir = Path(tmp)
        run_stage_in_conda(args, args.conda_env_name, "extract", cache_dir, source_ros=True)
        run_stage_in_conda(args, args.writer_conda_env_name, "write", cache_dir, source_ros=False)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def ensure_runtime_dependencies() -> None:
    global cv2, np

    if cv2 is not None and np is not None:
        return
    import cv2 as cv2_module
    import numpy as np_module

    cv2 = cv2_module
    np = np_module


def load_rosbag2() -> dict[str, Any]:
    try:
        from rclpy.serialization import deserialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise ImportError(
            "ROS bag Python APIs are required. Run through this script's conda entry so ROS is sourced "
            f"inside the conda environment. Original import error: {exc}"
        ) from exc

    return {
        "deserialize_message": deserialize_message,
        "ConverterOptions": ConverterOptions,
        "SequentialReader": SequentialReader,
        "StorageOptions": StorageOptions,
        "get_message": get_message,
    }


def open_reader(bag_uri: Path, rosbag2: dict[str, Any]) -> Any:
    reader = rosbag2["SequentialReader"]()
    storage_opts = rosbag2["StorageOptions"](uri=str(bag_uri), storage_id="mcap")
    converter_opts = rosbag2["ConverterOptions"]("", "")
    reader.open(storage_opts, converter_opts)
    return reader


def nearest_frame(ref_t: float, candidates: list[tuple[float, np.ndarray]]) -> tuple[float, np.ndarray] | None:
    if not candidates:
        return None
    times = np.asarray([item[0] for item in candidates], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - ref_t)))
    return candidates[idx]


def image_msg_to_numpy(msg: Any) -> np.ndarray:
    encoding = (msg.encoding or "").lower()
    height = int(msg.height)
    width = int(msg.width)

    if encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, int(msg.step))[:, : width * 3]
        arr = arr.reshape(height, width, 3)
        if encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr.copy()

    if encoding in ("rgba8", "bgra8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, int(msg.step))[:, : width * 4]
        arr = arr.reshape(height, width, 4)
        code = cv2.COLOR_BGRA2RGB if encoding == "bgra8" else cv2.COLOR_RGBA2RGB
        return cv2.cvtColor(arr, code).copy()

    if encoding in ("mono8", "8uc1"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(height, int(msg.step))[:, :width].copy()

    if encoding in ("mono16", "16uc1"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(height, int(msg.step) // 2)[:, :width].copy()

    if encoding == "32fc1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(height, int(msg.step) // 4)[:, :width].copy()

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def resize_image(image: np.ndarray, width: int, height: int, is_depth: bool = False) -> np.ndarray:
    if width <= 0 or height <= 0:
        return image
    if image.shape[1] == width and image.shape[0] == height:
        return image
    interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def normalize_depth_to_u8(depth: np.ndarray) -> np.ndarray:
    if depth.dtype == np.uint8:
        return depth
    depth_f = depth.astype(np.float32)
    finite = np.isfinite(depth_f)
    if not np.any(finite):
        return np.zeros(depth.shape, dtype=np.uint8)
    valid = depth_f[finite]
    lo = float(np.percentile(valid, 1))
    hi = float(np.percentile(valid, 99))
    if hi <= lo:
        return np.zeros(depth.shape, dtype=np.uint8)
    depth_f = np.clip((depth_f - lo) / (hi - lo), 0.0, 1.0)
    return (depth_f * 255.0).astype(np.uint8)


def joint_state_from_msg(msg: Any) -> np.ndarray:
    return np.asarray(
        [
            math.radians(msg.j1_cur_pos),
            math.radians(msg.j2_cur_pos),
            math.radians(msg.j3_cur_pos),
            math.radians(msg.j4_cur_pos),
            math.radians(msg.j5_cur_pos),
            math.radians(msg.j6_cur_pos),
            float(msg.gripper_position),
        ],
        dtype=np.float32,
    )


def find_mcap_bags(src_dir: Path) -> list[Path]:
    src_dir = src_dir.expanduser().resolve()
    if src_dir.is_file() and src_dir.suffix == ".mcap":
        return [src_dir]
    if src_dir.is_dir() and any(src_dir.glob("*.mcap")):
        return [src_dir]

    bags: list[Path] = []
    for path in sorted(src_dir.rglob("*.mcap")):
        parent = path.parent
        if parent not in bags:
            bags.append(parent)
    return bags


def load_mcap_episode(bag_uri: Path, image_width: int, image_height: int, include_depth: bool) -> McapEpisode:
    rosbag2 = load_rosbag2()
    reader = open_reader(bag_uri, rosbag2)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    missing = [topic for topic in (JOINT_TOPIC, COLOR_TOPIC) if topic not in topic_types]
    if missing:
        raise RuntimeError(f"Bag {bag_uri} is missing required topics: {missing}")

    msg_types = {topic: rosbag2["get_message"](msg_type) for topic, msg_type in topic_types.items()}
    deserialize_message = rosbag2["deserialize_message"]

    joint_data: list[tuple[float, np.ndarray]] = []
    color_data: list[tuple[float, np.ndarray]] = []
    depth_data: list[tuple[float, np.ndarray]] = []

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic not in (JOINT_TOPIC, COLOR_TOPIC, DEPTH_TOPIC):
            continue

        msg = deserialize_message(data, msg_types[topic])
        t_sec = timestamp_ns * 1e-9
        if topic == JOINT_TOPIC:
            joint_data.append((t_sec, joint_state_from_msg(msg)))
        elif topic == COLOR_TOPIC:
            image = resize_image(image_msg_to_numpy(msg), image_width, image_height)
            color_data.append((t_sec, image))
        elif topic == DEPTH_TOPIC and include_depth:
            image = resize_image(image_msg_to_numpy(msg), image_width, image_height, is_depth=True)
            depth_data.append((t_sec, image))

    if not joint_data:
        raise RuntimeError(f"No joint data found in bag: {bag_uri}")
    if not color_data:
        raise RuntimeError(f"No color image data found in bag: {bag_uri}")

    states: list[np.ndarray] = []
    main_images: list[np.ndarray] = []
    depth_images: list[np.ndarray] | None = [] if include_depth else None

    for t_sec, state in joint_data:
        color = nearest_frame(t_sec, color_data)
        if color is None:
            continue
        states.append(state)
        main_images.append(color[1])

        if include_depth and depth_images is not None:
            depth = nearest_frame(t_sec, depth_data)
            if depth is None:
                depth_img = np.zeros(main_images[-1].shape[:2], dtype=np.uint8)
            else:
                depth_img = normalize_depth_to_u8(depth[1])
            depth_images.append(depth_img[..., None])

    if len(states) < 2:
        raise RuntimeError(f"Need at least 2 frames for LeRobot episode: {bag_uri}")

    return McapEpisode(
        bag_uri=bag_uri,
        file_name=bag_uri.name,
        states=states,
        main_images=main_images,
        depth_images=depth_images,
    )


def motor_names() -> list[str]:
    return [f"joint_{i}" for i in range(1, 7)] + ["gripper"]


def dataset_features(config: ConverterConfig) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if config.use_videos else "image"
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": {"motors": motor_names()}},
        "observation.images.main": {
            "dtype": image_dtype,
            "shape": (config.image_height, config.image_width, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (7,), "names": {"motors": motor_names()}},
    }
    if config.include_depth:
        features["observation.images.depth"] = {
            "dtype": image_dtype,
            "shape": (config.image_height, config.image_width, 1),
            "names": ["height", "width", "channels"],
        }
    return features


def create_lerobot_dataset(config: ConverterConfig) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {
        "repo_id": config.repo_id,
        "fps": int(config.fps),
        "features": dataset_features(config),
        "root": config.output_dir,
        "robot_type": config.robot_type,
        "use_videos": config.use_videos,
        "image_writer_processes": config.image_writer_processes,
        "image_writer_threads": config.image_writer_threads,
    }
    signature = inspect.signature(LeRobotDataset.create)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return LeRobotDataset.create(**supported_kwargs)


def convert(config: ConverterConfig) -> None:
    logging.basicConfig(level=getattr(logging, config.log_level), format="%(message)s")
    ensure_runtime_dependencies()
    config.src_dir = config.src_dir.expanduser().resolve()
    config.output_dir = config.output_dir.expanduser().resolve()

    bag_uris = find_mcap_bags(config.src_dir)
    logging.info("Found %d MCAP episodes under %s", len(bag_uris), config.src_dir)
    if not bag_uris:
        raise RuntimeError(f"No MCAP files found under {config.src_dir}")

    if config.output_dir.exists():
        logging.info("Removing existing output directory: %s", config.output_dir)
        shutil.rmtree(config.output_dir)

    dataset = create_lerobot_dataset(config)
    saved_episodes = 0
    try:
        for bag_uri in bag_uris:
            try:
                episode = load_mcap_episode(
                    bag_uri,
                    image_width=config.image_width,
                    image_height=config.image_height,
                    include_depth=config.include_depth,
                )
            except Exception as exc:
                logging.warning("Skipping %s: %s", bag_uri, exc)
                continue

            frame_count = len(episode.states) - 1
            for idx in range(frame_count):
                frame = {
                    "observation.state": np.asarray(episode.states[idx], dtype=np.float32),
                    "observation.images.main": episode.main_images[idx],
                    "action": np.asarray(episode.states[idx + 1], dtype=np.float32),
                    "task": config.task_text,
                }
                if config.include_depth and episode.depth_images is not None:
                    frame["observation.images.depth"] = episode.depth_images[idx]
                dataset.add_frame(frame)

            dataset.save_episode()
            saved_episodes += 1
            logging.info("Saved episode %d from %s with %d frames", saved_episodes - 1, episode.bag_uri, frame_count)
    finally:
        dataset.finalize()

    if saved_episodes == 0:
        raise RuntimeError("No valid MCAP episodes were converted")

    if config.push_to_hub:
        dataset.push_to_hub(tags=["mtbot", "fairino", "manipulation", "mcap"], private=False, push_videos=True)

    logging.info("Latest LeRobot dataset ready: %s", config.output_dir)


def extract_to_cache(config: ConverterConfig, cache_dir: Path) -> None:
    logging.basicConfig(level=getattr(logging, config.log_level), format="%(message)s")
    ensure_runtime_dependencies()
    cache_dir.mkdir(parents=True, exist_ok=True)
    config.src_dir = config.src_dir.expanduser().resolve()

    bag_uris = find_mcap_bags(config.src_dir)
    logging.info("Found %d MCAP episodes under %s", len(bag_uris), config.src_dir)
    if not bag_uris:
        raise RuntimeError(f"No MCAP files found under {config.src_dir}")

    episodes: list[dict[str, Any]] = []
    for bag_uri in bag_uris:
        try:
            episode = load_mcap_episode(
                bag_uri,
                image_width=config.image_width,
                image_height=config.image_height,
                include_depth=config.include_depth,
            )
        except Exception as exc:
            logging.warning("Skipping %s: %s", bag_uri, exc)
            continue

        episode_index = len(episodes)
        episode_file = f"episode_{episode_index:06d}.npz"
        payload: dict[str, Any] = {
            "states": np.stack(episode.states).astype(np.float32),
            "main_images": np.stack(episode.main_images).astype(np.uint8),
        }
        if config.include_depth and episode.depth_images is not None:
            payload["depth_images"] = np.stack(episode.depth_images).astype(np.uint8)
        np.savez_compressed(cache_dir / episode_file, **payload)
        episodes.append({"file": episode_file, "bag_uri": str(episode.bag_uri), "frames": len(episode.states) - 1})
        logging.info("Extracted %s with %d training frames", episode.bag_uri, len(episode.states) - 1)

    if not episodes:
        raise RuntimeError("No valid MCAP episodes were extracted")

    manifest = {
        "repo_id": config.repo_id,
        "task_text": config.task_text,
        "fps": config.fps,
        "image_width": config.image_width,
        "image_height": config.image_height,
        "robot_type": config.robot_type,
        "include_depth": config.include_depth,
        "use_videos": config.use_videos,
        "image_writer_processes": config.image_writer_processes,
        "image_writer_threads": config.image_writer_threads,
        "push_to_hub": config.push_to_hub,
        "output_dir": str(config.output_dir),
        "episodes": episodes,
    }
    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_from_cache(config: ConverterConfig, cache_dir: Path) -> None:
    logging.basicConfig(level=getattr(logging, config.log_level), format="%(message)s")
    ensure_runtime_dependencies()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing extraction manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    config.output_dir = Path(manifest["output_dir"]).expanduser().resolve()
    if config.output_dir.exists():
        logging.info("Removing existing output directory: %s", config.output_dir)
        shutil.rmtree(config.output_dir)

    dataset = create_lerobot_dataset(config)
    saved_episodes = 0
    try:
        for item in manifest["episodes"]:
            data = np.load(cache_dir / item["file"])
            states = data["states"].astype(np.float32)
            main_images = data["main_images"]
            depth_images = data["depth_images"] if config.include_depth and "depth_images" in data.files else None
            frame_count = int(states.shape[0]) - 1
            for idx in range(frame_count):
                frame = {
                    "observation.state": states[idx],
                    "observation.images.main": main_images[idx],
                    "action": states[idx + 1],
                    "task": config.task_text,
                }
                if config.include_depth and depth_images is not None:
                    frame["observation.images.depth"] = depth_images[idx]
                dataset.add_frame(frame)
            dataset.save_episode()
            saved_episodes += 1
            logging.info("Saved episode %d from %s with %d frames", saved_episodes - 1, item["bag_uri"], frame_count)
    finally:
        dataset.finalize()

    if config.push_to_hub:
        dataset.push_to_hub(tags=["mtbot", "fairino", "manipulation", "mcap"], private=False, push_videos=True)
    logging.info("Latest LeRobot dataset ready: %s", config.output_dir)


def config_from_args(args: argparse.Namespace) -> ConverterConfig:
    return ConverterConfig(
        src_dir=args.src_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        task_text=args.task_text,
        fps=args.fps,
        image_width=args.image_width,
        image_height=args.image_height,
        robot_type=args.robot_type,
        include_depth=args.include_depth,
        use_videos=not args.no_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        push_to_hub=args.push_to_hub,
        log_level=args.log_level,
    )


def main() -> int:
    parser = build_parser()
    args = apply_interactive_prompts(parser.parse_args())
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    try:
        if args.setup_only:
            ros_python, writer_python = ensure_conda_environments(args)
            print(f"ROS extraction environment ready: {args.conda_env_name} ({ros_python})")
            print(f"LeRobot writer environment ready: {args.writer_conda_env_name} ({writer_python})")
            return 0
        if args.inside_env and args.stage == "extract":
            if args.cache_dir is None:
                raise ValueError("--cache-dir is required for extract stage")
            extract_to_cache(config_from_args(args), args.cache_dir)
            return 0
        if args.inside_env and args.stage == "write":
            if args.cache_dir is None:
                raise ValueError("--cache-dir is required for write stage")
            write_from_cache(config_from_args(args), args.cache_dir)
            return 0
        if not args.inside_env:
            orchestrate_two_env_conversion(args)
            return 0
        convert(config_from_args(args))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
