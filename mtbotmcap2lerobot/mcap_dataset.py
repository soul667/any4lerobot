import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


JOINT_TOPIC = "/slave/nonrt_state_data"
COLOR_TOPIC = "/camera/d405/color/image_raw"
DEPTH_TOPIC = "/camera/d405/aligned_depth_to_color/image_raw"


def _load_rosbag2():
    try:
        from rclpy.serialization import deserialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise ImportError(
            "ROS bag Python APIs are required. Source your ROS environment and use the same Python "
            f"minor version as ROS. Original import error: {exc}"
        ) from exc

    return {
        "deserialize_message": deserialize_message,
        "ConverterOptions": ConverterOptions,
        "SequentialReader": SequentialReader,
        "StorageOptions": StorageOptions,
        "get_message": get_message,
    }


def _open_reader(bag_uri: Path, rosbag2: Dict[str, Any]):
    reader = rosbag2["SequentialReader"]()
    storage_opts = rosbag2["StorageOptions"](uri=str(bag_uri), storage_id="mcap")
    converter_opts = rosbag2["ConverterOptions"]("", "")
    reader.open(storage_opts, converter_opts)
    return reader


def _nearest_frame(ref_t: float, candidates: List[tuple]):
    if not candidates:
        return None
    times = np.asarray([item[0] for item in candidates], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - ref_t)))
    return candidates[idx]


def _image_msg_to_numpy(msg) -> np.ndarray:
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
        if encoding == "bgra8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr.copy()

    if encoding in ("mono8", "8uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, int(msg.step))[:, :width]
        return arr.copy()

    if encoding in ("mono16", "16uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, int(msg.step) // 2)[:, :width]
        return arr.copy()

    if encoding == "32fc1":
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(height, int(msg.step) // 4)[:, :width]
        return arr.copy()

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def _resize_image(image: np.ndarray, width: int, height: int, is_depth: bool = False) -> np.ndarray:
    if width <= 0 or height <= 0:
        return image
    if image.shape[1] == width and image.shape[0] == height:
        return image
    interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _normalize_depth_to_u8(depth: np.ndarray) -> np.ndarray:
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


def _joint_state_from_msg(msg) -> np.ndarray:
    joints = np.asarray(
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
    return joints


@dataclass
class McapEpisode:
    bag_uri: Path
    file_name: str
    timestamps: List[float]
    states: List[np.ndarray]
    main_images: List[np.ndarray]
    depth_images: Optional[List[np.ndarray]] = None


def find_mcap_bags(src_dir: Path) -> List[Path]:
    src_dir = src_dir.expanduser().resolve()
    if src_dir.is_file() and src_dir.suffix == ".mcap":
        return [src_dir]

    if src_dir.is_dir() and any(src_dir.glob("*.mcap")):
        return [src_dir]

    bags = []
    for path in sorted(src_dir.rglob("*.mcap")):
        parent = path.parent
        if parent not in bags:
            bags.append(parent)
    return bags


def load_mcap_episode(
    bag_uri: Path,
    image_width: int,
    image_height: int,
    include_depth: bool = False,
) -> McapEpisode:
    rosbag2 = _load_rosbag2()
    reader = _open_reader(bag_uri, rosbag2)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    missing = [topic for topic in (JOINT_TOPIC, COLOR_TOPIC) if topic not in topic_types]
    if missing:
        raise RuntimeError(f"Bag {bag_uri} is missing required topics: {missing}")

    msg_types = {topic: rosbag2["get_message"](msg_type) for topic, msg_type in topic_types.items()}
    deserialize_message = rosbag2["deserialize_message"]

    joint_data = []
    color_data = []
    depth_data = []

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic not in (JOINT_TOPIC, COLOR_TOPIC, DEPTH_TOPIC):
            continue

        msg = deserialize_message(data, msg_types[topic])
        t_sec = timestamp_ns * 1e-9

        if topic == JOINT_TOPIC:
            joint_data.append((t_sec, _joint_state_from_msg(msg)))
        elif topic == COLOR_TOPIC:
            img = _image_msg_to_numpy(msg)
            img = _resize_image(img, image_width, image_height, is_depth=False)
            color_data.append((t_sec, img))
        elif topic == DEPTH_TOPIC and include_depth:
            img = _image_msg_to_numpy(msg)
            img = _resize_image(img, image_width, image_height, is_depth=True)
            depth_data.append((t_sec, img))

    if not joint_data:
        raise RuntimeError(f"No joint data found in bag: {bag_uri}")
    if not color_data:
        raise RuntimeError(f"No color image data found in bag: {bag_uri}")

    timestamps = []
    states = []
    main_images = []
    depth_images = [] if include_depth else None

    t0 = joint_data[0][0]
    for t_sec, state in joint_data:
        color = _nearest_frame(t_sec, color_data)
        if color is None:
            continue

        timestamps.append(float(t_sec - t0))
        states.append(state)
        main_images.append(color[1])

        if include_depth and depth_images is not None:
            depth = _nearest_frame(t_sec, depth_data)
            if depth is None:
                depth_img = np.zeros(main_images[-1].shape[:2], dtype=np.uint8)
            else:
                depth_img = _normalize_depth_to_u8(depth[1])
            depth_images.append(depth_img[..., None])

    if len(states) < 2:
        raise RuntimeError(f"Need at least 2 frames for LeRobot episode: {bag_uri}")

    logging.debug("Loaded %s with %d frames", bag_uri, len(states))
    return McapEpisode(
        bag_uri=bag_uri,
        file_name=bag_uri.name,
        timestamps=timestamps,
        states=states,
        main_images=main_images,
        depth_images=depth_images,
    )
