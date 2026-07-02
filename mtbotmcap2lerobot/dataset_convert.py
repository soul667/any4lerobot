import argparse
import inspect
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Task:
    task_file: Path = Path("./task_config.yml")
    task_rules: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.load_task_rules()

    def load_task_rules(self):
        if not self.task_file.exists():
            return
        import yaml

        with open(self.task_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if isinstance(raw, dict) and "task-rules" in raw:
            raw = raw["task-rules"]

        if isinstance(raw, dict):
            self.task_rules = {str(k): str(v) for k, v in raw.items()}
            return

        if isinstance(raw, list):
            rules = {}
            for item in raw:
                if isinstance(item, dict):
                    for key, value in item.items():
                        rules[str(key)] = str(value)
            self.task_rules = rules

    def get_task_by_name(self, name: str, default_task: str) -> str:
        for pattern, task in self.task_rules.items():
            if re.match(pattern, name):
                return task
        return default_task


@dataclass
class Config:
    src_dir: Path = Path("/home/mtbot/blackbox/robot_arm_data")
    output_dir: Path = Path("./data/lrobot/mtbot_mcap")
    log_level: str = "INFO"
    use_videos: bool = False
    image_writer_process: int = 2
    image_writer_threads: int = 4
    robot_type: str = "mtbot fairino arm"
    task_text: str = "robot arm demonstration"
    task_file: Path = Path("some/default/task/file")
    fps: int = 30
    repo_id: str = "username/mtbot_mcap"
    image_width: int = 640
    image_height: int = 480
    include_depth: bool = False
    push_to_hub: bool = False
    tags: List[str] = field(default_factory=list)

    task: Task = field(init=False)
    use_task_file: bool = field(init=False, default=False)

    def __post_init__(self):
        self.src_dir = Path(self.src_dir).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        self.task_file = Path(self.task_file).expanduser()
        self.use_task_file = self.task_file != Path("some/default/task/file")
        self.task = Task(task_file=self.task_file) if self.use_task_file else Task()

    def motor_names(self):
        return [f"joint_{i}" for i in range(1, 7)] + ["gripper"]

    def features(self):
        image_shape = [int(self.image_height), int(self.image_width), 3]
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"motors": self.motor_names()},
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"motors": self.motor_names()},
            },
            "observation.images.main": {
                "dtype": "video" if self.use_videos else "image",
                "shape": image_shape,
                "names": ["height", "width", "rgb"],
            },
            "timestamp": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
        }

        if self.include_depth:
            features["observation.images.depth"] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": [int(self.image_height), int(self.image_width), 1],
                "names": ["height", "width", "depth"],
            }

        return features


class DatasetConverter:
    def __init__(self, config: Config, episodes: List[Any]):
        self.config = config
        self.episodes = episodes

    def create_lerobot_dataset(self):
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            robot_type=self.config.robot_type,
            root=self.config.output_dir,
            fps=int(self.config.fps),
            use_videos=self.config.use_videos,
            features=self.config.features(),
            image_writer_threads=self.config.image_writer_threads,
            image_writer_processes=self.config.image_writer_process,
        )

        try:
            add_frame_params = list(inspect.signature(dataset.add_frame).parameters)
        except Exception:
            add_frame_params = []
        is_v21_api = len(add_frame_params) >= 2 and add_frame_params[1] == "task"

        for episode_index, episode in enumerate(self.episodes):
            task_text = (
                self.config.task.get_task_by_name(episode.file_name, self.config.task_text)
                if self.config.use_task_file
                else self.config.task_text
            )

            num_steps = len(episode.states)
            for t in range(num_steps - 1):
                state = np.asarray(episode.states[t], dtype=np.float32)
                next_state = np.asarray(episode.states[t + 1], dtype=np.float32)
                frame_data = {
                    "observation.state": state,
                    "observation.images.main": episode.main_images[t],
                    "action": next_state,
                }

                if self.config.include_depth and episode.depth_images is not None:
                    frame_data["observation.images.depth"] = episode.depth_images[t]

                if is_v21_api:
                    timestamp = float(t / float(self.config.fps))
                    dataset.add_frame(frame_data, task_text, timestamp=timestamp)
                else:
                    timestamp = np.float32(t / float(self.config.fps))
                    frame_data["timestamp"] = np.array([timestamp], dtype=np.float32)
                    frame_data["frame_index"] = np.array([t], dtype=np.int64)
                    frame_data["episode_index"] = np.array([episode_index], dtype=np.int64)
                    frame_data["task"] = task_text
                    dataset.add_frame(frame_data)

            dataset.save_episode()
            logging.info("Saved episode %d from %s with %d frames", episode_index, episode.bag_uri, num_steps - 1)

        return dataset

    def push_to_hub(self):
        if self.config.push_to_hub:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            LeRobotDataset(self.config.repo_id, root=self.config.output_dir).push_to_hub(
                tags=self.config.tags,
                private=False,
                push_videos=True,
            )


def reset_output_dir(output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)


def main(config: Config):
    from mcap_dataset import find_mcap_bags, load_mcap_episode

    reset_output_dir(config.output_dir)

    bag_uris = find_mcap_bags(config.src_dir)
    logging.info("Found %d MCAP episodes under %s", len(bag_uris), config.src_dir)
    if not bag_uris:
        raise RuntimeError(f"No MCAP files found under {config.src_dir}")

    episodes = []
    for bag_uri in bag_uris:
        try:
            episode = load_mcap_episode(
                bag_uri,
                image_width=config.image_width,
                image_height=config.image_height,
                include_depth=config.include_depth,
            )
            episodes.append(episode)
        except Exception as exc:
            logging.warning("Skipping %s: %s", bag_uri, exc)

    if not episodes:
        raise RuntimeError("No valid MCAP episodes were loaded")

    converter = DatasetConverter(config=config, episodes=episodes)
    converter.create_lerobot_dataset()
    converter.push_to_hub()
    logging.info("Conversion completed: %s", config.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MTBot MCAP bags to a LeRobot dataset")
    parser.add_argument("--src-dir", type=Path, default=Path("/home/mtbot/blackbox/robot_arm_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./data/lrobot/mtbot_mcap"))
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--image-writer-process", type=int, default=2)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--robot-type", type=str, default="mtbot fairino arm")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repo_id", type=str, default="username/mtbot_mcap")
    parser.add_argument("--task-text", type=str, default="robot arm demonstration")
    parser.add_argument("--task-file", type=Path, default=Path("some/default/task/file"))
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--tags", type=str, nargs="+", default=[])

    args = parser.parse_args()
    config = Config(**vars(args))
    logging.basicConfig(level=getattr(logging, config.log_level), format="%(message)s")
    main(config)
