#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


V30_CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 500
V30_DATA_PATH = "data/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet"
V30_VIDEO_PATH = "videos/chunk-{chunk_index:03d}/{video_key}/file_{file_index:03d}.mp4"
V30_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet"
V30_EPISODES_STATS_PATH = "meta/episodes_stats/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet"
V30_TASKS_PATH = "meta/tasks/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            items.append(item)
    return items


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def chunk_and_file_index(item_index: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[int, int]:
    return item_index // chunk_size, item_index % chunk_size


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, flat_key))
        else:
            flattened[flat_key] = value
    return flattened


def validate_v21_dataset(src_dir: Path) -> None:
    required = [
        src_dir / "meta" / "info.json",
        src_dir / "meta" / "episodes.jsonl",
        src_dir / "meta" / "tasks.jsonl",
        src_dir / "meta" / "episodes_stats.jsonl",
        src_dir / "data",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Input is not a complete local LeRobot v2.1 dataset. Missing:\n{missing_text}")


def sorted_episode_parquets(src_dir: Path) -> list[Path]:
    paths = sorted((src_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v2.1 episode parquet files found under {src_dir / 'data'}")
    return paths


def convert_info(src_dir: Path, dst_dir: Path, data_file_size_mb: int, video_file_size_mb: int) -> None:
    info = read_json(src_dir / "meta" / "info.json")
    info["codebase_version"] = V30_CODEBASE_VERSION
    info.pop("total_chunks", None)
    info.pop("total_videos", None)
    info["data_files_size_in_mb"] = int(data_file_size_mb)
    info["video_files_size_in_mb"] = int(video_file_size_mb)
    info["data_path"] = V30_DATA_PATH
    info["video_path"] = V30_VIDEO_PATH
    info["fps"] = int(info["fps"])

    for feature in info.get("features", {}).values():
        if isinstance(feature, dict) and feature.get("dtype") != "video":
            feature["fps"] = info["fps"]

    write_json(dst_dir / "meta" / "info.json", info)


def convert_tasks(src_dir: Path, dst_dir: Path) -> None:
    tasks = sorted(read_jsonl(src_dir / "meta" / "tasks.jsonl"), key=lambda item: item["task_index"])
    df = pd.DataFrame(tasks, columns=["task_index", "task"])
    path = dst_dir / V30_TASKS_PATH.format(chunk_index=0, file_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def convert_data(src_dir: Path, dst_dir: Path, max_file_size_mb: int) -> list[dict[str, int]]:
    episode_paths = sorted_episode_parquets(src_dir)
    metadata: list[dict[str, int]] = []
    pending_paths: list[Path] = []
    pending_size = 0.0
    file_number = 0
    chunk_index = 0
    file_index = 0
    dataset_from_index = 0

    def flush(paths: list[Path], chunk_idx: int, file_idx: int) -> None:
        if not paths:
            return
        frames = [pd.read_parquet(path) for path in paths]
        df = pd.concat(frames, ignore_index=True)
        out_path = dst_dir / V30_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

    for episode_index, episode_path in enumerate(episode_paths):
        episode_frame_count = len(pd.read_parquet(episode_path, columns=["timestamp"]))
        episode_size = file_size_mb(episode_path)

        if pending_paths and pending_size + episode_size >= max_file_size_mb:
            flush(pending_paths, chunk_index, file_index)
            file_number += 1
            chunk_index, file_index = chunk_and_file_index(file_number)
            pending_paths = []
            pending_size = 0.0

        metadata.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": dataset_from_index,
                "dataset_to_index": dataset_from_index + episode_frame_count,
            }
        )
        dataset_from_index += episode_frame_count
        pending_paths.append(episode_path)
        pending_size += episode_size

    if pending_paths:
        flush(pending_paths, chunk_index, file_index)

    return metadata


def video_keys_from_info(src_dir: Path) -> list[str]:
    features = read_json(src_dir / "meta" / "info.json").get("features", {})
    return sorted(key for key, feature in features.items() if isinstance(feature, dict) and feature.get("dtype") == "video")


def convert_videos(src_dir: Path, dst_dir: Path, total_episodes: int) -> list[dict[str, Any]] | None:
    video_keys = video_keys_from_info(src_dir)
    if not video_keys:
        return None

    per_episode: list[dict[str, Any]] = [{"episode_index": idx} for idx in range(total_episodes)]
    for video_key in video_keys:
        for episode_index in range(total_episodes):
            chunk_index, file_index = chunk_and_file_index(episode_index)
            old_path = src_dir / "videos" / f"chunk-{chunk_index:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
            if not old_path.exists():
                raise FileNotFoundError(f"Missing video for {video_key} episode {episode_index}: {old_path}")
            new_path = dst_dir / V30_VIDEO_PATH.format(chunk_index=chunk_index, video_key=video_key, file_index=file_index)
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, new_path)
            per_episode[episode_index][f"videos/{video_key}/chunk_index"] = chunk_index
            per_episode[episode_index][f"videos/{video_key}/file_index"] = file_index

    return per_episode


def episode_rows(
    legacy_episodes: Iterable[dict[str, Any]],
    data_metadata: list[dict[str, int]],
    stats_items: Iterable[dict[str, Any]],
    video_metadata: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    legacy_by_index = {item["episode_index"]: item for item in legacy_episodes}
    stats_by_index = {item["episode_index"]: item.get("stats", {}) for item in stats_items}
    rows: list[dict[str, Any]] = []

    for item in data_metadata:
        episode_index = item["episode_index"]
        if episode_index not in legacy_by_index:
            raise ValueError(f"episodes.jsonl is missing episode_index={episode_index}")
        if episode_index not in stats_by_index:
            raise ValueError(f"episodes_stats.jsonl is missing episode_index={episode_index}")
        row = dict(item)
        if video_metadata is not None:
            row.update(video_metadata[episode_index])
        row.update(legacy_by_index[episode_index])
        row.update(flatten_dict({"stats": stats_by_index[episode_index]}))
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        rows.append(row)

    return rows


def convert_episodes_metadata(
    src_dir: Path,
    dst_dir: Path,
    data_metadata: list[dict[str, int]],
    video_metadata: list[dict[str, Any]] | None,
) -> None:
    legacy_episodes = sorted(read_jsonl(src_dir / "meta" / "episodes.jsonl"), key=lambda item: item["episode_index"])
    legacy_stats = sorted(read_jsonl(src_dir / "meta" / "episodes_stats.jsonl"), key=lambda item: item["episode_index"])
    rows = episode_rows(legacy_episodes, data_metadata, legacy_stats, video_metadata)

    episodes_path = dst_dir / V30_EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(episodes_path, index=False)

    stats_rows = [
        {"episode_index": item["episode_index"], **flatten_dict(item.get("stats", {}))}
        for item in legacy_stats
    ]
    stats_path = dst_dir / V30_EPISODES_STATS_PATH.format(chunk_index=0, file_index=0)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats_rows).to_parquet(stats_path, index=False)


def convert_local_dataset(src_dir: Path, dst_dir: Path, data_file_size_mb: int, video_file_size_mb: int, overwrite: bool) -> None:
    src_dir = src_dir.expanduser().resolve()
    dst_dir = dst_dir.expanduser().resolve()
    validate_v21_dataset(src_dir)

    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst_dir}. Use --overwrite to replace it.")
        shutil.rmtree(dst_dir)

    convert_info(src_dir, dst_dir, data_file_size_mb, video_file_size_mb)
    convert_tasks(src_dir, dst_dir)
    data_metadata = convert_data(src_dir, dst_dir, data_file_size_mb)
    video_metadata = convert_videos(src_dir, dst_dir, total_episodes=len(data_metadata))
    convert_episodes_metadata(src_dir, dst_dir, data_metadata, video_metadata)
    logging.info("Converted local LeRobot dataset: %s -> %s", src_dir, dst_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a local LeRobot v2.1 dataset directory to v3.0 layout")
    parser.add_argument("--src-dir", type=Path, required=True, help="Input local LeRobot v2.1 dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output local LeRobot v3.0 dataset directory")
    parser.add_argument("--data-file-size-in-mb", type=int, default=DEFAULT_DATA_FILE_SIZE_IN_MB)
    parser.add_argument("--video-file-size-in-mb", type=int, default=DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    convert_local_dataset(
        src_dir=args.src_dir,
        dst_dir=args.output_dir,
        data_file_size_mb=args.data_file_size_in_mb,
        video_file_size_mb=args.video_file_size_in_mb,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
