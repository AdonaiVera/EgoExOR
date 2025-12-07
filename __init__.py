"""
EgoExOR: An Egocentric-Exocentric Operating Room Dataset for FiftyOne

Multi-modal surgical dataset with synchronized RGB video, eye gaze, hand tracking,
audio, 3D point clouds, and scene graph annotations.

Usage:
    # Option 1: Download specific files from HuggingFace (default: smallest)
    dataset = foz.load_zoo_dataset("ardamamur/EgoExOR", max_samples=10)

    # Option 2: Use local h5 files
    dataset = foz.load_zoo_dataset("ardamamur/EgoExOR", h5_path="/path/to/miss_4.h5")

    # Option 3: Download full dataset and merge
    dataset = foz.load_zoo_dataset("ardamamur/EgoExOR", download_full=True)
"""

import os
import glob
import h5py
import numpy as np
import fiftyone as fo
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from huggingface_hub import hf_hub_download

CAMERA_TYPE_MAPPING = {
    "head_surgeon": 1, "assistant": 2, "circulator": 3, "anesthetist": 4,
    "or_light": 5, "microscope": 6, "external_1": 7, "external_2": 8,
    "external_3": 9, "external_4": 10, "external_5": 11, "simstation": 12,
    "ultrasound": 13, "blank": -1
}

EGOCENTRIC_SOURCES = {"head_surgeon", "assistant", "circulator", "anesthetist"}
EXOCENTRIC_SOURCES = {"or_light", "microscope", "external_1", "external_2",
                      "external_3", "external_4", "external_5", "simstation", "ultrasound"}

HF_REPO = "ardamamur/EgoExOR"
AVAILABLE_H5_FILES = [
    "miss_1.h5", "miss_2.h5", "miss_3.h5", "miss_4.h5",
    "ultrasound_1.h5", "ultrasound_2.h5", "ultrasound_3.h5", "ultrasound_4.h5",
    "ultrasound_5_14.h5", "ultrasound_5_58.h5"
]
DEFAULT_H5_FILE = "miss_4.h5"


def _load_metadata(f: h5py.File) -> Dict[str, Any]:
    """Load dataset metadata from h5 file."""
    metadata = {}

    if "metadata/dataset" in f:
        ds_grp = f["metadata/dataset"]
        metadata["dataset_info"] = {
            "version": _decode(ds_grp.attrs.get("version", "unknown")),
            "creation_date": _decode(ds_grp.attrs.get("creation_date", "unknown")),
            "title": _decode(ds_grp.attrs.get("title", "EgoExOR")),
        }

    if "metadata/vocabulary/entity" in f:
        entities = f["metadata/vocabulary/entity"][:]
        metadata["entity_vocabulary"] = {
            int(e["id"]): _decode(e["name"]) for e in entities
        }

    if "metadata/vocabulary/relation" in f:
        relations = f["metadata/vocabulary/relation"][:]
        metadata["relation_vocabulary"] = {
            int(r["id"]): _decode(r["name"]) for r in relations
        }

    if "metadata/sources/sources" in f:
        sources = f["metadata/sources/sources"][:]
        metadata["sources"] = {
            int(s["id"]): _decode(s["name"]) for s in sources
        }

    return metadata


def download_and_prepare(
    dataset_dir: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    h5_path: Optional[str] = None,
    h5_files: Optional[List[str]] = None,
    download_full: bool = False,
    **kwargs
) -> Tuple[Optional[str], int, List[str]]:
    """
    Download and prepare EgoExOR dataset.

    Args:
        dataset_dir: Directory to store downloaded files and extracted frames
        split: Optional split (train/validation/test) - requires merged file
        max_samples: Max samples to report
        h5_path: Path to local h5 file or directory containing h5 files
        h5_files: List of specific h5 files to download from HuggingFace
        download_full: If True, download all files and merge into egoexor.h5

    Returns:
        Tuple of (dataset_type, num_samples, classes)
    """
    os.makedirs(dataset_dir, exist_ok=True)

    h5_paths = _resolve_h5_paths(dataset_dir, h5_path, h5_files, download_full)

    num_samples = sum(_count_samples_in_file(p) for p in h5_paths)
    classes = list(EGOCENTRIC_SOURCES | EXOCENTRIC_SOURCES)

    return None, num_samples, classes


def load_dataset(
    dataset: fo.Dataset,
    dataset_dir: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    h5_path: Optional[str] = None,
    h5_files: Optional[List[str]] = None,
    download_full: bool = False,
    **kwargs
) -> fo.Dataset:
    """
    Load EgoExOR dataset into FiftyOne with grouped samples.

    Args:
        dataset: FiftyOne dataset to populate
        dataset_dir: Directory containing h5 files or for downloads
        split: Optional split filter (requires merged file with splits)
        max_samples: Maximum number of timepoints to load
        h5_path: Path to local h5 file or directory containing h5 files
        h5_files: List of specific h5 files to download from HuggingFace
        download_full: If True, download all files and merge

    Returns:
        Populated FiftyOne dataset with grouped samples
    """
    h5_paths = _resolve_h5_paths(dataset_dir, h5_path, h5_files, download_full)

    _setup_group_slices(dataset)

    # Load and store metadata from first h5 file
    metadata = {}
    if h5_paths:
        with h5py.File(h5_paths[0], "r") as f:
            metadata = _load_metadata(f)

    if metadata:
        _store_dataset_metadata(dataset, metadata)

    sample_count = 0
    for h5_file_path in h5_paths:
        if max_samples and sample_count >= max_samples:
            break

        remaining = max_samples - sample_count if max_samples else None
        loaded = _load_from_h5(dataset, h5_file_path, dataset_dir, remaining, split, metadata)
        sample_count += loaded

    return dataset


def _store_dataset_metadata(dataset: fo.Dataset, metadata: Dict[str, Any]):
    """Store metadata as dataset-level info."""
    if "dataset_info" in metadata:
        dataset.info["version"] = metadata["dataset_info"]["version"]
        dataset.info["creation_date"] = metadata["dataset_info"]["creation_date"]
        dataset.info["title"] = metadata["dataset_info"]["title"]

    if "entity_vocabulary" in metadata:
        dataset.info["entity_classes"] = list(metadata["entity_vocabulary"].values())
        dataset.info["entity_id_to_name"] = {
            str(k): v for k, v in metadata["entity_vocabulary"].items()
        }

    if "relation_vocabulary" in metadata:
        dataset.info["relation_classes"] = list(metadata["relation_vocabulary"].values())
        dataset.info["relation_id_to_name"] = {
            str(k): v for k, v in metadata["relation_vocabulary"].items()
        }

    if "sources" in metadata:
        dataset.info["camera_sources"] = {
            str(k): v for k, v in metadata["sources"].items()
        }

    dataset.save()


def _resolve_h5_paths(dataset_dir: str, h5_path: Optional[str],
                      h5_files: Optional[List[str]], download_full: bool) -> List[str]:
    """
    Resolve which h5 files to use based on options.

    Priority:
    1. h5_path (local file or directory)
    2. download_full (download all + merge)
    3. h5_files (specific files from HF)
    4. Default: download smallest file
    """
    if h5_path:
        return _resolve_local_path(h5_path)

    if download_full:
        return [_download_and_merge(dataset_dir)]

    if h5_files:
        return _download_specific_files(dataset_dir, h5_files)

    return _download_specific_files(dataset_dir, [DEFAULT_H5_FILE])


def _resolve_local_path(h5_path: str) -> List[str]:
    """Resolve local h5 path - can be file or directory."""
    path = Path(h5_path)

    if path.is_file() and path.suffix == ".h5":
        return [str(path)]

    if path.is_dir():
        h5_files = list(path.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {h5_path}")
        return [str(f) for f in sorted(h5_files)]

    raise FileNotFoundError(f"Invalid h5_path: {h5_path}")


def _download_specific_files(dataset_dir: str, h5_files: List[str]) -> List[str]:
    """Download specific h5 files from HuggingFace."""
    paths = []
    for filename in h5_files:
        if filename not in AVAILABLE_H5_FILES:
            raise ValueError(f"Unknown file: {filename}. Available: {AVAILABLE_H5_FILES}")

        local_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename} from HuggingFace...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=dataset_dir,
                local_dir_use_symlinks=False
            )
        paths.append(local_path)
    return paths


def _download_and_merge(dataset_dir: str) -> str:
    """Download all h5 files and merge into single file."""
    merged_path = os.path.join(dataset_dir, "egoexor.h5")

    if os.path.exists(merged_path):
        print(f"Using existing merged file: {merged_path}")
        return merged_path

    print("Downloading all h5 files from HuggingFace...")
    downloaded = []
    for filename in AVAILABLE_H5_FILES:
        local_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(local_path):
            print(f"  Downloading {filename}...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=dataset_dir,
                local_dir_use_symlinks=False
            )
        downloaded.append(local_path)

    splits_path = os.path.join(dataset_dir, "splits.h5")
    if not os.path.exists(splits_path):
        print("  Downloading splits.h5...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename="splits.h5",
            repo_type="dataset",
            local_dir=dataset_dir,
            local_dir_use_symlinks=False
        )

    print("Merging h5 files...")
    from data.utils.merge_h5 import merge_files
    merge_files(downloaded, splits_path, merged_path)

    return merged_path


def _setup_group_slices(dataset: fo.Dataset):
    """Configure group field for all camera sources."""
    dataset.add_group_field("group", default="head_surgeon")


def _load_from_h5(dataset: fo.Dataset, h5_path: str, dataset_dir: str,
                  max_samples: Optional[int], split: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> int:
    """Load samples from a single h5 file."""
    samples_to_add = []
    sample_count = 0

    split_frames = None
    if split:
        split_frames = _get_split_frames(h5_path, split)

    with h5py.File(h5_path, "r") as f:
        for surgery_type in f["data"].keys():
            if max_samples and sample_count >= max_samples:
                break

            for procedure_id in f[f"data/{surgery_type}"].keys():
                if max_samples and sample_count >= max_samples:
                    break

                takes_path = f"data/{surgery_type}/{procedure_id}/take"
                if takes_path not in f:
                    continue

                for take_id in f[takes_path].keys():
                    if max_samples and sample_count >= max_samples:
                        break

                    take_path = f"{takes_path}/{take_id}"
                    take_samples, count = _load_take(
                        f, take_path, surgery_type, procedure_id, take_id,
                        dataset_dir, split_frames,
                        max_samples - sample_count if max_samples else None,
                        metadata
                    )
                    samples_to_add.extend(take_samples)
                    sample_count += count

    if samples_to_add:
        dataset.add_samples(samples_to_add)

    return sample_count


def _load_take(f: h5py.File, take_path: str, surgery_type: str,
               procedure_id: str, take_id: str, dataset_dir: str,
               split_frames: Optional[set],
               max_frames: Optional[int],
               metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[fo.Sample], int]:
    """Load frames from a single take."""
    samples = []
    loaded_count = 0

    source_map = _get_source_map(f, take_path)
    rgb_data = f[f"{take_path}/frames/rgb"]
    num_frames = rgb_data.shape[0]

    gaze_data = f.get(f"{take_path}/eye_gaze/coordinates")
    gaze_depth_data = f.get(f"{take_path}/eye_gaze_depth/values")
    hand_data = f.get(f"{take_path}/hand_tracking/positions")
    audio_snippets = f.get(f"{take_path}/audio/snippets")
    pc_coords = f.get(f"{take_path}/point_cloud/coordinates")
    pc_colors = f.get(f"{take_path}/point_cloud/colors")

    frames_dir = Path(dataset_dir) / "frames" / surgery_type / procedure_id / take_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(num_frames):
        if split_frames is not None:
            key = (surgery_type, int(procedure_id), int(take_id), frame_idx)
            if key not in split_frames:
                continue

        if max_frames and loaded_count >= max_frames:
            break

        group = fo.Group()
        text_annotations, tokenized_annotations = _get_frame_annotations(f, take_path, frame_idx, metadata)

        for cam_idx, cam_name in source_map.items():
            frame = rgb_data[frame_idx, cam_idx]
            filepath = frames_dir / f"frame_{frame_idx}_{cam_name}.jpg"

            if not filepath.exists():
                import cv2
                cv2.imwrite(str(filepath), frame)

            sample = fo.Sample(filepath=str(filepath), group=group.element(cam_name))
            sample["surgery_type"] = surgery_type
            sample["procedure_id"] = int(procedure_id)
            sample["take_id"] = int(take_id)
            sample["frame_idx"] = frame_idx
            sample["camera_name"] = cam_name
            sample["is_egocentric"] = cam_name in EGOCENTRIC_SOURCES

            if cam_name in EGOCENTRIC_SOURCES:
                _add_egocentric_data(sample, frame_idx, cam_name, frame.shape,
                                     gaze_data, gaze_depth_data, hand_data)

            if text_annotations:
                sample["scene_graph"] = text_annotations
                sample["relations"] = fo.Classifications(
                    classifications=[
                        fo.Classification(label=ann) for ann in text_annotations
                    ]
                )
            if tokenized_annotations:
                sample["scene_graph_tokens"] = tokenized_annotations
            if audio_snippets is not None:
                sample["has_audio"] = True
                audio_path = _save_audio_snippet(audio_snippets[frame_idx], frames_dir, frame_idx)
                if audio_path:
                    sample["audio_path"] = audio_path

            samples.append(sample)

        if pc_coords is not None and pc_colors is not None:
            pc_sample = _create_pointcloud_sample(
                group, pc_coords[frame_idx], pc_colors[frame_idx],
                frames_dir, frame_idx, surgery_type, procedure_id, take_id,
                text_annotations, tokenized_annotations
            )
            samples.append(pc_sample)

        loaded_count += 1

    return samples, loaded_count


def _get_source_map(f: h5py.File, take_path: str) -> Dict[int, str]:
    """Get camera index to name mapping."""
    sources_group = f[f"{take_path}/sources"]
    return {
        i: _decode(sources_group.attrs[f"source_{i}"])
        for i in range(sources_group.attrs["source_count"])
    }


def _get_split_frames(h5_path: str, split: str) -> Optional[set]:
    """Get set of (surgery_type, procedure_id, take_id, frame_idx) for a split."""
    frames = set()
    with h5py.File(h5_path, "r") as f:
        split_path = f"splits/{split}"
        if split_path not in f:
            return None

        split_data = f[split_path][:]
        for row in split_data:
            surgery_type = _decode(row[0]) if len(row) > 0 else None
            procedure_id = int(row[1]) if len(row) > 1 else None
            take_id = int(row[2]) if len(row) > 2 else None
            frame_id = int(row[3]) if len(row) > 3 else None
            if all(v is not None for v in [surgery_type, procedure_id, take_id, frame_id]):
                frames.add((surgery_type, procedure_id, take_id, frame_id))
    return frames if frames else None


def _add_egocentric_data(sample: fo.Sample, frame_idx: int, cam_name: str,
                         frame_shape: tuple, gaze_data, gaze_depth_data, hand_data):
    """Add gaze and hand tracking data to egocentric samples."""
    if gaze_data is not None:
        gaze_info = _get_gaze_for_camera(
            gaze_data[frame_idx],
            gaze_depth_data[frame_idx] if gaze_depth_data else None,
            cam_name, frame_shape
        )
        if gaze_info:
            depth_label = f"depth: {gaze_info['depth']:.2f}m" if gaze_info.get("depth") else "gaze"
            sample["gaze"] = fo.Keypoint(points=[gaze_info["point"]], label=depth_label)
            if gaze_info.get("depth") is not None:
                sample["gaze_depth"] = float(gaze_info["depth"])

    if hand_data is not None:
        left_hand, right_hand = _get_hands_for_camera(hand_data[frame_idx], cam_name, frame_shape)
        keypoints = []
        if left_hand:
            keypoints.append(fo.Keypoint(points=left_hand, label="left_hand"))
        if right_hand:
            keypoints.append(fo.Keypoint(points=right_hand, label="right_hand"))
        if keypoints:
            sample["hand_tracking"] = fo.Keypoints(keypoints=keypoints)


def _create_pointcloud_sample(group: fo.Group, coords: np.ndarray, colors: np.ndarray,
                               frames_dir: Path, frame_idx: int, surgery_type: str,
                               procedure_id: str, take_id: str,
                               text_annotations: Optional[List[str]],
                               tokenized_annotations: Optional[List[List[int]]] = None) -> fo.Sample:
    """Create a point cloud sample with fo3d scene file."""
    pc_dir = frames_dir / "pointcloud"
    pc_dir.mkdir(parents=True, exist_ok=True)
    ply_filepath = pc_dir / f"frame_{frame_idx}.ply"
    fo3d_filepath = pc_dir / f"frame_{frame_idx}.fo3d"

    if not ply_filepath.exists():
        _save_point_cloud_ply(coords, colors, str(ply_filepath))

    if not fo3d_filepath.exists():
        _create_fo3d_scene(str(ply_filepath), str(fo3d_filepath))

    sample = fo.Sample(filepath=str(fo3d_filepath), group=group.element("point_cloud"))
    sample["surgery_type"] = surgery_type
    sample["procedure_id"] = int(procedure_id)
    sample["take_id"] = int(take_id)
    sample["frame_idx"] = frame_idx
    sample["camera_name"] = "point_cloud"
    sample["is_egocentric"] = False
    if text_annotations:
        sample["scene_graph"] = text_annotations
    if tokenized_annotations:
        sample["scene_graph_tokens"] = tokenized_annotations
    return sample


def _create_fo3d_scene(ply_path: str, fo3d_path: str):
    """Create an fo3d scene file for a PLY point cloud."""
    from fiftyone.core.threed import Scene, PlyMesh

    scene = Scene()
    ply_mesh = PlyMesh("point_cloud", ply_path, is_point_cloud=True)
    scene.add(ply_mesh)
    scene.write(fo3d_path)


def _count_samples_in_file(h5_path: str) -> int:
    """Count total frames in an h5 file."""
    count = 0
    try:
        with h5py.File(h5_path, "r") as f:
            for surgery_type in f["data"].keys():
                for procedure_id in f[f"data/{surgery_type}"].keys():
                    takes_path = f"data/{surgery_type}/{procedure_id}/take"
                    if takes_path in f:
                        for take_id in f[takes_path].keys():
                            rgb_path = f"{takes_path}/{take_id}/frames/rgb"
                            if rgb_path in f:
                                count += f[rgb_path].shape[0]
    except Exception:
        pass
    return count


def _decode(val):
    return val.decode() if isinstance(val, bytes) else val


def _get_frame_annotations(f: h5py.File, take_path: str, frame_idx: int,
                           metadata: Optional[Dict[str, Any]] = None
                           ) -> Tuple[Optional[List[str]], Optional[List[List[int]]]]:
    """Get frame annotations in both text and tokenized form."""
    text_annotations = None
    tokenized_annotations = None

    ann_base = f"{take_path}/annotations/frame_{frame_idx}"

    rel_path = f"{ann_base}/rel_annotations"
    if rel_path in f:
        raw = f[rel_path][:]
        text_annotations = [" ".join(_decode(x) for x in row) for row in raw]

    sg_path = f"{ann_base}/scene_graph"
    if sg_path in f:
        raw = f[sg_path][:]
        tokenized_annotations = [[int(x) for x in row] for row in raw]

    return text_annotations, tokenized_annotations


def _get_gaze_for_camera(gaze_points: np.ndarray, gaze_depths: Optional[np.ndarray],
                         cam_name: str, frame_shape: tuple) -> Optional[Dict[str, Any]]:
    expected_type = CAMERA_TYPE_MAPPING.get(cam_name)
    if expected_type is None:
        return None
    h, w = frame_shape[:2]
    for idx, g in enumerate(gaze_points):
        if g[0] == expected_type and not np.allclose(g[1:], [-1, -1]):
            result = {"point": (float(g[1] / w), float(g[2] / h))}
            if gaze_depths is not None and idx < len(gaze_depths):
                result["depth"] = float(gaze_depths[idx])
            return result
    return None


def _get_hands_for_camera(hand_points: np.ndarray, cam_name: str,
                          frame_shape: tuple) -> Tuple[Optional[List], Optional[List]]:
    """Extract left and right hand keypoints for a specific camera."""
    expected_id = CAMERA_TYPE_MAPPING.get(cam_name)
    if expected_id is None:
        return None, None

    h, w = frame_shape[:2]

    for slot in hand_points:
        if np.isnan(slot[0]):
            continue
        cam_id = int(slot[0])
        if cam_id != expected_id:
            continue

        left_pts = []
        for i in range(4):
            x, y = slot[1 + i*2], slot[2 + i*2]
            if not np.isnan(x) and not np.isnan(y):
                left_pts.append((float(x / w), float(y / h)))

        right_pts = []
        for i in range(4):
            x, y = slot[9 + i*2], slot[10 + i*2]
            if not np.isnan(x) and not np.isnan(y):
                right_pts.append((float(x / w), float(y / h)))

        return (left_pts if left_pts else None, right_pts if right_pts else None)

    return None, None


def _save_audio_snippet(audio_data: np.ndarray, frames_dir: Path, frame_idx: int,
                        sample_rate: int = 48000) -> Optional[str]:
    """Save audio snippet as WAV file."""
    import wave

    audio_dir = frames_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"frame_{frame_idx}.wav"

    if wav_path.exists():
        return str(wav_path)

    try:
        audio_clean = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_clean = np.clip(audio_clean, -1.0, 1.0)
        audio_int16 = (audio_clean * 32767).astype(np.int16)

        with wave.open(str(wav_path), 'w') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return str(wav_path)
    except Exception:
        return None


def _save_point_cloud_ply(coords: np.ndarray, colors: np.ndarray, filepath: str):
    valid_mask = ~np.isnan(coords).any(axis=1)
    coords, colors = coords[valid_mask], colors[valid_mask]
    if len(coords) == 0:
        coords = np.array([[0, 0, 0]], dtype=np.float32)
        colors = np.array([[0, 0, 0]], dtype=np.float32)
    colors_uint8 = (colors * 255).astype(np.uint8)
    with open(filepath, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(coords)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(coords)):
            f.write(f"{coords[i,0]} {coords[i,1]} {coords[i,2]} "
                    f"{colors_uint8[i,0]} {colors_uint8[i,1]} {colors_uint8[i,2]}\n")
