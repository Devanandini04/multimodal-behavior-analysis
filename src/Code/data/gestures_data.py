from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from config import REQUIRED_KEYPOINTS
from pose.openpose_base import BODY_25_JOINTS


def load_gesture_time_ranges(filepath: str) -> List[Tuple[float, float]]:
    """
    Load gesture annotation time ranges from a CSV file.

    The CSV file must contain the following columns:
    - 'Begin Time - msec'
    - 'End Time - msec'

    Returns:
        List of (begin_time_ms, end_time_ms) tuples.
    """
    df = pd.read_csv(filepath)

    if "Begin Time - msec" not in df.columns or "End Time - msec" not in df.columns:
        raise ValueError("CSV must contain 'Begin Time - msec' and 'End Time - msec' columns")

    begin_times = df["Begin Time - msec"].tolist()
    end_times = df["End Time - msec"].tolist()

    return list(zip(begin_times, end_times))


def generate_dummy_keypoints() -> Dict:
    """
    Generate dummy keypoints for missing persons or joints.

    Dummy keypoints ensure consistent tensor dimensions
    when fewer persons are detected in a frame.
    """
    return {joint: [-1.0, -1.0, -1.0] for joint in BODY_25_JOINTS}


def arrange_train_data(
    keypoints: Dict,
    gesture_time_ranges: List[Tuple[float, float]],
    fps: float,
    max_persons: int
) -> Dict:
    """
    Convert pose keypoints and gesture time ranges into frame-level training data.

    Each frame is labeled with:
    - pose keypoints (with dummy padding if required)
    - gesture presence (True / False)

    Returns:
        Dictionary indexed by video/segment keys containing frame-wise data.
    """
    data = {}

    for segment_key, segment_data in keypoints.items():
        persons = [
            p for p in segment_data.keys()
            if p not in ("start_frame", "end_frame")
        ]
        num_persons = len(persons)

        start_frame = segment_data["start_frame"]
        end_frame = segment_data["end_frame"]

        start_time_ms = start_frame / fps * 1000
        end_time_ms = end_frame / fps * 1000

        person_keypoints = []
        for idx in range(1, num_persons + 1):
            person_keypoints.append(segment_data[str(idx)]["person_keypoints"])

        dummy = generate_dummy_keypoints()
        dummy_frames = [dummy for _ in range(start_frame, end_frame + 1)]

        for _ in range(max_persons - num_persons):
            person_keypoints.append(dummy_frames)

        frames_by_person = list(zip(*person_keypoints))
        frames_dict = {}

        for i, frame_data in enumerate(frames_by_person):
            frame_no = str(start_frame + i)
            frames_dict[frame_no] = {
                "frames": frame_data,
                "gesture": False
            }

        # Assign gesture labels using time ranges
        for begin_ms, end_ms in gesture_time_ranges:
            if end_ms < start_time_ms or begin_ms > end_time_ms:
                continue

            bt = max(begin_ms, start_time_ms)
            et = min(end_ms, end_time_ms)

            start_idx = int(bt * fps / 1000 + 0.5)
            end_idx = int(et * fps / 1000 + 0.5)

            max_frame = int(list(frames_dict.keys())[-1])
            end_idx = min(end_idx, max_frame)

            for frame_no in range(start_idx, end_idx + 1):
                frames_dict[str(frame_no)]["gesture"] = True

        data[segment_key] = frames_dict

    return data


def arrange_detect_data(keypoints: Dict, max_persons: int) -> Dict:
    """
    Prepare frame-level pose data for inference (no gesture labels).
    """
    data = {}

    for segment_key, segment_data in keypoints.items():
        persons = [
            p for p in segment_data.keys()
            if p not in ("start_frame", "end_frame")
        ]
        num_persons = len(persons)

        start_frame = segment_data["start_frame"]
        end_frame = segment_data["end_frame"]

        person_keypoints = []
        for idx in range(1, num_persons + 1):
            person_keypoints.append(segment_data[str(idx)]["person_keypoints"])

        dummy = generate_dummy_keypoints()
        dummy_frames = [dummy for _ in range(start_frame, end_frame + 1)]

        for _ in range(max_persons - num_persons):
            person_keypoints.append(dummy_frames)

        frames_by_person = list(zip(*person_keypoints))
        frames_dict = {}

        for i, frame_data in enumerate(frames_by_person):
            frame_no = str(start_frame + i)
            frames_dict[frame_no] = {
                "frames": frame_data,
                "gesture": False
            }

        data[segment_key] = frames_dict

    return data


def generate_npy_data_train(data: Dict, window_size: int) -> np.ndarray:
    """
    Convert frame-level data into sliding windows for training.
    """
    npy_data = []

    for segment_key, frames in data.items():
        frame_keys = list(frames.keys())

        for i in range(len(frame_keys) - window_size + 1):
            window_frames = []
            target_frame = frame_keys[i + window_size - 1]

            for j in range(window_size):
                frame_no = frame_keys[i + j]
                persons_data = []

                for person in frames[frame_no]["frames"]:
                    keypoint_vec = []
                    for kp in REQUIRED_KEYPOINTS:
                        keypoint_vec.extend(person[kp][:2])
                    persons_data.append(np.array(keypoint_vec, dtype=np.float16))

                window_frames.append(np.array(persons_data, dtype=np.float16))

            npy_data.append([
                int(target_frame),
                np.array(window_frames, dtype=np.float16),
                int(frames[target_frame]["gesture"])
            ])

    return np.array(npy_data, dtype=object)


def generate_npy_data_detect(data: Dict, window_size: int) -> np.ndarray:
    """
    Convert frame-level data into sliding windows for inference.
    """
    npy_data = []

    for segment_key, frames in data.items():
        frame_keys = list(frames.keys())

        for i in range(len(frame_keys) - window_size + 1):
            window_frames = []
            target_frame = frame_keys[i + window_size - 1]

            for j in range(window_size):
                frame_no = frame_keys[i + j]
                persons_data = []

                for person in frames[frame_no]["frames"]:
                    keypoint_vec = []
                    for kp in REQUIRED_KEYPOINTS:
                        keypoint_vec.extend(person[kp][:2])
                    persons_data.append(np.array(keypoint_vec, dtype=np.float16))

                window_frames.append(np.array(persons_data, dtype=np.float16))

            npy_data.append([
                int(target_frame),
                np.array(window_frames, dtype=np.float16)
            ])

    return np.array(npy_data, dtype=object)
