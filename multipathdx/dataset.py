import csv
import os
from typing import List, Tuple, Dict


def _read_csv_mean(path: str) -> List[float]:
    """Read a CSV file of numeric values and compute the mean for each column."""
    sums = []
    counts = 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not sums:
                sums = [0.0] * len(row)
            for i, value in enumerate(row):
                try:
                    sums[i] += float(value)
                except ValueError:
                    sums[i] += 0.0
            counts += 1
    if counts == 0:
        return [0.0] * len(sums)
    return [s / counts for s in sums]


def load_labels(labels_file: str) -> Dict[str, Dict[str, str]]:
    """Load label information from a split CSV file."""
    labels = {}
    with open(labels_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            participant_id = row["Participant_ID"].strip()
            labels[participant_id] = row
    return labels


class ParticipantSample:
    def __init__(self, pid: str, features: List[float], label: int):
        self.pid = pid
        self.features = features
        self.label = label


class EDAICDataset:
    """Simple loader for the E-DAIC dataset using precomputed features."""

    def __init__(self, root: str, split_file: str):
        self.root = root
        self.split_file = split_file
        self.samples: List[ParticipantSample] = []
        self._load()

    def _load(self) -> None:
        labels = load_labels(self.split_file)
        for pid, row in labels.items():
            part_dir = os.path.join(self.root, f"{pid}_P")
            feat_dir = os.path.join(part_dir, "features")
            # Primary feature files
            egemaps_path = os.path.join(
                feat_dir, f"{pid}_OpenSMILE2.3.0_egemaps.csv"
            )
            openface_path = os.path.join(
                feat_dir, f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"
            )
            features = []
            if os.path.exists(egemaps_path):
                features.extend(_read_csv_mean(egemaps_path))
            if os.path.exists(openface_path):
                features.extend(_read_csv_mean(openface_path))
            if not features:
                # Skip participants without supported features
                continue
            label = int(row.get("PHQ_Binary", "0"))
            self.samples.append(ParticipantSample(pid, features, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ParticipantSample:
        return self.samples[idx]

    def feature_dim(self) -> int:
        return len(self.samples[0].features) if self.samples else 0
