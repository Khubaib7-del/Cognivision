from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# FER2013 class IDs:
# 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
ATTENTIVE_IDS = {3, 6}  # happy, neutral
DISTRACTED_IDS = {0, 1, 2, 4, 5}  # angry, disgust, fear, sad, surprise


class Fer2013BinaryDataset(Dataset):
    """
    Converts FER2013 multi-class labels to binary labels:
    attentive=1, distracted=0
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def emotion_to_binary(emotion_id: int) -> int:
        if emotion_id in ATTENTIVE_IDS:
            return 1  # attentive
        if emotion_id in DISTRACTED_IDS:
            return 0  # distracted
        raise ValueError(f"Unsupported emotion id: {emotion_id}")

    @staticmethod
    def pixels_to_pil(pixel_string: str) -> Image.Image:
        values = np.fromstring(pixel_string, dtype=np.uint8, sep=" ")
        if values.size != 48 * 48:
            raise ValueError(f"Expected 2304 pixels, got {values.size}")

        img = values.reshape(48, 48)
        pil = Image.fromarray(img, mode="L").convert("RGB")
        return pil

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self.pixels_to_pil(row["pixels"])
        label = self.emotion_to_binary(int(row["emotion"]))

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def split_fer2013(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    FER2013 uses Usage column with values: Training, PublicTest, PrivateTest
    """
    train_df = df[df["Usage"] == "Training"].copy()
    val_df = df[df["Usage"] == "PublicTest"].copy()
    test_df = df[df["Usage"] == "PrivateTest"].copy()
    return train_df, val_df, test_df


def resolve_fer2013_csv(explicit_path: str = "") -> Path:
    """
    Resolve FER2013 csv path on Kaggle. Priority:
    1) explicit path
    2) env FER2013_CSV
    3) search /kaggle/input/**/fer2013.csv
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p

    env_path = Path(str(__import__("os").environ.get("FER2013_CSV", "")))
    if str(env_path) and env_path.exists():
        return env_path

    kaggle_input = Path("/kaggle/input")
    for p in kaggle_input.rglob("fer2013.csv"):
        return p

    raise FileNotFoundError(
        "Could not find fer2013.csv. Set FER2013_CSV or provide --csv-path."
    )
