import pickle
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
from trocr_russian_handwritten_text.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

RANDOM_STATE = 42

class HandwrittingDataset(Dataset):
    def __init__(self, image_dir: Path, df, processor, max_target_length=128):
        self.image_dir = image_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.image_dir / file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length
        ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels
        ]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


def make_dataset_main():
    logger.info("Processing raw data...")

    df_train = pd.read_csv(
        RAW_DATA_DIR / "train_labels.txt",
        header=None,
        names=["file_name", "text"],
        sep=",",
        quotechar='"',
        on_bad_lines="skip",
    )
    df_test = pd.read_csv(
        RAW_DATA_DIR / "test_labels.txt",
        header=None,
        names=["file_name", "text"],
        sep=",",
        quotechar='"',
        on_bad_lines="skip",
    )
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=RANDOM_STATE)

    # Reset_indexes to start from zero
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    # Importing processor for tokenizing
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    # Constructing pytorch datasets
    train_dataset = HandwrittingDataset(
        image_dir=RAW_DATA_DIR / "train", df=df_train, processor=processor
    )
    eval_dataset = HandwrittingDataset(
        image_dir=RAW_DATA_DIR / "train", df=df_val, processor=processor
    )
    test_dataset = HandwrittingDataset(
        image_dir=RAW_DATA_DIR / "test", df=df_test, processor=processor
    )

    # Saving the datasets
    with open(PROCESSED_DATA_DIR / "train_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)

    with open(PROCESSED_DATA_DIR / "eval_dataset.pkl", "wb") as f:
        pickle.dump(eval_dataset, f)

    with open(PROCESSED_DATA_DIR / "test_dataset.pkl", "wb") as f:
        pickle.dump(test_dataset, f)

    logger.success("Processing raw data complete.")


if __name__ == "__main__":
    make_dataset_main()
