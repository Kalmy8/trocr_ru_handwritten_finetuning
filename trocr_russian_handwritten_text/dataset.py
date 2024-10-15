from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import typer
from loguru import logger
from tqdm import tqdm
from transformers import TrOCRProcessor

from trocr_russian_handwritten_text.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    train_images, train_labels = 0,0
    text_images, text_labels = 0, 0

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=.2)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    from PIL import Image

    class HandwrittenDataset(Dataset):
        def __init__(self, root_dir, df, processor, max_target_length=128):
            self.root_dir = root_dir
            self.df = df
            self.processor = processor
            self.max_target_length = max_target_length

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            # get file name + text
            file_name = self.df['file_name'][idx]
            text = self.df['text'][idx]
            # prepare image (i.e. resize + normalize)
            image = Image.open(self.root_dir + file_name).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            # add labels (input_ids) by encoding the text
            labels = self.processor.tokenizer(text,
                                              padding="max_length",
                                              max_length=self.max_target_length).input_ids
            # important: make sure that PAD tokens are ignored by the loss function
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
            return encoding

    train_dataset = HandwritingsDataset(train_encodings, train_labels)
    val_dataset = HandwritingsDataset(val_encodings, val_labels)
    test_dataset = HandwritingsDataset(test_encodings, test_labels)


    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
