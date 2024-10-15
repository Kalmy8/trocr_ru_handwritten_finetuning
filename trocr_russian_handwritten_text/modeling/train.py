from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import typer
from loguru import logger
from tqdm import tqdm

from trocr_russian_handwritten_text.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):

               #image_source_here
        image = Image.open().convert("RGB")

        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
