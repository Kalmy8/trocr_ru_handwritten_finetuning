import pickle
import torch
from loguru import logger
from torch.utils.data import DataLoader
from trocr_russian_handwritten_text.config import MODELS_DIR, PROCESSED_DATA_DIR
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

def predict_main():
    logger.info("Performing inference for model...")
    # Loading the dataset
    with open(PROCESSED_DATA_DIR / "test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    # Initialize your dataset
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Set batch_size as needed

    # Innitializing tokenizer and a model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(MODELS_DIR)

    # Ensure your model is in evaluation mode
    model.eval()

    # List to hold the generated texts
    generated_texts = []

    with torch.no_grad():  # Disable gradient calculations for inference
        for batch in dataloader:
            pixel_values = batch["pixel_values"]  # Extract pixel values from the batch

            # Generate the predicted IDs
            generated_ids = model.generate(pixel_values)

            # Decode the generated IDs to get the text
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Append the generated text to the list
            generated_texts.append(generated_text)

    # Now you can print or process the generated texts
    for text in generated_texts:
        print(text)

    logger.success("Inference complete.")


if __name__ == "__main__":
    predict_main()
