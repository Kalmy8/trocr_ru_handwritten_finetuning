import pickle
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import typer
from loguru import logger
from tqdm import tqdm
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from evaluate import load
from transformers import default_data_collator
from trocr_russian_handwritten_text.dataset import HandwrittingDataset
from trocr_russian_handwritten_text.config import MODELS_DIR, PROCESSED_DATA_DIR
from models.models import model, processor
cer_metric = load("cer")

def compute_cer(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

def main():
    logger.info("Training model...")

    # Loading the datasets
    with open(PROCESSED_DATA_DIR / "train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    with open(PROCESSED_DATA_DIR / "eval_dataset.pkl", "rb") as f:
        eval_dataset = pickle.load(f)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True,
        output_dir=MODELS_DIR,
        logging_steps=2,
        save_steps=1000,
        eval_steps=200,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.image_processor,
        args=training_args,
        compute_metrics=compute_cer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,

    )
    trainer.train()
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    main()
