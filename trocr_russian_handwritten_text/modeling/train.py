import pickle
from functools import partial

from evaluate import load
from loguru import logger
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator, EvalPrediction, EarlyStoppingCallback,
)

from trocr_russian_handwritten_text.config import MODELS_DIR, PROCESSED_DATA_DIR
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

cer_metric = load("cer")
def compute_cer_base(processor, eval_pred: EvalPrediction):
    labels_ids = eval_pred.label_ids
    pred_ids = eval_pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

def train_main():
    logger.info("Training model...")

    # Loading the datasets
    with open(PROCESSED_DATA_DIR / "train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    with open(PROCESSED_DATA_DIR / "eval_dataset.pkl", "rb") as f:
        eval_dataset = pickle.load(f)

    # Initialize a tokenizer and a model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model.gradient_checkpointing_enable()

    # pass initialized processor to the compute function
    compute_cer = partial(compute_cer_base, processor)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        per_device_train_batch_size=16,  # Increase batch size if GPU allows
        per_device_eval_batch_size=16,
        fp16=True,
        gradient_accumulation_steps=4,  # Simulate larger batch size
        output_dir=MODELS_DIR,
        logging_steps=10,
        save_steps=500,  # Save more frequently
        eval_steps=200,
        max_steps=1000,  # Limit total training steps
    )

    # Add EarlyStoppingCallback
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.image_processor,
        args=training_args,
        train_dataset=train_dataset.shuffle(seed=42).select(range(50000)),  # Subset for quicker training
        eval_dataset=eval_dataset.shuffle(seed=42).select(range(5000)),
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
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
    trainer.train(resume_from_checkpoint=True)
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    train_main()
