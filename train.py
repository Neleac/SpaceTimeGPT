from datasets import load_from_disk
import evaluate
import numpy as np
import torch
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: %s" % device)

FRAMES_PER_VIDEO = 8
encoder = "facebook/timesformer-base-finetuned-k600"
decoder = "gpt2"

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained(decoder)
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder).to(device)

tokenizer.pad_token = tokenizer.eos_token
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
print(model.num_parameters())

dataset = load_from_disk("dataset/preprocessed")
dataset.set_format(type="torch", device=device)
print(dataset)

metric = evaluate.load("rouge")
def metrics(eval_preds):
    preds, labels = eval_preds
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=preds, references=labels)

training_args = Seq2SeqTrainingArguments(
    output_dir="training",
    tf32=True,
    dataloader_pin_memory=False,
    predict_with_generate=True,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=image_processor,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=default_data_collator,
    compute_metrics=metrics,
)

trainer.train()
