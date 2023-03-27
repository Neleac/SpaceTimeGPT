import argparse
import json
import os
import sys

from datasets import load_from_disk
import evaluate
import numpy as np
import torch

sys.path.append("../../transformers/src")
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, EarlyStoppingCallback

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train(args):
    encoder = "facebook/timesformer-base-finetuned-k600"
    decoder = "gpt2"
    
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained(decoder)
    tokenizer.pad_token = tokenizer.eos_token
    
    kwargs = {
        "encoder_hidden_dropout_prob": args.hidden_dropout_prob,
        "encoder_attention_probs_dropout_prob": args.attention_probs_dropout_prob,
        "encoder_drop_path_rate": args.drop_path_rate,
        "decoder_resid_pdrop": args.resid_pdrop,
        "decoder_embd_pdrop": args.embd_pdrop,
        "decoder_attn_pdrop": args.attn_pdrop,
    }
    
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder, **kwargs).to(device)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = 50
    model.config.num_beams = 4
    model.config.early_stopping = True

    dataset = load_from_disk("../dataset/processed/k600")
    
    train_output, val_output = {}, {}
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        fp16=True,
        predict_with_generate=True,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        dataloader_num_workers=8,
        num_train_epochs=100,
        learning_rate=args.learning_rate,
    )

    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")

    def metrics(eval_preds):
        preds, labels = eval_preds
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for i, label in enumerate(labels):
            if label in val_output:
                val_output[label].append(preds[i])
            else:
                val_output[label] = [preds[i]]
            
        try:
            bleu_scores = bleu_metric.compute(predictions=preds, references=labels, smooth=True)
            bleu = bleu_scores["bleu"]
        except:
            bleu = 0
            
        try:
            meteor_scores = meteor_metric.compute(predictions=preds, references=labels)
            meteor = meteor_scores["meteor"]
        except:
            meteor = 0
            
        try:
            rouge_scores = rouge_metric.compute(predictions=preds, references=labels, rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge1, rouge2, rougeL = rouge_scores["rouge1"], rouge_scores["rouge2"], rouge_scores["rougeL"]
        except:
            rouge1, rouge2, rougeL = 0, 0, 0
        
        return {"bleu": bleu, "meteor": meteor, "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}

    trainer = Seq2SeqTrainer(
        train_output,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=default_data_collator,
        compute_metrics=metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    
    with open(os.path.join(args.output_dir, "train_output.json"), "w") as file:
        file.write(json.dumps(train_output))
    
    with open(os.path.join(args.output_dir, "val_output.json"), "w") as file:
        file.write(json.dumps(val_output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    parser.add_argument("--hidden_dropout_prob", type=float, default=0)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0)
    parser.add_argument("--drop_path_rate", type=float, default=0)
    
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--embd_pdrop", type=float, default=0.1)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    
    parser.add_argument("--output_dir", type=str, default="../training")
    
    args = parser.parse_args()
    train(args)
