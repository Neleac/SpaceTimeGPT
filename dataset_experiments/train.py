import argparse
import json
import os
import random

from datasets import load_from_disk
import evaluate
import numpy as np
import torch
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

def train(random_frames, random_captions):
    encoder = "facebook/timesformer-base-finetuned-k600"
    decoder = "gpt2"

    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained(decoder)
    tokenizer.pad_token = tokenizer.eos_token

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder).to(device)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = 50
    model.config.num_beams = 4
    model.config.early_stopping = True

    dataset = load_from_disk("../dataset/processed/k600_16frames_10captions")
    dataset.set_format("torch")

    frame_strat = "randomFrames" if random_frames else "fixedFrames"
    caption_strat = "randomCaptions" if random_captions else "fixedCaptions"
    output_dir = "../training/%s_%s" % (frame_strat, caption_strat)
    print("OUTPUT DIR: %s" % output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        tf32=True,
        predict_with_generate=True,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        dataloader_drop_last=True,
        dataloader_num_workers=8,
        num_train_epochs=100,
        learning_rate=5e-7,
    )

    def collator(examples):
        pixel_values, labels = [], []
        for example in examples:
            # train
            if len(example["pixel_values"]) == 16:
                if random_frames:
                    frame_idxs = [i + random.randint(0, 1) for i in range(0, 16, 2)]
                else:
                    frame_idxs = [i for i in range(0, 16, 2)]
                pixel_values.append(example["pixel_values"][frame_idxs])
                
                caption_idx = random.randint(0, 9) if random_captions else 0
                labels.append(example["labels"][caption_idx])
            # val
            else:
                pixel_values.append(example["pixel_values"])
                labels.append(example["labels"])

        pixel_values = torch.stack(pixel_values)
        labels = torch.stack(labels)
        return {"pixel_values": pixel_values, "labels": labels}

    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    val_output = {}
    with open("../dataset/longestCaption_videoID.json") as file:
        longestCaption_videoID = json.load(file)
    with open("../dataset/videoID_captions.json") as file:
        videoID_captions = json.load(file)

    def metrics(eval_predictions):
        predictions, labels = eval_predictions
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        references = []
        for i, label in enumerate(labels):
            video_id = longestCaption_videoID[label]
            references.append(videoID_captions[video_id])
            if video_id in val_output:
                val_output[video_id].append(predictions[i])
            else:
                val_output[video_id] = [predictions[i]]

        bleu_scores = bleu.compute(predictions=predictions, references=references, smooth=True)
        meteor_scores = meteor.compute(predictions=predictions, references=references)
        rouge_scores = rouge.compute(predictions=predictions, references=references, rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)    
        return {"bleu": bleu_scores["bleu"], "meteor": meteor_scores["meteor"], "rouge1": rouge_scores["rouge1"], "rouge2": rouge_scores["rouge2"], "rougeL": rouge_scores["rougeL"]}

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        compute_metrics=metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    with open(os.path.join(output_dir, "val_output.json"), "w") as file:
        file.write(json.dumps(val_output))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_frames", type=bool, default=False)
    parser.add_argument("--random_captions", type=bool, default=False)
    args = parser.parse_args()
    train(args.random_frames, args.random_captions)