import json
import os

from datasets import Dataset, load_from_disk
import evaluate
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, default_data_collator, get_scheduler

device = "cuda"

# MODEL
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

# DATASET
class VatexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return 10 * len(self.dataset)
    
    def __getitems__(self, idxs):
        items = []
        for idx in idxs:
            video_idx = idx // 10
            caption_idx = idx % 10
            example = self.dataset[video_idx]
            items.append({
                "videoID": example["videoID"],
                "pixel_values": example["pixel_values"], 
                "labels": example["labels"][caption_idx]
            })
        return items

dataset = load_from_disk("/data1/caelen/dataset/vatex")
dataset.set_format("torch")
dataset_train = VatexDataset(dataset["train"])
dataset_val = VatexDataset(dataset["validation"])
print("DATASET: train - %d, validation - %d" % (len(dataset_train), len(dataset_val)))

def val_collator(examples):
    videoID, pixel_values, labels = [], [], []
    for example in examples:
        videoID.append(example["videoID"])
        pixel_values.append(example["pixel_values"])
        labels.append(example["labels"])

    pixel_values = torch.stack(pixel_values)
    labels = torch.stack(labels)
    return {"videoID": videoID, "pixel_values": pixel_values, "labels": labels}

kwargs = {
    "batch_size": 7,
    "drop_last": True,
    "num_workers": 16,
    "pin_memory": True,
}

train_dataloader = DataLoader(dataset_train, collate_fn=default_data_collator, shuffle=True, **kwargs)
val_dataloader = DataLoader(dataset_val, collate_fn=val_collator, **kwargs)

# TRAINING
OUTPUT_DIR = "/data1/caelen/training/vatex"
EPOCHS = 100

scaler = GradScaler()
optimizer = AdamW(model.parameters(), lr=5e-7)
training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=training_steps,
)

# VALIDATION
val_output = {}
bleu = evaluate.load("bleu")
with open("dataset/videoID_captions.json") as file:
    videoID_captions = json.load(file)

writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "runs"))
train_progress = tqdm(range(training_steps))
for epoch in range(EPOCHS):
    train_loss, val_loss = 0, 0
    
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.item()
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        train_progress.update(1)
    
    model.eval()
    seen_videos = set()
    val_progress = tqdm(range(len(val_dataloader)))
    for batch in val_dataloader:
        videoIDs = batch.pop("videoID")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        val_loss += outputs.loss.item()
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        
        preds, refs = [], []
        for videoID, prediction, label in zip(videoIDs, predictions, labels):
            # each video appears 10 times, only log metrics and caption once
            if videoID not in seen_videos:
                # use for metrics
                preds.append(prediction)
                refs.append(videoID_captions[videoID])
                
                # save generated captions
                if videoID in val_output:
                    val_output[videoID].append(prediction)
                else:
                    val_output[videoID] = [prediction]
                
                seen_videos.add(videoID)
        
        if len(preds) > 0:
            bleu.add_batch(predictions=preds, references=refs)  
        val_progress.update(1)
                
    bleu_scores = bleu.compute(smooth=True)
    writer.add_scalar("Loss/train", train_loss / len(train_dataloader), epoch)
    writer.add_scalar("Loss/val", val_loss / len(val_dataloader), epoch)
    writer.add_scalar("Metric/bleu", bleu_scores["bleu"], epoch)

    model.save_pretrained(os.path.join(OUTPUT_DIR, "checkpoint_%d" % (epoch + 1))) 
    
    with open(os.path.join(OUTPUT_DIR, "val_output.json"), "w") as file:
        file.write(json.dumps(val_output))
