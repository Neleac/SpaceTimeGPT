from datasets import load_from_disk
import multiprocessing as mp
import numpy as np
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import torch
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

device = "cuda" if torch.cuda.is_available() else "cpu"


# MODEL
encoder = "facebook/timesformer-base-finetuned-k600"
decoder = "gpt2"
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained(decoder)
tokenizer.pad_token = tokenizer.eos_token

def model_init(trial):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder).to(device)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = 50
    model.config.num_beams = 4
    model.config.early_stopping = True
    return model

# DATASET
dataset = load_from_disk("/home/922201615/video-caption/dataset/processed/k600")
dataset.set_format(type="torch")

# TRAINING
training_args = Seq2SeqTrainingArguments(
    output_dir="training/raytune",
    fp16=True,
    predict_with_generate=True,
    dataloader_num_workers=4,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="steps",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
)

trainer = Seq2SeqTrainer(
    model=None,
    tokenizer=image_processor,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=default_data_collator,
    model_init=model_init,
)

def ray_hp_space(trial):
    # return {
    #     "learning_rate": tune.loguniform(1e-5, 5e-4),
    #     "lr_scheduler_type": tune.choice(["linear", "cosine"]),
    #     "warmup_ratio": tune.loguniform(1e-3, 1e-1),
    #     "weight_decay": tune.loguniform(1e-4, 5e-4),
    # }
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-3),
        "lr_scheduler_type": tune.choice(["linear", "cosine"]),
        "warmup_ratio": tune.uniform(0.0, 0.2),
        "weight_decay": tune.uniform(0.0, 1e-3),
    }

best_trial = trainer.hyperparameter_search(
    resources_per_trial={"cpu": mp.cpu_count(), "gpu": torch.cuda.device_count()},
    direction="minimize",
    backend="ray",
    search_alg=HyperOptSearch(metric="eval_loss", mode="min"),
    scheduler=ASHAScheduler(metric="eval_loss", mode="min"),
    hp_space=ray_hp_space,
    n_trials=25,
)

print(best_trial)
