from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, TimesformerForVideoClassification

model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

actions = set()
for action in model.config.id2label.values():
    if len(action.split(" ")) == 1:
        actions.add(action)

dataset = load_from_disk("dataset/processed/k600")
combined = concatenate_datasets([dataset["train"], dataset["validation"]])

action_idxs = {}
for i, item in enumerate(combined):
    tokens = item["labels"]
    caption = tokenizer.decode(tokens, skip_special_tokens=True)
    for word in caption.split(" "):
        if word in actions:
            if word in action_idxs:
                action_idxs[word].append(i)
            else:
                action_idxs[word] = [i]
            break
            
action_train_val_idxs = {}
train_idxs, val_idxs = [], []
for action, idxs in action_idxs.items():
    if len(idxs) > 10:
        pivot = int(0.91 * len(idxs))
        train_split, val_split = idxs[:pivot], idxs[pivot:]
        
        train_lo, train_hi = len(train_idxs), len(train_idxs) + len(train_split)
        val_lo, val_hi = len(val_idxs), len(val_idxs) + len(val_split)
        action_train_val_idxs[action] = {"train": (train_lo, train_hi), "val": (val_lo, val_hi)}
        
        train_idxs.extend(train_split)
        val_idxs.extend(val_split)

dataset["train"] = combined.select(train_idxs)
dataset["validation"] = combined.select(val_idxs)
dataset.save_to_disk("dataset/processed/test")

print(action_train_val_idxs)