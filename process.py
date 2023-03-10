from datasets import load_from_disk
from transformers import AutoTokenizer, TimesformerForVideoClassification

model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_from_disk("dataset/processed/8frames_pt1")

actions = set()
for action in model.config.id2label.values():
    if len(action.split(" ")) == 1:
        actions.add(action)

train_idxs = []
for i, item in enumerate(dataset["train"]):
    if i % 1000 == 0:
        print("idx: %d, total: %d" % (i, len(train_idxs)))
    
    tokens = item["labels"]
    caption = tokenizer.decode(tokens, skip_special_tokens=True)
    for word in caption.split(" "):
        if word in actions:
            train_idxs.append(i)
            break
print(train_idxs)

val_idxs = [6, 40, 41, 66, 67, 105, 118, 136, 185, 188, 193, 194, 195, 196, 197, 230, 252, 259, 262, 263, 271, 332, 338, 354, 357, 358, 393, 444, 463, 496, 519, 566, 589, 611, 631, 633, 654, 707, 709, 714, 731, 781, 782, 908, 926, 946, 947, 948, 949, 950, 1010, 1012, 1013, 1014, 1015, 1028, 1031, 1066, 1069, 1078, 1080, 1087, 1088, 1091, 1103, 1104, 1145, 1171, 1173, 1226, 1230, 1231, 1250, 1304, 1348, 1399, 1433, 1440, 1483, 1525, 1526, 1529, 1657, 1674, 1675, 1727, 1736, 1737, 1762, 1802, 1882, 1926, 1927, 1930, 1931, 1932, 1933, 1934, 1994, 2021, 2032, 2081, 2082, 2085, 2086, 2098, 2100, 2102, 2107, 2120, 2123, 2125, 2129, 2180, 2193, 2196, 2198, 2262, 2316, 2317, 2341, 2347, 2507, 2512, 2533, 2536, 2571, 2590, 2592, 2594, 2597, 2600, 2609, 2618, 2632, 2635, 2637]

dataset["train"] = dataset["train"].select(train_idxs)
dataset["validation"] = dataset["validation"].select(val_idxs)
dataset.save_to_disk("dataset/processed/k600")