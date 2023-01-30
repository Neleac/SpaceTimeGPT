

# import requests
# from PIL import Image
# from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
# from transformers import AutoFeatureExtractor, AutoTokenizer

# encoder_name = "google/vit-base-patch16-224-in21k"
# decoder_name = "gpt2"

# model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name)
# image_encoder = AutoFeatureExtractor.from_pretrained(encoder_name)
# tokenizer = AutoTokenizer.from_pretrained(decoder_name)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# pixel_values = image_encoder(image, return_tensors="pt").pixel_values

# # autoregressively generate caption (greedy decoding by default)
# generated_ids = model.generate(pixel_values)
# generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)


import requests
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def generate_captions(image_urls):
    images = []
    for url in image_urls:
        image = Image.open(requests.get(url, stream=True).raw)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        images.append(image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    print(pixel_values.shape)
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]
    return captions

image_urls = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
captions = generate_captions(image_urls)
print(captions)
