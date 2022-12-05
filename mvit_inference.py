from torchvision.io.video import read_video
#from torchvision.models.video.mvit import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models.video.swin_transformer import swin3d_t, Swin3D_T_Weights

vid, _, _ = read_video("dataset/vatex_test_videos/_0ZBlXUcaOk_000013_000023.mp4", output_format="TCHW")
#vid = vid[:32]  # optionally shorten duration

# Step 1: Initialize model with the best available weights
# weights = MViT_V2_S_Weights.DEFAULT
# model = mvit_v2_s(weights=weights)
weights = Swin3D_T_Weights.DEFAULT
model = swin3d_t(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(vid).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
label = prediction.argmax().item()
score = prediction[label].item()
category_name = weights.meta["categories"][label]
print(f"{category_name}: {100 * score}%")
