from diffusers import FluxPipeline
from PIL import Image
import torch
from transformers import SiglipVisionModel, AutoProcessor
import torch.nn as nn

def resize_img(img, size=(1024, 1024)):
    return img.resize(size, Image.BILINEAR)

# Load models
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
).to("mps")

image_encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384").to("mps", dtype=torch.float16)
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

print("Models loaded. Ready.")

# Generation
ref_path = "inputs/BavarianInn.webp"   # Change this
ref_img = Image.open(ref_path).convert("RGB")
ref_img = resize_img(ref_img)

inputs = processor(images=ref_img, return_tensors="pt").to("mps")

with torch.no_grad():
    emb = image_encoder(**inputs).last_hidden_state

# Simple projection (adjust if needed)
proj = nn.Linear(emb.shape[-1], 2048).to("mps", dtype=torch.float16)
image_emb = proj(emb.mean(dim=1))

# Generate
result = pipe(
    prompt="Gothic cityscape reimagined with comic charm, crisp architecture, set a whimsical atmosphere that transforms the original photo.",
    image_emb=image_emb,
    num_inference_steps=20,
    guidance_scale=3.5,
    height=1024,
    width=1024,
).images[0]

result.save("output.png")
print("✅ Done! Check output.png")