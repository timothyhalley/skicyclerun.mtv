from pipeline import MultiversePipeline
from utils import load_config
import torch
from PIL import Image

config = load_config()
pipeline = MultiversePipeline(config)
pipeline.load_pipeline()

prompt = "a beautiful cyberpunk city at night, neon lights, highly detailed, cinematic"

generator = torch.Generator(device="mps").manual_seed(42)

images = pipeline.pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=20,
    generator=generator,
).images

images[0].save("outputs/test_base_output.png")
print("Saved outputs/test_base_output.png - check if it's black")