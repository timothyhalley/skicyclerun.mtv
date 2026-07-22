import torch
from PIL import Image
from pathlib import Path
import torch.nn as nn

# Diffusers
from diffusers import FluxImg2ImgPipeline
from transformers import SiglipVisionModel, AutoProcessor

# Utils
from utils import RunLogger, truncate_prompt

def resize_img(input_image, max_side=1280, min_side=1024, size=None, pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        scale = max(max_side / max(w, h), min_side / min(w, h))
        w_resize_new = int(w * scale)
        h_resize_new = int(h * scale)
    input_image = input_image.resize((w_resize_new, h_resize_new), mode)
    return input_image


class MLPProjModel(nn.Module):
    def __init__(self, id_embeddings_dim=1152, cross_attention_dim=2048, num_tokens=128):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        if id_embeds.dim() == 3 and id_embeds.shape[1] > 1:
            id_embeds = id_embeds.mean(dim=1)  # Global average pooling
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, x.shape[-1] // self.num_tokens)
        x = self.norm(x)
        return x


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=128):
        self.device = device
        self.num_tokens = num_tokens
        self.pipe = sd_pipe
        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(device, dtype=torch.float16)
        self.processor = AutoProcessor.from_pretrained(image_encoder_path)

        from safetensors.torch import load_file
        state_dict = load_file(ip_ckpt, device="cpu")
        self.image_proj_model = MLPProjModel(num_tokens=num_tokens).to(device, dtype=torch.float16)
        
        proj_key = "image_proj" if "image_proj" in state_dict else list(state_dict.keys())[0]
        self.image_proj_model.load_state_dict(state_dict[proj_key] if isinstance(state_dict[proj_key], dict) else state_dict, strict=False)
        print("✅ Kijai IP-Adapter loaded")

    def get_image_embeds(self, pil_image):
        pil_image = resize_img(pil_image, size=(384, 384))
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        inputs = {k: v.to(torch.float16) if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.no_grad():
            image_emb = self.image_encoder(**inputs).last_hidden_state
        return self.image_proj_model(image_emb)


class MultiversePipeline:
    def __init__(self, config, run_logger=None):
        self.config = config
        self.run_logger = run_logger
        self.pipe = None
        self.device = self.config.get("device", "mps")
        self.dtype = getattr(torch, self.config.get("model_dtype", "float16"))

    def _status(self, icon, message, next_step=None):
        if self.run_logger:
            self.run_logger.status(icon, message, next_step)
        else:
            line = f"{icon} {message}"
            if next_step:
                line = f"{line} | next: {next_step}"
            print(line)

    def load_pipeline(self):
        self._status("🟡", "Loading Flux Img2Img", "base model")
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            self.config["model_id"],
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )

        if self.device == "mps":
            self.pipe = self.pipe.to("mps")

        # Load LoRAs explicitly
        for lora in self.config.get("loras", []):
            path = lora.get("path")
            if path and Path(path).exists():
                scale = lora.get("scale", 0.8)
                try:
                    self.pipe.load_lora_weights(path)
                    print(f"✅ Loaded LoRA: {Path(path).stem} @ scale {scale}")
                except Exception as e:
                    print(f"❌ Failed to load LoRA {path}: {e}")
            else:
                print(f"⚠️ LoRA not found: {path}")

        self._status("✅", "Pipeline ready", "generation")

    def generate(self, input_image_path, base_prompt, reference_image, output_path, seed=None):
        self._status("🎨", f"Generate: {Path(input_image_path).name}", f"save -> {Path(output_path).name}")

        input_img = Image.open(input_image_path).convert("RGB")
        input_img = input_img.resize((1024, 1024), Image.BILINEAR)

        # Build with trigger first
        triggers = ", ".join(l.get("trigger", "") for l in self.config.get("loras", []))
        full_prompt = f"{triggers}, {base_prompt}, masterpiece, best quality, highly detailed"

        full_prompt = truncate_prompt(full_prompt, max_length=280)

        generator = torch.Generator(device=self.device).manual_seed(seed or self.config.get("seed", 42))

        images = self.pipe(
            prompt=full_prompt,
            image=input_img,
            strength=self.config.get("strength", 0.65),
            guidance_scale=self.config.get("guidance_scale", 3.8),
            num_inference_steps=self.config.get("num_inference_steps", 35),
            generator=generator,
            max_sequence_length=512,
        ).images

        result = images[0]
        result.save(output_path)
        self._status("✅", f"Saved: {output_path.name}")
        return result