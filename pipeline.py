import sys
import traceback
from pathlib import Path
import torch
from PIL import Image
import torch.nn as nn

# Standard Diffusers
from diffusers import FluxImg2ImgPipeline
from transformers import SiglipVisionModel, AutoProcessor

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
        # SigLIP returns (batch, tokens=729, dim=1152) for 384x384 images
        if id_embeds.dim() == 3 and id_embeds.shape[1] > 1:
            id_embeds = id_embeds.mean(dim=1)  # Global average pooling → (batch, 1152)
        
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, x.shape[-1] // self.num_tokens)
        x = self.norm(x)
        return x
    
class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=128):
        self.device = device
        self.num_tokens = num_tokens
        self.pipe = sd_pipe

        # SigLIP
        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(device, dtype=torch.float16)
        self.processor = AutoProcessor.from_pretrained(image_encoder_path)

        # Load Kijai safetensors
        from safetensors.torch import load_file
        state_dict = load_file(ip_ckpt, device="cpu")

        self.image_proj_model = MLPProjModel(
            num_tokens=num_tokens
        ).to(device, dtype=torch.float16)
        # Kijai weights may have different key
        proj_key = "image_proj" if "image_proj" in state_dict else list(state_dict.keys())[0]
        self.image_proj_model.load_state_dict(state_dict[proj_key] if isinstance(state_dict[proj_key], dict) else state_dict, strict=False)

        print("✅ Kijai IP-Adapter loaded")

    def get_image_embeds(self, pil_image):
        # SigLIP-so400m works best at 384px
        pil_image = resize_img(pil_image, size=(384, 384))
        
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        inputs = {k: v.to(torch.float16) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            image_emb = self.image_encoder(**inputs).last_hidden_state
        
        return self.image_proj_model(image_emb)

    def generate(self, pil_image=None, prompt=None, scale=0.7, **kwargs):
        if pil_image is None:
            raise ValueError("pil_image required")
        
        image_emb = self.get_image_embeds(pil_image)
        
        return self.pipe(
            prompt=prompt,
            image_emb=image_emb,
            guidance_scale=kwargs.get("guidance_scale", 3.5),
            num_inference_steps=kwargs.get("num_inference_steps", 24),
            generator=kwargs.get("generator"),
            **kwargs
        ).images

def _resolve_device(preferred_device):
    preferred_device = str(preferred_device or "mps").lower()
    if preferred_device != "mps":
        raise RuntimeError(f"Unsupported device '{preferred_device}'. This project is MPS-only.")
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")
    return "mps"

class MultiversePipeline:
    def __init__(self, config, run_logger=None):
        self.config = config
        self.run_logger = run_logger
        self.pipe = None
        self.device = _resolve_device(self.config.get("device", "mps"))
        self.config["device"] = self.device
        self.dtype = getattr(torch, self.config.get("model_dtype", "bfloat16"))

    def _status(self, icon, message, next_step=None):
        if self.run_logger:
            self.run_logger.status(icon, message, next_step)
        else:
            line = f"{icon} {message}"
            if next_step:
                line = f"{line} | next: {next_step}"
            print(line)

    def generate(self, input_image_path, base_prompt, reference_image, output_path, seed=None):
        self._status("🎨", f"Generate: {Path(input_image_path).name}", f"save -> {Path(output_path).name}")

        # Load primary image for img2img
        input_img = Image.open(input_image_path).convert("RGB")
        target_size = self.config.get("image_size", [1024, 1024])
        input_img = resize_img(input_img, size=target_size)

        triggers = ", ".join(l.get("trigger", "") for l in self.config.get("loras", []))
        full_prompt = f"{base_prompt}, {triggers}, masterpiece, best quality, highly detailed"

        generator = torch.Generator(device=self.device).manual_seed(seed or self.config["seed"])

        # Simple img2img (no IP-Adapter yet)
        images = self.pipe(
            prompt=full_prompt,
            image=input_img,
            strength=self.config.get("strength", 0.55),
            guidance_scale=self.config.get("guidance_scale", 3.5),
            num_inference_steps=self.config.get("num_inference_steps", 28),
            generator=generator,
            max_sequence_length=512,
        ).images

        result = images[0]
        result.save(output_path)
        self._status("✅", f"Saved: {output_path.name}")
        return result

    def load_pipeline(self):
        self._status("🟡", f"Loading Flux Img2Img ({self.dtype})", "base model")
        
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            self.config["model_id"],
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )

        if self.device == "mps":
            self.pipe = self.pipe.to("mps")

        self._status("✅", "Pipeline ready (img2img mode)", "generation")