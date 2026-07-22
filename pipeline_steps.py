import torch
from PIL import Image
from pathlib import Path
import cv2
from datetime import datetime
from diffusers import FluxImg2ImgPipeline

from utils import RunLogger, truncate_prompt, build_output_path

class MiniaturizationPipeline:
    def __init__(self, config, run_logger=None):
        self.config = config
        self.run_logger = run_logger
        self.pipe = None
        self.device = config.get("device", "mps")
        self.dtype = getattr(torch, config.get("model_dtype", "float16"))
        self.run_tag = None

    def _status(self, icon, message, next_step=None):
        if self.run_logger:
            self.run_logger.status(icon, message, next_step)
        else:
            line = f"{icon} {message}"
            if next_step:
                line += f" | next: {next_step}"
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

        for lora in self.config.get("loras", []):
            path = lora.get("path")
            if path and Path(path).exists():
                try:
                    self.pipe.load_lora_weights(path)
                    self._status("🔄", f"Loaded LoRA: {Path(path).stem}")
                except Exception as e:
                    self._status("❌", f"LoRA failed: {Path(path).stem}")
        self._status("✅", "Pipeline ready")

    def run_full(self, input_path):
        self.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + input_path.stem
        base_prompt = open(self.config["prompts_file"], "r", encoding="utf-8").read().strip()

        original_img = Image.open(input_path).convert("RGB")
        original_img = original_img.resize((1024, 1024), Image.BILINEAR)

        # Pass 1: Aggressive stylization
        self._status("🔥", "Pass 1: Aggressive miniaturization")
        full_prompt = base_prompt + ", miniature diorama, toy town, tilt-shift, exaggerated perspective, soft cartoonish style, whimsical, clean colors"
        full_prompt = truncate_prompt(full_prompt, max_length=280)

        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"])

        pass1 = self.pipe(
            prompt=full_prompt,
            image=original_img,
            strength=0.85,           # push hard
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
        ).images[0]

        pass1.save(build_output_path(self.config, input_path.stem, "pass1_aggressive"))

        # Pass 2: Recover structure from original
        self._status("🔄", "Pass 2: Structure recovery")
        final = self.pipe(
            prompt=base_prompt + ", highly detailed, realistic, coherent",
            image=pass1,
            strength=0.35,           # light recovery
            guidance_scale=4.0,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

        final.save(build_output_path(self.config, input_path.stem, "final"))
        self._status("🏁", f"Two-pass test complete for {input_path.name}")
        return final

    def stage1_base(self, input_path, base_prompt):
        self._status("🖼️", "Stage 1: Base structure preservation")
        img = Image.open(input_path).convert("RGB")
        img = img.resize((1024, 1024), Image.BILINEAR)

        full_prompt = f"{base_prompt}, clean, detailed, realistic, masterpiece"
        full_prompt = truncate_prompt(full_prompt, max_length=280)

        seed = self.config.get("seed", 42)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        images = self.pipe(
            prompt=full_prompt,
            image=img,
            strength=self.config.get("stage1_strength", 0.40),
            guidance_scale=self.config.get("stage1_guidance", 4.0),
            num_inference_steps=self.config.get("stage1_steps", 30),
            generator=generator,
        ).images

        result = images[0]
        result.save(build_output_path(self.config, input_path.stem, "stage1_base"))
        return result

    def stage2_miniaturize(self, img, input_stem):
        self._status("🧸", "Stage 2: Miniaturization pass")
        full_prompt = "miniature diorama style, toy town, slightly exaggerated perspective, soft outlines, gentle shading, simple yet realistic, clean colors, masterpiece"

        seed = self.config.get("seed", 42) + 100
        generator = torch.Generator(device=self.device).manual_seed(seed)

        images = self.pipe(
            prompt=full_prompt,
            image=img,
            strength=self.config.get("stage2_strength", 0.55),
            guidance_scale=self.config.get("stage2_guidance", 5.0),
            num_inference_steps=self.config.get("stage2_steps", 32),
            generator=generator,
        ).images

        result = images[0]
        result.save(build_output_path(self.config, input_stem, "stage2_mini"))
        return result

    def stage3_cartoon(self, img, input_stem):
        self._status("🎨", "Stage 3: Cartoonish refinement")
        full_prompt = "stylized illustration, soft cartoon style, simplified shapes, clear outlines, gentle shading, realistic lighting, coherent scene, masterpiece"

        seed = self.config.get("seed", 42) + 200
        generator = torch.Generator(device=self.device).manual_seed(seed)

        images = self.pipe(
            prompt=full_prompt,
            image=img,
            strength=self.config.get("stage3_strength", 0.60),
            guidance_scale=self.config.get("stage3_guidance", 5.5),
            num_inference_steps=self.config.get("stage3_steps", 34),
            generator=generator,
        ).images

        result = images[0]
        result.save(build_output_path(self.config, input_stem, "stage3_cartoon"))
        return result

    def stage4_refine(self, img, input_stem):
        self._status("✨", "Stage 4: Final refinement")
        full_prompt = "soft, cohesive color palette, gentle contrast, subtle shading, clean details, coherent miniature scene, simple yet realistic, masterpiece"

        seed = self.config.get("seed", 42) + 300
        generator = torch.Generator(device=self.device).manual_seed(seed)

        images = self.pipe(
            prompt=full_prompt,
            image=img,
            strength=self.config.get("stage4_strength", 0.35),
            guidance_scale=self.config.get("stage4_guidance", 4.0),
            num_inference_steps=self.config.get("stage4_steps", 28),
            generator=generator,
        ).images

        result = images[0]
        result.save(build_output_path(self.config, input_stem, "stage4_final"))
        return result