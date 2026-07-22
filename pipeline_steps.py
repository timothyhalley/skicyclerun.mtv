import torch
from PIL import Image
from pathlib import Path
from datetime import datetime
from diffusers import FluxImg2ImgPipeline

from utils import RunLogger, truncate_prompt, build_output_path

class StepPipeline:
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

    def run_full_pipeline(self, input_image_path):
        self.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + input_image_path.stem
        base_prompt = open(self.config["prompts_file"], "r", encoding="utf-8").read().strip()

        base_img = self.step1_base_generation(input_image_path, base_prompt)
        styled_img = self.step2_reference_style(base_img, input_image_path.stem)
        final_img = self.step3_lora_refine(styled_img, input_image_path.stem)

        self._status("🏁", f"Full pipeline completed for {input_image_path.name}")
        return final_img

    def step1_base_generation(self, input_path, base_prompt):
        self._status("🖼️", "Step 1: Base generation from primary image")
        input_img = Image.open(input_path).convert("RGB")
        input_img = input_img.resize((1024, 1024), Image.BILINEAR)

        triggers = ", ".join(l.get("trigger", "") for l in self.config.get("loras", []))
        full_prompt = f"{triggers}, {base_prompt}, masterpiece, best quality, highly detailed"
        full_prompt = truncate_prompt(full_prompt, max_length=280)

        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"])

        images = self.pipe(
            prompt=full_prompt,
            image=input_img,
            strength=self.config.get("strength", 0.65),
            guidance_scale=self.config.get("guidance_scale", 3.8),
            num_inference_steps=self.config.get("num_inference_steps", 35),
            generator=generator,
        ).images

        result = images[0]
        output_path = build_output_path(self.config, input_path.stem, "step1_base")
        result.save(output_path)
        return result

    def step2_reference_style(self, base_image, input_stem):
        self._status("🌆", "Step 2: Inject reference style")
        ref_name = Path(self.config["reference_image"]).stem
        full_prompt = f"in the style of {ref_name}, {self.config.get('style_prompt', '')}, masterpiece, best quality"

        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"] + 100)

        images = self.pipe(
            prompt=full_prompt,
            image=base_image,
            strength=self.config.get("style_strength", 0.50),
            guidance_scale=self.config.get("guidance_scale", 4.0),
            num_inference_steps=self.config.get("num_inference_steps", 30),
            generator=generator,
        ).images

        result = images[0]
        output_path = build_output_path(self.config, input_stem, "step2_styled")
        result.save(output_path)
        return result

    def step3_lora_refine(self, styled_image, input_stem):
        self._status("✨", "Step 3: LoRA + final refinement")
        triggers = ", ".join(l.get("trigger", "") for l in self.config.get("loras", []))
        full_prompt = f"{triggers}, {self.config.get('style_prompt', '')}, masterpiece, best quality, highly detailed"

        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"] + 200)

        images = self.pipe(
            prompt=full_prompt,
            image=styled_image,
            strength=self.config.get("final_strength", 0.40),
            guidance_scale=self.config.get("guidance_scale", 4.0),
            num_inference_steps=self.config.get("num_inference_steps", 28),
            generator=generator,
        ).images

        result = images[0]
        output_path = build_output_path(self.config, input_stem, "step3_final")
        result.save(output_path)
        return result