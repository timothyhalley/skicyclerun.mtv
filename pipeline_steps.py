import torch
from PIL import Image
from pathlib import Path
from datetime import datetime
from diffusers import FluxImg2ImgPipeline

from utils import RunLogger, truncate_prompt, build_output_path

class ControlledPipeline:
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
        self._status("✅", "Pipeline ready")

    def run_full(self, input_path):
        self.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + input_path.stem
        base_prompt = open(self.config["prompts_file"], "r", encoding="utf-8").read().strip()

        img = Image.open(input_path).convert("RGB")
        img = img.resize((1024, 1024), Image.BILINEAR)

        triggers = ", ".join(l.get("trigger", "") for l in self.config.get("loras", []))
        full_prompt = f"{triggers}, {base_prompt}, masterpiece, best quality, highly detailed"
        full_prompt = truncate_prompt(full_prompt, max_length=280)

        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"])

        images = self.pipe(
            prompt=full_prompt,
            image=img,
            strength=self.config.get("strength", 0.65),
            guidance_scale=self.config.get("guidance_scale", 3.8),
            num_inference_steps=self.config.get("num_inference_steps", 35),
            generator=generator,
        ).images

        result = images[0]
        result.save(build_output_path(self.config, input_path.stem, "final"))
        self._status("🏁", f"Completed {input_path.name}")
        return result