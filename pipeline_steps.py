import torch
from PIL import Image
from pathlib import Path
import cv2
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

        # Load LoRAs
        for lora in self.config.get("loras", []):
            path = lora.get("path")
            if not path or not Path(path).exists():
                self._status("⚠️", f"LoRA not found: {path}")
                continue
            scale = lora.get("scale", 0.7)
            try:
                self.pipe.load_lora_weights(path)
                self._status("🔄", f"Loaded LoRA: {Path(path).stem} (scale={scale})")
            except Exception as e:
                self._status("❌", f"Failed LoRA {Path(path).stem}: {str(e)[:80]}")
        self._status("✅", "Pipeline ready", "steps")

    def run_full_pipeline(self, input_image_path):
        self.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + input_image_path.stem
        base_prompt = open(self.config["prompts_file"], "r", encoding="utf-8").read().strip()

        # Step 1: Canny structure from primary
        edge_img = self.step1_canny(input_image_path)

        # Step 2: Generate logical merged scene
        base_img = self.step2_merge_generate(edge_img, base_prompt, input_image_path.stem)

        # Step 3: Style + LoRA refinement + reference influence
        final_img = self.step3_style_refine(base_img, input_image_path.stem)

        self._status("🏁", f"Full pipeline completed for {input_image_path.name}")
        return final_img

    def step1_canny(self, input_path):
        self._status("🔍", "Step 1: Canny edge detection (structure)")
        img = cv2.imread(str(input_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.config.get("canny_low", 100), self.config.get("canny_high", 200))
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edge_img = Image.fromarray(edges)
        output_path = build_output_path(self.config, input_path.stem, "step1_canny")
        edge_img.save(output_path)
        self._status("💾", f"Saved Canny: {output_path.name}")
        return edge_img

    def step2_merge_generate(self, control_image, base_prompt, input_stem):
        self._status("🎨", "Step 2: Generate logical merged scene")
        triggers = ", ".join(l.get("trigger", "") for l in self.config.get("loras", []))
        full_prompt = f"{triggers}, {base_prompt}, masterpiece, best quality, highly detailed"
        full_prompt = truncate_prompt(full_prompt, max_length=280)

        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"])

        images = self.pipe(
            prompt=full_prompt,
            image=control_image,
            strength=self.config.get("strength", 0.65),
            guidance_scale=self.config.get("guidance_scale", 3.8),
            num_inference_steps=self.config.get("num_inference_steps", 35),
            generator=generator,
            max_sequence_length=512,
        ).images

        result = images[0]
        output_path = build_output_path(self.config, input_stem, "step2_base")
        result.save(output_path)
        self._status("💾", f"Saved Step 2: {output_path.name}")
        return result

    def step3_style_refine(self, base_image, input_stem):
        self._status("🌟", "Step 3: Style refinement + reference influence")
        ref_name = Path(self.config["reference_image"]).stem
        full_prompt = f"in the style of {ref_name}, {self.config.get('style_prompt', '')}, masterpiece, best quality"

        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"] + 100)

        images = self.pipe(
            prompt=full_prompt,
            image=base_image,
            strength=self.config.get("style_strength", 0.55),
            guidance_scale=self.config.get("guidance_scale", 4.0),
            num_inference_steps=self.config.get("num_inference_steps", 30),
            generator=generator,
        ).images

        result = images[0]
        output_path = build_output_path(self.config, input_stem, "step3_final")
        result.save(output_path)
        self._status("💾", f"Saved Final: {output_path.name}")
        return result