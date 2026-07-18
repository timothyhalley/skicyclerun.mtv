from pipeline import MultiversePipeline
from utils import RunLogger, load_config, load_images_from_dir, load_prompt
from pathlib import Path
from datetime import datetime
import os
import tqdm

def main():
    config = load_config()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logger = RunLogger(config.get("log_file", "logs/run.log"))
    run_logger.status("🚀", "Start run", "load pipeline")
    pipeline = MultiversePipeline(config, run_logger=run_logger)
    pipeline.load_pipeline()

    input_images = load_images_from_dir(config["input_dir"])
    if not input_images:
        run_logger.status("❌", "No input images found", "add images to input_dir and rerun")
        raise FileNotFoundError(
            f"No images found in '{config['input_dir']}'. Supported extensions: "
            ".png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff"
        )
    reference_path = Path(config["reference_image"])
    if not reference_path.exists():
        run_logger.status("❌", "Reference image missing", "fix config.reference_image and rerun")
        raise FileNotFoundError(f"Configured reference image does not exist: {reference_path}")
    prompt_file = config.get("prompt_file") or config.get("prompts_file")
    if not prompt_file:
        raise KeyError("Missing prompt file in config. Set either 'prompt_file' or 'prompts_file'.")
    
    # Load single prompt from chosen file
    base_prompt = load_prompt(prompt_file)
    run_logger.status("📝", f"Prompt loaded from {Path(prompt_file).name}", f"process {len(input_images)} images")
    run_logger.info(f"Prompt text: {base_prompt}")

    os.makedirs(config["output_dir"], exist_ok=True)

    for idx, img_path in enumerate(tqdm.tqdm(input_images)):
        run_logger.status("⏳", f"Queue image {idx + 1}/{len(input_images)}: {img_path.name}")
        output_name = f"transformed_{Path(prompt_file).stem}_{idx:02d}_{img_path.stem}.png"
        output_path = Path(config["output_dir"]) / output_name

        if output_path.exists() and not config.get("overwrite_outputs", False):
            rolled_name = f"{output_path.stem}__{run_tag}{output_path.suffix}"
            output_path = output_path.with_name(rolled_name)
            run_logger.status("♻️", "Output exists, rolling filename", f"save -> {output_path.name}")
        
        pipeline.generate(
            input_image_path=img_path,
            base_prompt=base_prompt,          # Same prompt for entire batch
            reference_image=reference_path,
            output_path=output_path,
            seed=config["seed"] + idx
        )

    run_logger.status("🏁", "Run complete", "review logs/run.log for loader details")

if __name__ == "__main__":
    main()