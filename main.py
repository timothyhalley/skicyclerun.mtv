from pipeline_steps import StepPipeline
from utils import RunLogger, load_config, load_images_from_dir, generate_seed
from pathlib import Path
import tqdm

def main():
    config = load_config()
    seed = generate_seed(config)
    config["seed"] = seed

    run_logger = RunLogger(config.get("log_file", "logs/run.log"))
    run_logger.status("🚀", f"Start run | Seed: {seed}", "load pipeline")

    pipeline = StepPipeline(config, run_logger=run_logger)
    pipeline.load_pipeline()

    input_images = load_images_from_dir(config["input_dir"])
    if not input_images:
        run_logger.status("❌", "No input images found", "")
        raise FileNotFoundError(f"No images in '{config['input_dir']}'.")

    for idx, img_path in enumerate(tqdm.tqdm(input_images)):
        run_logger.status("⏳", f"Queue image {idx + 1}/{len(input_images)}: {img_path.name}")
        pipeline.run_full_pipeline(img_path)

    run_logger.status("🏁", "Run complete", "review outputs/ folder")

if __name__ == "__main__":
    main()