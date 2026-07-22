import os
from PIL import Image
from pathlib import Path
import json
import sys
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import random
from pathlib import Path
from datetime import datetime


class RunLogger:
    def __init__(self, log_path="logs/run.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"comfy.run.{self.log_path}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)

    def status(self, icon, message, next_step=None):
        line = f"{icon} {message}"
        if next_step:
            line = f"{line} | next: {next_step}"
        print(line)
        self.logger.info(line)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    @contextmanager
    def _redirect_logging_streams(self, stream):
        logger_names = ["", "diffusers", "transformers", "huggingface_hub"]
        redirected = []

        for name in logger_names:
            logger = logging.getLogger(name)
            redirected.append((logger, list(logger.handlers), logger.level, logger.propagate))

            handler = logging.StreamHandler(stream)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            logger.handlers = [handler]
            logger.setLevel(logging.INFO)
            logger.propagate = False

        try:
            yield
        finally:
            for logger, handlers, level, propagate in reversed(redirected):
                logger.handlers = handlers
                logger.setLevel(level)
                logger.propagate = propagate

    @contextmanager
    def capture_output(self, label):
        self.logger.info("BEGIN %s", label)
        with open(self.log_path, "a", encoding="utf-8") as stream:
            stream.write(f"\n--- BEGIN {label} ---\n")
            stream.flush()
            with self._redirect_logging_streams(stream), redirect_stdout(stream), redirect_stderr(stream):
                yield
            stream.write(f"--- END {label} ---\n")
            stream.flush()
        self.logger.info("END %s", label)

def load_config(config_path="config.json"):
    with open(config_path) as f:
        return json.load(f)

def load_images_from_dir(dir_path):
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    paths = sorted([p for p in Path(dir_path).glob("*.*") if p.suffix.lower() in allowed_suffixes])
    return paths

def load_prompt(prompt_file):
    """Load a single prompt from a .txt file"""
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    return prompt

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

def truncate_prompt(prompt: str, max_length: int = 280, log=True) -> str:
    """
    Preserve trigger section at the beginning, then truncate the rest.
    """
    original = prompt.strip()
    
    if len(original) <= max_length:
        if log:
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("\t✅ Prompt length OK")
            print(f"\t🟩 ORIGINAL ({len(original)} chars)\n")
            print(original)
            print("──────────────────────────────────────────────────────────────────────────────\n")

        return original

    # Split into triggers + rest of prompt
    # We assume triggers come first, separated by a clear point
    if '[' in original and ']' in original:
        # Protect everything inside first [] as sacred triggers
        start = original.find('[')
        end = original.find(']') + 1
        triggers_section = original[start:end]
        rest = original[end:].strip()
    else:
        # Fallback: take first part as triggers
        parts = original.split(',', 3)  # keep first few comma groups
        triggers_section = ','.join(parts[:2]).strip()
        rest = ','.join(parts[2:]).strip() if len(parts) > 2 else ""

    # Truncate the descriptive part
    available = max_length - len(triggers_section) - 60
    if available > 50 and rest:
        rest = rest[:available]
        last_comma = rest.rfind(',')
        if last_comma > 30:
            rest = rest[:last_comma]

    truncated = f"{triggers_section} {rest}, masterpiece, best quality, highly detailed"

    if log:
        print("\n⚠️ PROMPT TRUNCATED")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🟥 ORIGINAL ({len(original)} chars)")
        print("──────────────────────────────────────────────────────────────────────────────")
        print(original)
        print(f"\n🟩 TRUNCATED ({len(truncated)} chars)")
        print("──────────────────────────────────────────────────────────────────────────────")
        print(truncated)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    return truncated

def generate_seed(config):
    """
    Generate a single seed for the entire pipeline run.
    Uses fixed seed if provided, otherwise random within bounds.
    """
    lower = int(config.get("seed_lower_bound"))
    upper = int(config.get("seed_upper_bound"))
    fixed = config.get("seed_fixed_value")

    if fixed is not None:
        seed = int(fixed)
        print(f"🌱 Using fixed seed: {seed}")
        return seed

    seed = random.randint(lower, upper)
    print(f"🎲 Generated random seed: {seed}")
    return seed

def build_output_path(config, input_stem, step_name):
    """
    Build a consistent output filename:
    YYYYMMDD_seed_inputstem_step.png
    """
    date_str = datetime.now().strftime("%Y%m%d")
    seed = config["seed"]
    output_dir = Path(config["output_dir"])

    filename = f"{date_str}_{seed}_{input_stem}_{step_name}.png"
    return output_dir / filename
