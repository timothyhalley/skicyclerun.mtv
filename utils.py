import os
from PIL import Image
from pathlib import Path
import json
import sys
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout


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