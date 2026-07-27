import json
import time
import copy
import csv
import requests
from pathlib import Path
from datetime import datetime
import itertools
import numpy as np

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_config(path="run_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_workflow(path):
    with open(path, "r") as f:
        return json.load(f)

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def submit_prompt(base_url, workflow):
    url = f"{base_url}/prompt"
    response = requests.post(url, json={"prompt": workflow})
    if response.status_code != 200:
        raise RuntimeError(f"Submit failed {response.status_code}: {response.text}")
    return response.json()["prompt_id"]

def wait_for_completion(base_url, prompt_id, poll_interval=2.0, max_wait=600):
    history_url = f"{base_url}/history/{prompt_id}"
    start = time.time()

    while True:
        if time.time() - start > max_wait:
            raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")

        r = requests.get(history_url)
        if r.status_code == 200:
            data = r.json()
            if prompt_id in data:
                return data[prompt_id]

        time.sleep(poll_interval)

def download_images(base_url, history_entry, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    outputs = history_entry.get("outputs", {})

    for node_id, node_output in outputs.items():
        if "images" not in node_output:
            continue
        for img_info in node_output["images"]:
            filename = img_info["filename"]
            subfolder = img_info.get("subfolder", "")
            img_type = img_info.get("type", "output")

            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": img_type
            }
            r = requests.get(f"{base_url}/view", params=params)
            if r.status_code != 200:
                print(f"   ⚠️  Failed to download {filename}")
                continue

            local_name = f"{prefix}_{filename}"
            local_path = output_dir / local_name
            with open(local_path, "wb") as f:
                f.write(r.content)
            saved_files.append(str(local_path))
            print(f"   💾 Saved: {local_name}")

    return saved_files

def generate_sweep_values(sweeps):
    value_lists = []
    labels = []

    for s in sweeps:
        vals = np.linspace(s["start"], s["end"], s["steps"]).tolist()
        value_lists.append(vals)
        labels.append(s.get("label", f"{s['node_id']}.{s['key']}"))

    combinations = list(itertools.product(*value_lists))
    return combinations, labels

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    cfg = load_config()
    workflow_template = load_workflow(cfg["workflow_path"])
    base_url = cfg["comfy_url"].rstrip("/")

    # Load prompt from file
    prompt_text = load_prompt(cfg["prompt_file"])
    print(f"📝 Prompt loaded from {cfg['prompt_file']} ({len(prompt_text)} chars)")

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Prepare CSV log
    log_path = Path(cfg["log_csv"])
    log_exists = log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)

    combinations, labels = generate_sweep_values(cfg["sweeps"])

    if not log_exists:
        header = ["timestamp", "prompt_id"] + labels + ["output_files", "status"]
        writer.writerow(header)

    print(f"🚀 Starting creativity sweep (Low → High)")
    print(f"   Total combinations: {len(combinations)}")
    print(f"   Parameters: {labels}")
    print("-" * 70)

    for i, combo in enumerate(combinations):
        workflow = copy.deepcopy(workflow_template)

        # Inject images + prompt
        workflow["40"]["inputs"]["image"] = cfg["primary_image"]
        workflow["47"]["inputs"]["image"] = cfg["style_image"]
        workflow["6"]["inputs"]["text"] = prompt_text

        # Apply all swept values
        param_str_parts = []
        for s, value in zip(cfg["sweeps"], combo):
            node_id = s["node_id"]
            key = s["key"]
            workflow[node_id]["inputs"][key] = float(value)
            label = s.get("label", key)
            param_str_parts.append(f"{label}={value:.3f}")

        param_str = "__".join(param_str_parts)
        print(f"[{i+1:03d}/{len(combinations)}] {param_str}")

        # Optional fixed seed
        if cfg.get("fixed_seed") is not None:
            workflow["25"]["inputs"]["noise_seed"] = int(cfg["fixed_seed"])

        # Unique filename prefix
        prefix = f"{cfg['filename_prefix']}_{param_str}"
        workflow["9"]["inputs"]["filename_prefix"] = prefix

        # Submit
        try:
            prompt_id = submit_prompt(base_url, workflow)
            print(f"   → Submitted. ID: {prompt_id}")
        except Exception as e:
            print(f"   → Submit ERROR: {e}")
            writer.writerow([datetime.now().isoformat(), "", *combo, "", "submit_failed"])
            log_file.flush()
            continue

        # Wait for completion
        try:
            history = wait_for_completion(
                base_url,
                prompt_id,
                poll_interval=cfg.get("poll_interval", 2.0),
                max_wait=cfg.get("max_wait_seconds", 600)
            )
            print(f"   → Completed")
        except Exception as e:
            print(f"   → Wait ERROR: {e}")
            writer.writerow([datetime.now().isoformat(), prompt_id, *combo, "", "timeout"])
            log_file.flush()
            continue

        # Download images
        try:
            saved = download_images(base_url, history, cfg["output_dir"], prefix)
            status = "success" if saved else "no_images"
        except Exception as e:
            print(f"   → Download ERROR: {e}")
            saved = []
            status = "download_failed"

        # Log
        writer.writerow([
            datetime.now().isoformat(),
            prompt_id,
            *[f"{v:.4f}" for v in combo],
            ";".join(saved),
            status
        ])
        log_file.flush()

        time.sleep(1.0)

    log_file.close()
    print("-" * 70)
    print(f"✅ Sweep finished. Log: {log_path}")
    print(f"   Images: {cfg['output_dir']}")

if __name__ == "__main__":
    main()