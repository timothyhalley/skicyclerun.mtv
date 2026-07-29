import json
import time
import copy
import csv
import requests
from pathlib import Path
from datetime import datetime
import itertools
import piexif


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
        start = s["start"]
        step = s["step"]
        iterations = s["iterations"]
        vals = [(start + step * i) for i in range(iterations)]
        value_lists.append(vals)
        labels.append(s.get("label", f"{s['node_id']}.{s['key']}"))

    combinations = list(itertools.product(*value_lists))
    return combinations, labels

def list_images(folder):
    p = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [str(f) for f in p.iterdir() if f.suffix.lower() in exts]

def build_prefix(prefix_base, image_name, seed, combo, labels):
    parts = [prefix_base, image_name, f"s{seed}"]

    for label, value in zip(labels, combo):
        short = label[0].lower()  # d, s, g
        val = int(value * 1000)   # 0.950 → 950
        parts.append(f"{short}{val}")

    return "_".join(parts)

def embed_metadata(path, metadata):
    try:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # Build a readable metadata block
        comment = "\n".join([f"{k}: {v}" for k, v in metadata.items()])

        # EXIF requires bytes, not str
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = comment.encode("utf-8")

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, path)

        print(f"   📝 Metadata embedded")

    except Exception as e:
        print(f"   ⚠️ Metadata failed: {e}")




# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    cfg = load_config()
    workflow_template = load_workflow(cfg["workflow_path"])
    base_url = cfg["comfy_url"].rstrip("/")

    prompt_text = load_prompt(cfg["prompt_file"])
    print(f"📝 Prompt loaded from {cfg['prompt_file']} ({len(prompt_text)} chars)")

    print("──────────────────────────────────────────────")
    print("📣 PROMPT CONTENT")
    print("──────────────────────────────────────────────")
    print(prompt_text)
    print("──────────────────────────────────────────────\n")

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    log_path = Path(cfg["log_csv"])
    log_exists = log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)

    combinations, labels = generate_sweep_values(cfg["sweeps"])

    if not log_exists:
        header = ["timestamp", "prompt_id"] + labels + ["seed", "image", "output_files", "status"]
        writer.writerow(header)

    images = list_images(cfg["input_folder"])
    if not images:
        print(f"⚠️ No images found in {cfg['input_folder']}")
        return

    print(f"🚀 Starting sweep")
    print(f"   Total images: {len(images)}")
    print(f"   Total combinations per image: {len(combinations)}")
    print(f"   Parameters: {labels}")
    print(f"   Dry run: {cfg.get('dry_run', False)}")
    print("-" * 70)

    dry_run = cfg.get("dry_run", False)

    for img_idx, image_path in enumerate(images, start=1):
        image_name = Path(image_path).stem
        print(f"📷 [{img_idx}/{len(images)}] Image: {image_path}")

        for i, combo in enumerate(combinations):
            workflow = copy.deepcopy(workflow_template)

            # inject images + prompt
            workflow["40"]["inputs"]["image"] = image_path
            workflow["47"]["inputs"]["image"] = cfg["style_image"]
            workflow["6"]["inputs"]["text"] = prompt_text

            # apply swept values
            param_str_parts = []
            for s, value in zip(cfg["sweeps"], combo):
                node_id = s["node_id"]
                key = s["key"]
                workflow[node_id]["inputs"][key] = float(value)
                label = s.get("label", key)
                param_str_parts.append(f"{label}={value:.3f}")

            param_str = "__".join(param_str_parts)

            # seed logic: fixed or per-image deterministic
            if cfg.get("fixed_seed") is not None:
                seed = int(cfg["fixed_seed"])
            else:
                # simple deterministic seed from image name + combo index
                seed = abs(hash(f"{image_name}_{i}")) % (2**31)
            workflow["25"]["inputs"]["noise_seed"] = int(seed)

            prefix = build_prefix(cfg['filename_prefix'], image_name, seed, combo, labels)
            workflow["9"]["inputs"]["filename_prefix"] = prefix

            print(f"[{i+1:03d}/{len(combinations)}]")
            icons = {
                "denoise": "🔧",
                "style_strength": "🎨",
                "guidance": "⚡"
            }

            for label, value in zip(labels, combo):
                icon = icons.get(label, "🔧")
                print(f"   {icon} {label:<12} = {value:.3f}")

            if dry_run:
                # sanity check only
                writer.writerow([
                    datetime.now().isoformat(),
                    "",
                    *[f"{v:.4f}" for v in combo],
                    seed,
                    image_path,
                    "",
                    "dry_run"
                ])
                log_file.flush()
                continue

            # submit
            try:
                prompt_id = submit_prompt(base_url, workflow)
                print(f"   → Submitted. ID: {prompt_id}")
            except Exception as e:
                print(f"   → Submit ERROR: {e}")
                writer.writerow([datetime.now().isoformat(), "", *[f"{v:.4f}" for v in combo], seed, image_path, "", "submit_failed"])
                log_file.flush()
                continue

            # wait
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
                writer.writerow([datetime.now().isoformat(), prompt_id, *[f"{v:.4f}" for v in combo], seed, image_path, "", "timeout"])
                log_file.flush()
                continue

            # download
            try:
                saved = download_images(base_url, history, cfg["output_dir"], prefix)
                status = "success" if saved else "no_images"

                for saved_path in saved:
                    denoise_val = combo[labels.index("denoise")]
                    style_val = combo[labels.index("style_strength")]
                    guidance_val = combo[labels.index("guidance")]

                    metadata = {
                        "seed": seed,
                        "denoise": denoise_val,
                        "style_strength": style_val,
                        "guidance": guidance_val,
                        "prompt": prompt_text,
                        "timestamp": datetime.now().isoformat()
                    }

                    embed_metadata(saved_path, metadata)

            except Exception as e:
                print(f"   → Download ERROR: {e}")
                saved = []
                status = "download_failed"

            writer.writerow([
                datetime.now().isoformat(),
                prompt_id,
                *[f"{v:.4f}" for v in combo],
                seed,
                image_path,
                ";".join(saved),
                status
            ])
            log_file.flush()

            time.sleep(1.0)

    log_file.close()
    print("-" * 70)
    print(f"✅ Sweep finished. Log: {log_path}")
    print(f"   Images: {cfg['output_dir']}")

    total_images = len(images)
    total_combos = len(combinations)
    total_runs = total_images * total_combos

    print("\n📊 SUMMARY")
    print(f"   Images processed: {total_images}")
    print(f"   Sweep combinations: {total_combos}")
    print(f"   Total runs: {total_runs}")
    print(f"   Steps: {workflow_template['17']['inputs']['steps']}")
    print(f"   Sampler: {workflow_template['16']['inputs']['sampler_name']}")
    print(f"   Estimated time: ~{total_runs * cfg.get('poll_interval', 2.0):.1f} seconds")


if __name__ == "__main__":
    main()
