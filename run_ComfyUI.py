import json
import time
import copy
import csv
import requests
from pathlib import Path
from datetime import datetime
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
    for node_id, node_output in history_entry.get("outputs", {}).items():
        if "images" not in node_output:
            continue
        for img_info in node_output["images"]:
            filename = img_info["filename"]
            params = {
                "filename": filename,
                "subfolder": img_info.get("subfolder", ""),
                "type": img_info.get("type", "output")
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

def list_images(folder):
    p = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [str(f) for f in p.iterdir() if f.suffix.lower() in exts]

def generate_values(start, step, iterations):
    return [round(start + step * i, 4) for i in range(iterations)]

def build_prefix(prefix_base, image_name, seed, label, value):
    short = label[0].lower()
    val = int(value * 1000)
    return f"{prefix_base}{image_name}_s{seed}_{short}{val}"

def embed_metadata(path, metadata):
    """
    Embed metadata into the image.
    - JPEG/TIFF → uses piexif (EXIF)
    - PNG → uses tEXt chunks via Pillow
    """
    from pathlib import Path
    path = Path(path)

    try:
        comment = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        print(f"   📝 Attempting to embed metadata into {path.name}...")

        suffix = path.suffix.lower()

        if suffix in {".jpg", ".jpeg", ".tif", ".tiff"}:
            # --- JPEG / TIFF path (piexif) ---
            import piexif
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = comment.encode("utf-8")
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(path))
            print(f"   ✅ Metadata embedded (EXIF / piexif)")

        elif suffix == ".png":
            # --- PNG path (Pillow tEXt) ---
            from PIL import Image, PngImagePlugin
            img = Image.open(path)

            meta = PngImagePlugin.PngInfo()
            # Store the full block under a single key
            meta.add_text("parameters", comment)
            # Also store individual keys for easier reading later
            for k, v in metadata.items():
                meta.add_text(str(k), str(v))

            img.save(path, pnginfo=meta)
            print(f"   ✅ Metadata embedded (PNG tEXt)")

        else:
            print(f"   ⚠️ Unsupported format for metadata: {suffix}")
            return

    except Exception as e:
        import traceback
        print(f"   ⚠️ Metadata failed: {type(e).__name__}: {e}")
        print("   ── Full traceback ──")
        traceback.print_exc()


# --------------------------------------------------
# Pretty printing helpers
# --------------------------------------------------

def print_header(title):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)

def print_section(title):
    print(f"\n── {title} ──────────────────────────────────────")

def print_kv(emoji, key, value, width=18):
    print(f"   {emoji} {key:<{width}} : {value}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    cfg = load_config()
    workflow_template = load_workflow(cfg["workflow_path"])
    base_url = cfg["comfy_url"].rstrip("/")
    prompt_text = load_prompt(cfg["prompt_file"])

    # --- Sweep setup ---
    sweep = cfg["sweep"]
    target_name = sweep["target"]
    node_def = cfg["nodes"][target_name]
    node_id = node_def["node_id"]
    key = node_def["key"]
    values = generate_values(sweep["start"], sweep["step"], sweep["iterations"])

    # --- Startup banner ---
    print_header("🚀  COMFYUI PARAMETER SWEEP  🚀")
    print_kv("📝", "Prompt file", cfg["prompt_file"])
    print_kv("📄", "Prompt length", f"{len(prompt_text)} chars")
    print_kv("🖼️", "Input folder", cfg["input_folder"])
    print_kv("🎨", "Style image", cfg["style_image"])
    print_kv("📁", "Output dir", cfg["output_dir"])
    print_kv("🧪", "Dry run", cfg.get("dry_run", False))

    print_section("PROMPT")
    print(prompt_text)

    print_section("SWEEP TARGET")
    print_kv("🎯", "Target", target_name)
    print_kv("🔢", "Node", f"{node_id}.{key}")
    print_kv("📈", "Values", values)
    print_kv("🔄", "Iterations", len(values))

    # Show what the other important nodes will be set to
    print_section("FIXED SETTINGS (will be the same for every run)")
    # These are the common ones people care about
    fixed_map = {
        "denoise":        ("17", "denoise",   "🔧"),
        "style_strength": ("41", "strength",  "🎨"),
        "guidance":       ("26", "guidance",  "⚡"),
    }
    for name, (nid, k, emoji) in fixed_map.items():
        if name == target_name:
            print_kv(emoji, name, f"(being swept → {values})")
        else:
            # Try to read the default from the workflow if possible
            try:
                default = workflow_template[nid]["inputs"].get(k, "unknown")
            except Exception:
                default = "unknown"
            print_kv(emoji, name, default)

    images = list_images(cfg["input_folder"])
    if not images:
        print(f"\n⚠️  No images found in {cfg['input_folder']}")
        return

    print_section("IMAGES TO PROCESS")
    print_kv("📷", "Count", len(images))
    for i, img in enumerate(images, 1):
        print(f"      {i:2d}. {Path(img).name}")

    total_runs = len(images) * len(values)
    print_kv("🧮", "Total runs", total_runs)
    print("═" * 70)

    # --- Logging ---
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg["log_csv"])
    log_exists = log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if not log_exists:
        writer.writerow(["timestamp", "prompt_id", target_name, "seed", "image", "output_files", "status"])

    dry_run = cfg.get("dry_run", False)
    run_counter = 0

    for img_idx, image_path in enumerate(images, start=1):
        image_name = Path(image_path).stem

        print_header(f"📷  IMAGE {img_idx}/{len(images)}  →  {Path(image_path).name}")

        for i, value in enumerate(values):
            run_counter += 1
            workflow = copy.deepcopy(workflow_template)

            # Inject fixed inputs
            workflow["40"]["inputs"]["image"] = image_path
            workflow["47"]["inputs"]["image"] = cfg["style_image"]
            workflow["6"]["inputs"]["text"] = prompt_text

            # The single swept value
            workflow[node_id]["inputs"][key] = float(value)

            # Seed
            if cfg.get("fixed_seed") is not None:
                seed = int(cfg["fixed_seed"])
            else:
                seed = abs(hash(f"{image_name}_{i}")) % (2**31)
            workflow["25"]["inputs"]["noise_seed"] = seed

            # Filename
            prefix = build_prefix(cfg["filename_prefix"], image_name, seed, target_name, value)
            workflow["9"]["inputs"]["filename_prefix"] = prefix

            # ---- Verbose status block ----
            print(f"\n┌─ Run {run_counter:03d}/{total_runs} ─────────────────────────────────────┐")
            print_kv("🎯", "Sweeping", f"{target_name} = {value:.4f}")
            print_kv("🌱", "Seed", seed)
            print_kv("🖼️", "Subject", Path(image_path).name)
            print_kv("🎨", "Style ref", Path(cfg["style_image"]).name)
            print_kv("📝", "Prompt", prompt_text[:60] + ("…" if len(prompt_text) > 60 else ""))
            print_kv("💾", "Prefix", prefix)

            # Show the other parameters that are currently fixed
            for name, (nid, k, emoji) in fixed_map.items():
                if name == target_name:
                    continue
                try:
                    current = workflow[nid]["inputs"].get(k, "unknown")
                except Exception:
                    current = "unknown"
                print_kv(emoji, name, current)

            print("└──────────────────────────────────────────────────────────┘")

            if dry_run:
                print("   🧪 DRY RUN – skipping submit")
                writer.writerow([
                    datetime.now().isoformat(), "", f"{value:.4f}",
                    seed, image_path, "", "dry_run"
                ])
                log_file.flush()
                continue

            # Submit
            try:
                print("   📤 Submitting to ComfyUI…")
                prompt_id = submit_prompt(base_url, workflow)
                print(f"   ✅ Submitted  →  prompt_id: {prompt_id}")
            except Exception as e:
                print(f"   ❌ Submit ERROR: {e}")
                writer.writerow([datetime.now().isoformat(), "", f"{value:.4f}", seed, image_path, "", "submit_failed"])
                log_file.flush()
                continue

            # Wait
            try:
                print("   ⏳ Waiting for completion…")
                history = wait_for_completion(
                    base_url, prompt_id,
                    poll_interval=cfg.get("poll_interval", 2.0),
                    max_wait=cfg.get("max_wait_seconds", 600)
                )
                print("   ✅ Generation complete")
            except Exception as e:
                print(f"   ❌ Wait ERROR: {e}")
                writer.writerow([datetime.now().isoformat(), prompt_id, f"{value:.4f}", seed, image_path, "", "timeout"])
                log_file.flush()
                continue

            # Download + metadata
            try:
                print("   📥 Downloading images…")
                saved = download_images(base_url, history, cfg["output_dir"], prefix)
                status = "success" if saved else "no_images"
                for path in saved:
                    embed_metadata(path, {
                        "seed": seed,
                        target_name: value,
                        "prompt": prompt_text,
                        "timestamp": datetime.now().isoformat()
                    })
                if saved:
                    print(f"   🎉 Success – {len(saved)} image(s) saved")
            except Exception as e:
                print(f"   ❌ Download ERROR: {e}")
                saved = []
                status = "download_failed"

            writer.writerow([
                datetime.now().isoformat(), prompt_id, f"{value:.4f}",
                seed, image_path, ";".join(saved), status
            ])
            log_file.flush()
            time.sleep(1.0)

    log_file.close()

    print_header("✅  SWEEP FINISHED")
    print_kv("📄", "Log file", str(log_path))
    print_kv("📁", "Output folder", cfg["output_dir"])
    print_kv("🧮", "Total runs completed", run_counter)
    print("═" * 70)


if __name__ == "__main__":
    main()