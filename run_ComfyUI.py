import json
import time
import copy
import csv
import shutil
import requests
import traceback
from pathlib import Path
from datetime import datetime
from PIL import Image, PngImagePlugin

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

def resolve(base_dir: str, path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path(base_dir) / p).resolve()

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

def download_images(base_url, history_entry, output_dir: Path, prefix: str):
    """Save images using the filename ComfyUI already generated (prefix is already applied)."""
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
                print(f" ⚠️ Failed to download {filename}")
                continue
            local_path = output_dir / filename
            with open(local_path, "wb") as f:
                f.write(r.content)
            saved_files.append(str(local_path))
            print(f" 💾 Saved: {filename}")
    return saved_files

def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted([str(f) for f in folder.iterdir() if f.suffix.lower() in exts])

def generate_values(start, step, iterations):
    return [round(start + step * i, 4) for i in range(iterations)]

def build_prefix(prefix_base, image_name, seed, label, value):
    short = label[0].lower()
    val = int(value * 1000)
    return f"{prefix_base}{image_name}_s{seed}_{short}{val}"

def stage_image_for_comfy(src_path: str | Path, comfy_input_dir: Path) -> str:
    src = Path(src_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source image does not exist: {src}")
    dest_dir = comfy_input_dir / "staged"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if (not dest.exists()) or (dest.stat().st_mtime < src.stat().st_mtime):
        shutil.copy2(src, dest)
        print(f" 📦 Staged → {dest}")
    else:
        print(f" 📦 Already staged → {dest}")
    return f"staged/{src.name}"

def embed_metadata(path, metadata):
    path = Path(path)
    try:
        comment = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        print(f" 📝 Embedding metadata into {path.name}...")
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".tif", ".tiff"}:
            import piexif
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = comment.encode("utf-8")
            piexif.insert(piexif.dump(exif_dict), str(path))
            print(f" ✅ Metadata embedded (EXIF)")
        elif suffix == ".png":
            img = Image.open(path)
            meta = PngImagePlugin.PngInfo()
            meta.add_text("parameters", comment)
            for k, v in metadata.items():
                meta.add_text(str(k), str(v))
            img.save(path, pnginfo=meta)
            print(f" ✅ Metadata embedded (PNG tEXt)")
        else:
            print(f" ⚠️ Unsupported format for metadata: {suffix}")
    except Exception as e:
        print(f" ⚠️ Metadata failed: {type(e).__name__}: {e}")
        traceback.print_exc()

# --------------------------------------------------
# Pretty printing
# --------------------------------------------------
def print_header(title):
    print("\n" + "═" * 70)
    print(f" {title}")
    print("═" * 70)

def print_section(title):
    print(f"\n── {title} ──────────────────────────────────────")

def print_kv(emoji, key, value, width=22):
    print(f" {emoji} {key:<{width}} : {value}")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    cfg = load_config()
    base = cfg["base_dir"]

    # ---------- Resolve every path once ----------
    workflow_path = resolve(base, cfg["workflow_path"])
    comfy_input_dir = resolve(base, cfg["comfy_input_dir"])
    output_dir = resolve(base, cfg["output_dir"])
    log_csv = resolve(base, cfg["log_csv"])
    prompt_file = resolve(base, cfg["prompt_file"])
    style_image = resolve(base, cfg["style_image"])
    input_folder = resolve(base, cfg["input_folder"])

    workflow_template = load_workflow(workflow_path)
    prompt_text = load_prompt(prompt_file)
    base_url = cfg["comfy_url"].rstrip("/")

    # ---------- Nodes + validation ----------
    nodes = cfg.get("nodes", {})
    if not nodes:
        raise ValueError("Config is missing the 'nodes' section")

    # Emoji mapping (extend as needed)
    emoji_map = {
        "denoise": "🔧",
        "primary_strength": "🖼️",
        "style_strength": "🎨",
        "guidance": "⚡",
    }

    # Validate every declared node exists and has the expected key
    print_section("NODE VALIDATION")
    for name, defn in nodes.items():
        nid = str(defn["node_id"])
        key = defn["key"]
        if nid not in workflow_template:
            raise ValueError(f"Node '{name}' → node_id '{nid}' does not exist in the workflow")
        if key not in workflow_template[nid].get("inputs", {}):
            raise ValueError(f"Node '{name}' → key '{key}' does not exist on node {nid}")
        current = workflow_template[nid]["inputs"][key]
        fixed_val = defn.get("value")
        status = f"fixed={fixed_val}" if fixed_val is not None else f"template={current}"
        print_kv(emoji_map.get(name, "•"), name, f"{nid}.{key}  ({status})")

    # ---------- Sweep setup ----------
    sweep = cfg["sweep"]
    target_name = sweep["target"]
    if target_name not in nodes:
        raise ValueError(f"Sweep target '{target_name}' is not defined in nodes")
    node_def = nodes[target_name]
    node_id = str(node_def["node_id"])
    key = node_def["key"]
    values = generate_values(sweep["start"], sweep["step"], sweep["iterations"])

    # ---------- Startup banner ----------
    print_header("🚀 COMFYUI PARAMETER SWEEP 🚀")
    print_kv("📝", "Prompt file", prompt_file)
    print_kv("📄", "Prompt length", f"{len(prompt_text)} chars")
    print_kv("🖼️", "Input folder", input_folder)
    print_kv("🎨", "Style image", style_image)
    print_kv("📁", "Output dir", output_dir)
    print_kv("📦", "Comfy input (abs)", comfy_input_dir)
    print_kv("🧪", "Dry run", cfg.get("dry_run", False))

    print_section("PROMPT")
    print(prompt_text)

    print_section("SWEEP TARGET")
    print_kv("🎯", "Target", target_name)
    print_kv("🔢", "Node", f"{node_id}.{key}")
    print_kv("📈", "Values", values)
    print_kv("🔄", "Iterations", len(values))

    print_section("FIXED / OVERRIDDEN SETTINGS")
    for name, defn in nodes.items():
        emoji = emoji_map.get(name, "•")
        if name == target_name:
            print_kv(emoji, name, f"(being swept → {values})")
        else:
            if "value" in defn:
                print_kv(emoji, name, f"{defn['value']}  (from config)")
            else:
                try:
                    default = workflow_template[str(defn["node_id"])]["inputs"].get(defn["key"], "unknown")
                except Exception:
                    default = "unknown"
                print_kv(emoji, name, f"{default}  (from template)")

    images = list_images(input_folder)
    if not images:
        print(f"\n⚠️ No images found in {input_folder}")
        return

    print_section("IMAGES TO PROCESS")
    print_kv("📷", "Count", len(images))
    for i, img in enumerate(images, 1):
        print(f"  {i:2d}. {Path(img).name}")

    total_runs = len(images) * len(values)
    print_kv("🧮", "Total runs", total_runs)
    print("═" * 70)

    # ---------- Logging ----------
    output_dir.mkdir(parents=True, exist_ok=True)
    log_exists = log_csv.exists()
    log_file = open(log_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if not log_exists:
        writer.writerow(["timestamp", "prompt_id", target_name, "seed", "image", "output_files", "status"])

    dry_run = cfg.get("dry_run", False)
    run_counter = 0

    for img_idx, image_path in enumerate(images, start=1):
        image_name = Path(image_path).stem
        print_header(f"📷 IMAGE {img_idx}/{len(images)} → {Path(image_path).name}")

        for i, value in enumerate(values):
            run_counter += 1
            workflow = copy.deepcopy(workflow_template)

            # ---- Stage images ----
            try:
                staged_subject = stage_image_for_comfy(image_path, comfy_input_dir)
                staged_style = stage_image_for_comfy(style_image, comfy_input_dir)
            except Exception as e:
                print(f" ❌ Staging failed: {e}")
                writer.writerow([datetime.now().isoformat(), "", f"{value:.4f}", "", image_path, "", "staging_failed"])
                log_file.flush()
                continue

            # Inject images + prompt
            workflow["40"]["inputs"]["image"] = staged_subject
            workflow["47"]["inputs"]["image"] = staged_style
            workflow["6"]["inputs"]["text"] = prompt_text

            # ---- Apply ALL node values (fixed + swept) ----
            for name, defn in nodes.items():
                nid = str(defn["node_id"])
                k = defn["key"]
                if name == target_name:
                    workflow[nid]["inputs"][k] = float(value)
                elif "value" in defn:
                    # Apply the fixed value from config
                    workflow[nid]["inputs"][k] = float(defn["value"]) if isinstance(defn["value"], (int, float)) else defn["value"]

            # Seed
            if cfg.get("fixed_seed") is not None:
                seed = int(cfg["fixed_seed"])
            else:
                seed = abs(hash(f"{image_name}_{i}")) % (2**31)
            workflow["25"]["inputs"]["noise_seed"] = seed

            # Filename prefix
            prefix = build_prefix(cfg["filename_prefix"], image_name, seed, target_name, value)
            workflow["9"]["inputs"]["filename_prefix"] = prefix

            # ---- Verbose status ----
            print(f"\n┌─ Run {run_counter:03d}/{total_runs} ─────────────────────────────────────┐")
            print_kv("🎯", "Sweeping", f"{target_name} = {value:.4f}")
            print_kv("🌱", "Seed", seed)
            print_kv("🖼️", "Subject", Path(image_path).name)
            print_kv("📦", "Staged as", staged_subject)
            print_kv("🎨", "Style ref", Path(style_image).name)
            print_kv("📦", "Staged as", staged_style)
            print_kv("📝", "Prompt", prompt_text[:60] + ("…" if len(prompt_text) > 60 else ""))
            print_kv("💾", "Prefix", prefix)

            # Show every declared node value for this run
            for name, defn in nodes.items():
                emoji = emoji_map.get(name, "•")
                nid = str(defn["node_id"])
                k = defn["key"]
                current = workflow[nid]["inputs"].get(k, "???")
                label = f"{name} ({'swept' if name == target_name else 'fixed'})"
                print_kv(emoji, label, current)

            print("└──────────────────────────────────────────────────────────┘")

            if dry_run:
                print(" 🧪 DRY RUN – skipping submit")
                writer.writerow([
                    datetime.now().isoformat(), "", f"{value:.4f}",
                    seed, image_path, "", "dry_run"
                ])
                log_file.flush()
                continue

            # Submit
            try:
                print(" 📤 Submitting to ComfyUI…")
                prompt_id = submit_prompt(base_url, workflow)
                print(f" ✅ Submitted → prompt_id: {prompt_id}")
            except Exception as e:
                print(f" ❌ Submit ERROR: {e}")
                writer.writerow([datetime.now().isoformat(), "", f"{value:.4f}", seed, image_path, "", "submit_failed"])
                log_file.flush()
                continue

            # Wait
            try:
                print(" ⏳ Waiting for completion…")
                history = wait_for_completion(
                    base_url, prompt_id,
                    poll_interval=cfg.get("poll_interval", 2.0),
                    max_wait=cfg.get("max_wait_seconds", 600)
                )
                print(" ✅ Generation complete")
            except Exception as e:
                print(f" ❌ Wait ERROR: {e}")
                writer.writerow([datetime.now().isoformat(), prompt_id, f"{value:.4f}", seed, image_path, "", "timeout"])
                log_file.flush()
                continue

            # Download + metadata
            try:
                print(" 📥 Downloading images…")
                saved = download_images(base_url, history, output_dir, prefix)
                status = "success" if saved else "no_images"
                for p in saved:
                    embed_metadata(p, {
                        "seed": seed,
                        target_name: value,
                        "prompt": prompt_text,
                        "timestamp": datetime.now().isoformat()
                    })
                if saved:
                    print(f" 🎉 Success – {len(saved)} image(s) saved")
            except Exception as e:
                print(f" ❌ Download ERROR: {e}")
                saved = []
                status = "download_failed"

            writer.writerow([
                datetime.now().isoformat(), prompt_id, f"{value:.4f}",
                seed, image_path, ";".join(saved), status
            ])
            log_file.flush()
            time.sleep(1.0)

    log_file.close()
    print_header("✅ SWEEP FINISHED")
    print_kv("📄", "Log file", log_csv)
    print_kv("📁", "Output folder", output_dir)
    print_kv("📦", "Staged folder", comfy_input_dir / "staged")
    print_kv("🧮", "Total runs completed", run_counter)
    print("═" * 70)
    print("💡 Tip: You can delete the contents of the staged/ folder when finished.")

if __name__ == "__main__":
    main()