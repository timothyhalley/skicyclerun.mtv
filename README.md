Multiverse Image Transformer
Generate unified, stylized images as if they were captured from another dimension.
Multiverse is a Python-based pipeline that blends FLUX.1‑dev, SigLIP, IP‑Adapter, and optional LoRA styles to transform ordinary images into coherent visuals from an alternate universe.
It takes multiple input images, applies a shared “multiverse prompt,” and outputs unified, cinematic results.
This project is ideal for:
• Stylizing photo sets into a single aesthetic
• Creating alternate‑dimension versions of real images
• Applying LoRA-driven artistic styles
• Experimenting with FLUX + IP‑Adapter conditioning
---
✨ Features
• Unified Multiverse Transformation
Feed multiple images and generate outputs that share a single dimensional aesthetic.
• FLUX.1‑dev Integration
Uses the latest FLUX transformer, VAE, and text encoders.
• IP‑Adapter Conditioning
Injects reference-image identity/style into the generation process.
• SigLIP Vision Encoder
Extracts high‑quality embeddings from input images.
• LoRA Support
Drop LoRA files into models/loras/ and activate them via config.
• Prompt Packs
Store multiverse prompts in prompts/ for reusable dimensional themes.
• Config‑Driven Pipeline
All model paths, LoRA weights, and generation parameters live in config.json.
---
📁 Project Structure
.
├── main.py                 # Entry point for multiverse generation
├── pipeline.py             # Core FLUX + IP-Adapter pipeline
├── utils.py                # Helpers for loading, preprocessing, logging
├── config.json             # Unified configuration for models & settings
├── prompts/                # Prompt packs (multiverse themes)
├── inputs/                 # Input images to transform
├── outputs/                # Generated multiverse images
├── models/
│   ├── flux/               # FLUX.1-dev model files
│   ├── flux_ipa/           # IP-Adapter weights
│   ├── google/             # SigLIP vision encoder
│   └── loras/              # Optional LoRA styles
├── reference/              # Reference images for conditioning
├── logs/                   # Runtime logs
└── requirements.txt        # Python dependencies

🚀 Usage

1. Install dependencies

pip install -r requirements.txt
