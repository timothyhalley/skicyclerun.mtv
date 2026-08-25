Flux Dual‑Image Workflow — Settings & Best‑Practice Guide
For ComfyUI (Flux1‑Dev + Redux Style Model + Dual CLIP + Vision Conditioning)
This guide explains how to tune the four key parameters in your workflow:
• Denoise (Node 17)
• Primary Strength (Node 41)
• Style Strength (Node 45)
• Guidance (Node 26)
It also includes recommended ranges and presets for different artistic outcomes.
---
1. Denoise Strength
Node: 17
Key: denoise
Your value: 1.00
What it does
Controls how strongly the sampler modifies the latent image.
Higher values produce more creative freedom and more deviation from the source.
Recommended ranges
Preserve structure: 0.65 to 0.85
Balanced creativity: 0.90 to 0.97
Maximum transformation: 0.98 to 1.00
Best practice
Use 1.00 when generating from an empty latent.
Lower only when doing img2img or revisions.
---
2. Primary Strength
Node: 41
Key: strength
Your value: 1.000
What it does
Controls how strongly the primary reference image influences the output.
Recommended ranges
Strong adherence to primary image: 0.90 to 1.00
Balanced with prompt: 0.70 to 0.85
Prompt‑dominant: 0.40 to 0.60
Best practice
Keep 0.95 to 1.00 when the primary image defines composition.
Reduce to around 0.70 if you want the prompt to reshape the scene.
---
3. Style Strength
Node: 45
Key: strength
Your value: 0.955
What it does
Controls how strongly the secondary style image affects the output.
Recommended ranges
Subtle style influence: 0.30 to 0.55
Balanced style: 0.60 to 0.80
Strong stylization: 0.85 to 1.00
Best practice
For Flux Redux, 0.85 to 0.95 gives strong but controlled stylization.
If the output becomes too stylized or monochrome, reduce to 0.65 to 0.75.
---
4. Guidance Scale
Node: 26
Key: guidance
Your value: 3.3
What it does
Controls how strongly the model follows the text prompt.
Lower values produce more natural, image‑driven results.
Higher values produce more literal prompt adherence.
Recommended ranges
Natural, image‑driven: 2.0 to 3.0
Balanced: 3.0 to 4.0
Strong prompt adherence: 4.0 to 5.0
Best practice
3.3 is a sweet spot for mixed prompt + image workflows.
Avoid going above 4.2 unless you need strict prompt control.
---
Recommended Presets
Preset A — Balanced Output (Default)
denoise: 1.00
primary_strength: 0.95
style_strength: 0.85
guidance: 3.3
Preset B — Strong Stylization (Artistic / Ink / BW)
denoise: 1.00
primary_strength: 0.80
style_strength: 0.95
guidance: 2.8
Preset C — Scene Accuracy (Architecture / Landscapes)
denoise: 0.95
primary_strength: 1.00
style_strength: 0.70
guidance: 3.5
Preset D — Prompt‑Driven (Creative Reinterpretation)
denoise: 1.00
primary_strength: 0.60
style_strength: 0.60
guidance: 4.0
---
Additional Best Practices
Use consistent image resolutions.
Flux performs best at 1024×1024 or 1152×1152.
Keep style and primary images visually distinct.
If both images share similar tones, the style model may overpower the primary.
Avoid extreme guidance.
High guidance can cause over‑sharpening, repetitive patterns, or washed‑out contrast.
Adjust strengths gradually.
Changes of plus or minus 0.05 often produce noticeable differences.