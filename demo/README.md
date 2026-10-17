---
title: RiSE-9-DCT
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: 3.10
app_file: app.py
pinned: false
license: mit
models:
  - Jason37437/RiSE
---

# RiSE-9-DCT

Document forgery localization with nine residual experts, JPEG DCT features, and LPC expert selection.

Upload an image to obtain its forgery overlay, probability heatmap, and binary mask. Every upload is automatically resized to 512 × 512 and JPEG-encoded once at quality 100 before inference. Outputs correspond to the processed 512 × 512 image.

The app loads the released checkpoint strictly, fuses DCT at block index 10, and uses sequential expert evaluation to reduce GPU memory. ADF quadrature order is 5 and mask threshold is 0.5. The full backbone is included in the checkpoint; the bundled backbone configuration contains no weights or credentials.

[Source code](https://github.com/KahimWong/RiSE) · [Released models](https://huggingface.co/Jason37437/RiSE)

## Deploy

From the RiSE repository root, authenticate with `hf auth login`, then run:

```bash
python tools/deploy_space.py --repo-id YOUR_USERNAME/RiSE-9-DCT
```

This assembles the demo with the shared inference code and requests **ZeroGPU** hardware. It does not provision paid GPU hardware. ZeroGPU queues and daily visitor quotas apply.
