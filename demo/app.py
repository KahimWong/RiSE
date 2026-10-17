"""RiSE-9-DCT Gradio demo; deploy with tools/deploy_space.py."""

# ZeroGPU must patch CUDA before PyTorch or any model module is imported.
import spaces

import io
from pathlib import Path
import sys

import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

# Works both from this repository and from the assembled Space root.
ROOT = Path(__file__).resolve().parent
if (ROOT.parent / "rise").is_dir():
    sys.path.insert(0, str(ROOT.parent))

from rise.checkpoint import load_model
from rise.config import ModelConfig
from rise.data import read_jpeg
from rise.inference import predict

MODEL_REPO = "Jason37437/RiSE"
MODEL_REVISION = "37a9588590a59de22c22a301d0d9d63333e9c464"
PRESERVE = "Original JPEG (512 × 512; no recompression)"
CONVERT = "Resize to 512 × 512 and encode JPEG at quality 90"

# The released checkpoint contains the full backbone. Only its architecture
# config is bundled here, so startup needs no gated backbone download or token.
config = ModelConfig(
    backbone=str(ROOT / "backbone_config"),
    num_experts=9,
    use_dct=True,
    dct_fuse_block_idx=10,
    dct_fuse_prob=1.0,
)
checkpoint = hf_hub_download(MODEL_REPO, "RiSE-9-DCT.pth", revision=MODEL_REVISION)
# CUDA placement at startup is required by ZeroGPU's CUDA emulation.
MODEL, _ = load_model(checkpoint, config, device="cuda")


@spaces.GPU(duration=30)
def predict_gpu(sample):
    """Only GPU inference consumes visitors' ZeroGPU quota."""
    with torch.inference_mode():
        inputs = {name: value.unsqueeze(0).to("cuda") for name, value in sample.items()}
        output = predict(MODEL, **inputs, gh_order=5, mode="sequential")
        return output["probability"][0].float().cpu().numpy()


def analyze(upload, preprocessing):
    if not upload:
        raise gr.Error("Upload an image first.")
    path = Path(upload)
    if path.stat().st_size > 20 * 1024 * 1024:
        raise gr.Error("Please upload an image smaller than 20 MB.")
    raw = path.read_bytes()
    try:
        with Image.open(io.BytesIO(raw)) as source:
            original_size = source.size
            if preprocessing == CONVERT:
                # Explicit conversion supports PNG and arbitrary image sizes.
                # It changes JPEG traces, so it is not the benchmark protocol.
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                raw = buffer.getvalue()
        sample = read_jpeg(raw, use_dct=True)
    except (
        ValueError,
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise gr.Error(
            f"Cannot process this image: {error}. For other sizes or formats, "
            "select the resize/JPEG quality 90 option."
        ) from error
    image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    probability = predict_gpu(sample)
    # Blue → cyan → yellow → red, with a fixed probability scale [0, 1].
    anchors = np.array([[20, 40, 140], [0, 180, 220], [255, 225, 70], [210, 20, 30]])
    x = np.linspace(0, 1, len(anchors))
    heatmap = np.stack([np.interp(probability, x, anchors[:, c]) for c in range(3)], -1)
    mask = (probability >= 0.5).astype(np.uint8) * 255
    alpha = (0.65 * probability)[..., None]
    overlay = (image * (1 - alpha) + np.array([255, 35, 35]) * alpha).astype(np.uint8)
    details = (
        f"Input: {original_size[0]} × {original_size[1]}. "
        + (
            "Resized to 512 × 512 and JPEG-encoded once at quality 90."
            if preprocessing == CONVERT
            else "Original 512 × 512 JPEG bytes preserved."
        )
        + " Mask threshold: 0.5. Heatmap: blue = low, red = high forgery probability."
    )
    return image, overlay, heatmap.astype(np.uint8), mask, details


with gr.Blocks(
    title="RiSE-9-DCT", analytics_enabled=False, delete_cache=(3600, 3600)
) as demo:
    gr.Markdown(
        "# RiSE-9-DCT · Document Forgery Localization\n"
        "Upload a document image to highlight suspected manipulated regions. "
        "[Code](https://github.com/KahimWong/RiSE) · "
        "[Checkpoint](https://huggingface.co/Jason37437/RiSE)\n\n"
        "The model processes **512 × 512** images. The default option resizes and "
        "JPEG-encodes your image at quality 90, which changes its compression traces. "
        "For benchmark-style inference, select **Original JPEG** to preserve its bytes."
    )
    with gr.Row():
        with gr.Column():
            upload = gr.File(
                label="Upload image", file_types=["image"], type="filepath"
            )
            preprocessing = gr.Radio(
                [CONVERT, PRESERVE], value=CONVERT, label="Image preparation"
            )
            run = gr.Button("Localize forgery", variant="primary")
        with gr.Column():
            preview = gr.Image(label="Processed input", interactive=False)
    with gr.Row():
        overlay = gr.Image(label="Forgery overlay", interactive=False)
        heatmap = gr.Image(label="Forgery probability", interactive=False)
        mask = gr.Image(label="Binary mask", interactive=False)
    status = gr.Textbox(label="Processing details", interactive=False)
    run.click(
        analyze,
        [upload, preprocessing],
        [preview, overlay, heatmap, mask, status],
        concurrency_limit=1,
        api_name="localize",
    )
    gr.Markdown(
        "Runs on shared ZeroGPU hardware. Requests may queue or reach daily GPU quotas."
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
