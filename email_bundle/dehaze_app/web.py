from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageOps

from .inference import available_models, dehaze_pil_image


APP_CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top left, #f5f9ff 0%, #eef4ff 28%, #f8fbff 55%, #ffffff 100%);
}
.hero {
  padding: 18px 22px;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(38,99,235,0.10), rgba(15,118,110,0.08));
  border: 1px solid rgba(38,99,235,0.12);
  margin-bottom: 10px;
}
.hero h1 {
  margin: 0 0 8px 0;
  font-size: 2rem;
}
.hero p {
  margin: 0;
  color: #334155;
  line-height: 1.6;
}
.metric-card {
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(15,23,42,0.08);
  box-shadow: 0 10px 24px rgba(15,23,42,0.06);
}
.metric-card strong {
  color: #0f172a;
}
"""


def build_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size="lg",
        text_size="md",
    )


def build_comparison_image(input_image: Image.Image, output_image: Image.Image) -> Image.Image:
    """Create a labeled side-by-side panel for easier visual comparison."""
    panel_width = 320
    panel_height = 260
    header_height = 44
    padding = 16

    canvas = Image.new("RGB", (panel_width * 2 + padding * 3, panel_height + header_height + padding * 2), "#f8fafc")
    draw = ImageDraw.Draw(canvas)

    for index, (title, image) in enumerate([("Uploaded Image", input_image), ("Dehazed Result", output_image)]):
        x0 = padding + index * (panel_width + padding)
        y0 = padding
        x1 = x0 + panel_width
        y1 = y0 + panel_height + header_height

        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill="#ffffff", outline="#dbe4f0", width=2)
        draw.rounded_rectangle((x0 + 1, y0 + 1, x1 - 1, y0 + header_height), radius=18, fill="#e8f0ff", outline=None)
        draw.text((x0 + 14, y0 + 12), title, fill="#0f172a")

        fitted = ImageOps.pad(image.convert("RGB"), (panel_width - 24, panel_height - 24), color="#ffffff")
        canvas.paste(fitted, (x0 + 12, y0 + header_height + 12))

    return canvas


def build_info_html(info: dict) -> str:
    return f"""
    <div class="metric-card">
      <div><strong>Model:</strong> {info['model_name'].title()}</div>
      <div><strong>Device:</strong> {info['device']}</div>
      <div><strong>Original Size:</strong> {info['input_size']}</div>
      <div><strong>Network Input:</strong> {info['processed_size']}</div>
      <div style="margin-top:8px;"><strong>Note:</strong> Best results are expected on hazy or low-contrast outdoor images.</div>
    </div>
    """


def run_inference(uploaded_image: Image.Image, model_name: str):
    if uploaded_image is None:
        raise gr.Error("Upload an image first.")

    output_image, info = dehaze_pil_image(uploaded_image, model_name)
    comparison_image = build_comparison_image(uploaded_image, output_image)
    info_html = build_info_html(info)
    return output_image, comparison_image, info_html


def build_interface() -> gr.Blocks:
    models = available_models()
    if not models:
        raise RuntimeError(
            "No checkpoints were found. Train at least one model first with tools/train_web_models.py."
        )

    with gr.Blocks(title="AOD-Net Dehazing Demo") as demo:
        gr.Markdown(
            """
            <div class="hero">
              <h1>AOD-Net Interactive Dehazing Studio</h1>
              <p>
                Upload a hazy image, pick one of the trained AOD-Net variants, and inspect the restored output instantly.
                This interface is designed for academic presentation, side-by-side comparison, and quick demonstration.
              </p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                image_input = gr.Image(type="pil", label="Upload Hazy Image", height=360)
                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=models[0],
                    label="Model Variant",
                    info="Choose the trained model you want to use for restoration.",
                )
                with gr.Row():
                    submit_button = gr.Button("Run Dehazing", variant="primary")
                    clear_button = gr.ClearButton(value="Clear", components=[image_input])

            with gr.Column(scale=5):
                image_output = gr.Image(type="pil", label="Dehazed Output", height=360)
                comparison_output = gr.Image(type="pil", label="Before and After Comparison", height=360)
                info_output = gr.HTML(label="Run Details")

        gr.Markdown("### Try a sample image")

        gr.Examples(
            examples=[
                [str(Path("data/hazy/0001_0.8_0.2.jpg")), models[0]],
                [str(Path("data/hazy/0104_0.95_0.12.jpg")), models[0]],
            ],
            inputs=[image_input, model_dropdown],
            outputs=[image_output, comparison_output, info_output],
            fn=run_inference,
            cache_examples=False,
        )

        submit_button.click(
            fn=run_inference,
            inputs=[image_input, model_dropdown],
            outputs=[image_output, comparison_output, info_output],
        )

    return demo
