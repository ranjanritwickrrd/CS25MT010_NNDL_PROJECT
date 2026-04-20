from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageOps

from .config import PROJECT_ROOT
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
.compare-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-top: 8px;
}
.compare-tile {
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 10px 24px rgba(15,23,42,0.06);
}
.compare-tile-header {
  padding: 10px 14px;
  background: linear-gradient(90deg, rgba(37,99,235,0.10), rgba(14,165,233,0.06));
  font-weight: 700;
  color: #0f172a;
}
.compare-tile-body {
  padding: 12px;
}
.compare-tile-body img {
  width: 100%;
  height: 220px;
  object-fit: contain;
  background: #f8fafc;
  border-radius: 12px;
}
.compare-caption {
  margin-top: 10px;
  color: #475569;
  font-size: 0.92rem;
}
.workspace-card {
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(15,23,42,0.08);
  box-shadow: 0 10px 24px rgba(15,23,42,0.05);
  margin-bottom: 12px;
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


def build_status_html(message: str) -> str:
    return f"<div class='workspace-card'><strong>Status:</strong> {message}</div>"


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_model_cards_html(model_outputs: list[tuple[str, Image.Image, dict]]) -> str:
    cards = []
    for model_name, output_image, info in model_outputs:
        image_b64 = pil_to_base64(output_image)
        cards.append(
            f"""
            <div class="compare-tile">
              <div class="compare-tile-header">{model_name.title()}</div>
              <div class="compare-tile-body">
                <img src="data:image/png;base64,{image_b64}" alt="{model_name} output" />
                <div class="compare-caption">
                  Input {info['input_size']} -> Output {info['processed_size']} on {info['device']}
                </div>
              </div>
            </div>
            """
        )
    return f"<div class='compare-grid'>{''.join(cards)}</div>"


def compare_all_models(uploaded_image: Image.Image):
    if uploaded_image is None:
        raise gr.Error("Upload an image first.")

    model_outputs = []
    summary_lines = []

    for model_name in available_models():
        output_image, info = dehaze_pil_image(uploaded_image, model_name)
        model_outputs.append((model_name, output_image, info))
        summary_lines.append(
            f"<div><strong>{model_name.title()}:</strong> processed at {info['processed_size']} on {info['device']}</div>"
        )

    return build_model_cards_html(model_outputs), f"<div class='metric-card'>{''.join(summary_lines)}</div>", build_status_html("All trained models were compared successfully.")


def run_inference(uploaded_image: Image.Image, model_name: str):
    if uploaded_image is None:
        raise gr.Error("Upload an image first.")

    output_image, info = dehaze_pil_image(uploaded_image, model_name)
    comparison_image = build_comparison_image(uploaded_image, output_image)
    info_html = build_info_html(info)
    status_html = build_status_html(f"{model_name.title()} model finished dehazing the uploaded image.")
    return output_image, comparison_image, info_html, status_html


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
            with gr.Column(scale=4):
                image_input = gr.Image(type="pil", label="Upload Hazy Image", height=320)
                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=models[0],
                    label="Model Variant",
                    info="Choose the trained model you want to use for restoration.",
                )
                with gr.Row():
                    submit_button = gr.Button("Run Dehazing", variant="primary")
                    compare_button = gr.Button("Compare All Models")

            with gr.Column(scale=6):
                status_output = gr.HTML(value=build_status_html("Upload an image and choose an action to begin."))
                info_output = gr.HTML(value=build_status_html("Run details will appear here after inference."))
                with gr.Tabs():
                    with gr.Tab("Selected Model Output"):
                        image_output = gr.Image(type="pil", label="Dehazed Output", height=300)
                    with gr.Tab("Before / After View"):
                        comparison_output = gr.Image(type="pil", label="Before and After Comparison", height=300)
                    with gr.Tab("All Models Comparison"):
                        comparison_gallery = gr.HTML(value=build_status_html("Use Compare All Models to inspect every trained variant."))
                        compare_info = gr.HTML(value=build_status_html("Comparison details will appear here after multi-model evaluation."))

        clear_button = gr.ClearButton(
            value="Clear",
            components=[
                image_input,
                image_output,
                comparison_output,
                comparison_gallery,
                compare_info,
                info_output,
                status_output,
            ],
        )

        gr.Markdown("### Try a sample image")

        sample_images = [
            str(PROJECT_ROOT / "data" / "hazy" / "0001_0.8_0.2.jpg"),
            str(PROJECT_ROOT / "data" / "hazy" / "0104_0.95_0.12.jpg"),
        ]

        gr.Examples(
            examples=[
                [sample_images[0], models[0]],
                [sample_images[1], models[0]],
            ],
            inputs=[image_input, model_dropdown],
            outputs=[image_output, comparison_output, info_output, status_output],
            fn=run_inference,
            cache_examples=False,
        )

        submit_button.click(
            fn=run_inference,
            inputs=[image_input, model_dropdown],
            outputs=[image_output, comparison_output, info_output, status_output],
        )

        compare_button.click(
            fn=compare_all_models,
            inputs=[image_input],
            outputs=[comparison_gallery, compare_info, status_output],
        )

    return demo
