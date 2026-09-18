"""Local and Hugging Face Spaces interface for actual compact world models."""
from __future__ import annotations
import json
import os
from pathlib import Path
import uuid

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(__file__).resolve().parent / "outputs" / "gradio_cache"))
import gradio as gr
from backend import HERE, available, checkpoint_catalog, forecast, load_clip, observed_images, render_plot, samples


def show_frames(sample_id, appearance):
    item, frames, _ = load_clip(sample_id)
    displayed = (observed_images(frames, int(appearance), item["appearances"]).permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")
    return [(frame, f"Observed support t={i}") for i, frame in enumerate(displayed[:3])], displayed[-1], [(frame, f"Recorded future t={i+3}") for i, frame in enumerate(displayed[3:])]


def infer(checkpoint, episode, appearance):
    if not checkpoint:
        raise gr.Error("No eligible checkpoint for this environment. Complete training or add a verified frozen export.")
    record, frames, metadata = forecast(checkpoint, episode, int(appearance))
    output = HERE / "outputs"
    output.mkdir(exist_ok=True)
    path = output / f"forecast_{uuid.uuid4().hex[:12]}.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    data = [[r["horizon"]] + [round(r[key], 5) for key in ("recorded_action_mse", "zero_action_mismatch", "persistence_mse", "action_effect_mse")] for r in record["rows"]]
    status = (f"**{metadata['status']}** · mode `{metadata['mode']}` · checkpoint epoch {record['checkpoint_epoch']}\n\n"
              f"Actual serialized weights: {metadata['bytes']/1024**2:.1f} MiB. "
              f"This clip was computed on {record['device']} in {record['elapsed_seconds_excluding_load']:.2f} s, excluding model load. "
              "These five forecasts are a single example, not aggregate benchmark performance.")
    package = Path(metadata["directory"])
    return render_plot(record), data, status, record, str(path), [str(package / "model.pt"), str(package / "config.json")]


def build_app():
    catalog = checkpoint_catalog()
    # Freeze choices for this server lifetime to match its file-download allowlist.
    def session_choices(environment):
        checkpoints = [name for name, item in catalog.items() if item["environment"] == environment]
        episodes = [key for key, value in samples().items() if value["environment"] == environment]
        return gr.update(choices=checkpoints, value=checkpoints[0] if checkpoints else None), gr.update(choices=episodes, value=episodes[0])
    environments = sorted({c["environment"] for c in catalog.values()}) or ["pusht", "reacher"]
    initial_environment = environments[0]
    initial_checkpoints = available(initial_environment)
    initial_episodes = [key for key, value in samples().items() if value["environment"] == initial_environment]
    support, goal, future = show_frames(initial_episodes[0], 2)
    with gr.Blocks(title="What Changed? World Model Explorer", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
                   css=".gradio-container {max-width: 1150px !important;} footer {display: none !important;} .prose h1 {font-size: 2rem !important;} @media(max-width:700px) {.stacked-row {flex-direction:column !important;} .stacked-row > * {width:100% !important; flex-basis:auto !important;}}") as app:
        gr.Markdown("# What Changed? World Model Explorer\nCompare action-conditioned forecasts from reusable compact world-model checkpoints. All images below are real simulator observations; the model predicts **latent states, not future video**.")
        with gr.Row(elem_classes="stacked-row"):
            environment = gr.Dropdown(environments, value=initial_environment, label="Environment", scale=1)
            checkpoint = gr.Dropdown(initial_checkpoints, value=initial_checkpoints[0] if initial_checkpoints else None, label="Checkpoint — frozen or training complete", scale=3)
        with gr.Row(elem_classes="stacked-row"):
            episode = gr.Dropdown(initial_episodes, value=initial_episodes[0], label="Held-out test clip", scale=3)
            appearance = gr.Dropdown([("Canonical", 0), ("Warm", 1), ("Cool · held-out combination", 2), ("Dim · appearance extrapolation", 3)], value=2, label="Observation condition", scale=2)
        with gr.Row(elem_classes="stacked-row"):
            with gr.Column(scale=3, min_width=0):
                gr.Markdown("**Actual chronological support — model input**")
                support_gallery = gr.Gallery(support, show_label=False, columns=3, rows=1, height=245, object_fit="contain")
            with gr.Column(scale=1, min_width=0):
                gr.Markdown("**Actual recorded endpoint**")
                goal_image = gr.Image(goal, show_label=False, height=245, interactive=False)
        gr.Markdown("The model sees the three support images and their two executed action blocks. Future images are used **only to score predictions**. Both action candidates are scored against the recorded-action trajectory: the zero-action curve is a sensitivity control, not a verified counterfactual error.")
        run = gr.Button("Run real checkpoint inference", variant="primary")
        status = gr.Markdown("Select a checkpoint and run inference. No numerical result is shown before computation.")
        with gr.Tabs():
            with gr.Tab("Forecast comparison"):
                plot = gr.Plot(show_label=False)
                table = gr.Dataframe(headers=["Horizon", "Recorded-action MSE", "Zero-action mismatch", "Persistence MSE", "Action-effect MSE"], datatype=["number"] * 5, interactive=False, label="Actual computed values")
            with gr.Tab("Recorded future frames"):
                future_gallery = gr.Gallery(future, label="Actual saved trajectory — not generated predictions", columns=5, rows=1, height=230, object_fit="contain")
            with gr.Tab("Provenance and downloads"):
                details = gr.JSON(label="Checkpoint, sample hashes, and interpretation")
                with gr.Row():
                    record_file = gr.File(label="Download this inference record", interactive=False)
                    weights_file = gr.File(label="Download reusable checkpoint files (keep together)", file_count="multiple", interactive=False)
        with gr.Accordion("Model scope and limitations", open=True):
            gr.Markdown((HERE / "MODEL_CARD.md").read_text())
        environment.change(session_choices, environment, [checkpoint, episode], api_name="list_environment")
        episode.change(show_frames, [episode, appearance], [support_gallery, goal_image, future_gallery], api_name=False)
        appearance.change(show_frames, [episode, appearance], [support_gallery, goal_image, future_gallery], api_name=False)
        run.click(infer, [checkpoint, episode, appearance], [plot, table, status, details, record_file, weights_file], api_name="forecast", concurrency_limit=1)
    return app


if __name__ == "__main__":
    catalog = checkpoint_catalog()
    allowed = [str(HERE / "outputs")] + [str(Path(item["directory"]) / name) for item in catalog.values() for name in ("model.pt", "config.json")]
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")), allowed_paths=allowed,
        share=False, show_error=True)
