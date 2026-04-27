import os
from pathlib import Path
from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER

# ── Character voice loader ──────────────────────────────────────────────────
VOICES_DIR = Path(__file__).resolve().parent.parent.parent / "voices"


def get_character_choices():
    """Return a list of character names (folder names) found in /voices/."""
    if not VOICES_DIR.exists():
        return []
    return sorted([d.name for d in VOICES_DIR.iterdir() if d.is_dir()])


def get_clips_for_character(character: str):
    """Return a list of .wav file paths inside voices/<character>/."""
    if not character:
        return []
    char_dir = VOICES_DIR / character
    if not char_dir.exists():
        return []
    clips = sorted(char_dir.glob("*.wav"))
    return [str(c) for c in clips]


def get_first_clip(character: str):
    """Return the path of the first .wav clip for a character, or None."""
    clips = get_clips_for_character(character)
    return clips[0] if clips else None


def on_character_change(character: str):
    """When character dropdown changes, update the clip dropdown and audio preview."""
    clips = get_clips_for_character(character)
    clip_labels = [Path(c).name for c in clips]
    first_clip = clips[0] if clips else None
    return gr.update(choices=clip_labels, value=clip_labels[0] if clip_labels else None), first_clip


def on_clip_select(character: str, clip_name: str):
    """When a specific clip is chosen, return its full path for the audio component."""
    if not character or not clip_name:
        return None
    return str(VOICES_DIR / character / clip_name)


# ── App builder ─────────────────────────────────────────────────────────────

def build_app(inference_fct: Callable, theme: str = "light") -> gr.Blocks:
    characters = get_character_choices()
    initial_character = characters[0] if characters else None
    initial_clips = get_clips_for_character(initial_character)
    initial_clip_labels = [Path(c).name for c in initial_clips]
    initial_clip_path = initial_clips[0] if initial_clips else None

    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=10
                )

                with gr.Row():
                    with gr.Column():
                        with gr.Tab(label=i18n("Advanced Config")):
                            with gr.Row():
                                chunk_length = gr.Slider(
                                    label=i18n("Iterative Prompt Length, 0 means off"),
                                    minimum=100,
                                    maximum=400,
                                    value=300,
                                    step=8,
                                )

                                max_new_tokens = gr.Slider(
                                    label=i18n(
                                        "Maximum tokens per batch, 0 means no limit"
                                    ),
                                    minimum=0,
                                    maximum=2048,
                                    value=0,
                                    step=8,
                                )

                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-P",
                                    minimum=0.7,
                                    maximum=0.95,
                                    value=0.8,
                                    step=0.01,
                                )

                                repetition_penalty = gr.Slider(
                                    label=i18n("Repetition Penalty"),
                                    minimum=1,
                                    maximum=1.2,
                                    value=1.1,
                                    step=0.01,
                                )

                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.7,
                                    maximum=1.0,
                                    value=0.8,
                                    step=0.01,
                                )
                                seed = gr.Number(
                                    label="Seed",
                                    info="0 means randomized inference, otherwise deterministic",
                                    value=0,
                                )

                        with gr.Tab(label=i18n("Reference Audio")):
                            gr.Markdown("### 🎙️ Character Voice")

                            with gr.Row():
                                character_dropdown = gr.Dropdown(
                                    label="Character",
                                    choices=characters,
                                    value=initial_character,
                                    interactive=True,
                                )

                            with gr.Row():
                                clip_dropdown = gr.Dropdown(
                                    label="Voice Clip",
                                    choices=initial_clip_labels,
                                    value=initial_clip_labels[0] if initial_clip_labels else None,
                                    interactive=True,
                                )

                            with gr.Row():
                                reference_audio = gr.Audio(
                                    label=i18n("Reference Audio Preview"),
                                    value=initial_clip_path,
                                    type="filepath",
                                    interactive=False,
                                )

                            with gr.Row():
                                reference_id = gr.Textbox(
                                    label=i18n("Reference ID"),
                                    placeholder="Leave empty to use selected character above",
                                    visible=False,  # hidden, not needed with dropdown
                                )

                            with gr.Row():
                                use_memory_cache = gr.Radio(
                                    label=i18n("Use Memory Cache"),
                                    choices=["on", "off"],
                                    value="on",
                                )

                            with gr.Row():
                                reference_text = gr.Textbox(
                                    label=i18n("Reference Text"),
                                    lines=1,
                                    placeholder="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                                    value="",
                                )

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001f3a7 " + i18n("Generate"),
                            variant="primary",
                        )

        # ── Dropdown interactions ────────────────────────────────────────────
        character_dropdown.change(
            fn=on_character_change,
            inputs=[character_dropdown],
            outputs=[clip_dropdown, reference_audio],
        )

        clip_dropdown.change(
            fn=on_clip_select,
            inputs=[character_dropdown, clip_dropdown],
            outputs=[reference_audio],
        )

        # ── Submit ───────────────────────────────────────────────────────────
        generate.click(
            inference_fct,
            [
                text,
                reference_id,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache,
            ],
            [audio, error],
            concurrency_limit=1,
        )

    return app
