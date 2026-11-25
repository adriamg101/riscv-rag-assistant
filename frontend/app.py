"""Minimal interface that only talks to the API. Run: python -m frontend.app"""

import os

import gradio as gr
import httpx

DEFAULT_API_URL = "http://localhost:8000"
AUTO = "auto"
TIMEOUT_SECONDS = 150.0
EXCERPT_CHARS = 300


def api_url() -> str:
    return (os.getenv("API_URL") or DEFAULT_API_URL).rstrip("/")


def model_choices() -> list[tuple[str, str]]:
    """Dropdown options with each provider's current status."""
    choices = [("Auto", AUTO)]
    try:
        response = httpx.get(f"{api_url()}/models", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return choices
    for provider in response.json()["providers"]:
        label = f"{provider['name']} · {provider['model']} ({provider['status']})"
        choices.append((label, provider["name"]))
    return choices


def refresh_models() -> gr.Dropdown:
    return gr.Dropdown(choices=model_choices(), value=AUTO)


def format_response(body: dict) -> str:
    answer = body["answer"]
    if answer is None:
        answer = "*Retrieval-only mode: these are the most relevant excerpts.*"
    if not body["sources"]:
        return answer

    lines = [answer, "", "**Sources**"]
    for source in body["sources"]:
        excerpt = " ".join(source["text"].split())[:EXCERPT_CHARS]
        lines.append(
            f"- **[{source['label']}]** [{source['section']}]({source['source_url']}) "
            f"(similarity {source['score']:.2f})\n  > {excerpt}…"
        )
    return "\n".join(lines)


def respond(message: str, _history: list, model: str) -> str:
    payload = {"question": message, "model": None if model == AUTO else model}
    try:
        response = httpx.post(f"{api_url()}/ask", json=payload, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return f"Cannot reach the API at {api_url()}."
    if response.status_code == 200:
        return format_response(response.json())
    if response.status_code == 422:
        return "The question or model is not valid."
    if response.status_code == 502:
        return f"{response.json()['detail']}. Try another model."
    return f"API error ({response.status_code})."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="RISC-V RAG Assistant", analytics_enabled=False) as demo:
        dropdown = gr.Dropdown(choices=[("Auto", AUTO)], value=AUTO, label="Model")
        gr.ChatInterface(
            respond,
            additional_inputs=[dropdown],
            title="RISC-V RAG Assistant",
            description="Ask about the RV32I ISA and get answers that cite the specification.",
            analytics_enabled=False,
        )
        demo.load(refresh_models, outputs=dropdown)
    return demo


if __name__ == "__main__":
    build_demo().launch()
