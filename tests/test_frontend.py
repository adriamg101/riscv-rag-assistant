import httpx

from frontend import app as frontend

SOURCE = {
    "label": "S1",
    "section": "1.1.5.2. Conditional Branches",
    "path": "RV32I > Conditional Branches",
    "source_url": "https://example.test/rv32.html#branches",
    "score": 0.78,
    "text": "The conditional branch range is ±4 KiB.",
}


def response(status, body):
    request = httpx.Request("POST", "http://api.test/ask")
    return httpx.Response(status, json=body, request=request)


def test_format_response_shows_answer_and_sources():
    text = frontend.format_response({"answer": "±4 KiB [S1].", "sources": [SOURCE]})

    assert text.startswith("±4 KiB [S1].")
    assert "[1.1.5.2. Conditional Branches](https://example.test/rv32.html#branches)" in text


def test_format_response_in_retrieval_only_mode():
    text = frontend.format_response({"answer": None, "sources": [SOURCE]})
    assert "Retrieval-only" in text


def test_respond_sends_no_model_in_auto_mode(monkeypatch):
    sent = {}

    def post(url, json, timeout):
        sent.update(json)
        return response(200, {"answer": "OK [S1]", "sources": [SOURCE]})

    monkeypatch.setattr(frontend.httpx, "post", post)

    assert "OK [S1]" in frontend.respond("q", [], frontend.AUTO)
    assert sent == {"question": "q", "model": None}


def test_respond_explains_llm_failure(monkeypatch):
    monkeypatch.setattr(
        frontend.httpx,
        "post",
        lambda url, json, timeout: response(502, {"detail": "The LLM provider failed"}),
    )
    assert "Try another model" in frontend.respond("q", [], "groq")


def test_respond_handles_unreachable_api(monkeypatch):
    def post(url, json, timeout):
        raise httpx.ConnectError("no connection")

    monkeypatch.setattr(frontend.httpx, "post", post)
    assert "Cannot reach the API" in frontend.respond("q", [], frontend.AUTO)


def test_model_choices_show_status(monkeypatch):
    body = {
        "retrieval_only": False,
        "providers": [{"name": "groq", "model": "m", "status": "rate limited until 10:05"}],
    }
    monkeypatch.setattr(frontend.httpx, "get", lambda url, timeout: response(200, body))

    choices = frontend.model_choices()

    assert choices[0] == ("Auto", frontend.AUTO)
    assert choices[1] == ("groq · m (rate limited until 10:05)", "groq")


def test_model_choices_survive_api_failure(monkeypatch):
    def get(url, timeout):
        raise httpx.ConnectError("no connection")

    monkeypatch.setattr(frontend.httpx, "get", get)
    assert frontend.model_choices() == [("Auto", frontend.AUTO)]
