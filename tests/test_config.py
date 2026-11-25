from app.config import is_retrieval_only, load_providers


def test_loads_providers_in_declared_order():
    env = {
        "LLM_PROVIDERS": " groq, local ",
        "GROQ_BASE_URL": "https://groq.test/v1",
        "GROQ_API_KEY": "secret",
        "GROQ_MODEL": "m1",
        "LOCAL_BASE_URL": "http://local.test/v1",
        "LOCAL_MODEL": "m2",
    }
    providers = load_providers(env)
    assert list(providers) == ["groq", "local"]
    assert providers["groq"].api_key == "secret"
    assert providers["local"].api_key == ""


def test_incomplete_provider_is_skipped():
    env = {"LLM_PROVIDERS": "groq", "GROQ_BASE_URL": "https://groq.test/v1"}
    assert load_providers(env) == {}


def test_no_providers_means_retrieval_only():
    assert is_retrieval_only(load_providers({"LLM_PROVIDERS": ""}))


def test_api_key_is_not_in_repr():
    env = {
        "LLM_PROVIDERS": "groq",
        "GROQ_BASE_URL": "https://groq.test/v1",
        "GROQ_API_KEY": "secret",
        "GROQ_MODEL": "m1",
    }
    assert "secret" not in repr(load_providers(env)["groq"])
