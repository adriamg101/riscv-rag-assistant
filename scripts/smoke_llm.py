"""Live check of the providers in LLM_PROVIDERS. Run: python -m scripts.smoke_llm"""

import time

from openai import APIConnectionError, APIStatusError, OpenAIError

from app.config import is_retrieval_only, load_providers
from app.llm import make_client

# Reasoning models spend tokens before writing, so a tiny limit gives an empty answer.
MAX_TOKENS = 512


def main() -> None:
    providers = load_providers()
    if is_retrieval_only(providers):
        print("No providers configured: retrieval-only mode.")
        return

    for name, provider in providers.items():
        try:
            start = time.perf_counter()
            response = make_client(provider).chat.completions.create(
                model=provider.model,
                temperature=0,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": "Reply only with: OK"}],
            )
            elapsed = time.perf_counter() - start
            if not response.choices:
                print(f"{name} -> (no choices) ({elapsed:.1f}s)")
                continue
            choice = response.choices[0]
            text = choice.message.content or "(empty)"
            print(f"{name} -> {text} [finish={choice.finish_reason}] ({elapsed:.1f}s)")
        except APIStatusError as exc:
            print(f"{name} ERROR HTTP {exc.status_code} {type(exc).__name__}")
        except APIConnectionError as exc:
            cause = type(exc.__cause__).__name__ if exc.__cause__ else "unknown"
            print(f"{name} ERROR {type(exc).__name__} (cause: {cause})")
        except OpenAIError as exc:
            print(f"{name} ERROR {type(exc).__name__}")


if __name__ == "__main__":
    main()
