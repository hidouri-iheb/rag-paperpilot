"""Send one tiny request to each configured Gemini model to confirm the setup works."""

from google import genai

from paperpilot.config import get_settings


def main() -> None:
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise SystemExit("GEMINI_API_KEY is missing in .env")

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    models = [("main", settings.gemini_main_model), ("light", settings.gemini_light_model)]
    for role, model in models:
        if not model:
            raise SystemExit(f"GEMINI_{role.upper()}_MODEL is missing in .env")
        response = client.models.generate_content(model=model, contents="Reply with exactly: OK")
        print(f"{role:5} | {model} | {response.text!r}")


if __name__ == "__main__":
    main()