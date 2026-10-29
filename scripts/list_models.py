"""List the Gemini Flash models this API key can call. Prints names only, never the key."""

from google import genai

from paperpilot.config import get_settings


def main() -> None:
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise SystemExit("GEMINI_API_KEY is missing in .env")

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    for model in client.models.list():
        name = model.name or ""
        if "flash" in name.lower():
            print(name, "|", model.display_name)


if __name__ == "__main__":
    main()