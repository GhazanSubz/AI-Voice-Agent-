"""Application settings. Everything configurable comes from environment variables (or .env)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./data/patients.db"
    seed_demo_data: bool = True
    log_level: str = "INFO"

    # Public https origin Vapi reaches us on (the ngrok static domain), no trailing slash.
    public_base_url: str = ""

    # Shared secret Vapi sends on every webhook and TTS request (generate a long random string).
    vapi_secret: str = ""
    # Open-weights model Vapi runs on Groq for the conversation.
    vapi_llm_model: str = "openai/gpt-oss-120b"

    # Only used by scripts/chat_cli.py, which talks to Groq directly to test the agent in text.
    groq_api_key: str = ""

    # Kokoro-82M runs locally on CPU. The int8 model is ~4x smaller and faster than fp32.
    kokoro_model_path: str = "models/kokoro-v1.0.int8.onnx"
    kokoro_voices_path: str = "models/voices-v1.0.bin"
    kokoro_voice_en: str = "af_heart"
    kokoro_voice_es: str = "ef_dora"
    # Set false to skip loading Kokoro at startup (tests, API-only runs).
    tts_warm_up: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
