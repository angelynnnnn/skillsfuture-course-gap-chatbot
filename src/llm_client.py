import os
import requests


def generate_with_ollama(prompt: str, json_mode: bool = False) -> str:
    """
    Local/free LLM option using Ollama.
    """
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    # Ollama supports JSON mode for models that can follow it.
    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=400,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def generate_with_gemini(prompt: str, json_mode: bool = False) -> str:
    """
    Google AI Studio / Gemini API option.

    Required environment variable:
    - GEMINI_API_KEY

    Optional:
    - GEMINI_MODEL, default gemini-2.5-flash
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    generation_config = {
        "temperature": 0.1,
    }
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": generation_config,
    }

    response = requests.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError(f"Gemini returned no content parts: {data}")

    return "".join(part.get("text", "") for part in parts).strip()


def call_llm(prompt: str, json_mode: bool = False):
    """
    Returns (text, error).

    Supported providers:
    - gemini: Google AI Studio / Gemini API
    - ollama: local Ollama model
    - none: disabled
    """
    provider = os.getenv("LLM_PROVIDER", "none").lower().strip()

    if provider in ("", "none", "off", "false"):
        return None, "LLM_PROVIDER is not enabled. Set LLM_PROVIDER=gemini or LLM_PROVIDER=ollama."

    if provider == "gemini":
        try:
            return generate_with_gemini(prompt, json_mode=json_mode), None
        except Exception as exc:
            return None, f"Gemini call failed: {exc}"

    if provider == "ollama":
        try:
            return generate_with_ollama(prompt, json_mode=json_mode), None
        except Exception as exc:
            return None, f"Ollama call failed: {exc}"

    return None, f"Unknown LLM_PROVIDER: {provider}"
