"""Local LLM client via Ollama HTTP API.

Provides embedding generation and text inference using models
running locally on the Mac via Ollama. No data leaves the machine.

Setup (run once on your M4 Mac):
    brew install ollama
    ollama serve &
    ollama pull nomic-embed-text       # 137M param embedding model, ~274MB
    ollama pull qwen3:8b               # 8B param, ~4.9GB, fast structured extraction
    ollama pull qwen3:30b-a3b          # 30B MoE (3B active), ~18GB, deep reasoning

The 30B MoE variant (qwen3:30b-a3b) is ideal for your M4/48GB:
  - Only 3B parameters active per token → fast inference (~40 tok/s)
  - Full 30B knowledge base for reasoning quality
  - Leaves ~25GB free for graph, embeddings, and OS

For embedding, nomic-embed-text produces 768-dim vectors and runs
near-instantly on M4. Alternative: all-minilm (384-dim, faster, slightly
less accurate).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Model recommendations for M4 48GB
RECOMMENDED_MODELS = {
    "embedding": "nomic-embed-text",
    "fast": "qwen3:8b",           # fast classification, entity extraction
    "reasoning": "qwen3:30b-a3b", # deep reasoning, summarization
}

EMBEDDING_DIMS = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "mxbai-embed-large": 1024,
}


class OllamaClient:
    """Client for Ollama's local inference API.

    All calls go to localhost — nothing leaves the machine.
    """

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL):
        self.base_url = base_url.rstrip("/")
        self._available_models: set[str] | None = None

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """List locally available models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                self._available_models = set(models)
                return models
        except requests.RequestException:
            pass
        return []

    def has_model(self, model: str) -> bool:
        """Check if a specific model is available."""
        if self._available_models is None:
            self.list_models()
        return model in (self._available_models or set())

    # ══════════════════════════════════════════════════════════════
    # Embedding
    # ══════════════════════════════════════════════════════════════

    def embed(
        self,
        text: str,
        model: str = "nomic-embed-text",
    ) -> Optional[np.ndarray]:
        """Generate an embedding vector for text.

        Returns numpy array of shape (dim,) or None on failure.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=30,
            )

            if resp.status_code != 200:
                logger.error("Ollama embed failed: %s %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                return None

            return np.array(embeddings[0], dtype=np.float32)

        except requests.RequestException as exc:
            logger.error("Ollama embed request failed: %s", exc)
            return None

    def embed_batch(
        self,
        texts: list[str],
        model: str = "nomic-embed-text",
        batch_size: int = 32,
    ) -> list[Optional[np.ndarray]]:
        """Embed multiple texts. Returns list parallel to input."""
        results: list[Optional[np.ndarray]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": model, "input": batch},
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    for emb in embeddings:
                        results.append(np.array(emb, dtype=np.float32))
                    # Pad if fewer results than inputs
                    while len(results) < i + len(batch):
                        results.append(None)
                else:
                    results.extend([None] * len(batch))

            except requests.RequestException:
                results.extend([None] * len(batch))

        return results

    # ══════════════════════════════════════════════════════════════
    # Text generation
    # ══════════════════════════════════════════════════════════════

    def generate(
        self,
        prompt: str,
        model: str = "qwen3:8b",
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        format_json: bool = False,
    ) -> Optional[str]:
        """Generate text from a prompt.

        Args:
            prompt: User message.
            model: Which local model to use.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            format_json: If True, request JSON output format.

        Returns text response or None on failure.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        if format_json:
            payload["format"] = "json"
            # Disable thinking mode — it conflicts with JSON format
            # and produces empty responses on many model sizes.
            payload["think"] = False

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )

            if resp.status_code != 200:
                logger.error("Ollama generate failed: %s %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            return data.get("response", "")

        except requests.RequestException as exc:
            logger.error("Ollama generate request failed: %s", exc)
            return None

    def generate_json(
        self,
        prompt: str,
        model: str = "qwen3:8b",
        system: str = "",
        temperature: float = 0.1,
    ) -> Optional[dict]:
        """Generate and parse JSON output from a prompt."""
        raw = self.generate(
            prompt=prompt,
            model=model,
            system=system,
            temperature=temperature,
            format_json=True,
        )

        if not raw:
            return None

        # Clean markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Ollama JSON response: %s...", raw[:200])
            return None

    # ══════════════════════════════════════════════════════════════
    # Chat (multi-turn)
    # ══════════════════════════════════════════════════════════════

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "qwen3:8b",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        format_json: bool = False,
    ) -> Optional[str]:
        """Multi-turn chat with a local model.

        messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if format_json:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )

            if resp.status_code != 200:
                logger.error("Ollama chat failed: %s", resp.status_code)
                return None

            data = resp.json()
            return data.get("message", {}).get("content", "")

        except requests.RequestException as exc:
            logger.error("Ollama chat request failed: %s", exc)
            return None


def check_local_llm_setup() -> dict[str, Any]:
    """Check Ollama installation and available models.

    Returns a dict with status info and setup instructions if needed.
    """
    client = OllamaClient()
    result: dict[str, Any] = {
        "ollama_running": False,
        "models_available": [],
        "models_missing": [],
        "ready": False,
        "instructions": [],
    }

    if not client.is_available():
        result["instructions"] = [
            "Ollama is not running. Install and start it:",
            "  brew install ollama",
            "  ollama serve &",
            "",
            "Then pull the required models:",
            f"  ollama pull {RECOMMENDED_MODELS['embedding']}",
            f"  ollama pull {RECOMMENDED_MODELS['fast']}",
            f"  ollama pull {RECOMMENDED_MODELS['reasoning']}",
        ]
        return result

    result["ollama_running"] = True
    available = client.list_models()
    result["models_available"] = available

    for role, model in RECOMMENDED_MODELS.items():
        # Check if model name matches (ollama uses name:tag format)
        found = any(model in m for m in available)
        if not found:
            result["models_missing"].append(f"{model} ({role})")
            result["instructions"].append(f"  ollama pull {model}")

    result["ready"] = len(result["models_missing"]) == 0

    if result["models_missing"]:
        result["instructions"].insert(0, "Missing models. Pull them:")

    return result
