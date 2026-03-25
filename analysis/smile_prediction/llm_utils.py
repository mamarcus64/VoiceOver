"""
Shared utilities for OpenRouter LLM calls.

Provides async batch calling with concurrency control, retry logic,
and .env loading for OPENROUTER_API_KEY.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_CONCURRENCY = 50


def load_api_key() -> str:
    """Load OPENROUTER_API_KEY from environment or .env file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    return key

    raise RuntimeError(
        "OPENROUTER_API_KEY not found in environment or .env file"
    )


async def call_openrouter(
    session: aiohttp.ClientSession,
    messages: list[dict],
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> dict:
    """
    Single OpenRouter chat completion call with retry logic.
    Returns the parsed JSON content from the response.
    """
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            async with session.post(
                OPENROUTER_URL, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 429:
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = await resp.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            parsed = json.loads(content)
            return {"result": parsed, "usage": usage, "error": None}

        except Exception as e:
            if attempt == max_retries - 1:
                return {"result": None, "usage": {}, "error": str(e)}
            await asyncio.sleep(2 ** attempt)

    return {"result": None, "usage": {}, "error": "max retries exceeded"}


async def batch_call(
    prompts: list[dict],
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress_every: int = 20,
) -> list[dict]:
    """
    Run many OpenRouter calls concurrently with a semaphore.

    Args:
        prompts: list of {"messages": [...], "metadata": {...}} dicts.
            metadata is passed through to the result unchanged.
        progress_every: print progress every N completions.

    Returns list of {"result": ..., "usage": ..., "error": ..., "metadata": ...}.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)
    completed = 0
    t0 = time.time()

    async def _run_one(idx: int, prompt: dict):
        nonlocal completed
        async with semaphore:
            resp = await call_openrouter(
                session, prompt["messages"], api_key,
                model=model, temperature=temperature,
            )
            resp["metadata"] = prompt.get("metadata", {})
            results[idx] = resp

            completed += 1
            if completed % progress_every == 0 or completed == len(prompts):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(prompts) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(prompts)}] "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_run_one(i, p) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)

    return results
