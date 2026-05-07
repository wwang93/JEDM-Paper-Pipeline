from __future__ import annotations

import time
from pathlib import Path

from openai import OpenAI

from .io_utils import ensure_dir, iter_txt_files, read_text, write_text
from .text_cleaning import chunk_paragraphs


class OpenAITextEnhancer:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_chars: int = 2000,
        sleep_seconds: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_chars = max_chars
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
                try:
                    response = self.client.chat.completions.create(**payload)
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    if "temperature" in msg and "default (1)" in msg:
                        payload.pop("temperature", None)
                        payload.pop("top_p", None)
                        response = self.client.chat.completions.create(**payload)
                    else:
                        raise
                return response.choices[0].message.content.strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.0)
        raise RuntimeError(f"OpenAI request failed after retries: {last_err}")

    def transform_text(self, text: str, prompt_template: str, system_prompt: str) -> str:
        chunks = chunk_paragraphs(text, max_chars=self.max_chars)
        outputs: list[str] = []
        for chunk in chunks:
            user_prompt = prompt_template.format(text=chunk)
            outputs.append(self._chat(system_prompt=system_prompt, user_prompt=user_prompt))
            time.sleep(self.sleep_seconds)
        return "\n\n".join(outputs)


def batch_enhance_dir(
    enhancer: OpenAITextEnhancer,
    input_dir: Path,
    output_dir: Path,
    prompt_template: str,
    system_prompt: str,
    skip_existing: bool = True,
) -> tuple[int, int]:
    ensure_dir(output_dir)
    processed = 0
    skipped = 0
    for txt_path in iter_txt_files(input_dir):
        out_path = output_dir / txt_path.name
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        source = read_text(txt_path)
        transformed = enhancer.transform_text(source, prompt_template, system_prompt)
        write_text(out_path, transformed)
        processed += 1
    return processed, skipped
