"""Utility wrapper around *agents* for interacting with the OpenAI API.

This module defines :class:`OpenAIAgent`, a thin convenience wrapper that
configures the underlying *agents* library (https://github.com/f/agents) for
both text-only and vision-enabled prompts.  It also adds robust retry logic
with exponential back-off.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, cast

from agents import Agent, ModelSettings, Runner
from dotenv import load_dotenv
from pydantic import BaseModel

logger = logging.getLogger(__name__)

logger.info("Loading environment variables")
load_dotenv(override=True)
logger.info("OPENAI_API_KEY: %s", os.getenv("OPENAI_API_KEY"))


class OpenAIAgent:
    """High-level helper for running text or vision extraction prompts against OpenAI."""

    def __init__(self, model_name: str = "gpt-4.1-nano", temperature: float = 0.0):
        """Create a new agent wrapper.

        Parameters
        ----------
        model_name:
            The OpenAI model name to use (e.g. ``"gpt-4o-mini"``).
        temperature:
            Sampling temperature; higher values yield more diverse outputs.
        """
        self.model_name = model_name
        self.temperature = temperature
        # Chat payload passed to ``Runner``. Each item follows the OpenAI chat
        # format specification (role/content pair), thus we type it broadly as
        # ``dict[str, Any]``.
        self.messages: list[dict[str, Any]] = []
        # The underlying *agents* Agent instance. ``None`` until either
        # :pymeth:`_vision_agent` or :pymeth:`_text_agent` has been invoked.
        self.agent: Agent[Any] | None = None

    async def _vision_agent(self, prompt: str, image_base64: str, temperature: float | None = None) -> None:
        if temperature is None:
            temperature = self.temperature

        self.agent = Agent(
            name="VisionExtractor",
            instructions=(
                "You are a vision model that extracts text from images following the"
                " provided instructions. Respond with *only* the extracted markdown "
                "content—no additional commentary."
            ),
            model=self.model_name,
            model_settings=ModelSettings(temperature=temperature),
        )

        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}",
                        "detail": "low",
                    },
                ],
            }
        ]

    async def _text_agent(
        self,
        prompt: str,
        context: str,
        temperature: float | None = None,
        output_format: str | type[BaseModel] = "markdown",
    ) -> None:
        if temperature is None:
            temperature = self.temperature

        if output_format == "markdown":
            self.agent = Agent(
                name="TextExtractor",
                instructions=prompt,
                model=self.model_name,
                model_settings=ModelSettings(temperature=temperature),
            )
        elif isinstance(output_format, type) and issubclass(output_format, BaseModel):
            self.agent = Agent(
                name="TextExtractor",
                instructions=prompt,
                model=self.model_name,
                output_type=output_format,
                model_settings=ModelSettings(temperature=temperature),
            )

        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": context},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ]

    async def run(
        self,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Any:
        """Run the configured agent with automatic retries.

        The call is retried up to *max_retries* times with exponential back-off
        (doubling after each failed attempt). A response with no usable content
        is treated as a failure and will be retried as well.
        """
        attempt = 0

        while True:
            try:
                if self.agent is None:  # Defensive check for mypy
                    raise RuntimeError("Agent has not been initialised.")

                result = await Runner.run(self.agent, cast("Any", self.messages))
            except Exception as exc:
                attempt += 1
                logger.warning("Runner attempt %d/%d failed: %s", attempt, max_retries, exc)

                if attempt >= max_retries:
                    logger.exception("Max retries exceeded. Raising last error.")
                    raise

                # Exponential back-off before the next attempt
                delay = base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
            else:

                def _is_empty_or_no_content(value: object) -> bool:
                    """Return *True* if *value* is empty or resembles a 204 response.

                    A value is considered empty if it is an empty/whitespace string or
                    contains the phrase "no content" (case-insensitive), including
                    responses such as "204 No Content".
                    """
                    if not isinstance(value, str):
                        return False

                    cleaned = value.strip().lower()
                    return cleaned == "" or "no content" in cleaned or cleaned == "204"

                final_output = getattr(result, "final_output", None)

                if (
                    result is None
                    or (isinstance(result, str) and not result.strip())
                    or final_output is None
                    or _is_empty_or_no_content(final_output)
                ):
                    raise RuntimeError("Runner returned no or empty content; retrying …")

                return result  # Successful run; break the retry loop
