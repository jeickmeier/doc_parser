from agents import Agent, Runner
from dotenv import load_dotenv
import logging
import os
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading environment variables")
load_dotenv(override=True)
logger.info("OPENAI_API_KEY: %s", os.getenv("OPENAI_API_KEY"))

class OpenAIAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.messages = []
        self.agent = None
        self.runner = None

    async def _vision_agent(self, prompt: str, image_base64: str):
        
        self.agent = Agent(
            name="VisionExtractor",
            instructions=(
                "You are a vision model that extracts text from images following the"
                " provided instructions. Respond with *only* the extracted markdown "
                "content—no additional commentary."
            ),
            model=self.model_name,
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

    async def run(
        self,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """Run the agent asynchronously with automatic retries.

        Args:
            max_retries: Maximum number of retry attempts before giving up.
            base_delay: Base delay (in seconds) used for exponential backoff
                between retries.

        Returns:
            The result returned by :pyfunc:`Runner.run`.

        Raises:
            Exception: Re-raises the last encountered exception if all retry
                attempts fail.
        """

        attempt = 0

        while True:
            try:
                result = await Runner.run(self.agent, self.messages)

                # Treat `None` or empty responses as failures warranting a
                # retry—this covers cases where the run "didn't finish" but
                # did not throw an exception.
                if result is None or (isinstance(result, str) and not result.strip()):
                    raise RuntimeError("Runner returned no result; retrying …")

                return result

            except Exception as exc:
                attempt += 1
                logger.warning(
                    "Runner attempt %d/%d failed: %s", attempt, max_retries, exc
                )

                if attempt >= max_retries:
                    logger.error("Max retries exceeded. Raising last error.")
                    raise

                # Exponential back-off before the next attempt
                delay = base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)