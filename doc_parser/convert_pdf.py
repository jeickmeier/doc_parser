"""Convert PDF pages to Markdown via an LLM-powered vision extractor.

This module defines `PDFConverter`, a utility class that renders each page of a PDF
into an image (using PyMuPDF) and then uses an OpenAI vision model to extract the
page content as GitHub-flavoured Markdown.  The extracted Markdown for all pages
can be retrieved via `PDFConverter.get_markdown()`.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path  # NEW: prefer pathlib over os
from typing import Any

import fitz  # PyMuPDF

from doc_parser.utils.openai_agent import OpenAIAgent

DEFAULT_VISION_PROMPT = r"""
    You are DocVision-Extractor, an expert LLM that transforms images into well-structured, readable Markdown.
    Your output preserves content accuracy, faithfully replicating tables, lists, figure/chart descriptions, and maintaining contextual coherence.

    Your task is to convert the provided image into **valid Markdown** format.

    ## Global Formatting Rules

    1. **DO NOT** output any explanatory text like "Here is the markdown", "TEXT EXTRACTION",  "Based on the image", etc. Start directly with the actual content.
    2. **DO NOT** include any page numbers, headers, publication dates, vendor or publisher names, or footers.
    3. **Preserve** the document's heading hierarchy.
    4. **DO NOT** wrap the content in ```markdown fences.
    5. **DO NOT** add any horizontal rules or page breaks.

    ## Content-Specific Rules

    ### Tables
    - Use proper GitHub-flavored Markdown table syntax.
    - Align columns for readability.
    - Include header separators.
    - For complex tables with merged cells, use a simplified representation.
    - **DO NOT** create a table from a line chart.

    ### Lists
    - Maintain original list hierarchy (numbered, bulleted, nested).
    - Use consistent list markers.

    ### Formulas and Equations
    - Write mathematical formulas in LaTeX format enclosed in $ for inline or $$ for block equations.
    - Example: $E = mc^2$ or $$\int_{a}^{b} f(x) dx$$

    ### Figures, Charts, and Diagrams
    - For diagrams, the description should be a text-based representation of the diagram with enough detail to understand the content.
    - For figures/charts, describe the content of the figure with a short description describing the chart or
    figure with enough detail to understand the content. Also include any annotations or labels that are present in the figure/charts.

"""


class PDFConverter:
    """Convert a PDF into Markdown using an LLM vision model.

    The workflow is as follows:

    1. Render each PDF page into a raster image (`fitz.Pixmap`).
    2. Pass the image bytes to an OpenAI vision model with a carefully crafted
       prompt (see `DEFAULT_VISION_PROMPT`).
    3. Collect the Markdown output for every page.

    Parameters
    ----------
    pdf_path:
        Path to the input PDF file.
    dpi:
        Resolution used when rasterising pages.  Higher values yield sharper
        images at the cost of increased processing time and memory.
    max_workers:
        Maximum number of threads to use for concurrent operations.  Defaults to
        the number of pages in the document.
    """

    def __init__(self, pdf_path: str, dpi: int = 150, max_workers: int | None = None):
        """Create a new :class:`~doc_parser.convert_pdf.PDFConverter` instance."""
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.page_count = len(self.doc)
        self.dpi = dpi
        self.max_workers = max_workers
        # Rendered pages as ``fitz.Pixmap`` instances.
        self.images: list[fitz.Pixmap] = []
        # Collected model outputs for all pages; ``None`` until populated.
        self.output_parsed: list[Any] | None = None

    def __del__(self) -> None:
        """Ensure the underlying *fitz* document is closed on garbage collection."""
        self.doc.close()

    def _pdf_to_images(self) -> None:
        # Use a thread pool to render pages concurrently. Each call returns a
        # pixmap which we collect into a list, preserving page order.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.images = list(executor.map(self._render_page, range(self.page_count)))

    def _render_page(self, page_number: int) -> fitz.Pixmap:
        """Render a single PDF page to a pixmap.

        Args:
            page_number: Zero-based index of the page to render.

        Returns:
            The rendered PyMuPDF pixmap for the requested page.
        """
        page = self.doc.load_page(page_number)
        return page.get_pixmap(dpi=self.dpi)

    def _save_images(self, output_folder: str) -> None:
        """Save rendered images to *output_folder* concurrently."""
        # Ensure the target directory exists before saving images (PTH103)
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        def _save(idx_img: tuple[int, fitz.Pixmap]) -> None:
            """Helper to save a single image."""
            i, image = idx_img
            image.save(f"{output_folder}/page_{i}.png")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # executor.map automatically iterates through enumerate(self.images)
            list(executor.map(_save, enumerate(self.images)))

    async def _convert_to_markdown(
        self,
        page_number: int,
        prompt: str,
        model_name: str = "gpt-4.1-nano",
        temperature: float = 0.0,
    ) -> Any:
        """Convert a single PDF page (already rendered) to Markdown."""
        agent = OpenAIAgent(model_name=model_name)
        image_bytes = self.images[page_number].tobytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        await agent._vision_agent(prompt, image_b64, temperature)
        result = await agent.run()
        return result

    async def _convert_to_markdown_all(
        self, prompt: str, model_name: str = "gpt-4.1-nano", temperature: float = 0.0
    ) -> None:
        """Convert **all** pages to Markdown concurrently."""
        import asyncio

        # Fallback to using number of pages as workers if not specified
        worker_count = self.max_workers or self.page_count

        results: list[str] = []
        page_numbers = list(range(self.page_count))

        for i in range(0, self.page_count, worker_count):
            batch = page_numbers[i : i + worker_count]
            tasks = [self._convert_to_markdown(page_number, prompt, model_name, temperature) for page_number in batch]
            batch_results = await asyncio.gather(*tasks)
            # `asyncio.gather` returns a list; directly extend `results`
            results.extend(batch_results)

        self.output_parsed = results

    async def convert_to_markdown(
        self,
        prompt: str = DEFAULT_VISION_PROMPT,
        model_name: str = "gpt-4.1-nano",
        temperature: float = 0.0,
    ) -> None:
        """Convert the PDF to markdown."""
        # Check if images are available
        if len(self.images) == 0:
            self._pdf_to_images()

        # Check if markdown is available
        await self._convert_to_markdown_all(prompt, model_name, temperature)

    def get_markdown(self) -> str:
        """Return the concatenated Markdown extracted from all pages.

        Raises:
        ------
        RuntimeError
            If :pymeth:`convert_to_markdown` has not been called yet.
        """
        if self.output_parsed is None:
            raise RuntimeError("No parsed output available; call convert_to_markdown() first.")

        return "\n".join([getattr(i, "final_output", "") for i in self.output_parsed])
