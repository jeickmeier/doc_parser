import fitz  # PyMuPDF
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from doc_parser.utils.openai_agent import OpenAIAgent
import os
import base64

class PDFConverter:

    def __init__(self, pdf_path: str, dpi: int = 150, max_workers: Optional[int] = None):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.page_count = len(self.doc)
        self.dpi = dpi
        self.max_workers = max_workers
        self.images = []
        self.output_parsed = None

    def __del__(self):
        self.doc.close()

    def _pdf_to_images(self):
        # Use a thread pool to render pages concurrently. Each call returns a
        # pixmap which we collect into a list, preserving page order.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.images = list(executor.map(self._render_page, range(self.page_count)))

    def _render_page(self, page_number: int):
        """Render a single PDF page to a pixmap.

        Args:
            page_number: Zero-based index of the page to render.

        Returns:
            The rendered PyMuPDF pixmap for the requested page.
        """
        page = self.doc.load_page(page_number)
        return page.get_pixmap(dpi=self.dpi)

    def _save_images(self, output_folder: str):
        """Save rendered images to *output_folder* concurrently.

        Args:
            output_folder: Directory path where PNGs will be written.
        """

        # Ensure the target directory exists before saving images
        os.makedirs(output_folder, exist_ok=True)

        def _save(idx_img):
            """Helper to save a single image.

            Args:
                idx_img: Tuple of (index, pixmap).
            """
            i, image = idx_img
            image.save(f"{output_folder}/page_{i}.png")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # executor.map automatically iterates through enumerate(self.images)
            list(executor.map(_save, enumerate(self.images)))

    async def _convert_to_markdown(self, page_number: int, prompt: str):
        """Convert the PDF to markdown.
        """
        agent = OpenAIAgent(model_name="gpt-4.1-nano")
        image_bytes = self.images[page_number].tobytes()
        image_b64   = base64.b64encode(image_bytes).decode("utf-8")
        await agent._vision_agent(prompt, image_b64)
        result = await agent.run()
        return result
    
    async def _convert_to_markdown_all(self, prompt: str):
        """Convert the PDF to markdown.
        """
        import asyncio

        # Fallback to using number of pages as workers if not specified
        worker_count = self.max_workers or self.page_count

        results: list[str] = []
        page_numbers = list(range(self.page_count))

        for i in range(0, self.page_count, worker_count):
            batch = page_numbers[i : i + worker_count]
            tasks = [
                self._convert_to_markdown(page_number, prompt)
                for page_number in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            # `asyncio.gather` returns a list; directly extend `results`
            results.extend(batch_results)

        self.output_parsed = results

    async def convert_to_markdown(self, prompt: str):
        """Convert the PDF to markdown.
        """
        #Check if images are available
        if len(self.images) == 0:
            self._pdf_to_images()

        #Check if markdown is available
        await self._convert_to_markdown_all(prompt)

