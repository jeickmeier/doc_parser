# doc_parser

`doc_parser` is a lightweight Python library that turns PDFs into clean, GitHub-flavoured Markdown with the help of OpenAI's vision-enabled models. It also ships a small wrapper around the [agents](https://github.com/f/agents) framework that makes it easy to run text- and image-based prompts with robust retry logic.

## Features

- 🚀 **One-liner PDF → Markdown** – convert entire documents with a single call.
- 🖼️ **Vision model support** – uses OpenAI's multimodal endpoints under the hood.
- 🧵 **Automatic concurrency** – pages are rendered & processed in parallel for speed.
- 🔁 **Resilient execution** – built-in retries with exponential back-off.
- 📄 **Markdown-first output** – produces ready-to-render GitHub Markdown, including tables, lists and LaTeX maths.

## Installation

```bash
# clone the repo
$ git clone https://github.com/your-org/doc_parser.git && cd doc_parser

# create a virtual env (any tool works – here we use python -m venv)
$ python -m venv .venv && source .venv/bin/activate

# install in editable mode + dev extras
$ pip install -e .[dev]
```

Make sure you have an **OpenAI API key** available in your environment:

```bash
export OPENAI_API_KEY="sk-..."
```

## Quickstart

Below is a minimal, end-to-end example identical to the one found in `scratch2.ipynb`.
It converts a PDF into Markdown **and** asks the model for a short summary of the extracted text.

```python
import asyncio
import logging

from doc_parser.convert_pdf import PDFConverter
from doc_parser.utils.openai_agent import OpenAIAgent

logging.basicConfig(level=logging.INFO)

async def main() -> None:
    pdf_file = "inputs/your_document.pdf"  # path to your PDF

    # 1️⃣  Convert the PDF to Markdown ---------------------------------------
    pdf_converter = PDFConverter(pdf_file, dpi=300, max_workers=10)
    await pdf_converter.convert_to_markdown(temperature=0.0)  # vision model
    markdown_text = pdf_converter.get_markdown()

    # 2️⃣  Ask a follow-up question on the extracted text --------------------
    agent = OpenAIAgent(model_name="gpt-4.1-nano")
    await agent._text_agent(
        "Extract a three-word summary from the provided text",
        markdown_text,
    )
    result = await agent.run()

    print("Markdown characters:", len(markdown_text))
    print("Three-word summary :", result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

> **Tip**: The library is fully asynchronous; wrap calls in `asyncio.run()` when
> using scripts or notebooks without an active event loop.

## API Overview

### `PDFConverter`

| Method | Description |
| ------ | ----------- |
| `await convert_to_markdown(...)` | Converts all pages to Markdown. |
| `get_markdown()` | Returns the concatenated Markdown string. |

### `OpenAIAgent`

| Method | Description |
| ------ | ----------- |
| `await _vision_agent(prompt, image_b64)` | Low-level helper for vision requests. |
| `await _text_agent(prompt, context)` | Low-level helper for text-only prompts. |
| `await run()` | Executes the prepared agent with automatic retries. |

## License

Distributed under the MIT license. See `LICENSE` for more information.
