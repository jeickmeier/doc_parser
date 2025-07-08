import asyncio
from doc_parser.convert_pdf import PDFConverter
from doc_parser.utils.openai_agent import OpenAIAgent
import base64

pdf_file = "JPM_An_Introduction_to_C_2025-07-02_5004439.pdf"
pdf_converter = PDFConverter(pdf_file, dpi=300, max_workers=10)
pdf_converter._pdf_to_images()

async def extract_first_page_text():
    agent = OpenAIAgent(model_name="gpt-4.1-nano")

    prompt = "Extract the text from the image"

    # Encode image bytes as base-64 for the vision model
    image_bytes = pdf_converter.images[0].tobytes()
    image_b64   = base64.b64encode(image_bytes).decode("utf-8")

    await agent._vision_agent(prompt, image_b64)
    result = await agent.run()
    return result

result = asyncio.run(extract_first_page_text())
print(result)