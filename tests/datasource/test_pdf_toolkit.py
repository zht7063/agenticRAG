from typing import List
from langchain_core.documents import Document
from pathlib import Path
from src.services.datasource.pdf_toolkit import PDFToolkit


def test_pdf_toolkit() -> List[Document]:
    pdf_toolkit = PDFToolkit()
    pdf_path = Path("assets/pdf/xianfa.pdf")
    splits = pdf_toolkit.get_splits(pdf_path = pdf_path)
    return splits


if __name__ == "__main__":

    splits = test_pdf_toolkit()
    for split in splits:
        print(split.page_content)
        print(split.metadata)
        print("-" * 100)
