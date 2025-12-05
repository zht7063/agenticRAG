from typing import List

from langchain_core.documents import Document
from src.services.datasource.html_toolkit import HTMLToolkit

def test_html_toolkit(url: str): 
    html_toolkit = HTMLToolkit()
    splits = html_toolkit.get_splits(url)
    for split in splits[:10]:
        print(split.page_content)
        print(split.metadata)
        print("-" * 100)
    