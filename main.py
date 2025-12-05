from tests.datasource.test_pdf_toolkit import test_pdf_toolkit
from tests.datasource.test_html_toolkit import test_html_toolkit


if __name__ == "__main__":
    # splits = test_pdf_toolkit()
    test_html_toolkit(url = "https://arxiv.org/html/2402.08954v1")
