import sys
import types
import pytest

# Provide dummy loader modules so utils.doc_helper can be imported without optional deps
loader_module = types.ModuleType("langchain_community.document_loaders")
loader_module.UnstructuredHTMLLoader = object
loader_module.BSHTMLLoader = object
package_module = types.ModuleType("langchain_community")
package_module.document_loaders = loader_module
sys.modules.setdefault("langchain_community", package_module)
sys.modules.setdefault("langchain_community.document_loaders", loader_module)

from utils.doc_helper import clean_text, clean_document_content

class MinimalDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def test_clean_text_removes_blank_lines():
    text = "Line1\n\nLine2\n   \nLine3"
    assert clean_text(text) == "Line1\nLine2\nLine3"

def test_clean_document_content_preserves_metadata():
    doc = MinimalDocument(page_content="A\n\nB", metadata={"key": "value"})
    returned = clean_document_content(doc)
    assert returned is doc
    assert doc.page_content == "A\nB"
    assert doc.metadata == {"key": "value"}
