from langchain_community.document_loaders import BSHTMLLoader
from langchain.schema import Document
import os
import logging
from typing import List

# Configure the logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Function to load HTML files from a directory and return an array of document objects
def load_html_documents(directory_path: str) -> List[Document]:
    """
    Load all HTML files from the given directory and return an array of document objects.

    Args:
        directory_path (str): Path to the directory containing HTML files.

    Returns:
        List[Document]: A list of Document objects.
    """
    documents = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)

        # Check if the file is an HTML file
        if os.path.isfile(file_path) and filename.lower().endswith(".html"):
            try:
                # Use the BSHTMLLoader to load the file
                loader = BSHTMLLoader(file_path)
                doc = loader.load()
                documents.extend(doc)  # Extend the documents list with loaded documents
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

    return documents

# Example usage
if __name__ == "__main__":
    directory_path = "/Users/jzhang/git-repo/rag/html-docs/test-docs"
    documents = load_html_documents(directory_path)

    # Print the number of documents loaded
    print(f"Loaded {len(documents)} documents.")
    # Print each document
    for i, doc in enumerate(documents, start=1):
        print(f"Document {i}: {doc.metadata}")
