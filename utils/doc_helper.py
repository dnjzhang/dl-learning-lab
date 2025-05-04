import os

from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader


def clean_text(text):
    """
    Remove blank lines from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: Cleaned text with blank lines removed.
    """
    return "\n".join([line for line in text.splitlines() if line.strip()])


def clean_document_content(document):
    """
    Clean the content of a Document object by removing blank lines while preserving metadata.

    Args:
        document (Document): A LangChain Document object.

    Returns:
        Document: A new Document object with cleaned content and preserved metadata.
    """

    # Clean the content using clean_text
    cleaned_content = clean_text(document.page_content)

    document.page_content = cleaned_content
    # Create a new Document object with cleaned content and original metadata
    return document

def load_html_files_from_directory(directory_path, loader_type='unstructured'):
    """
    Load HTML files from the specified directory using the chosen loader.

    Args:
        directory_path (str): Path to the directory containing HTML files.
        loader_type (str): Type of loader to use ('unstructured' or 'bs4').

    Returns:
        list: A list of Document objects loaded from the HTML files.
    """
    documents = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.html') or filename.endswith('.htm'):
            file_path = os.path.join(directory_path, filename)

            if loader_type == 'unstructured':
                loader = UnstructuredHTMLLoader(file_path)
            elif loader_type == 'bs4':
                loader = BSHTMLLoader(file_path)
            else:
                raise ValueError("Invalid loader_type. Choose 'unstructured' or 'bs4'.")

            docs = loader.load()
            clean_document_content(docs[0])
            documents.extend(docs)

    return documents