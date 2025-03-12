from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_negative_samples(
    file_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 400,
) -> List[str]:
    """
    Split a text file into overlapping chunks.

    Args:
        file_path (str, optional): Path to the negative file.
        max_char (int, optional): Maximum number of characters per chunk. Defaults to 800.
        overlap (int, optional): Number of characters to overlap between chunks. Defaults to 300.

    Returns:
        List[str]: List of text chunks.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(text)
