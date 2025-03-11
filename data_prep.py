from typing import List


def get_negative_samples(
    file_path: str,
    max_char: int = 1000,
    overlap: int = 300,
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
    chunks = [
        text[i: i + max_char]
        for i in range(0, len(text), max_char - overlap)
        if i + max_char <= len(text)
    ]
    if len(text) % (max_char - overlap) != 0:
        chunks.append(text[len(text) - max_char:])  # add last chunk
    return chunks
