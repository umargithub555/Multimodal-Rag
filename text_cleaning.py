import re

def clean_page_text(text: str) -> str:
    """
    Cleans OCR or scanned book text for better embeddings.
    Removes unwanted symbols, headers/footers, URLs, emails, and page numbers.
    """
    # 1. Replace unicode bullets or weird characters
    text = text.replace("\uf0b7", "-")
    text = text.replace("\uf0d8", "")
    # 2. Remove URLs or hyperlinks
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # 3. Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
    # 4. Remove page numbers or indicators like "Page 12" or "- 45 -"
    text = re.sub(r"(\bpage\s*\d+\b|\b\d+\b|-\s*\d+\s*-)", "", text, flags=re.IGNORECASE)
    # 5. Remove excessive newlines
    text = re.sub(r"\n+", "\n", text)
    # 6. Remove book headers/footers (case-insensitive)
    text = re.sub(r"(pakistan\s*culture\s*&?\s*society|lecture\s*#?\s*\d+)", "", text, flags=re.IGNORECASE)
    # 7. Merge broken lines (convert line breaks inside paragraphs to spaces)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # 8. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # 9. Strip leading/trailing spaces
    text = text.strip()
    return text
