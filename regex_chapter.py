import fitz  # PyMuPDF
import re
import json
import uuid
# Path to your PDF
pdf_path = "Topic REFRIGERATION.pdf"
output_json = "output/regex_chapters.json"
# Regex to detect chapter titles
chapter_pattern = re.compile(
    r"(Chapter\s*[–-]\s*[IVXLC\d]+\s*\([^)]+\))",
    re.IGNORECASE
)
# Open PDF
doc = fitz.open(pdf_path)
chapter_data = []
text_pages = []
# :white_check_mark: Extract text starting from page 6 (skip contents)
for page_num, page in enumerate(doc, start=1):
    if page_num >= 6:
        text = page.get_text("text")
        text_pages.append((page_num, text))
# Combine text from page 6 onward
full_text = "\n".join([t for _, t in text_pages])
# Find all chapter title matches
matches = list(chapter_pattern.finditer(full_text))
for i, match in enumerate(matches):
    chapter_title = match.group(1).strip()
    start_idx = match.start()
    end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
    chapter_text = full_text[start_idx:end_idx].strip()
    # Estimate start and end pages based on chapter title location
    start_page = None
    end_page = None
    for page_num, text in text_pages:
        if start_page is None and chapter_title in text:
            start_page = page_num
        if start_page is not None and chapter_title not in text:
            end_page = page_num - 1
    if end_page is None:
        end_page = doc.page_count
    # :white_check_mark: Added chapter_title field
    chapter_data.append({
        "chapter_number": i + 1,
        "chapter_title": chapter_title,
        "chapter_id": uuid.uuid4().hex[:8],
        "start_page": start_page or 6,
        "end_page": end_page,
        "chapter_text_length": len(chapter_text),
        "chapter_text": chapter_text
    })
# Save as JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(chapter_data, f, ensure_ascii=False, indent=4)
print(f":white_check_mark: Extracted {len(chapter_data)} chapters starting from page 6.")
print(f":file_folder: Output saved to: {output_json}")















