# import fitz  # PyMuPDF
# import re
# import json
# import uuid
# # Path to your PDF
# pdf_path = "Topic REFRIGERATION.pdf"
# output_json = "output/regex_chapters.json"
# # Regex to detect chapter titles
# chapter_pattern = re.compile(
#     r"(Chapter\s*[–-]\s*[IVXLC\d]+\s*\([^)]+\))",
#     re.IGNORECASE
# )
# # Open PDF
# doc = fitz.open(pdf_path)
# chapter_data = []
# text_pages = []
# # :white_check_mark: Extract text starting from page 6 (skip contents)
# for page_num, page in enumerate(doc, start=1):
#     if page_num >= 6:
#         text = page.get_text("text")
#         text_pages.append((page_num, text))
# # Combine text from page 6 onward
# full_text = "\n".join([t for _, t in text_pages])
# # Find all chapter title matches
# matches = list(chapter_pattern.finditer(full_text))
# for i, match in enumerate(matches):
#     chapter_title = match.group(1).strip()
#     start_idx = match.start()
#     end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
#     chapter_text = full_text[start_idx:end_idx].strip()
#     # Estimate start and end pages based on chapter title location
#     start_page = None
#     end_page = None
#     for page_num, text in text_pages:
#         if start_page is None and chapter_title in text:
#             start_page = page_num
#         if start_page is not None and chapter_title not in text:
#             end_page = page_num - 1
#     if end_page is None:
#         end_page = doc.page_count
#     # :white_check_mark: Added chapter_title field
#     chapter_data.append({
#         "chapter_number": i + 1,
#         "chapter_title": chapter_title,
#         "chapter_id": uuid.uuid4().hex[:8],
#         "start_page": start_page or 6,
#         "end_page": end_page,
#         "chapter_text_length": len(chapter_text),
#         "chapter_text": chapter_text
#     })
# # Save as JSON
# with open(output_json, "w", encoding="utf-8") as f:
#     json.dump(chapter_data, f, ensure_ascii=False, indent=4)
# print(f":white_check_mark: Extracted {len(chapter_data)} chapters starting from page 6.")
# print(f":file_folder: Output saved to: {output_json}")










import fitz  # PyMuPDF
import re
import json
import uuid
import os
pdf_path = "Topic REFRIGERATION.pdf"
output_json = "output/regex_chapters.json"
images_dir = "output/images"
FIRST_CONTENT_PAGE = 6  # skip TOC pages
# Create folder for images
os.makedirs(images_dir, exist_ok=True)
# Regex to detect chapter titles (flexible)
chapter_pattern = re.compile(
    r"Chapter\s*(?:–|—|-)?\s*([IVXLCDM0-9]+)\s*\(?([A-Z][^)]+)?\)?",
    re.IGNORECASE | re.DOTALL
)
doc = fitz.open(pdf_path)
n_pages = doc.page_count
chapter_starts = []
# --- Step 1: Find all chapters ---
for pno in range(FIRST_CONTENT_PAGE, n_pages + 1):
    page = doc.load_page(pno - 1)
    text = re.sub(r"\s+", " ", page.get_text("text"))
    for m in chapter_pattern.finditer(text):
        roman_or_num = m.group(1) or ""
        title_part = (m.group(2) or "").strip()
        full_title = f"Chapter – {roman_or_num}"
        if title_part:
            full_title += f" ({title_part})"
        chapter_starts.append({
            "page": pno,
            "chapter_number_raw": roman_or_num,
            "chapter_title": full_title
        })
if not chapter_starts:
    print(":x: No chapters found. Try lowering FIRST_CONTENT_PAGE to 1.")
    raise SystemExit
# --- Step 2: Build chapter metadata ---
chapters = []
roman_map = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
def roman_to_int(s):
    total, prev = 0, 0
    for c in reversed(s):
        val = roman_map.get(c, 0)
        total += -val if val < prev else val
        prev = val
    return total
for idx, ch in enumerate(chapter_starts):
    start_page = ch["page"]
    end_page = chapter_starts[idx + 1]["page"] - 1 if idx + 1 < len(chapter_starts) else n_pages
    raw_num = ch["chapter_number_raw"].upper()
    try:
        chapter_num = roman_to_int(raw_num) if all(c in roman_map for c in raw_num) else int(raw_num)
    except:
        chapter_num = idx + 1
    # Clean folder name
    safe_title = re.sub(r'[^A-Za-z0-9]+', '_', ch["chapter_title"]).strip('_')
    chapter_folder = os.path.join(images_dir, f"Chapter_{chapter_num}_{safe_title}")
    os.makedirs(chapter_folder, exist_ok=True)
    # --- Step 3: Extract images for this chapter ---
    chapter_images = []
    for p in range(start_page, end_page + 1):
        page = doc.load_page(p - 1)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_filename = f"page_{p}_img_{img_index + 1}.{img_ext}"
            img_path = os.path.join(chapter_folder, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            chapter_images.append(img_path)
    # --- Step 4: Combine text for this chapter ---
    chapter_text = ""
    for p in range(start_page, end_page + 1):
        chapter_text += doc.load_page(p - 1).get_text("text") + "\n"
    # --- Step 5: Append final data ---
    chapters.append({
        "chapter_number": chapter_num,
        "chapter_title": ch["chapter_title"],
        "chapter_id": uuid.uuid4().hex[:8],
        "start_page": start_page,
        "end_page": end_page,
        "chapter_text_length": len(chapter_text),
        "chapter_text": chapter_text.strip(),
        "chapter_images": chapter_images
    })
# --- Step 6: Save everything to JSON ---
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(chapters, f, ensure_ascii=False, indent=4)
print(f"\n:white_check_mark: Extracted {len(chapters)} chapters and their images.")
print(f":file_folder: JSON saved to: {output_json}")
print(f":frame_with_picture: Images saved under: {images_dir}/")




