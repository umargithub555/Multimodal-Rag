# import re 
import fitz

# from app.utils.id_generator import generate_id, generate_simple_id
# from app.utils.text_cleaning import clean_page_text



# async def extract_text_from_pdf(pdf_bytes,file_name:None):
#     """
#     The issue with text extraction from book is that books are very diverse and thats why we cant use plain regix so 
#     instead we will extract text page by page and follow embeddings approach
#     """
#     text = ""
#     try:
#         # book_id = generate_id("book")
#         if not file_name:
#             book_id = generate_id('book')

#         book_id = generate_id(file_name)
#         doc = fitz.open(stream=pdf_bytes, filetype='pdf')
#         total_pages = doc.page_count
#         print("Total pages:", total_pages)
#         page_number = 1
#         all_data = []

#         for page in doc:
#             text = page.get_text()

#             cleaned_text = clean_page_text(text)

#             all_data.append({
#                 'page_number':page_number,
#                 'page_content':cleaned_text,
#                 'page_content_length':len(cleaned_text)
#             })

#             page_number += 1

#             if page_number >= total_pages:
#                 break

#     except Exception as e:
#         print(f"An error occurred while [extracting text from file {pdf_bytes}]: {e}")
#         return None,'', 0 

#     finally:
#         # 4. Close the document (good practice)
#         if 'doc' in locals() and doc:
#             doc.close()

#     return all_data, book_id, total_pages






# import fitz


# def extract_text_and_images(pdf_path):
#     doc = fitz.open(pdf_path)
#     texts = []
#     images = []
#     for i, page in enumerate(doc):
#         text = page.get_text("text")
#         if text.strip():
#             texts.append({"page": i, "content": text})
#         for img in page.get_images(full=True):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_bytes = base_image["image"]
#             images.append({"page": i, "image": image_bytes})
#     return texts, images

# texts, images = extract_text_and_images('')
# print(f":white_check_mark: Extracted {len(texts)} text chunks and {len(images)} images.")


import fitz
import os
import json

# def extract_text_and_images(pdf_path, output_dir="output"):
#     # Ensure output folder exists
#     os.makedirs(output_dir, exist_ok=True)
#     images_dir = os.path.join(output_dir, "images")
#     os.makedirs(images_dir, exist_ok=True)

#     doc = fitz.open(pdf_path)
#     texts = []
#     images = []

#     for i, page in enumerate(doc):
#         # --- Extract text ---
#         text = page.get_text("text")
#         if text.strip():
#             texts.append({"page": i + 1, "content": text})

#         # --- Extract images ---
#         for j, img in enumerate(page.get_images(full=True)):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_bytes = base_image["image"]
#             image_ext = base_image["ext"]
#             image_filename = f"page_{i+1}_img_{j+1}.{image_ext}"
#             image_path = os.path.join(images_dir, image_filename)
#             with open(image_path, "wb") as img_file:
#                 img_file.write(image_bytes)

#             images.append({
#                 "page": i + 1,
#                 "path": image_path,
#                 "ext": image_ext
#             })

#     # --- Save text data ---
#     text_output_path = os.path.join(output_dir, "texts.json")
#     with open(text_output_path, "w", encoding="utf-8") as f:
#         json.dump(texts, f, indent=4, ensure_ascii=False)

#     print(f"✅ Extracted {len(texts)} text chunks and {len(images)} images.")
#     print(f"📝 Text saved at: {text_output_path}")
#     print(f"🖼️ Images saved in: {images_dir}")

#     return texts, images


# # Example usage:
# pdf_path = "Topic REFRIGERATION.pdf"  # Replace with your actual PDF path
# texts, images = extract_text_and_images(pdf_path)






# along with images 




import fitz # PyMuPDF
import os
import json
import base64
from tqdm import tqdm # Assuming you use this for progress bars later

# --- Helper Function: Convert Image to Base64 Data URI ---
def image_to_base64(image_path):
    """Converts a local image file to a Base64 data URI string."""
    try:
        with open(image_path, "rb") as image_file:
            # Determine mime type from extension
            ext = image_path.split('.')[-1].lower()
            # Handle common variations
            mime_type = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"
            
            # Encode and return with data URI prefix
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

# --- Main Extraction Function (Modified) ---
def extract_text_and_images(pdf_path, output_dir="output"):
    # Ensure output folders exist
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return [], []

    texts = []
    all_images_meta = []
    
    # Dictionary to hold Base64 images grouped by page for easy linking
    page_images_map = {} 

    for i, page in enumerate(doc):
        page_num = i + 1
        
        # 1. --- Image Extraction and Base64 Conversion ---
        current_page_images = []
        for j, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{page_num}_img_{j+1}.{image_ext}"
            image_path = os.path.join(images_dir, image_filename)
            
            # Save the file to disk first
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            # Get the Base64 data URI from the saved file
            base64_data_uri = image_to_base64(image_path)

            image_metadata = {
                "page": page_num,
                "path": image_path,
                "ext": image_ext,
                "base64_uri": base64_data_uri # <-- Store the Base64 URI here
            }
            current_page_images.append(image_metadata)
            all_images_meta.append(image_metadata)

        page_images_map[page_num] = current_page_images
        
        # 2. --- Text Extraction and Linking ---
        text = page.get_text("text")
        if text.strip():
            # Find the relevant images for this page/text chunk
            linked_images = page_images_map.get(page_num, [])
            
            # Remove the local 'path' key before storing in the final data structure 
            # (only keep 'base64_uri' for web display)
            web_ready_images = [
                {"base64_uri": img['base64_uri'], "alt_text": f"Image from page {page_num}"}
                for img in linked_images
            ]
            
            texts.append({
                "page": page_num, 
                "content": text,
                "images": web_ready_images # <-- This links the image data directly
            })

    # --- Save text data ---
    text_output_path = os.path.join(output_dir, "texts_with_images.json")
    with open(text_output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=4, ensure_ascii=False)

    print(f"✅ Extracted {len(texts)} text chunks and {len(all_images_meta)} images.")
    print(f"📝 Text (with linked images) saved at: {text_output_path}")
    print(f"🖼️ Images saved in: {images_dir}")

    return texts, all_images_meta