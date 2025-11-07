from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
from text_extractor import extract_text_and_images
from text_cleaning import clean_page_text




def get_embeddings(texts, model_name="nomic-ai/nomic-embed-text-v1"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    embeddings = []
    for text in tqdm(texts, desc="🔹 Generating embeddings"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=8192
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
    return np.array(embeddings)


def detect_boundaries(embeddings, threshold=0.70):
    boundaries = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        if sim < threshold:
            boundaries.append(i + 1)
    return boundaries


import uuid

# def group_into_chapters(texts, boundaries):
#     chapters = []
#     start = 0
#     chapter_num = 1  # 👈 Track chapter number

#     for b in boundaries + [len(texts)]:
#         chapter_text = "\n".join(texts[start:b])
#         chapters.append({
#             "chapter_number": chapter_num,  # 👈 New field
#             "chapter_id": str(uuid.uuid4())[:8],
#             "start_page": start + 1,
#             "end_page": b,
#             "chapter_text_length": len(chapter_text),
#             "chapter_text": clean_page_text(chapter_text)
#         })
#         start = b
#         chapter_num += 1  # Increment for next chapter

#     return chapters


 # Ensure this import is available

def group_into_chapters(extracted_chunks, boundaries): # <-- Renamed 'texts' to 'extracted_chunks' for clarity
    chapters = []
    start = 0
    chapter_num = 1
    
    for b in boundaries + [len(extracted_chunks)]:
        # Slice the list of dictionaries for the current chapter
        chapter_chunks = extracted_chunks[start:b]
        
        # Aggregate text content and image metadata
        chapter_text = "\n".join([c["content"] for c in chapter_chunks])
        
        # Collect all unique images from all chunks in this chapter
        chapter_images = []
        for chunk in chapter_chunks:
            # Safely extend the image list (each chunk's images are already Base64)
            chapter_images.extend(chunk.get("images", [])) 
        
        chapters.append({
            "chapter_number": chapter_num,
            "chapter_id": str(uuid.uuid4())[:8],
            "start_page": chapter_chunks[0]["page"] if chapter_chunks else 0, # Get starting page
            "end_page": chapter_chunks[-1]["page"] if chapter_chunks else 0,   # Get ending page
            "chapter_text_length": len(chapter_text),
            "chapter_text": clean_page_text(chapter_text),
            # NEW FIELD: Include the collected images 🖼️
            "chapter_images": chapter_images 
        })
        start = b
        chapter_num += 1

    return chapters







# def make_semantic_chapters(extracted_texts, threshold=0.70):
#     print("📘 Creating semantic chapters...")

#     texts = [t["content"] for t in extracted_texts]

#     embeddings = get_embeddings(texts)
#     boundaries = detect_boundaries(embeddings, threshold)
#     chapters = group_into_chapters(texts, boundaries)

#     print(f"✅ Created {len(chapters)} semantic chapters.")
#     return chapters



def make_semantic_chapters(extracted_texts, threshold=0.70):
    print("📘 Creating semantic chapters...")

    # CHANGE 1: Create a list of ONLY the text content for embedding/boundary detection
    texts_for_embedding = [t["content"] for t in extracted_texts]

    embeddings = get_embeddings(texts_for_embedding)
    boundaries = detect_boundaries(embeddings, threshold)
    
    # CHANGE 2: Pass the original, rich dictionaries (extracted_texts) 
    # to the grouping function, not just the plain text.
    chapters = group_into_chapters(extracted_texts, boundaries) # <-- Key change here

    print(f"✅ Created {len(chapters)} semantic chapters.")
    return chapters


pdf_path = "Topic REFRIGERATION.pdf"
texts, images = extract_text_and_images(pdf_path)

semantic_chapters = make_semantic_chapters(texts, threshold=0.75)

# Save chapters
with open("output/semantic_chapters.json", "w", encoding="utf-8") as f:
    json.dump(semantic_chapters, f, indent=4, ensure_ascii=False)

print("📁 Semantic chapters saved to output/semantic_chapters.json")
