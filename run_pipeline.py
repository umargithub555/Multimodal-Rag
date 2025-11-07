# import os
# import json
# from text_extractor import extract_text_and_images
# from semantic_chapters import make_semantic_chapters
# from embeddings import store_embeddings  # You'll need to create this from your embedding code

# def main():
#     pdf_path = "Topic REFRIGERATION.pdf"
    
#     print("🚀 Starting Multimodal RAG Pipeline...")
#     print("=" * 50)
    
#     # Step 1: Extract text and images
#     print("📄 Step 1: Extracting text and images from PDF...")
#     texts, images = extract_text_and_images(pdf_path)
#     print("✅ Extraction complete!")
    
#     # Step 2: Create semantic chapters
#     print("\n📘 Step 2: Creating semantic chapters...")
#     semantic_chapters = make_semantic_chapters(texts, threshold=0.75)
    
#     # Save chapters
#     with open("output/semantic_chapters.json", "w", encoding="utf-8") as f:
#         json.dump(semantic_chapters, f, indent=4, ensure_ascii=False)
#     print("✅ Semantic chapters saved!")
    
#     # Step 3: Store embeddings in ChromaDB
#     print("\n💾 Step 3: Storing embeddings in ChromaDB...")
#     store_embeddings(semantic_chapters)  # You need to wrap your embedding code in this function
#     print("✅ Embeddings stored!")
    
#     print("\n🎉 Pipeline complete! You can now run multimodal_rag.py")
#     print("📁 Files created:")
#     print("   - output/texts_with_images.json")
#     print("   - output/semantic_chapters.json") 
#     print("   - chroma_db/ (with embeddings)")
#     print("   - output/images/ (extracted images)")

# if __name__ == "__main__":
#     main()




import json
import os
import io
import base64
from tqdm import tqdm
import numpy as np
import torch
from typing import List
import fitz  # PyMuPDF
from PIL import Image

# Model imports
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
TEXT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHROMA_DB_PATH = "./chroma_db"

print(f"Using device: {DEVICE}")

# --- 1. Text Extraction Functions ---
def image_to_base64(image_path):
    """Converts a local image file to a Base64 data URI string."""
    try:
        with open(image_path, "rb") as image_file:
            ext = image_path.split('.')[-1].lower()
            mime_type = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def extract_text_and_images(pdf_path, output_dir="output"):
    """Extract text and images from PDF"""
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
    page_images_map = {}

    for i, page in enumerate(doc):
        page_num = i + 1
        
        # Extract images
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

            # Get the Base64 data URI
            base64_data_uri = image_to_base64(image_path)

            image_metadata = {
                "page": page_num,
                "path": image_path,
                "ext": image_ext,
                "base64_uri": base64_data_uri
            }
            current_page_images.append(image_metadata)
            all_images_meta.append(image_metadata)

        page_images_map[page_num] = current_page_images
        
        # Extract text
        text = page.get_text("text")
        if text.strip():
            linked_images = page_images_map.get(page_num, [])
            
            web_ready_images = [
                {"base64_uri": img['base64_uri'], "alt_text": f"Image from page {page_num}"}
                for img in linked_images
            ]
            
            texts.append({
                "page": page_num, 
                "content": text,
                "images": web_ready_images
            })

    # Save text data
    text_output_path = os.path.join(output_dir, "texts_with_images.json")
    with open(text_output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=4, ensure_ascii=False)

    print(f"✅ Extracted {len(texts)} text chunks and {len(all_images_meta)} images.")
    return texts, all_images_meta

# --- 2. Semantic Chapter Creation ---
def clean_page_text(text):
    """Basic text cleaning"""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_embeddings(texts, model_name="nomic-ai/nomic-embed-text-v1"):
    """Generate embeddings for texts"""
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
    """Detect chapter boundaries based on cosine similarity"""
    from sklearn.metrics.pairwise import cosine_similarity
    boundaries = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        if sim < threshold:
            boundaries.append(i + 1)
    return boundaries

def group_into_chapters(extracted_chunks, boundaries):
    """Group text chunks into semantic chapters"""
    import uuid
    chapters = []
    start = 0
    chapter_num = 1
    
    for b in boundaries + [len(extracted_chunks)]:
        chapter_chunks = extracted_chunks[start:b]
        
        chapter_text = "\n".join([c["content"] for c in chapter_chunks])
        
        # Collect all images from this chapter
        chapter_images = []
        for chunk in chapter_chunks:
            chapter_images.extend(chunk.get("images", []))
        
        if chapter_chunks:  # Only add if there are chunks
            chapters.append({
                "chapter_number": chapter_num,
                "chapter_id": str(uuid.uuid4())[:8],
                "start_page": chapter_chunks[0]["page"],
                "end_page": chapter_chunks[-1]["page"],
                "chapter_text_length": len(chapter_text),
                "chapter_text": clean_page_text(chapter_text),
                "chapter_images": chapter_images
            })
            chapter_num += 1
        start = b

    return chapters

def make_semantic_chapters(extracted_texts, threshold=0.70):
    """Create semantic chapters from extracted texts"""
    print("📘 Creating semantic chapters...")
    texts_for_embedding = [t["content"] for t in extracted_texts]
    
    embeddings = get_embeddings(texts_for_embedding)
    boundaries = detect_boundaries(embeddings, threshold)
    chapters = group_into_chapters(extracted_texts, boundaries)

    print(f"✅ Created {len(chapters)} semantic chapters.")
    return chapters

# --- 3. Embedding Storage ---
class EmbeddingStorage:
    def __init__(self):
        self.text_model = SentenceTransformer(TEXT_MODEL_NAME, trust_remote_code=True).to(DEVICE)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Custom embedding function
        class NomicTextEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, text_model):
                self.text_model = text_model
                
            def __call__(self, texts: List[str]) -> List[List[float]]:
                return self.text_model.encode(texts).tolist()
        
        self.nomic_ef = NomicTextEmbeddingFunction(self.text_model)
    
    def store_text_embeddings(self, chapters):
        """Store text embeddings in ChromaDB"""
        # Delete existing collection to avoid dimension issues
        try:
            self.chroma_client.delete_collection("pdf_chapter_embeddings")
        except:
            pass
            
        text_collection = self.chroma_client.get_or_create_collection(
            name="pdf_chapter_embeddings",
            embedding_function=self.nomic_ef
        )
        
        def split_text(text, max_length=5000):
            return [text[i:i + max_length] for i in range(0, len(text), max_length)]
        
        print("📝 Storing text embeddings...")
        for i, ch in enumerate(tqdm(chapters, desc="Embedding chapters")):
            text = ch.get("chapter_text", "").strip()
            if not text:
                continue

            total_length = ch.get("chapter_text_length", len(text))
            if total_length > 5000:
                chunks = split_text(text)
                for j, chunk in enumerate(chunks):
                    emb = self.text_model.encode(chunk).tolist()
                    text_collection.add(
                        documents=[chunk],
                        embeddings=[emb],
                        metadatas=[{
                            "chapter_id": ch["chapter_id"],
                            "chunk_index": j,
                            "start_page": ch["start_page"],
                            "end_page": ch["end_page"],
                            "chapter_title": f"Chapter {ch['chapter_number']}"
                        }],
                        ids=[f"chapter_{ch['chapter_id']}_chunk_{j}"]
                    )
            else:
                emb = self.text_model.encode(text).tolist()
                text_collection.add(
                    documents=[text],
                    embeddings=[emb],
                    metadatas=[{
                        "chapter_id": ch["chapter_id"],
                        "start_page": ch["start_page"],
                        "end_page": ch["end_page"],
                        "chapter_text_length": total_length,
                        "chapter_title": f"Chapter {ch['chapter_number']}"
                    }],
                    ids=[f"chapter_{ch['chapter_id']}"]
                )
        
        print(f"✅ Text embeddings stored! Total: {text_collection.count()} documents")
    
    def store_image_embeddings(self, chapters):
        """Store image embeddings using CLIP"""
        # Delete existing collection
        try:
            self.chroma_client.delete_collection("pdf_image_embeddings")
        except:
            pass
            
        image_collection = self.chroma_client.get_or_create_collection(
            name="pdf_image_embeddings"
        )
        
        def base64_uri_to_tensor(base64_uri):
            try:
                _, encoded_data = base64_uri.split(',', 1)
                image_bytes = base64.b64decode(encoded_data)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                return image
            except Exception:
                return None
        
        image_embeddings = []
        image_metadatas = []
        image_ids = []
        total_images = 0
        
        print("🖼️ Storing image embeddings...")
        for i, ch in enumerate(tqdm(chapters, desc="Embedding images")):
            chapter_id = ch["chapter_id"]
            
            for j, img_data in enumerate(ch.get("chapter_images", [])):
                base64_uri = img_data.get("base64_uri")
                if not base64_uri:
                    continue
                
                total_images += 1
                image = base64_uri_to_tensor(base64_uri)
                if image is None:
                    continue
                
                try:
                    inputs = self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        image_features = self.clip_model.get_image_features(**inputs).cpu().numpy()[0]
                    
                    image_embeddings.append(image_features.tolist())
                    image_metadatas.append({
                        "chapter_id": chapter_id,
                        "image_index": j,
                        "start_page": ch["start_page"],
                        "end_page": ch["end_page"],
                        "base64_uri": base64_uri,
                        "chapter_title": f"Chapter {ch['chapter_number']}"
                    })
                    image_ids.append(f"img_{chapter_id}_{j}")
                except Exception as e:
                    print(f"Error embedding image {chapter_id}_{j}: {e}")
        
        if image_embeddings:
            image_collection.add(
                embeddings=image_embeddings,
                metadatas=image_metadatas,
                ids=image_ids
            )
        
        print(f"✅ Image embeddings stored! Total: {len(image_embeddings)} images")

# --- Main Pipeline ---
def main():
    pdf_path = "Topic REFRIGERATION.pdf"  # Change this to your PDF path
    
    print("🚀 Starting Multimodal RAG Pipeline...")
    print("=" * 50)
    
    # Step 1: Extract text and images
    print("📄 Step 1: Extracting text and images from PDF...")
    texts, images = extract_text_and_images(pdf_path)
    print("✅ Extraction complete!")
    
    # Step 2: Create semantic chapters
    print("\n📘 Step 2: Creating semantic chapters...")
    semantic_chapters = make_semantic_chapters(texts, threshold=0.75)
    
    # Save chapters
    with open("output/semantic_chapters.json", "w", encoding="utf-8") as f:
        json.dump(semantic_chapters, f, indent=4, ensure_ascii=False)
    print("✅ Semantic chapters saved!")
    
    # Step 3: Store embeddings in ChromaDB
    print("\n💾 Step 3: Storing embeddings in ChromaDB...")
    storage = EmbeddingStorage()
    storage.store_text_embeddings(semantic_chapters)
    storage.store_image_embeddings(semantic_chapters)
    print("✅ Embeddings stored!")
    
    print("\n🎉 Pipeline complete! You can now run the RAG system.")
    print("📁 Files created:")
    print("   - output/texts_with_images.json")
    print("   - output/semantic_chapters.json") 
    print("   - chroma_db/ (with embeddings)")
    print("   - output/images/ (extracted images)")

if __name__ == "__main__":
    main()