# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.utils import embedding_functions # <-- NEW IMPORT
# import json
# from tqdm import tqdm
# from typing import List



# # --- Load chapters ---
# with open("output/semantic_chapters.json", "r", encoding="utf-8") as file:
#     chapters = json.load(file)

# # --- Load model ---
# model_name = "nomic-ai/nomic-embed-text-v1"
# class NomicTextEmbeddingFunction(embedding_functions.EmbeddingFunction):
#     def __call__(self, texts: List[str]) -> List[List[float]]:
#         # text_model must be defined globally or passed in
#         return text_model.encode(texts).tolist()

# # ... (Loading the SentenceTransformer model)
# text_model = SentenceTransformer(model_name, trust_remote_code=True)
# nomic_ef = NomicTextEmbeddingFunction() # <-- Initialize it

# # --- Persistent ChromaDB ---
# chroma_client = chromadb.PersistentClient(path="./chroma_db") # path fix from before

# collection_name = "pdf_chapter_embeddings"

# # CRITICAL FIX: Pass the custom embedding function here!
# text_collection = chroma_client.get_or_create_collection(
#     name=collection_name, 
#     embedding_function=nomic_ef # <-- PASS THE EF!
# )
# print(f"📦 Using Chroma collection: {collection_name}")

# # ... (The rest of your code remains the same)

# # --- Split text helper ---
# def split_text(text, max_length=5000):
#     return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# # --- Embed and store ---
# for i, ch in tqdm(enumerate(chapters), total=len(chapters), desc="📘 Embedding chapters"):
#     text = ch.get("chapter_text", "").strip()
#     if not text:
#         continue

#     total_length = ch.get("chapter_text_length", len(text))
#     if total_length > 5000:
#         chunks = split_text(text)
#         for j, chunk in enumerate(chunks):
#             emb = text_model.encode(chunk).tolist()
#             text_collection.add(
#                 documents=[chunk],
#                 embeddings=[emb],
#                 metadatas=[{
#                     "chapter_id": ch["chapter_id"],
#                     "chunk_index": j,
#                     "start_page": ch["start_page"],
#                     "end_page": ch["end_page"]
#                 }],
#                 ids=[f"chapter_{i}_chunk_{j}"]
#             )
#     else:
#         emb = text_model.encode(text).tolist()
#         text_collection.add(
#             documents=[text],
#             embeddings=[emb],
#             metadatas=[{
#                 "chapter_id": ch["chapter_id"],
#                 "start_page": ch["start_page"],
#                 "end_page": ch["end_page"],
#                 "chapter_text_length": total_length
#             }],
#             ids=[f"chapter_{i}"]
#         )

# # chroma_client.persist()  # ✅ Save permanently
# print("✅ All chapter embeddings stored successfully!")
# print("📄 Total documents in collection:", text_collection.count())





import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from typing import List
import torch
from PIL import Image
import io
import base64
from transformers import CLIPProcessor, CLIPModel

# Configuration
TEXT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EmbeddingStorage:
    def __init__(self):
        self.text_model = SentenceTransformer(TEXT_MODEL_NAME, trust_remote_code=True).to(DEVICE)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Custom embedding function
        class NomicTextEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __call__(self, texts: List[str]) -> List[List[float]]:
                return self.text_model.encode(texts).tolist()
        
        self.nomic_ef = NomicTextEmbeddingFunction()
    
    def store_text_embeddings(self, chapters):
        """Store text embeddings"""
        text_collection = self.chroma_client.get_or_create_collection(
            name="pdf_chapter_embeddings", 
            embedding_function=self.nomic_ef
        )
        
        def split_text(text, max_length=5000):
            return [text[i:i + max_length] for i in range(0, len(text), max_length)]
        
        for i, ch in tqdm(enumerate(chapters), total=len(chapters), desc="📘 Embedding chapters"):
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
                            "end_page": ch["end_page"]
                        }],
                        ids=[f"chapter_{i}_chunk_{j}"]
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
                        "chapter_text_length": total_length
                    }],
                    ids=[f"chapter_{i}"]
                )
        
        print(f"✅ Text embeddings stored! Total documents: {text_collection.count()}")
    
    def store_image_embeddings(self, chapters):
        """Store image embeddings using CLIP"""
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
        
        for i, ch in enumerate(chapters):
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
                        "base64_uri": base64_uri
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
        
        print(f"✅ Image embeddings stored! Total images: {total_images}")

def store_embeddings(chapters):
    """Main function to store both text and image embeddings"""
    storage = EmbeddingStorage()
    storage.store_text_embeddings(chapters)
    storage.store_image_embeddings(chapters)

if __name__ == "__main__":
    # Load chapters and store embeddings
    with open("output/semantic_chapters.json", "r", encoding="utf-8") as file:
        chapters = json.load(file)
    store_embeddings(chapters)