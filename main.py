import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import Dict, List, Any
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel

load_dotenv()

# --- Configuration Constants ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_chapter_embeddings"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
CHAPTERS_FILE = "output/semantic_chapters.json"
GEMINI_MODEL = "gemini-2.0-flash-exp"

# --- Global Variables ---
text_model = None
nomic_ef = None
chroma_client = None
collection = None
chapters = []

# --- Request Models ---
class QuestionRequest(BaseModel):
    question: str

class QuizGenerateRequest(BaseModel):
    chapter: str

class AnswerEvaluationRequest(BaseModel):
    question: str
    user_answer: str
    chapter_context: str


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_model, nomic_ef, chroma_client, collection, chapters
    
    print("🚀 Starting up application...")
    
    # Configure Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️ WARNING: GEMINI_API_KEY environment variable not set.")
    else:
        genai.configure(api_key=gemini_api_key)
        print("✅ Gemini API configured")
    
    # Initialize SentenceTransformer
    print(f"🔹 Loading embedding model: {MODEL_NAME}")
    text_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print("✅ Embedding model loaded")
    
    # Define custom Embedding Function
    class NomicTextEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, texts: List[str]) -> List[List[float]]:
            return text_model.encode(texts).tolist()
    
    nomic_ef = NomicTextEmbeddingFunction()
    
    # Initialize ChromaDB
    print(f"📦 Loading ChromaDB from: {CHROMA_DB_PATH}")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=nomic_ef
        )
        print(f"✅ Chroma Collection '{COLLECTION_NAME}' loaded with {collection.count()} documents.")
    except Exception as e:
        print(f"❌ Error loading ChromaDB: {e}")
        chroma_client = None
        collection = None
    
    # Load Chapter Metadata
    try:
        with open(CHAPTERS_FILE, "r", encoding="utf-8") as f:
            chapters = json.load(f)
            print(f"📚 Loaded {len(chapters)} chapter metadata items.")
    except Exception as e:
        print(f"❌ Error loading chapters: {e}")
        chapters = []
    
    print("✅ Application startup complete!\n")
    yield
    print("🔄 Shutting down application...")


# --- Setup FastAPI ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- Helper: Query Gemini ---
def ask_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error querying Gemini: {e}"


# --- Route: Serve Home Page ---
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found in static folder</h1>"


# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Route: Ask Question (RAG) ---
@app.post("/api/learning/ask")
async def ask_question(req: QuestionRequest) -> Dict[str, str]:
    if not collection:
        return {"answer": "Error: ChromaDB not initialized. Please check server logs."}
    
    question = req.question
    if not question:
        return {"answer": "Please provide a question."}

    try:
        q_emb = nomic_ef([question])[0]
        results = collection.query(query_embeddings=[q_emb], n_results=5)

        if not results["documents"] or not results["documents"][0]:
            return {"answer": "No relevant information found in the database."}
        
        context = "\n---\n".join(results["documents"][0])
        
#         prompt = f"""You are a helpful tutor. Use the following book context to answer the question.
                
# Context:
# {context}

# Question:
# {question}

# Provide a clear and factual answer based on the context provided. If the context doesn't contain relevant information, say so."""
        prompt = f"""
        You are a knowledgeable tutor. Use only the information from the following text to answer the question.
        Do not mention that the information came from the text, context, or any source.
        If the question cannot be answered from the provided information, politely respond that the information is not available in the study material.

        Text:
        {context}

        Question:
        {question}

        Instructions for your answer:
        - Respond directly and naturally, as if you already know the material.
        - Do not mention or reference the text, source, or context.
        - If the answer cannot be found, say something like:
        "I'm sorry, but the provided material doesn't include information about that topic."
        """

        answer = ask_gemini(prompt)
        return {"answer": answer}
            
    except Exception as e:
        print(f"Error in ask_question route: {e}")
        return {"answer": f"An error occurred during processing: {str(e)}"}


# --- Route: Generate Quiz (Dynamic Questions Only) ---
@app.post("/api/quiz/generate")
async def generate_quiz(req: QuizGenerateRequest) -> Dict[str, Any]:
    if not chapters:
        return {"error": "Chapter data not loaded. Please check server logs."}
    
    chapter_value = req.chapter
    
    # Extract chapter number
    try:
        ch_num = int(chapter_value.replace("chapter", ""))
    except ValueError:
        return {"error": "Invalid chapter format. Expected 'chapterN' (e.g., 'chapter1')."}

    # Find matching chapter
    chapter_data = next((ch for ch in chapters if ch["chapter_number"] == ch_num), None)
    if not chapter_data:
        return {"error": f"Chapter {ch_num} not found in the database."}

    chapter_text = chapter_data["chapter_text"]
    
    # Generate questions using Gemini
    prompt = f"""Based on the following chapter content, generate exactly 5 open-ended questions that test understanding of key concepts.

Requirements:
- Questions should be clear and specific
- Questions should require explanatory answers (not yes/no)
- Cover different topics from the chapter
- Return ONLY a JSON array of questions in this exact format:
[
  {{"question": "Question text here"}},
  {{"question": "Question text here"}},
  ...
]

Chapter Content:
{chapter_text[:8000]}

Return only the JSON array, no additional text."""
    
    try:
        response = ask_gemini(prompt)
        
        # Parse JSON response
        # Remove markdown code blocks if present
        response = response.replace("```json", "").replace("```", "").strip()
        questions = json.loads(response)
        
        return {
            "questions": questions,
            "chapter_number": ch_num,
            "chapter_context": chapter_text[:8000]  # Send context for evaluation
        }
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response}")
        return {"error": "Failed to parse questions from AI response"}
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return {"error": f"Error generating quiz: {str(e)}"}


# --- Route: Evaluate Answer ---
@app.post("/api/quiz/evaluate")
async def evaluate_answer(req: AnswerEvaluationRequest) -> Dict[str, Any]:
    question = req.question
    user_answer = req.user_answer
    chapter_context = req.chapter_context
    
    if not user_answer.strip():
        return {"error": "Please provide an answer"}
    
    prompt = f"""You are an expert tutor evaluating a student's answer.

Context from the chapter:
{chapter_context}

Question:
{question}

Student's Answer:
{user_answer}

Please evaluate the answer and provide:
1. An accuracy score (0-100%)
2. Detailed feedback on what was correct and what was missing
3. An improved/model answer

Return your response in this EXACT JSON format:
{{
  "score": <number between 0-100>,
  "feedback": "<detailed feedback text>",
  "improved_answer": "<model answer text>"
}}

Return only the JSON object, no additional text."""
    
    try:
        response = ask_gemini(prompt)
        
        # Clean response
        response = response.replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(response)
        
        return evaluation
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response}")
        return {"error": "Failed to parse evaluation from AI response"}
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return {"error": f"Error evaluating answer: {str(e)}"}


# --- Route: Health Check ---
@app.get("/health")
def health_check():
    return {
        'status': 'Ok',
        'chroma_loaded': collection is not None,
        'chroma_count': collection.count() if collection else 0,
        'chapters_loaded': len(chapters),
        'model_loaded': text_model is not None
    }


# --- Route: Get Available Chapters ---
@app.get("/api/chapters")
def get_chapters():
    if not chapters:
        return {"chapters": []}
    
    chapter_list = [
        {
            "number": ch["chapter_number"],
            "title": ch.get("title", f"Chapter {ch['chapter_number']}"),
            "value": f"chapter{ch['chapter_number']}"
        }
        for ch in chapters
    ]
    return {"chapters": chapter_list}








# import os
# import json
# import io
# import base64
# import torch
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# import chromadb
# from chromadb.utils import embedding_functions
# from sentence_transformers import SentenceTransformer
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import google.generativeai as genai
# from google.generativeai.types import Part
# from typing import Dict, List, Any
# from dotenv import load_dotenv
# from contextlib import asynccontextmanager
# from pydantic import BaseModel

# load_dotenv()

# # --- Configuration Constants ---
# CHROMA_DB_PATH = "./chroma_db"
# TEXT_COLLECTION_NAME = "pdf_chapter_embeddings"
# IMAGE_COLLECTION_NAME = "pdf_image_embeddings"
# TEXT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
# CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CHAPTERS_FILE = "output/semantic_chapters.json"
# GEMINI_MODEL = "gemini-2.0-pro" # Optimized for fast, multimodal RAG

# # --- Global Variables ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# text_model = None
# nomic_ef = None
# clip_model = None
# clip_processor = None
# chroma_client = None
# text_collection = None
# image_collection = None
# chapters = []

# # --- Request Models ---
# class QuestionRequest(BaseModel):
#     question: str

# class QuizGenerateRequest(BaseModel):
#     chapter: str

# class AnswerEvaluationRequest(BaseModel):
#     question: str
#     user_answer: str
#     chapter_context: str

# # --- Helper: Decode Base64 URI to Bytes for Gemini Part ---
# def base64_uri_to_bytes_and_mime(base64_uri: str) -> tuple[bytes, str] | tuple[None, None]:
#     """Decodes a Base64 data URI string into raw bytes and MIME type."""
#     try:
#         if ';' in base64_uri:
#             metadata, encoded_data = base64_uri.split(',', 1)
#             mime_type = metadata.split(':')[1].split(';')[0]
#         else:
#             # Assume common image type if metadata is missing (e.g., png)
#             encoded_data = base64_uri
#             mime_type = 'image/png'
        
#         image_bytes = base64.b64decode(encoded_data)
#         return image_bytes, mime_type
#     except Exception:
#         return None, None

# # --- Helper: Query Gemini Multimodal ---
# def ask_gemini_multimodal(content_parts: List[Any]) -> str:
#     """Queries the Gemini model with a list of text and image parts."""
#     try:
#         model = genai.GenerativeModel(GEMINI_MODEL)
#         response = model.generate_content(content_parts)
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error querying Gemini: {e}")
#         return f"Error querying Gemini: {e}"

# # --- Lifespan Context Manager ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global text_model, nomic_ef, clip_model, clip_processor, chroma_client, text_collection, image_collection, chapters
    
#     print("🚀 Starting up multimodal RAG application...")
    
#     # Configure Gemini
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     if not gemini_api_key:
#         print("⚠️ WARNING: GEMINI_API_KEY environment variable not set.")
#     else:
#         genai.configure(api_key=gemini_api_key)
#         print("✅ Gemini API configured")
    
#     # Initialize Text Embedding Model
#     print(f"🔹 Loading text model: {TEXT_MODEL_NAME} on {DEVICE}")
#     text_model = SentenceTransformer(TEXT_MODEL_NAME, trust_remote_code=True).to(DEVICE)
#     print("✅ Text model loaded")
    
#     # Initialize CLIP Multimodal Model
#     print(f"🖼️ Loading CLIP model: {CLIP_MODEL_NAME} on {DEVICE}")
#     clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
#     clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
#     print("✅ CLIP model loaded")

#     # Define custom Text Embedding Function for Chroma
#     class NomicTextEmbeddingFunction(embedding_functions.EmbeddingFunction):
#         def __call__(self, texts: List[str]) -> List[List[float]]:
#             return text_model.encode(texts).tolist()
    
#     nomic_ef = NomicTextEmbeddingFunction()
    
#     # Initialize ChromaDB
#     print(f"📦 Loading ChromaDB from: {CHROMA_DB_PATH}")
#     try:
#         chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
#         # Text Collection
#         text_collection = chroma_client.get_or_create_collection(
#             name=TEXT_COLLECTION_NAME,
#             embedding_function=nomic_ef
#         )
#         print(f"✅ Text Collection '{TEXT_COLLECTION_NAME}' loaded with {text_collection.count()} documents.")
        
#         # Image Collection (No EF needed, embeddings pre-computed)
#         image_collection = chroma_client.get_or_create_collection(name=IMAGE_COLLECTION_NAME)
#         print(f"✅ Image Collection '{IMAGE_COLLECTION_NAME}' loaded with {image_collection.count()} images.")
        
#     except Exception as e:
#         print(f"❌ Error loading ChromaDB collections: {e}")
#         text_collection = None
#         image_collection = None
    
#     # Load Chapter Metadata
#     try:
#         with open(CHAPTERS_FILE, "r", encoding="utf-8") as f:
#             chapters = json.load(f)
#             print(f"📚 Loaded {len(chapters)} chapter metadata items.")
#     except Exception as e:
#         print(f"❌ Error loading chapters: {e}")
#         chapters = []
    
#     print("✅ Application startup complete!\n")
#     yield
#     print("🔄 Shutting down application...")


# # --- Setup FastAPI ---
# app = FastAPI(lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )


# # --- Helper: Multimodal Search Logic ---
# def multimodal_rag_search(question: str, text_k: int = 5, image_k: int = 3) -> Dict[str, Any]:
#     """Searches both text and image collections and combines results."""
    
#     if not text_collection or not image_collection:
#         return {"context": "Database not available.", "image_parts": []}
    
#     # 1. --- Text Search (Nomic Embedding) ---
#     text_emb = nomic_ef([question])[0]
#     text_results = text_collection.query(
#         query_embeddings=[text_emb], 
#         n_results=text_k
#     )
    
#     context_chunks = text_results["documents"][0] if text_results.get("documents") else []
    
#     # 2. --- Image Search (CLIP Text Embedding) ---
#     # Generate CLIP text embedding for the question
#     inputs = clip_processor(text=[question], return_tensors="pt", padding=True).to(DEVICE)
#     with torch.no_grad():
#         clip_emb = clip_model.get_text_features(**inputs).cpu().numpy()[0]
    
#     image_results = image_collection.query(
#         query_embeddings=[clip_emb.tolist()], 
#         n_results=image_k
#     )
    
#     # 3. --- Combine and Process Results ---
#     image_parts = []
    
#     if image_results.get("metadatas") and image_results["metadatas"][0]:
#         unique_uris = set()
        
#         for metadata in image_results["metadatas"][0]:
#             uri = metadata.get("base64_uri")
#             if uri and uri not in unique_uris:
#                 unique_uris.add(uri)
                
#                 # Convert URI to Part object for Gemini
#                 image_bytes, mime_type = base64_uri_to_bytes_and_mime(uri)
#                 if image_bytes and mime_type:
#                     image_parts.append(Part.from_bytes(data=image_bytes, mime_type=mime_type))

#     # Construct combined text context
#     context = "\n---\n".join(context_chunks)
    
#     return {
#         "context": context,
#         "image_parts": image_parts,
#         "source_count": len(context_chunks) + len(image_parts)
#     }

# # --- Route: Serve Home Page ---
# @app.get("/", response_class=HTMLResponse)
# async def home():
#     try:
#         with open("static/index.html", "r", encoding="utf-8") as f:
#             return f.read()
#     except FileNotFoundError:
#         return "<h1>Error: index.html not found in static folder</h1>"


# # Mount Static Files
# app.mount("/static", StaticFiles(directory="static"), name="static")


# # --- Route: Ask Question (RAG) ---
# @app.post("/api/learning/ask")
# async def ask_question(req: QuestionRequest) -> Dict[str, str]:
#     if not text_collection or not image_collection:
#         return {"answer": "Error: Database not initialized. Please check server logs."}
    
#     question = req.question
#     if not question:
#         return {"answer": "Please provide a question."}

#     try:
#         # Perform Multimodal Search
#         search_results = multimodal_rag_search(question, text_k=5, image_k=3)
#         context = search_results["context"]
#         image_parts = search_results["image_parts"]
#         source_count = search_results["source_count"]

#         if not context and not image_parts:
#             return {"answer": "No relevant information (text or images) found in the study material."}

#         # 1. Build the Text Prompt
#         text_prompt = f"""
#         You are a knowledgeable tutor. Use ONLY the information from the provided text and images to answer the question.
        
#         Do not mention that the information came from a source.
#         If the question can be answered better using the images, prioritize the visual information.
#         If the question cannot be answered from the provided information, politely respond that the information is not available in the study material.

#         Text Context:
#         {context}

#         Question:
#         {question}

#         Instructions for your answer:
#         - Respond directly and naturally, as if you already know the material.
#         - Do not mention or reference the text, source, or context.
#         - If the answer cannot be found, say something like:
#         "I'm sorry, but the provided material doesn't include information about that topic."
#         """

#         # 2. Build the Content List for Gemini (Text Prompt + Images)
#         content_parts = [text_prompt] + image_parts
        
#         # 3. Query Gemini Multimodally
#         answer = ask_gemini_multimodal(content_parts)
        
#         return {
#             "answer": answer,
#             "sources_used": source_count
#         }
            
#     except Exception as e:
#         print(f"Error in ask_question route: {e}")
#         return {"answer": f"An error occurred during processing: {str(e)}"}


# # --- Route: Generate Quiz (Dynamic Questions Only) ---
# @app.post("/api/quiz/generate")
# async def generate_quiz(req: QuizGenerateRequest) -> Dict[str, Any]:
#     if not chapters:
#         return {"error": "Chapter data not loaded. Please check server logs."}
    
#     chapter_value = req.chapter
    
#     # Extract chapter number
#     try:
#         ch_num = int(chapter_value.replace("chapter", ""))
#     except ValueError:
#         return {"error": "Invalid chapter format. Expected 'chapterN' (e.g., 'chapter1')."}

#     # Find matching chapter
#     chapter_data = next((ch for ch in chapters if ch["chapter_number"] == ch_num), None)
#     if not chapter_data:
#         return {"error": f"Chapter {ch_num} not found in the database."}

#     chapter_text = chapter_data["chapter_text"]
    
#     # Use the first image from the chapter as a visual grounding element (if available)
#     chapter_image_parts = []
#     if chapter_data.get("chapter_images"):
#         first_uri = chapter_data["chapter_images"][0].get("base64_uri")
#         if first_uri:
#             image_bytes, mime_type = base64_uri_to_bytes_and_mime(first_uri)
#             if image_bytes and mime_type:
#                 chapter_image_parts.append(Part.from_bytes(data=image_bytes, mime_type=mime_type))

#     # Generate questions using Gemini
#     prompt = f"""Based on the following chapter content and any provided image, generate exactly 5 open-ended questions that test understanding of key concepts.

# Requirements:
# - Questions should be clear and specific
# - Questions should require explanatory answers (not yes/no)
# - Cover different topics from the chapter, including the visual information in the image if one is provided.
# - Return ONLY a JSON array of questions in this exact format:
# [
#   {{"question": "Question text here"}},
#   ...
# ]

# Chapter Content (Up to 8000 chars):
# {chapter_text[:8000]}

# Return only the JSON array, no additional text."""
    
#     try:
#         # Build multimodal content for the quiz generation prompt
#         content_parts = chapter_image_parts + [prompt]
#         response = ask_gemini_multimodal(content_parts)
        
#         # Parse JSON response
#         response = response.replace("```json", "").replace("```", "").strip()
#         questions = json.loads(response)
        
#         return {
#             "questions": questions,
#             "chapter_number": ch_num,
#             "chapter_context": chapter_text[:8000] # Send context for evaluation
#         }
#     except json.JSONDecodeError as e:
#         print(f"JSON parsing error: {e}")
#         print(f"Response was: {response}")
#         return {"error": "Failed to parse questions from AI response"}
#     except Exception as e:
#         print(f"Error generating quiz: {e}")
#         return {"error": f"Error generating quiz: {str(e)}"}


# # --- Route: Evaluate Answer ---
# @app.post("/api/quiz/evaluate")
# async def evaluate_answer(req: AnswerEvaluationRequest) -> Dict[str, Any]:
#     # This remains text-only for simplicity, as image grounding is complex for structured evaluation
#     question = req.question
#     user_answer = req.user_answer
#     chapter_context = req.chapter_context
    
#     if not user_answer.strip():
#         return {"error": "Please provide an answer"}
    
#     prompt = f"""You are an expert tutor evaluating a student's answer.

# Context from the chapter:
# {chapter_context}

# Question:
# {question}

# Student's Answer:
# {user_answer}

# Please evaluate the answer and provide:
# 1. An accuracy score (0-100%)
# 2. Detailed feedback on what was correct and what was missing
# 3. An improved/model answer

# Return your response in this EXACT JSON format:
# {{
#   "score": <number between 0-100>,
#   "feedback": "<detailed feedback text>",
#   "improved_answer": "<model answer text>"
# }}

# Return only the JSON object, no additional text."""
    
#     try:
#         response = ask_gemini_multimodal([prompt])
        
#         # Clean response
#         response = response.replace("```json", "").replace("```", "").strip()
#         evaluation = json.loads(response)
        
#         return evaluation
#     except json.JSONDecodeError as e:
#         print(f"JSON parsing error: {e}")
#         print(f"Response was: {response}")
#         return {"error": "Failed to parse evaluation from AI response"}
#     except Exception as e:
#         print(f"Error evaluating answer: {e}")
#         return {"error": f"Error evaluating answer: {str(e)}"}


# # --- Route: Health Check ---
# @app.get("/health")
# def health_check():
#     return {
#         'status': 'Ok',
#         'text_chroma_loaded': text_collection is not None,
#         'image_chroma_loaded': image_collection is not None,
#         'text_chroma_count': text_collection.count() if text_collection else 0,
#         'image_chroma_count': image_collection.count() if image_collection else 0,
#         'chapters_loaded': len(chapters),
#         'multimodal_models_loaded': text_model is not None and clip_model is not None
#     }


# # --- Route: Get Available Chapters ---
# @app.get("/api/chapters")
# def get_chapters():
#     if not chapters:
#         return {"chapters": []}
    
#     chapter_list = [
#         {
#             "number": ch["chapter_number"],
#             # Title extraction is often better done during the semantic grouping phase
#             "title": ch.get("title", f"Chapter {ch['chapter_number']}"),
#             "value": f"chapter{ch['chapter_number']}"
#         }
#         for ch in chapters
#     ]
#     return {"chapters": chapter_list}






# import os
# import json
# import io
# import base64
# import torch
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# import chromadb
# from chromadb.utils import embedding_functions
# from sentence_transformers import SentenceTransformer
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import google.generativeai as genai
# from typing import Dict, List, Any
# from dotenv import load_dotenv
# from contextlib import asynccontextmanager
# from pydantic import BaseModel

# load_dotenv()

# # --- Configuration Constants ---
# CHROMA_DB_PATH = "./chroma_db"
# TEXT_COLLECTION_NAME = "pdf_chapter_embeddings"
# IMAGE_COLLECTION_NAME = "pdf_image_embeddings"
# TEXT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
# CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CHAPTERS_FILE = "output/semantic_chapters.json"
# GEMINI_MODEL = "gemini-2.0-pro"  # Optimized for multimodal RAG

# # --- Global Variables ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# text_model = None
# nomic_ef = None
# clip_model = None
# clip_processor = None
# chroma_client = None
# text_collection = None
# image_collection = None
# chapters = []

# # --- Request Models ---
# class QuestionRequest(BaseModel):
#     question: str

# class QuizGenerateRequest(BaseModel):
#     chapter: str

# class AnswerEvaluationRequest(BaseModel):
#     question: str
#     user_answer: str
#     chapter_context: str

# class AskResponse(BaseModel):
#     answer: str
#     sources_used: str

# # --- Helper: Decode Base64 URI to Bytes for Gemini ---
# def base64_uri_to_bytes_and_mime(base64_uri: str) -> tuple[bytes, str] | tuple[None, None]:
#     """Decodes a Base64 data URI string into raw bytes and MIME type."""
#     try:
#         if ';' in base64_uri:
#             metadata, encoded_data = base64_uri.split(',', 1)
#             mime_type = metadata.split(':')[1].split(';')[0]
#         else:
#             encoded_data = base64_uri
#             mime_type = 'image/png'
        
#         image_bytes = base64.b64decode(encoded_data)
#         return image_bytes, mime_type
#     except Exception:
#         return None, None

# # --- Helper: Query Gemini Multimodal ---
# def ask_gemini_multimodal(content_parts: List[Any]) -> str:
#     """Queries the Gemini model with a list of text and image parts."""
#     try:
#         model = genai.GenerativeModel(GEMINI_MODEL)
#         response = model.generate_content(content_parts)
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error querying Gemini: {e}")
#         return f"Error querying Gemini: {e}"

# # --- Lifespan Context Manager ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global text_model, nomic_ef, clip_model, clip_processor, chroma_client, text_collection, image_collection, chapters
    
#     print("🚀 Starting up multimodal RAG application...")
    
#     # Configure Gemini
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     if not gemini_api_key:
#         print("⚠️ WARNING: GEMINI_API_KEY environment variable not set.")
#     else:
#         genai.configure(api_key=gemini_api_key)
#         print("✅ Gemini API configured")
    
#     # Initialize Text Embedding Model
#     print(f"🔹 Loading text model: {TEXT_MODEL_NAME} on {DEVICE}")
#     text_model = SentenceTransformer(TEXT_MODEL_NAME, trust_remote_code=True).to(DEVICE)
#     print("✅ Text model loaded")
    
#     # Initialize CLIP Model
#     print(f"🖼️ Loading CLIP model: {CLIP_MODEL_NAME} on {DEVICE}")
#     clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
#     clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
#     print("✅ CLIP model loaded")

#     # Define custom Text Embedding Function for Chroma
#     class NomicTextEmbeddingFunction(embedding_functions.EmbeddingFunction):
#         def __call__(self, texts: List[str]) -> List[List[float]]:
#             return text_model.encode(texts).tolist()
    
#     nomic_ef = NomicTextEmbeddingFunction()
    
#     # Initialize ChromaDB
#     print(f"📦 Loading ChromaDB from: {CHROMA_DB_PATH}")
#     try:
#         chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
#         text_collection = chroma_client.get_or_create_collection(
#             name=TEXT_COLLECTION_NAME,
#             embedding_function=nomic_ef
#         )
#         print(f"✅ Text Collection loaded with {text_collection.count()} documents.")
        
#         image_collection = chroma_client.get_or_create_collection(name=IMAGE_COLLECTION_NAME)
#         print(f"✅ Image Collection loaded with {image_collection.count()} images.")
        
#     except Exception as e:
#         print(f"❌ Error loading ChromaDB collections: {e}")
#         text_collection = None
#         image_collection = None
    
#     # Load Chapter Metadata
#     try:
#         with open(CHAPTERS_FILE, "r", encoding="utf-8") as f:
#             chapters = json.load(f)
#             print(f"📚 Loaded {len(chapters)} chapters.")
#     except Exception as e:
#         print(f"❌ Error loading chapters: {e}")
#         chapters = []
    
#     print("✅ Application startup complete!\n")
#     yield
#     print("🔄 Shutting down application...")

# # --- Setup FastAPI ---
# app = FastAPI(lifespan=lifespan)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# # --- Helper: Multimodal Search ---
# def multimodal_rag_search(question: str, text_k: int = 5, image_k: int = 3) -> Dict[str, Any]:
#     if not text_collection or not image_collection:
#         return {"context": "Database not available.", "image_parts": []}
    
#     # --- Text Search ---
#     text_emb = nomic_ef([question])[0]
#     text_results = text_collection.query(query_embeddings=[text_emb], n_results=text_k)
#     context_chunks = text_results["documents"][0] if text_results.get("documents") else []
    
#     # --- Image Search ---
#     inputs = clip_processor(text=[question], return_tensors="pt", padding=True).to(DEVICE)
#     with torch.no_grad():
#         clip_emb = clip_model.get_text_features(**inputs).cpu().numpy()[0]
    
#     image_results = image_collection.query(query_embeddings=[clip_emb.tolist()], n_results=image_k)
#     image_parts = []

#     if image_results.get("metadatas") and image_results["metadatas"][0]:
#         unique_uris = set()
#         for metadata in image_results["metadatas"][0]:
#             uri = metadata.get("base64_uri")
#             if uri and uri not in unique_uris:
#                 unique_uris.add(uri)
#                 image_bytes, mime_type = base64_uri_to_bytes_and_mime(uri)
#                 if image_bytes and mime_type:
#                     image_parts.append({"mime_type": mime_type, "data": image_bytes})

#     context = "\n---\n".join(context_chunks)
#     return {"context": context, "image_parts": image_parts, "source_count": len(context_chunks) + len(image_parts)}

# # --- Routes ---
# @app.get("/", response_class=HTMLResponse)
# async def home():
#     try:
#         with open("static/index.html", "r", encoding="utf-8") as f:
#             return f.read()
#     except FileNotFoundError:
#         return "<h1>Error: index.html not found in static folder</h1>"

# app.mount("/static", StaticFiles(directory="static"), name="static")

# # --- Ask Question ---
# @app.post("/api/learning/ask")
# async def ask_question(req: QuestionRequest) -> Dict[str, str]:
#     if not text_collection or not image_collection:
#         return {"answer": "Error: Database not initialized."}
    
#     question = req.question
#     if not question:
#         return {"answer": "Please provide a question."}

#     try:
#         search_results = multimodal_rag_search(question)
#         context, image_parts = search_results["context"], search_results["image_parts"]
#         source_count = search_results["source_count"]

#         if not context and not image_parts:
#             return {"answer": "No relevant information found in the study material."}

#         text_prompt = f"""
#         You are a knowledgeable tutor. Use ONLY the provided text and images to answer.
#         Do not mention sources or context.
#         If the question cannot be answered, politely respond that the information is not available.

#         Text Context:
#         {context}

#         Question:
#         {question}
#         """

#         content_parts = [text_prompt] + image_parts
#         answer = ask_gemini_multimodal(content_parts)
#         return {"answer": answer, "sources_used": str(source_count)}

#     except Exception as e:
#         print(f"Error in ask_question: {e}")
#         return {"answer": f"Error: {str(e)}"}

# # --- Generate Quiz ---
# @app.post("/api/quiz/generate")
# async def generate_quiz(req: QuizGenerateRequest) -> Dict[str, Any]:
#     if not chapters:
#         return {"error": "Chapter data not loaded."}
    
#     try:
#         ch_num = int(req.chapter.replace("chapter", ""))
#     except ValueError:
#         return {"error": "Invalid chapter format."}

#     chapter_data = next((c for c in chapters if c["chapter_number"] == ch_num), None)
#     if not chapter_data:
#         return {"error": f"Chapter {ch_num} not found."}

#     chapter_text = chapter_data["chapter_text"]
#     chapter_image_parts = []

#     if chapter_data.get("chapter_images"):
#         first_uri = chapter_data["chapter_images"][0].get("base64_uri")
#         if first_uri:
#             image_bytes, mime_type = base64_uri_to_bytes_and_mime(first_uri)
#             if image_bytes and mime_type:
#                 chapter_image_parts.append({"mime_type": mime_type, "data": image_bytes})

#     prompt = f"""Generate exactly 5 open-ended questions from the following chapter.
# Return ONLY JSON:
# [
#   {{"question": "..." }}
# ]

# Content:
# {chapter_text[:8000]}"""

#     try:
#         content_parts = chapter_image_parts + [prompt]
#         response = ask_gemini_multimodal(content_parts)
#         response = response.replace("```json", "").replace("```", "").strip()
#         questions = json.loads(response)

#         return {"questions": questions, "chapter_number": ch_num, "chapter_context": chapter_text[:8000]}
#     except Exception as e:
#         print(f"Quiz generation error: {e}")
#         return {"error": f"Failed to generate quiz: {str(e)}"}

# # --- Evaluate Answer ---
# @app.post("/api/quiz/evaluate")
# async def evaluate_answer(req: AnswerEvaluationRequest) -> Dict[str, Any]:
#     if not req.user_answer.strip():
#         return {"error": "Please provide an answer."}

#     prompt = f"""Evaluate the student's answer based on the chapter.

# Context:
# {req.chapter_context}

# Question:
# {req.question}

# Student's Answer:
# {req.user_answer}

# Return JSON:
# {{
#   "score": <0-100>,
#   "feedback": "<detailed feedback>",
#   "improved_answer": "<model answer>"
# }}"""

#     try:
#         response = ask_gemini_multimodal([prompt])
#         response = response.replace("```json", "").replace("```", "").strip()
#         evaluation = json.loads(response)
#         return evaluation
#     except Exception as e:
#         print(f"Evaluation error: {e}")
#         return {"error": "Failed to evaluate answer"}

# # --- Health Check ---
# @app.get("/health")
# def health_check():
#     return {
#         'status': 'Ok',
#         'text_chroma_loaded': text_collection is not None,
#         'image_chroma_loaded': image_collection is not None,
#         'text_chroma_count': text_collection.count() if text_collection else 0,
#         'image_chroma_count': image_collection.count() if image_collection else 0,
#         'chapters_loaded': len(chapters),
#         'models_loaded': text_model is not None and clip_model is not None
#     }

# # --- Get Chapters ---
# @app.get("/api/chapters")
# def get_chapters():
#     if not chapters:
#         return {"chapters": []}
#     return {
#         "chapters": [
#             {
#                 "number": ch["chapter_number"],
#                 "title": ch.get("title", f"Chapter {ch['chapter_number']}"),
#                 "value": f"chapter{ch['chapter_number']}"
#             }
#             for ch in chapters
#         ]
#     }
