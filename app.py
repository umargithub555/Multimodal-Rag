import base64
import mimetypes
import re
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse,JSONResponse,StreamingResponse
import os
import shutil
import asyncio
from dotenv import load_dotenv
import json
from fastapi.staticfiles import StaticFiles
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
# import google.generativeai as 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import HumanMessage
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel
import threading
from eleven_tts import ElevenTTS
import speech_recognition as sr
from pathlib import Path
from multimodal_pipeline import (
    process_book,
    rebuild_embeddings,
    process_query,
)



load_dotenv()


# --- Configuration Constants ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_chapter_embeddings"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
# CHAPTERS_FILE = "output/semantic_chapters.json"
# CHAPTERS_FILE= 'output/regex_chapters.json'
GEMINI_MODEL = "gemini-2.0-flash"
CHAPTERS_FILE = "output/regex_chapters.json"
TEXTS_FILE = "output/texts.json"
IMAGES_ROOT = (Path.cwd() / "output" / "images").resolve()
MAX_IMAGE_QUESTIONS = 3

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

text_model = None
nomic_ef = None
chroma_client = None
collection = None
chapters = []
page_text_map: Dict[int, str] = {}
retriever = None
elevenlabs_client = None
google_api_key = None  # Added for multimodal pipeline
model_loaded_event = None  
elevenlabs_client = None


# --- Request Models ---
class QuestionRequest(BaseModel):
    question: str

class QuizGenerateRequest(BaseModel):
    chapter: str

class AnswerEvaluationRequest(BaseModel):
    question: str
    user_answer: str
    chapter_context: str
    image_base64: str = None


class InterviewGenerateRequest(BaseModel):
    chapter: str
    interviewee_name: str
    interviewee_email: str
    interviewee_address: str
    num_questions: int = 3

class TTSRequest(BaseModel):
    text: str
    voice_id: str = "CwhRBWXzGAHq8TQ4Fs17"

class InterviewFeedbackRequest(BaseModel):
    interview_script: List[Dict[str, str]]
    user_answers: List[Dict[str, str]]
    chapter_context: str
    interviewee_name: str

class STTRequest(BaseModel):
    audio_url: str  # Base64 encoded audio or URL

class InterviewResponseRequest(BaseModel):
    conversation_history: List[Dict[str, str]]  # List of {role: "interviewer"/"interviewee", text: "..."}
    chapter_context: str
    interviewee_name: str
    current_stage: str  # "greeting", "introduction", "question", "closing"


# 1. Define the message format (must match what the front-end sends)
class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

# 2. Define the full request body structure
class ChatRequest(BaseModel):
    question: str
 # Assuming this is still needed for context selection
    messages: List[ChatMessage] # This is the conversation history





google_key_value = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if google_key_value:
    os.environ["GOOGLE_API_KEY"] = google_key_value
    print("✅ API Key successfully set.")
else:
    print("❌ No API key found in environment!")



# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_model, nomic_ef, chroma_client, collection, chapters, retriever
    
    print("🚀 Starting up application...")
    
    # Configure Gemini
    # gemini_api_key = os.getenv("GEMINI_API_KEY")
    # if not gemini_api_key:
    #     print("⚠️ WARNING: GEMINI_API_KEY environment variable not set.")
    # else:
    #     genai.configure(api_key=gemini_api_key)
    #     print("✅ Gemini API configured")
    print("✅ Building Retriever")
    # retriever = await rebuild_embeddings("./chroma_db_latest", "./docstore.json")
    retriever = await rebuild_embeddings("./chroma_db_final", "./docstore_final.json")



    global elevenlabs_client
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if elevenlabs_api_key:
        try:
            from elevenlabs import ElevenLabs
            elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
            print("✅ ElevenLabs API configured")
        except Exception as e:
            print(f"⚠️ WARNING: ElevenLabs not available: {e}")
            elevenlabs_client = None
    else:
        print("⚠️ WARNING: ELEVENLABS_API_KEY environment variable not set.")
        elevenlabs_client = None

    
    # global model_loaded_event
    # model_loaded_event = threading.Event()
    
    # Initialize SentenceTransformer
    # print(f"🔹 Loading embedding model: {MODEL_NAME}")
    # text_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    # print("✅ Embedding model loaded")
    
    # # Define custom Embedding Function
    # class NomicTextEmbeddingFunction(embedding_functions.EmbeddingFunction):
    #     def __call__(self, texts: List[str]) -> List[List[float]]:
    #         return text_model.encode(texts).tolist()
    
    # nomic_ef = NomicTextEmbeddingFunction()
    
    # # Initialize ChromaDB
    # print(f"📦 Loading ChromaDB from: {CHROMA_DB_PATH}")
    # try:
    #     chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    #     collection = chroma_client.get_or_create_collection(
    #         name=COLLECTION_NAME,
    #         embedding_function=nomic_ef
    #     )
    #     print(f"✅ Chroma Collection '{COLLECTION_NAME}' loaded with {collection.count()} documents.")
    # except Exception as e:
    #     print(f"❌ Error loading ChromaDB: {e}")
    #     chroma_client = None
    #     collection = None
    
    # Load Chapter Metadata
    print("✅ Loading Semantic Chapters for Quiz Generation")
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


app = FastAPI(lifespan=lifespan)

# Allow frontend access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # You can restrict to specific domains
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*","ngrok-skip-browser-warning"],
)




def ask_gemini_stream(prompt: str):
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)
    
    response = llm.invoke([HumanMessage(content=prompt)])
      
    return response.content


def ask_gemini_multimodal(prompt: str, image_paths: List[str] = None):
    """Generate response using Gemini with multimodal support (text + images)"""
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)
    # Build content list
    content = [{"type": "text", "text": prompt}]
    # Add images if provided
    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                # Read and encode image
                with open(img_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    # Determine mime type
                    ext = img_path.split('.')[-1].lower()
                    mime_type = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
                    })
    response = llm.invoke([HumanMessage(content=content)])
    return response.content




def image_path_to_base64(image_path: str) -> str:
    """Convert image file path to base64 string (without data URI prefix)"""
    try:
        if not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None



# def ask_gemini(prompt: str) -> str:
#     try:
#         model = genai.GenerativeModel(GEMINI_MODEL)
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"Error querying Gemini: {e}"


def _clean_json_response(response: str) -> str:
    if not response:
        return ""
    return response.replace("```json", "").replace("```", "").strip()


def _safe_resolve_image_path(image_path_str: str) -> Optional[Path]:
    candidate = Path(image_path_str)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve()
    except Exception:
        return None
    try:
        resolved.relative_to(IMAGES_ROOT)
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _extract_page_number(image_path: Path) -> Optional[int]:
    match = re.search(r"page_(\d+)", image_path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None
    

def _encode_image_to_data_uri(image_path: Path) -> Optional[str]:
    try:
        byte_content = image_path.read_bytes()
    except Exception:
        return None
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(byte_content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _build_image_question(
    image_path: Path,
    page_number: Optional[int],
    chapter_title: str,
    chapter_excerpt: str,
) -> Optional[Dict[str, Any]]:
    page_excerpt = ""
    if page_number is not None:
        page_excerpt = page_text_map.get(page_number, "")
    if not page_excerpt:
        page_excerpt = chapter_excerpt
    prompt = f"""You are creating one open-ended quiz question for learners.
The question must explicitly tell the learner to refer to an accompanying image from the chapter "{chapter_title}".
Use the provided textbook excerpt to understand what the image is likely illustrating and craft a meaningful question that encourages explanation or analysis.
Text excerpt:
{page_excerpt[:1500]}
Return exactly one JSON object with the following shape:
{{
  "question": "A single question that references the image (e.g., 'Refer to the image showing ... and explain ...').",
  "answer_guidance": "One or two sentences describing the key ideas a strong answer should cover."
}}
Return only the JSON object."""
    response = ask_gemini_stream(prompt)
    if response.startswith("Error"):
        return None
    cleaned = _clean_json_response(response)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    question_text = payload.get("question", "")
    if not question_text:
        return None
    answer_guidance = payload.get("answer_guidance", "").strip()
    image_data_uri = _encode_image_to_data_uri(image_path)
    if not image_data_uri:
        return None
    return {
        "type": "image",
        "question": question_text.strip(),
        "answer_guidance": answer_guidance or None,
        "image_data": image_data_uri,
        "image_filename": image_path.name,
        "page_number": page_number,
    }










@app.post("/api/upload-book")
async def upload_book(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ✅ Pass the Google API key explicitly
        result = await process_book(file_path=file_path, google_api_key=google_key_value)
        return {"status": "success", "message": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}




@app.get("/api/rebuild")
async def rebuild():
    try:
        # retriever = await rebuild_embeddings("./chroma_db_latest", "./docstore.json")
        if not retriever:
            return {"status": "error", "message": "Failed to rebuild retriever"}
        return {"status": "success", "message": "Retriever rebuilt successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}



# Working 10/11/2025 ⬇️

# @app.post("/api/ask")
# async def ask_question(question: str = Form(...)):
#     """
#     Handle a POST request with a 'question' field and return AI-generated response + context.
#     """
#     try:
#         # retriever = await rebuild_embeddings("./chroma_db_latest", "./docstore.json")

#         response = await process_query(
#             query=question,
#             retriever=retriever,
#             google_api_key=google_key_value,
#         )

#         return {
#             "status": "success",
#             "data": response,
#         }

#     except Exception as e:
#         return {
#             "status": "error",
#             "message": str(e),
#         }


@app.post("/api/ask")
async def ask_question(request: ChatRequest): # Change from Form(...) to ChatRequest
    """
    Handle a POST request with question, selected PDF, and conversation history.
    """
    try:
        # NOTE: You will need to manage the retriever loading/caching before this point.
        # Assuming 'retriever' and 'google_key_value' are globally available or handled.

        response = await process_query(
            query=request.question,              # Pass the new question
            messages=request.messages,           # <-- NEW: Pass the conversation history
            retriever=retriever,
            google_api_key=google_key_value,
        )

        return {
            "status": "success",
            "data": response,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }










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
    
#     # Generate questions using Gemini
#     prompt = f"""Based on the following chapter content, generate exactly 5 open-ended questions that test understanding of key concepts.

# Requirements:
# - Questions should be clear and specific
# - Questions should require explanatory answers (not yes/no)
# - Cover different topics from the chapter
# - Return ONLY a JSON array of questions in this exact format:
# [
#   {{"question": "Question text here"}},
#   {{"question": "Question text here"}},
#   ...
# ]

# Chapter Content:
# {chapter_text[:8000]}

# Return only the JSON array, no additional text."""
    
#     try:
#         response = ask_gemini_stream(prompt)
        
#         # Parse JSON response
#         # Remove markdown code blocks if present
#         response = response.replace("```json", "").replace("```", "").strip()
#         questions = json.loads(response)
        
#         return {
#             "questions": questions,
#             "chapter_number": ch_num,
#             "chapter_context": chapter_text[:8000]  # Send context for evaluation
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
# 2. Detailed feedback on what was correct and what was missing and do not add or mention student answer.
# 3. An improved/model answer keep it short and to the point

# Return your response in this EXACT JSON format:
# {{
#   "score": <number between 0-100>,
#   "feedback": "<detailed feedback text>",
#   "improved_answer": "<model answer text>"
# }}

# Return only the JSON object, no additional text."""
    
#     try:
#         response = ask_gemini_stream(prompt)
        
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



@app.get("/api/chapters")
def get_chapters():
    if not chapters:
        return {"chapters": []}
    chapter_list = [
        {
            "number": ch["chapter_number"],
            "title": ch.get("chapter_title") or ch.get("title") or f"Chapter {ch['chapter_number']}",
            "value": f"chapter{ch['chapter_number']}"
        }
        for ch in chapters
    ]
    return {"chapters": chapter_list}





# --- Route: Generate Interview Script ---
@app.post("/api/interview/generate")
async def generate_interview_script(req: InterviewGenerateRequest) -> Dict[str, Any]:
    """Generate a complete interview script based on chapter and interviewee info"""
    if not chapters:
        return {"error": "Chapter data not loaded. Please check server logs."}
    
    chapter_value = req.chapter
    interviewee_name = req.interviewee_name
    num_questions = req.num_questions
    
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
    
    # Generate initial interview greeting using Gemini
    prompt = f"""You are a professional interviewer conducting a technical interview. Start the interview with a warm greeting for {interviewee_name}.

The interview will be based on the following chapter content from a technical book about refrigeration and air conditioning.

Start with a warm, professional greeting. Keep it natural and conversational.

Chapter Content:
{chapter_text[:8000]}

Return ONLY a JSON object in this exact format:
{{
  "type": "greeting",
  "text": "Hi, how are you {interviewee_name}? How's your day?"
}}

Return only the JSON object, no additional text."""
    
    try:
        response = ask_gemini_stream(prompt)
        
        # Parse JSON response
        response = response.replace("```json", "").replace("```", "").strip()
        greeting = json.loads(response)
        
        # Return just the greeting - the rest will be generated dynamically
        return {
            "greeting": greeting,
            "chapter_number": ch_num,
            "chapter_context": chapter_text[:8000],
            "interviewee_name": interviewee_name,
            "current_stage": "greeting"
        }
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response}")
        return {"error": "Failed to parse interview script from AI response"}
    except Exception as e:
        print(f"Error generating interview script: {e}")
        return {"error": f"Error generating interview script: {str(e)}"}


# --- Route: Generate Interview Feedback ---
@app.post("/api/interview/feedback")
async def generate_interview_feedback(req: InterviewFeedbackRequest) -> Dict[str, Any]:
    """Generate feedback and suggestions after the interview"""
    interview_script = req.interview_script
    user_answers = req.user_answers
    chapter_context = req.chapter_context
    interviewee_name = req.interviewee_name
    
    # Build conversation summary
    conversation_summary = "Interview Script:\n"
    for i, item in enumerate(interview_script):
        conversation_summary += f"{i+1}. Interviewer: {item.get('text', '')}\n"
    
    conversation_summary += "\nInterviewee Answers:\n"
    for i, answer in enumerate(user_answers):
        question = answer.get('question', '')
        answer_text = answer.get('answer', '')
        conversation_summary += f"Q{i+1}: {question}\nAnswer: {answer_text}\n\n"
    
    prompt = f"""You are a professional interviewer providing feedback to {interviewee_name} after a technical interview.

Based on the following interview conversation and the chapter context, provide:
1. Overall assessment
2. Strengths shown
3. Areas for improvement
4. Technical knowledge evaluation
5. Suggestions for better preparation

Be constructive, encouraging, and professional.

Chapter Context:
{chapter_context[:6000]}

Interview Conversation:
{conversation_summary}

Return your feedback in this EXACT JSON format:
{{
  "overall_assessment": "Overall assessment text",
  "strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "improvements": ["Improvement 1", "Improvement 2", "Improvement 3"],
  "technical_score": <number between 0-100>,
  "suggestions": "Suggestions text for better preparation"
}}

Return only the JSON object, no additional text."""
    
    try:
        response = ask_gemini_stream(prompt)
        
        # Clean response
        response = response.replace("```json", "").replace("```", "").strip()
        feedback = json.loads(response)
        
        return feedback
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response}")
        return {"error": "Failed to parse feedback from AI response"}
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return {"error": f"Error generating feedback: {str(e)}"}


# --- Route: Stream TTS Audio with Interrupt Support ---
@app.post("/api/interview/tts")
async def stream_tts(req: TTSRequest):
    """Stream TTS audio from ElevenLabs with interrupt capability"""
    if not elevenlabs_client:
        error_msg = (
            "ElevenLabs API not configured. Please set ELEVENLABS_API_KEY in your .env file. "
            "Get your API key from https://elevenlabs.io/"
        )
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=503, detail=error_msg)
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        
        
        tts = ElevenTTS(
            client=elevenlabs_client,
            voice=req.voice_id,
            model_id="eleven_multilingual_v2"
        )
        
        async def generate_audio():
            try:
                async for chunk in tts.stream(req.text):
                    yield chunk
            except asyncio.CancelledError:
                print("TTS stream cancelled by client")
                raise
        
        return StreamingResponse(
            generate_audio(),
            media_type="audio/pcm",
            headers={
                "Content-Type": "audio/pcm",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        print(f"❌ Error in TTS stream: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")


# --- Route: Speech-to-Text (STT) ---
@app.post("/api/interview/stt")
async def speech_to_text(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Transcribe audio to text using speech_recognition library"""
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Save temporarily to process
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(tmp_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Record audio
                audio = recognizer.record(source)
            
            # Try Google Speech Recognition (free, no API key needed)
            try:
                transcript = recognizer.recognize_google(audio, language="en-US")
                return {
                    "text": transcript,
                    "confidence": 0.95  # Google Speech Recognition doesn't return confidence, use high default
                }
            except sr.UnknownValueError:
                return {
                    "text": "Could not understand the audio. Please speak clearly.",
                    "confidence": 0.0
                }
            except sr.RequestError as e:
                # If Google API fails, try offline recognition
                try:
                    # Try using Sphinx (offline, requires pocketsphinx)
                    transcript = recognizer.recognize_sphinx(audio, language="en-US")
                    return {
                        "text": transcript,
                        "confidence": 0.8
                    }
                except:
                    return {
                        "text": f"Speech recognition service error: {e}. Please try again.",
                        "confidence": 0.0
                    }
            
        except ImportError:
            return {
                "text": "[Speech Recognition library not available. Please install SpeechRecognition]",
                "confidence": 0.0
            }
        except Exception as e:
            print(f"❌ Error in speech recognition: {e}")
            return {
                "text": f"Error processing audio: {str(e)}",
                "confidence": 0.0
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ Error in STT: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")


# --- Route: Generate Dynamic Interview Response ---
@app.post("/api/interview/response")
async def generate_interview_response(req: InterviewResponseRequest) -> Dict[str, Any]:
    """Generate dynamic interview response based on conversation history"""
    conversation_history = req.conversation_history
    chapter_context = req.chapter_context
    interviewee_name = req.interviewee_name
    current_stage = req.current_stage
    
    # Build conversation context
    conversation_text = "\n".join([
        f"{'Interviewer' if msg.get('role') == 'interviewer' else interviewee_name}: {msg.get('text', '')}"
        for msg in conversation_history[-10:]  # Last 10 messages for context
    ])
    
    # Determine what to generate next
    if current_stage == "greeting":
        # After greeting, respond to their greeting and ask for introduction
        prompt = f"""You are a professional interviewer. The interviewee {interviewee_name} just responded to your greeting.

Conversation so far:
{conversation_text}

First, acknowledge their response naturally (e.g., if they said "I'm good, thanks", respond appropriately). Then naturally ask them to introduce themselves. Keep it conversational and professional.

Return ONLY a JSON object in this exact format:
{{
  "type": "introduction",
  "text": "Your natural response acknowledging their greeting and asking them to introduce themselves"
}}

Return only the JSON object, no additional text."""
    
    elif current_stage == "introduction":
        # After introduction, ask first technical question
        prompt = f"""You are a professional interviewer conducting a technical interview. The interviewee {interviewee_name} just introduced themselves.

Conversation so far:
{conversation_text}

Chapter Content:
{chapter_context[:6000]}

Based on their introduction and the chapter content, ask your first technical question related to refrigeration and air conditioning. Make it relevant and engaging.

Return ONLY a JSON object in this exact format:
{{
  "type": "question",
  "text": "Your technical question here"
}}

Return only the JSON object, no additional text."""
    
    elif current_stage == "question":
        # After a question, either ask follow-up or next question
        # Check how many questions asked (count interviewer messages with type='question')
        question_count = sum(1 for msg in conversation_history if msg.get('role') == 'interviewer' and msg.get('type') == 'question')
        
        if question_count < 3:
            # Ask next technical question - respond to their answer first
            prompt = f"""You are a professional interviewer. The interviewee {interviewee_name} just answered your technical question.

Conversation so far:
{conversation_text}

Chapter Content:
{chapter_context[:6000]}

IMPORTANT: First, briefly acknowledge or respond to what they said (e.g., "That's good", "I see", "Interesting", or provide brief feedback). Then naturally transition to the next technical question related to the chapter content. Don't ask the same question. Keep it conversational and engaging.

Return ONLY a JSON object in this exact format:
{{
  "type": "question",
  "text": "Brief acknowledgment of their answer, then the next technical question"
}}

Return only the JSON object, no additional text."""
        else:
            # End interview
            prompt = f"""You are a professional interviewer. The interviewee {interviewee_name} has answered all your questions.

Conversation so far:
{conversation_text}

Thank them for their time and ask if they have any questions for you.

Return ONLY a JSON object in this exact format:
{{
  "type": "closing",
  "text": "Thank you message and asking if they have questions"
}}

Return only the JSON object, no additional text."""
    
    else:
        # Closing stage
        return {"type": "end", "text": "Interview completed. Thank you!"}
    
    try:
        response = ask_gemini_stream(prompt)
        
        # Parse JSON response
        response = response.replace("```json", "").replace("```", "").strip()
        next_response = json.loads(response)
        
        return next_response
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response}")
        return {"error": "Failed to parse interview response from AI response"}
    except Exception as e:
        print(f"Error generating interview response: {e}")
        return {"error": f"Error generating interview response: {str(e)}"}












@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("static/test.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found in static folder</h1>"




@app.get("/health")
def home():
    return {"message": "📘 Multimodal RAG API is running!"}




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
#     # Generate questions using Gemini
#     prompt = f"""Based on the following chapter content, generate exactly 5 open-ended questions that test understanding of key concepts.
# Requirements:
# - Questions should be clear and specific
# - Questions should require explanatory answers (not yes/no)
# - Cover different topics from the chapter
# - Return ONLY a JSON array of questions in this exact format:
# [
#   {{"question": "Question text here"}},
#   {{"question": "Question text here"}},
#   ...
# ]
# Chapter Content:
# {chapter_text[:8000]}
# Return only the JSON array, no additional text."""
#     try:
#         response = ask_gemini_stream(prompt)
#         # Parse JSON response
#         # Remove markdown code blocks if present
#         cleaned_response = _clean_json_response(response)
#         raw_questions = json.loads(cleaned_response)
#         chapter_title = (
#             chapter_data.get("chapter_title")
#             or chapter_data.get("title")
#             or f"Chapter {ch_num}"
#         )
#         text_questions: List[Dict[str, Any]] = []
#         if isinstance(raw_questions, list):
#             for item in raw_questions:
#                 question_text = None
#                 if isinstance(item, dict):
#                     question_text = item.get("question") or item.get("text")
#                 elif isinstance(item, str):
#                     question_text = item
#                 if question_text:
#                     text_questions.append(
#                         {
#                             "type": "text",
#                             "question": question_text.strip(),
#                         }
#                     )
#         image_questions: List[Dict[str, Any]] = []
#         chapter_excerpt_for_images = chapter_text[:2000]
#         image_paths = chapter_data.get("chapter_images") or []
#         for image_path_str in image_paths:
#             if len(image_questions) >= MAX_IMAGE_QUESTIONS:
#                 break
#             resolved_image = _safe_resolve_image_path(image_path_str)
#             if not resolved_image:
#                 continue
#             page_number = _extract_page_number(resolved_image)
#             image_question = _build_image_question(
#                 image_path=resolved_image,
#                 page_number=page_number,
#                 chapter_title=chapter_title,
#                 chapter_excerpt=chapter_excerpt_for_images,
#             )
#             if image_question:
#                 image_questions.append(image_question)
#         combined_questions = text_questions + image_questions
#         return {
#             "questions": combined_questions,
#             "chapter_number": ch_num,
#             "chapter_title": chapter_title,
#             "chapter_context": chapter_text[:8000]  # Send context for evaluation
#         }
#     except json.JSONDecodeError as e:
#         print(f"JSON parsing error: {e}")
#         print(f"Response was: {response}")
#         return {"error": "Failed to parse questions from AI response"}
#     except Exception as e:
#         print(f"Error generating quiz: {e}")
#         return {"error": f"Error generating quiz: {str(e)}"}
    

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
    chapter_images = chapter_data.get("chapter_images", [])

    # Filter out empty or missing images
    valid_images = [img for img in chapter_images if img and os.path.exists(img)]

    all_questions = []

    try:
        # --- Generate 5 text-based questions ---
        text_prompt = f"""Based on the following chapter content, generate exactly 5 open-ended questions that test understanding of key concepts.
Requirements:
- Questions should be clear and specific
- Questions should require explanatory answers (not yes/no)
- Cover different topics from the chapter
- Return ONLY a JSON array of questions in this exact format:
[
  {{"question": "Question text here"}},
  {{"question": "Question text here"}},
  {{"question": "Question text here"}},
  {{"question": "Question text here"}},
  {{"question": "Question text here"}}
]
Chapter Content:
{chapter_text[:8000]}
Return only the JSON array, no additional text."""
        text_response = ask_gemini_stream(text_prompt)
        text_response = text_response.replace("```json", "").replace("```", "").strip()
        text_questions = json.loads(text_response)

        # Add metadata
        for q in text_questions:
            q["type"] = "text"
            q["image_base64"] = None

        all_questions.extend(text_questions)

        # --- Generate up to 3 image-based questions ---
        if valid_images:
            selected_images = valid_images[:3]  # ✅ exactly 3 if available
            for img_path in selected_images:
                try:
                    img_prompt = f"""Based on the image provided and the following chapter context, generate exactly 1 open-ended question that tests understanding of concepts shown in the image.
Requirements:
- The question should be directly related to what is shown in the image
- The question should be specific and focus on all parts of the image including minor things
- The question should require an explanatory answer
- Return ONLY a JSON object in this format:
{{"question": "Question text here"}}
Chapter Context (for reference):
{chapter_text[:4000]}
Return only the JSON object, no additional text."""
                    img_response = ask_gemini_multimodal(img_prompt, [img_path])
                    img_response = img_response.replace("```json", "").replace("```", "").strip()
                    img_question = json.loads(img_response)
                    img_question["type"] = "image"
                    img_question["image_base64"] = image_path_to_base64(img_path)
                    all_questions.append(img_question)
                except Exception as e:
                    print(f"Error generating question for image {img_path}: {e}")
                    continue

        # --- Ensure we have 8 questions total (5 text + 3 image) ---
        if len(all_questions) < 8:
            remaining = 8 - len(all_questions)
            fill_prompt = f"""Based on the following chapter content, generate exactly {remaining} more open-ended questions that test understanding of key concepts.
Requirements:
- Questions should be clear and specific
- Questions should require explanatory answers (not yes/no)
- Return ONLY a JSON array of questions in this format:
[
  {{"question": "Question text here"}},
  ...
]
Chapter Content:
{chapter_text[:8000]}
Return only the JSON array, no additional text."""
            fill_response = ask_gemini_stream(fill_prompt)
            fill_response = fill_response.replace("```json", "").replace("```", "").strip()
            fill_questions = json.loads(fill_response)
            if isinstance(fill_questions, dict):
                fill_questions = [fill_questions]
            for q in fill_questions[:remaining]:
                q["type"] = "text"
                q["image_base64"] = None
                all_questions.append(q)

        # ✅ limit final output to exactly 8
        all_questions = all_questions[:8]

        return {
            "questions": all_questions,
            "chapter_number": ch_num,
            "chapter_context": chapter_text[:8000]
        }

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {"error": "Failed to parse questions from AI response"}
    except Exception as e:
        print(f"Error generating quiz: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error generating quiz: {str(e)}"}








# --- Route: Evaluate Answer ---
# @app.post("/api/quiz/evaluate")
# async def evaluate_answer(req: AnswerEvaluationRequest) -> Dict[str, Any]:
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
# Return only the JSON object, no additional text.
# You MUST NOT include or mention the student answers or anything like this in your response.
# """
#     try:
#         response = ask_gemini_stream(prompt)
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



@app.post("/api/quiz/evaluate")
async def evaluate_answer(req: AnswerEvaluationRequest) -> Dict[str, Any]:
    question = req.question
    user_answer = req.user_answer
    chapter_context = req.chapter_context
    image_base64 = req.image_base64
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
2. Detailed feedback on what was correct and what was missing and do not add or mention student answer.
3. An improved/model answer keep it short and to the point
{"Note: This question includes an image. Consider the image content when evaluating the answer." if image_base64 else ""}
Return your response in this EXACT JSON format:
{{
  "score": <number between 0-100>,
  "feedback": "<detailed feedback text>",
  "improved_answer": "<model answer text>"
}}
Return only the JSON object, no additional text. and dont mention "that the student" anywhere in the response"""
    try:
        # Use multimodal evaluation if image is provided
        if image_base64:
            # Convert base64 string to image file temporarily for multimodal API
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                img_bytes = base64.b64decode(image_base64)
                tmp_file.write(img_bytes)
                tmp_path = tmp_file.name
            try:
                response = ask_gemini_multimodal(prompt, [tmp_path])
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        else:
            response = ask_gemini_stream(prompt)
        # Clean response
        response = response.replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(response)
        print(evaluation)
        return evaluation
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response}")
        return {"error": "Failed to parse evaluation from AI response"}
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error evaluating answer: {str(e)}"}