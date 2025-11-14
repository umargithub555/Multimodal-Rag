import os
import base64
import json
import uuid
import nest_asyncio
import asyncio
from base64 import b64decode
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Dict,Any,AsyncGenerator,List
from pydantic import BaseModel

nest_asyncio.apply()


class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str


CHROMA_DB_PATH = "./chroma_db_latest"
DOCSTORE_PATH = "./dosctore.json"
EMBEDDING_MODEL='gemini-embedding-001'


POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
TESSERACT_PATH=r"C:\Program Files\Tesseract-OCR"

os.environ["PATH"] += os.pathsep + POPPLER_PATH  # ✅ Add poppler to PATH
os.environ["PATH"] += os.pathsep + TESSERACT_PATH





def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            # if decoding succeeds, it's an image
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}



def display_base64_image(b64_str):
    """Display base64 image in Jupyter or VSCode notebooks."""
    image_bytes = base64.b64decode(b64_str)
    display(Image(data=image_bytes))





def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    chat_history = kwargs["history"] # <-- NEW: Get the history

    system_instruction = """
    Your answer MUST be based SOLELY on the provided Context: text, tables, and images. 
    If the context does not contain the answer, you MUST respond with the exact and complete phrase: 
    "The information is not available in the provided knowledge base." 

    Exception: 
    If the user's message is a greeting, farewell, or polite small-talk (e.g., "hi", "hello", "how are you", "thanks", "bye"),
    respond naturally and politely (e.g., "Hello! How can I help you today?").
    DO NOT mention based on the context or provided context or any other wording in the response.
    DO NOT use your general knowledge. Be concise.
    """
    
    # 1. Start with the System instruction
    prompt_messages = [SystemMessage(content=system_instruction)]
    
    # 2. Add the previous messages (History)
    prompt_messages.extend(chat_history) 
    
    # 3. Compile the new context and current question for the LLM
    context_text = ""
    # ... (Your context_text compilation logic remains the same) ...
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
             # ... (logic to extract page_content or text) ...
             if hasattr(text_element, "page_content"):
                 context_text += text_element.page_content
             elif hasattr(text_element, "text"):
                 context_text += text_element.text
             else:
                 context_text += str(text_element)

    # prompt_template = f"""

    # Context: {context_text}
    # Question: {user_question}
    # """

    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images
    Make sure to answer correctly to user queries about past conversations aka context_text.
    Context: {context_text}
    Question: {user_question}
    """

    # 4. Compile the final HumanMessage with context and images
    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    # 5. Add the final HumanMessage (the current user turn)
    prompt_messages.append(HumanMessage(content=prompt_content))

    # Return the ChatPromptTemplate containing the full message list
    return ChatPromptTemplate.from_messages(prompt_messages)




# Working 10/11/2025 ⬇️ single query


# async def make_retiever_structure(retriever,google_api_key):
        
#         llm1 = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",  # or gemini-1.5-pro for better reasoning
#         temperature=0.3,
#         google_api_key=google_api_key,
#         )


#         # ---- Define the runnable chain ----
#         chain_rebuild = (
#             {
#                 "context": retriever | RunnableLambda(parse_docs),
#                 "question": RunnablePassthrough(),
#             }
#             | RunnableLambda(build_prompt)
#             | llm1
#             | StrOutputParser()
#         )


#         # ---- Optional: Chain that also returns sources ----
#         chain_with_sources_rebuild = (
#             {
#                 "context": retriever | RunnableLambda(parse_docs),
#                 "question": RunnablePassthrough(),
#             }
#             | RunnablePassthrough().assign(
#                 response=(RunnableLambda(build_prompt) | llm1 | StrOutputParser())
#             )
#         )

#         print("✅ Gemini multimodal chain ready!")

#         return chain_with_sources_rebuild


async def make_retiever_structure(retriever, google_api_key, messages: List[ChatMessage]): # <-- UPDATED SIGNATURE
        
    llm1 = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=google_api_key,
    )

    # Convert the messages list to LangChain's format (HumanMessage/AIMessage)
    # def format_history(history: List[Dict[str, str]]):
    #     formatted_messages = []
    #     for msg in history:
    #         role = msg.get('role')
    #         content = msg.get('content', '')
    #         if role == 'user':
    #             formatted_messages.append(HumanMessage(content=content))
    #         elif role == 'assistant':
    #             # Note: Assuming 'assistant' messages are text only for history
    #             formatted_messages.append(SystemMessage(content=content)) 
    #     return formatted_messages
    count = 0
    def format_history(history: List[ChatMessage]):  # Change type hint
        formatted_messages = []
        for msg in history:
            # Use dot notation for Pydantic models, not .get()
            role = msg.role
            content = msg.content
            
            if role == 'user':
                formatted_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                formatted_messages.append(AIMessage(content=content))  # Use AIMessage, not SystemMessage
        return formatted_messages

    # Define a custom input dictionary that includes the formatted history
    chain_input = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
        "history": RunnableLambda(lambda x: format_history(messages)) # Pass history
    }

    print(messages)

    # ---- Optional: Chain that also returns sources ----
    chain_with_sources_rebuild = (
        chain_input
        | RunnablePassthrough().assign(
            response=(RunnableLambda(build_prompt) | llm1 | StrOutputParser())
        )
    )

    print("✅ Gemini multimodal chat chain ready!")
    return chain_with_sources_rebuild




async def process_images(images, google_api_key):
    prompt_template = """Describe the image in detail. For context, this image is from a marine refrigeration and air conditioning technical manual used in maritime engineering education. Be specific about graphs, such as bar plots and any other components given inside an image."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    # This Base64 structure is fully compatible with Gemini
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    # 💡 THE CHANGE: Use ChatGoogleGenerativeAI and pass the key explicitly
    chain = prompt | ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key # Pass the variable to bypass env visibility issues
    ) | StrOutputParser()

    try:
    # The batch invocation remains the same
        image_summaries = chain.batch(images)
    except Exception as e:
        print(f"An error occur while generating image summaries [multimodal_pipeline[process_image]] {e}")

    return image_summaries






async def generate_save_embeddings(CHROMA_PATH,DOCSTORE_PATH,embedding_model, texts, images, text_summaries, image_summaries):
    
        # Ensure the Chroma path exists
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH)

        # The vectorstore to use to index the child chunks
        # 💡 IMPORTANT: Use a Gemini Embedding function, as OpenAIEmbeddings requires an OpenAI key
        vectorstore = Chroma(
            collection_name="multi_modal_rag", 
            embedding_function=GoogleGenerativeAIEmbeddings(model=embedding_model), # Changed to Gemini Embeddings
            persist_directory=CHROMA_PATH # 🌟 CHROMA SAVES HERE 🌟
        )

        # The storage layer for the parent documents
        store = InMemoryStore()
        id_key = "doc_id"

        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        # ... (Your logic to add texts, tables, and images remains the same) ...

        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

        # Add tables
        # table_ids = [str(uuid.uuid4()) for _ in tables]
        # summary_tables = [
        #     Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
        # ]
        # retriever.vectorstore.add_documents(summary_tables)
        # retriever.docstore.mset(list(zip(table_ids, tables)))

        # Add image summaries
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, images)))



        # ----------------------------------------------------------------------
        # 🌟 STEP 2: Save the Parent Documents (from InMemoryStore) to a file 🌟
        # ----------------------------------------------------------------------

        # 1. Collect all parent documents from the InMemoryStore
        # The InMemoryStore's underlying storage is a dictionary accessible via its `store` attribute.
        # The keys are the doc_ids and the values are the parent documents (text, tables, or image base64).
        parent_docs_to_save = dict(retriever.docstore.store)

        # 2. Save the dictionary to a JSON file
        with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
            json.dump(parent_docs_to_save, f, ensure_ascii=False, default=str, indent=2)

        # 3. Trigger Chroma persistence (optional, as it often saves on additions, but good practice)
        vectorstore.persist()

        print(f"Data saved: Chroma DB to {CHROMA_PATH} and Document Store to {DOCSTORE_PATH}")

        return retriever




async def process_book(file_path, google_api_key):
    
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",

        extract_image_block_types=["Image"],

        extract_image_block_to_payload=True,
        
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    
    print(set([str(type(el)) for el in chunks]))

    print('\n\n', chunks[3].metadata.orig_elements)


    # separate tables from texts
    print("\n\n Saving texts and tables in separate arrays")
    tables = []
    texts = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)
    

    print("\n\n Saving images in separate arrays")


    # Get the images from the CompositeElement objects
    def get_images_base64(chunks):
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64

    images = get_images_base64(chunks)


    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """

    print("\n\n Making the summary chain")

    # google_key = os.getenv("GOOGLE_API_KEY")
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
   
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.5,
        google_api_key=google_api_key
        
    )
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    print("\n\n Using summary chain to generate text summaries")
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    print("\n\n Generating Image summaries")
    image_summaries = await process_images(images, google_api_key)


    await generate_save_embeddings(CHROMA_DB_PATH, DOCSTORE_PATH, EMBEDDING_MODEL,texts, images, text_summaries, image_summaries)

    return {'status':'Embeddings generated and saved sucessfully'}



async def rebuild_embeddings(chroma_path,doc_path):
    try:

        vectorstore = Chroma(
        collection_name="multi_modal_final_rag",
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=chroma_path,
        )

        # Rebuild docstore
        store = InMemoryStore()
        with open(doc_path, "r", encoding="utf-8") as f:
            store.store = json.load(f)

        # Rebuild retriever
        retriever_rebuild = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key="doc_id",
        )

    except Exception as e: 
        print(f"An Error occured while fetching and rebuilding embeddings [multimodal_pipeline[rebuild_embeddings]] - {e}")

    if not retriever_rebuild:
        return f"Unable to process retriever rebuilding"
    
    else:
        return retriever_rebuild
    




# async def process_query(query,retriever,google_api_key):

#     chain_with_sources = await make_retiever_structure(retriever,google_api_key)
#     # Invoke the chain
#     response = chain_with_sources.invoke(query)

#     # Print response text
#     print("🧠 Response:\n", response['response'])

#     print("\n📚 Context Texts:")
#     for text in response['context']['texts']:
#         if hasattr(text, "page_content"):
#             print(text.page_content)
#         elif hasattr(text, "text"):
#             print(text.text)
#         else:
#             print(str(text))
        
#         # ✅ FIX: use getattr() instead of "in"
#         page_number = getattr(text.metadata, "page_number", None)
#         if page_number is not None:
#             print("Page number:", page_number)

#         print("\n" + "-" * 50 + "\n")


#     print("🖼️ Context Images:")
#     for image in response['context']['images']:
#         display_base64_image(image)









# async def process_query(query: str, retriever, google_api_key: str) -> Dict[str, Any]:
#     """
#     Process a user query, retrieve relevant context, and query the Gemini multimodal chain.
#     """
#     chain_with_sources = await make_retiever_structure(retriever, google_api_key)

#     # Run the chain
#     response = chain_with_sources.invoke(query)

#     final_response = {
#         "answer": response.get("response", "No answer generated."),
#         "context_texts": [],
#         "context_images": [],
#     }

    
#     # Extract text contexts
#     for text in response.get("context", {}).get("texts", []):
#         text_entry = {}

#         if hasattr(text, "page_content"):
#             text_entry["content"] = text.page_content
#         elif hasattr(text, "text"):
#             text_entry["content"] = text.text
#         else:
#             text_entry["content"] = str(text)

#         metadata = getattr(text, "metadata", {}) or {}
#         text_entry["page_number"] = metadata.get("page_number", None)

#         final_response["context_texts"].append(text_entry)

#     # Extract image contexts
#     for image in response.get("context", {}).get("images", []):
#         # ensure it's base64-encoded string (no display)
#         if isinstance(image, bytes):
#             image_b64 = base64.b64encode(image).decode("utf-8")
#         elif isinstance(image, str):
#             image_b64 = image
#         else:
#             continue
#         final_response["context_images"].append(image_b64)

#     return final_response







# async def process_query(query: str,messages: List[Dict[str, str]], retriever, google_api_key: str) -> Dict[str, Any]:
#     """
#     Process a user query, retrieve relevant context, and query the Gemini multimodal chain.
#     """
#     # Define the refusal phrase (must match the instruction in build_prompt)
#     REFUSAL_PHRASE = "The information is not available in the provided knowledge base."



#     # 1. Handle Conversational/Trivial Queries First
#     trivial_response = None
#     if query.lower() in ['hi', 'hello', 'how are u']:
#         trivial_response = "Hello! I'm your AI assistant. How can I help you today?"
#     elif query.lower() in ['thanks', 'good', 'amazing']:
#         trivial_response = "Welcome! Is there anything else i can help u with?"

#     if trivial_response:
#         return {
#             "answer": trivial_response,
#             "context_texts": [],
#             "context_images": [],
#         }

#     # chain_with_sources = await make_retiever_structure(retriever, google_api_key)
#     chain_with_sources = await make_retiever_structure(retriever, google_api_key,messages)


#     # Run the chain
#     response = chain_with_sources.invoke(query)

#     answer = response.get("response", "No answer generated.")


#     if answer in ["Hello! How can I help you today?",'Welcome! Is there anything else I can help you with?', "Hello! I'm your AI assistant. How can I help you today?"]:
#         final_response = {
#         "answer": answer,
#         "context_texts": [],
#         "context_images": [],
#     }
    
#     # Initialize response structure with the answer
#     final_response = {
#         "answer": answer,
#         "context_texts": [],
#         "context_images": [],
#     }

#     # --- Core Correction Logic: Determine which contexts to process ---
#     if answer.strip().startswith(REFUSAL_PHRASE):
#         # If model refuses, we process EMPTY lists (no context shown)
#         context_texts_to_process = []
#         context_images_to_process = []
#     else:
#         # If model answers, we process the retrieved lists
#         context_texts_to_process = response.get("context", {}).get("texts", [])
#         context_images_to_process = response.get("context", {}).get("images", [])
#     # ------------------------------------------------------------------

#     # Extract text contexts from the determined list (either retrieved or empty)
#     for text in context_texts_to_process:
#         text_entry = {}

#         if hasattr(text, "page_content"):
#             text_entry["content"] = text.page_content
#         elif hasattr(text, "text"):
#             text_entry["content"] = text.text
#         else:
#             text_entry["content"] = str(text)

#         metadata = getattr(text, "metadata", {}) or {}
#         text_entry["page_number"] = metadata.get("page_number", None)

#         final_response["context_texts"].append(text_entry)

#     # Extract image contexts from the determined list (either retrieved or empty)
#     # NOTE: You must have 'import base64' at the top of your file for this to work.
#     for image in context_images_to_process:
#         # ensure it's base64-encoded string (no display)
#         if isinstance(image, bytes):
#             image_b64 = base64.b64encode(image).decode("utf-8")
#         elif isinstance(image, str):
#             image_b64 = image
#         else:
#             continue
#         final_response["context_images"].append(image_b64)

#     # Return the complete structure outside of any conditional block
#     return final_response




async def is_greeting_or_smalltalk(query: str, google_api_key: str) -> tuple[bool, str]:
    """
    Use LLM to determine if query is greeting/small-talk, and generate appropriate response.
    Returns: (is_trivial, response_text)
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=google_api_key,
    )
    
    classification_prompt = f"""Analyze this user message and determine if it's a greeting, farewell, or polite small-talk that doesn't require knowledge base lookup.

Examples of greetings/small-talk: "hi", "hello", "how are you", "thanks", "thank you", "bye", "good morning", "what's up", "hey there", "appreciate it", "see you later"

Examples of knowledge queries: "what is X", "explain Y", "show me Z", "tell me about", "how does", "when did"

User message: "{query}"

Respond in this EXACT format:
CLASSIFICATION: [GREETING or QUERY]
RESPONSE: [If GREETING, provide a natural polite response. If QUERY, write "N/A"]"""

    result = llm.invoke(classification_prompt)
    response_text = result.content.strip()
    
    # Parse the response
    lines = response_text.split('\n')
    classification = ""
    response_msg = ""
    
    for line in lines:
        if line.startswith("CLASSIFICATION:"):
            classification = line.split(":", 1)[1].strip().upper()
        elif line.startswith("RESPONSE:"):
            response_msg = line.split(":", 1)[1].strip()
    
    is_trivial = classification == "GREETING"
    
    return is_trivial, response_msg if is_trivial else ""


async def process_query(query: str, messages: List[Dict[str, str]], retriever, google_api_key: str) -> Dict[str, Any]:
    """
    Process a user query, retrieve relevant context, and query the Gemini multimodal chain.
    """
    REFUSAL_PHRASE = "The information is not available in the provided knowledge base."

    

    # Check if query is greeting/small-talk using LLM
    is_trivial, trivial_response = await is_greeting_or_smalltalk(query, google_api_key)
    
    if is_trivial:
        return {
            "answer": trivial_response,
            "context_texts": [],
            "context_images": [],
        }

    # For knowledge queries, use the retriever chain
    chain_with_sources = await make_retiever_structure(retriever, google_api_key, messages)
    response = chain_with_sources.invoke(query)
    answer = response.get("response", "No answer generated.")

    # Initialize response structure
    final_response = {
        "answer": answer,
        "context_texts": [],
        "context_images": [],
    }

    # Determine which contexts to process
    if answer.strip().startswith(REFUSAL_PHRASE):
        context_texts_to_process = []
        context_images_to_process = []
    else:
        context_texts_to_process = response.get("context", {}).get("texts", [])
        context_images_to_process = response.get("context", {}).get("images", [])

    # Extract text contexts
    for text in context_texts_to_process:
        text_entry = {}
        if hasattr(text, "page_content"):
            text_entry["content"] = text.page_content
        elif hasattr(text, "text"):
            text_entry["content"] = text.text
        else:
            text_entry["content"] = str(text)

        metadata = getattr(text, "metadata", {}) or {}
        text_entry["page_number"] = metadata.get("page_number", None)
        final_response["context_texts"].append(text_entry)

    # Extract image contexts
    for image in context_images_to_process:
        if isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode("utf-8")
        elif isinstance(image, str):
            image_b64 = image
        else:
            continue
        final_response["context_images"].append(image_b64)

    return final_response