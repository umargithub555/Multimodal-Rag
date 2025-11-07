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
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Dict,Any,AsyncGenerator


nest_asyncio.apply()





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



# ---- Function to build the multimodal prompt ----
# def build_prompt(kwargs):
#     docs_by_type = kwargs["context"]
#     user_question = kwargs["question"]

#     context_text = ""
#     if len(docs_by_type["texts"]) > 0:
#         for text_element in docs_by_type["texts"]:
#             # if text elements are Document objects
#             if hasattr(text_element, "page_content"):
#                 context_text += text_element.page_content
#             elif hasattr(text_element, "text"):
#                 context_text += text_element.text
#             else:
#                 context_text += str(text_element)

#     # Base text part of prompt
#     prompt_template = f"""
#     Answer the question based only on the following context, which can include text, tables, and images.

#     Context: {context_text}
#     Question: {user_question}
#     """

#     # The prompt content (text + base64 images)
#     prompt_content = [{"type": "text", "text": prompt_template}]

#     if len(docs_by_type["images"]) > 0:
#         for image in docs_by_type["images"]:
#             prompt_content.append(
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{image}"},
#                 }
#             )

#     # Return ChatPromptTemplate
#     return ChatPromptTemplate.from_messages(
#         [
#             HumanMessage(content=prompt_content),
#         ]
#     )

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    # --- START OF MODIFICATION ---
    # 1. Define the strict refusal instruction
    system_instruction = """
    Your answer MUST be based SOLELY on the provided Context: text, tables, and images. 
    If the context does not contain the answer, you MUST respond with the exact and complete phrase: 
    "The information is not available in the provided knowledge base." 
    DO NOT mention based on the context or provided context or any other wording in the response.
    DO NOT use your general knowledge. Be concise.
    """
    # --- END OF MODIFICATION ---

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            # if text elements are Document objects
            if hasattr(text_element, "page_content"):
                context_text += text_element.page_content
            elif hasattr(text_element, "text"):
                context_text += text_element.text
            else:
                context_text += str(text_element)

    # Base text part of prompt now just includes the context and question
    prompt_template = f"""
    Context: {context_text}
    Question: {user_question}
    """

    # The prompt content (text + base64 images)
    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    # Return ChatPromptTemplate with the new SystemMessage
    # You will need to ensure SystemMessage and HumanMessage are imported 
    # (e.g., from langchain_core.messages import SystemMessage, HumanMessage)
    return ChatPromptTemplate.from_messages(
        [
            # --- START OF MODIFICATION ---
            SystemMessage(content=system_instruction),
            # --- END OF MODIFICATION ---
            HumanMessage(content=prompt_content),
        ]
    )




async def make_retiever_structure(retriever,google_api_key):
        
        llm1 = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # or gemini-1.5-pro for better reasoning
        temperature=0.3,
        google_api_key=google_api_key,
        )


        # ---- Define the runnable chain ----
        chain_rebuild = (
            {
                "context": retriever | RunnableLambda(parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(build_prompt)
            | llm1
            | StrOutputParser()
        )


        # ---- Optional: Chain that also returns sources ----
        chain_with_sources_rebuild = (
            {
                "context": retriever | RunnableLambda(parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough().assign(
                response=(RunnableLambda(build_prompt) | llm1 | StrOutputParser())
            )
        )

        print("✅ Gemini multimodal chain ready!")

        return chain_with_sources_rebuild




async def process_images(images, google_api_key):
    prompt_template = """Describe the image in detail. For context,
                     the image is part of a research paper explaining the transformers
                        architecture. Be specific about graphs, such as bar plots."""
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
        collection_name="multi_modal_rag",
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







async def process_query(query: str, retriever, google_api_key: str) -> Dict[str, Any]:
    """
    Process a user query, retrieve relevant context, and query the Gemini multimodal chain.
    """
    # Define the refusal phrase (must match the instruction in build_prompt)
    REFUSAL_PHRASE = "The information is not available in the provided knowledge base."

    chain_with_sources = await make_retiever_structure(retriever, google_api_key)

    # Run the chain
    response = chain_with_sources.invoke(query)

    answer = response.get("response", "No answer generated.")
    
    # Initialize response structure with the answer
    final_response = {
        "answer": answer,
        "context_texts": [],
        "context_images": [],
    }

    # --- Core Correction Logic: Determine which contexts to process ---
    if answer.strip().startswith(REFUSAL_PHRASE):
        # If model refuses, we process EMPTY lists (no context shown)
        context_texts_to_process = []
        context_images_to_process = []
    else:
        # If model answers, we process the retrieved lists
        context_texts_to_process = response.get("context", {}).get("texts", [])
        context_images_to_process = response.get("context", {}).get("images", [])
    # ------------------------------------------------------------------

    # Extract text contexts from the determined list (either retrieved or empty)
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

    # Extract image contexts from the determined list (either retrieved or empty)
    # NOTE: You must have 'import base64' at the top of your file for this to work.
    for image in context_images_to_process:
        # ensure it's base64-encoded string (no display)
        if isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode("utf-8")
        elif isinstance(image, str):
            image_b64 = image
        else:
            continue
        final_response["context_images"].append(image_b64)

    # Return the complete structure outside of any conditional block
    return final_response