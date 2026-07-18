# imports 
import google.generativeai as genai
import os 
from dotenv import load_dotenv
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
import faiss
import numpy as np 
GLOBAL_CHUNKS=None
GLOBAL_INDEX=None


#  authentication 
load_dotenv()
api_key=os.getenv("api_key")
genai.configure(api_key=api_key)

# system prompt to determine state behaviour
SYSTEM_PROMPT = """
You are a helpful and friendly AI assistant.
Use the provided CONTEXT to answer questions about the user's documents.
If the context does not contain the answer, use your general knowledge to answer politely and clearly.
"""

# file loader functions to load the files
def read_pdf(path):
    text = ""
    with open(path,"rb") as f:
        reader=PyPDF2.PdfReader(f)
        for page in reader.pages:
            text=text+page.extract_text() + "\n"
    return text


def read_word(path):
    doc=docx.Document(path)
    text="\n".join([para.text for para in doc.paragraphs])
    return text


def load_text(path):
    with open(path,"r",encoding="utf-8") as f:
        return f.read()

# universal loader
def load_file_text(path):
    ext=path.lower().split(".")[-1]
    if ext == "txt":
        return load_text(path)
    elif ext == "pdf":
        return read_pdf(path)
    elif ext == "docx":
        return read_word(path)
    else:
        raise ValueError("Unsupported file type:" + ext)


# chunking of text
def chunk_data(text,chunk_size=500,overlap=100):
    chunks=[]
    start=0
    max=len(text)

    while (start<max):
        end=start + chunk_size
        chunk=text[start:end]
        chunks.append(chunk.strip())
        start=end-overlap
    return chunks


# embedding of text
def embed_chunks(chunks):
    embeddings=embedder.encode(chunks)
    return embeddings


# faiss index creation 
def create_faiss_index(chunks,embeddings):
    global GLOBAL_CHUNKS,GLOBAL_INDEX
    embeddings=np.array(embeddings).astype('float32')
    dim=embeddings.shape[1]

    index=faiss.IndexFlatL2(dim)
    index.add(embeddings)
    GLOBAL_CHUNKS=chunks
    GLOBAL_INDEX=index
    # print("faiss index created with",len(chunks),"chunks")


# retrieve chunks relevant to user query from faiss
def retrieve_relevant_chunks(query,top_k=3):
    global GLOBAL_CHUNKS, GLOBAL_INDEX, embedder

    query_vector=embedder.encode([query]).astype("float32")
    distance,indices=GLOBAL_INDEX.search(query_vector,top_k)

    results=[]
    for idx in indices[0]:
        results.append(GLOBAL_CHUNKS[idx])
    return results


# #  check
# text=load_file_text("makerspace.docx")
# chunks=chunk_data(text)
# embeddings=embed_chunks(chunks)
# # print(embeddings)
# # print(len(embeddings),len(embeddings[0]))
# create_faiss_index(chunks,embeddings)


# build_prompt()
def build_prompt(history,user_message,max_history_turns=6,extra_context=None):
    parts=[SYSTEM_PROMPT.strip()]
    # add rag context if it exists
    if extra_context:
        parts.append("CONTEXT FROM DOCUMENT:")
        parts.append("Note: Use the context only if relevant; otherwise answer from general knowledge.")
        for c in extra_context:
            parts.append(c)
            parts.append("")

   # add chat history and finally user query
    trim=history[-max_history_turns:] if history else []
    
    for role,text in trim:
        if role.lower()=="user":
            parts.append(f"\nUser: {text.strip()}")
        else:
            parts.append(f"\nAssistant: {text.strip()}")
     
            
    parts.append(f"\nUser: {user_message.strip()}")
    parts.append("\nAssistant:")

    final_prompt="\n".join(parts)
    return final_prompt


# ask gemini
def ask_gemini(prompt):
 model = genai.GenerativeModel("gemini-2.5-flash")  #select your model
 response=model.generate_content(prompt)
 return response.text

# loop
# history=[]
# while True:
#    User=input("USER:").strip()
#    if User.lower() in ["exit","quit"]:
#       break

   # build the final prompt
def build_index_from_file(path):
    text = load_file_text("makerspace.docx")
    chunks = chunk_data(text)
    embeddings = embed_chunks(chunks)
    create_faiss_index(chunks, embeddings)

def answer_query(query, history):
    retrieved = retrieve_relevant_chunks(query)
    final_prompt = build_prompt(history, query, extra_context=retrieved)
    reply = ask_gemini(final_prompt)
    return reply


# save into history
# history.append(("user",User))
# history.append(("Assistant",reply))