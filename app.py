from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *
import os
from google import genai
from google.genai import types


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Initialize Gemini client
if not GEMINI_API_KEY:
    raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file")

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("Successfully created Gemini client")
except Exception as e:
    print("Warning creating Gemini client:", e)
    gemini_client = None


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

def gemini_answer_from_context(question, context_docs, model: str = "gemini-1.5-flash"):
    """Ask Gemini a question with the retrieved context_docs (list of Documents).
    Returns text answer (best candidate) or raw response as fallback.
    """
    if gemini_client is None:
        raise RuntimeError("Gemini client is not initialized. See earlier warning.")

    context = "\n\n".join([d.page_content for d in context_docs]) if context_docs else ""
    system_text = system_prompt.format(context=context)
    
    # Create the content using the correct API structure
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"{system_text}\n\nQuestion: {question}"),
            ],
        ),
    ]
    
    # Generate content using the correct API method
    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        # Extract text from response
        if response.candidates and response.candidates[0].content:
            return response.candidates[0].content.parts[0].text
        else:
            return "No response generated"
            
    except Exception as e:
        print(f"Error generating content: {e}")
        return f"Error: {str(e)}"

def run_gemini_rag(question: str, k: int = 3, model: str = "gemini-1.5-flash"):
    """Retrieve top-k documents using the existing `retriever`, then ask Gemini for an answer.
    Returns a dict with answer, sources (metadata.source), and the raw docs.
    """
    # Retrieve documents
    docs = []
    try:
        if hasattr(retriever, "get_relevant_documents"):
            try:
                docs = retriever.get_relevant_documents(question)
            except TypeError:
                docs = retriever.get_relevant_documents(question, k=k)
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(question)
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(question)
        else:
            docs = retriever(question)
    except Exception as e:
        raise RuntimeError(f"Error retrieving documents: {e}")

    # Ensure docs is a list
    if docs is None:
        docs = []

    # Ask Gemini for the answer using the assembled context
    answer = gemini_answer_from_context(question, docs, model=model)

    return {"answer": answer, "sources": [d.metadata.get("source") if getattr(d, 'metadata', None) else None for d in docs], "docs": docs}



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User question: {input}")
    
    try:
        response = run_gemini_rag(question=msg, k=3)
        print(f"Gemini response: {response['answer']}")
    return str(response["answer"])
    except Exception as e:
        print(f"Error in chat: {e}")
        return f"Sorry, I encountered an error: {str(e)}"



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
