# Import necessary libraries
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import cohere
import os
import pdfplumber
import re
from fastapi.middleware.cors import CORSMiddleware


# Initialize the Cohere client
API_KEY='API_KEY'
cohere_client = cohere.Client(API_KEY)

# Load FAISS index from disk
def load_faiss_index(index_path="ncert_index.faiss"):
    return faiss.read_index(index_path)

# Generate an embedding for the user's query
def get_query_embedding(query, model):
    return model.encode([query])[0]

# Perform similarity search using FAISS
def search_faiss(query_embedding, faiss_index, chunks, top_k=5):
    query_embedding = np.expand_dims(query_embedding, axis=0)  # Reshape to 2D array
    distances, indices = faiss_index.search(query_embedding, top_k)  # Search the top_k closest chunks

    # Handle the case where no results are found
    if len(indices) == 0 or len(indices[0]) == 0:
        return []

    # Retrieve the original text chunks
    results = [chunks[idx] for idx in indices[0] if idx < len(chunks)]  # Ensure valid indices
    return results

# Function to generate content using Cohere's API
def generate_cohere_content(prompt):
    try:
        # Send the prompt to Cohere and get the response
        response = cohere_client.generate(
            model='command-r-plus',  # Choose the model size
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return str(e)

# FastAPI app initialization
app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to the specific domain if required, e.g., "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model to parse user input
class QueryRequest(BaseModel):
    query: str

# Load the FAISS index and chunks at startup
faiss_index = load_faiss_index()

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Clean the extracted text by removing unwanted characters
def clean_text(text):
    clean_text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    return clean_text.strip()

# Split the cleaned text into chunks of a specified size
def chunk_text(text, chunk_size=100):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Extract and clean text from the PDF
ncert_text = extract_text_from_pdf("iesc111.pdf")
cleaned_ncert_text = clean_text(ncert_text)

# Chunk the cleaned text
chunks = chunk_text(cleaned_ncert_text, chunk_size=100)

# Load Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# FastAPI route to handle user queries and return results
@app.post("/query/")
async def query_faiss(request: QueryRequest):
    try:
        # Generate the query embedding
        query_embedding = get_query_embedding(request.query, model)

        # Search the FAISS index
        top_k_results = search_faiss(query_embedding, faiss_index, chunks, top_k=3)  # Retrieve top 3 matches

        if not top_k_results:
            # If no matching content is found, send the user query alone to Cohere
            prompt = f"User query: {request.query}\n"
            generated_text = generate_cohere_content(prompt)
            return {"generated_text": generated_text}

        # Combine the top results from FAISS with the user query
        combined_prompt = f"User query: {request.query}\n\nRelevant content from database:\n"
        combined_prompt += "\n".join(top_k_results)  # Append the top K results to the prompt

        # Use the combined prompt to generate content via Cohere
        generated_text = generate_cohere_content(combined_prompt)

        return {"top_k_results": top_k_results, "generated_text": generated_text}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
