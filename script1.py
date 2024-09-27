# Import necessary libraries
import pdfplumber
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

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

# Generate embeddings for the text chunks
def generate_embeddings(chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Load pre-trained Sentence-BERT model
    return model.encode(chunks)

# Store the embeddings in a FAISS index for similarity search
def store_embeddings_in_faiss(embeddings):
    dimension = embeddings.shape[1]  # Number of dimensions in the embeddings
    index = faiss.IndexFlatL2(dimension)  # Create an index for L2 (Euclidean) similarity
    index.add(np.array(embeddings))  # Add embeddings to the FAISS index
    return index

if __name__ == "__main__":
    # Path to the NCERT PDF
    ncert_pdf = 'iesc111.pdf'

    # Step 1: Extract text from the NCERT PDF
    ncert_text = extract_text_from_pdf(ncert_pdf)

    # Step 2: Clean the extracted PDF text
    cleaned_ncert_text = clean_text(ncert_text)

    # Step 3: Create chunks of 100 words from the cleaned text
    chunks = chunk_text(cleaned_ncert_text, chunk_size=100)

    # Step 4: Generate embeddings for the chunks
    embeddings = generate_embeddings(chunks)
    print(f"Generated {embeddings.shape[0]} embeddings.")

    # Step 5: Create and store the FAISS index
    faiss_index = store_embeddings_in_faiss(embeddings)

    # Step 6: Save FAISS index to disk
    faiss.write_index(faiss_index, "ncert_index.faiss")
    print("FAISS index saved as 'ncert_index.faiss'.")
