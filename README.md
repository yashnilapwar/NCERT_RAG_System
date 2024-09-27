
# Part 1: Smart AI Query System

This project demonstrates the implementation of a FastAPI-based system that combines FAISS (Facebook AI Similarity Search) for vector database search and a Large Language Model (LLM) API from Cohere to respond to user queries. The application can search for relevant content from a document, perform similarity searches, and generate meaningful responses.

## Project Flow

1. **PDF Extraction & Chunking**:
   - The project first extracts text from a given PDF (`iesc111.pdf`), cleans it by removing unwanted characters, and splits the cleaned text into smaller chunks. These chunks form the basis of the searchable database.

2. **Embedding Creation**:
   - The application uses a pre-trained Sentence-BERT model (`paraphrase-MiniLM-L6-v2`) to generate embeddings for the text chunks. These embeddings are stored in a FAISS index to perform efficient similarity searches.

3. **FAISS Search**:
   - When a user sends a query, the system encodes it into an embedding and searches the FAISS index for the most relevant text chunks. The system retrieves the top-k most similar chunks.

4. **LLM API**:
   - If the system finds relevant content from the FAISS search, it sends the user query along with the relevant content to the LLM API (Cohere). If no content is found, only the user query is sent to the LLM to generate a response.

5. **Frontend Integration**:
   - The system integrates with a simple frontend where users can input their queries and view the generated responses from the backend.

## LLM API Used

We use the **Cohere API** in this project to generate responses to user queries. The model used is `command-r-plus`, which is suitable for generating coherent and contextual responses.

- API: Cohere's LLM API
- Model: `command-r-plus`
- Function: The LLM generates responses based on the user’s query, and when applicable, the relevant chunks retrieved from the FAISS index.

## How to Run the Project

### Prerequisites

Ensure you have Python 3.9+ installed and create a virtual environment.

### Steps to Run:

1. **Clone the Repository**:
   ```bash
   git clone <your-github-repo-url>
   cd <project-directory>
   ```

2. **Install Dependencies**:
   Install all the required packages by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Cohere API**:
   You will need to set your Cohere API key as an environment variable. Create an account at [Cohere](https://cohere.ai/) and get your API key.

   ```bash
   export COHERE_API_KEY='your-cohere-api-key'
   ```

4. **Run the FastAPI App**:
   Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the API**:
   - You can test the API by visiting `http://127.0.0.1:8000/docs` in your browser.
   - You can also use Postman to test the API by sending a POST request to the `/query/` endpoint with a JSON body containing the `query` field.

### Project Structure:

```
- Part 1/
  - env/                 # Python virtual environment (not included in Git)
  - iesc111.pdf          # PDF document used for the vector search
  - index.html           # Frontend HTML file
  - __pycache__/         # Python cache (not included in Git)
  - main.py              # Main FastAPI backend file
  - script1.py           # Backend script handling chunking and embedding
  - requirements.txt     # Project dependencies
  - part1_screenshot/
      - postman response.png  # Screenshot of Postman test
      - Frontend.png          # Screenshot of frontend
```

## Screenshots

### 1. Postman Response

Here’s a screenshot of testing the FastAPI `/query/` endpoint using Postman:

![Postman Response](Part%201/postman%20response.png)

### 2. Frontend Interaction

Here’s a screenshot of the frontend where users can input queries and get responses:

![Frontend Interaction](Part%201/Frontend.png)

## Conclusion

This project/Assignment demonstrates how to build a smart AI query system by combining FAISS, Cohere's LLM, and FastAPI. The system allows users to search documents efficiently and generate meaningful responses using AI.
