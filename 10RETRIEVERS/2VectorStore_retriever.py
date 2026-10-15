
# Searching and fetching documents from a vector store
# using vector embeddings and semantic similarity

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Step 3: Create vector store
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="retrieve_chroma_db",
    collection_name="sample"
)

# Step 4: Add documents to vector store
vector_store.add_documents(documents)

# Step 5: Convert vector store into a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# Step 6: Search relevant documents
query = "What is Chroma used for?"
results = retriever.invoke(query)

# Step 7: Print results
for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
