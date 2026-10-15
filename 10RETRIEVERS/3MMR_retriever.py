#maxium marginal relevance
#how can we pick the results that are not only relevant to the query but also different frm each other
#picks most relevant first doc...then second most relevant not similar to the first doc

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
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
    search_type='mmr',
    search_kwargs={"k": 2,"lambda_mult":0.5}#lambda=1 then it behaves as similarit search
)

# Step 6: Search relevant documents
query = "What is langchain?"
results = retriever.invoke(query)

# Step 7: Print results
for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
