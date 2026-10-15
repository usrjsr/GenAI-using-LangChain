import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=FAISS.load_local(
    "12YOUTUBECHATBOT/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever=vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k':4}
)

query = input("Ask your question: ")

retrieved_docs=retriever.invoke(query)

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nChunk {i}:\n")
    print(doc.page_content)

print("\nRetrieval completed!")

results = []

for doc in retrieved_docs:
    results.append({
        "content": doc.page_content,
        "metadata": doc.metadata
    })

with open("12YOUTUBECHATBOT/retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "query": query,
        "documents": results
    }, f, ensure_ascii=False, indent=4)

print("Query and retrieved documents saved successfully!")