#sometimes a single query might not capture all the ways information is phrased in your document
#how can i stay healthy could meann about food,exercise and all

#queryyy sent to LLM and it generates multiple queries from that ambiguous queries and its sent to retriever and they answer according to the generated multiple queries


# MultiQueryRetriever using Chroma and Gemini

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever

load_dotenv()

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Gemini embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding_model,
    collection_name="multiquery_demo",
    persist_directory="./multiquery_chroma_db"
)

# Create similarity retriever
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Create MultiQueryRetriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 5}
    ),
    llm=llm
)

# Query
query = "How to improve energy levels and maintain balance?"

# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

print("NORMAL SIMILARITY SEARCH")

for i, doc in enumerate(similarity_results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)

print("*" * 100)

print("MULTI QUERY SEARCH")

for i, doc in enumerate(multiquery_results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
