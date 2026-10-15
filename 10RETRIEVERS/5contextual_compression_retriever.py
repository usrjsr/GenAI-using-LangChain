#advance retriever that improves retrieval quality by compressing documents after retrieveal-kepping only the relevant content based on the user's query


#k=2 so 2 docs
#query->retriever->Doc1+query(again)->LLM->doc1
#                ->Doc2+query(again)->LLM->doc2

#llm acts as compressor which keeps the only part which is related to the query


from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

load_dotenv()

# Recreate the document objects
docs = [
    Document(
        page_content="""The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years.""",
        metadata={"source": "Doc1"}
    ),

    Document(
        page_content="""In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls.""",
        metadata={"source": "Doc2"}
    ),

    Document(
        page_content="""Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league.""",
        metadata={"source": "Doc3"}
    ),

    Document(
        page_content="""The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design.""",
        metadata={"source": "Doc4"}
    )
]

# Create Gemini embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="compression_demo",
    persist_directory="./compression_chroma_db"
)

# Create base retriever
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Create the compressor
compressor = LLMChainExtractor.from_llm(llm)

# Create contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Query the retriever
query = "What is photosynthesis?"

compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
