
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "12YOUTUBECHATBOT/faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

model = ChatGoogleGenerativeAI(
    model="models/gemini-3.8-flash",
    temperature=0
)

prompt = PromptTemplate(
    template="""
Answer the question using only the context provided below.

If the answer is not present in the context,
say "I don't know based on the provided transcript."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
)
parser=StrOutputParser()

chain = (parallel_chain| prompt| model| parser)

print("YouTube Chatbot is ready!")
print("Type 'exit' to quit.\n")

while True:
    query = input("You: ")

    if query.lower().strip() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break

    if not query.strip():
        continue

    response = chain.invoke(query)

    print("\nChatbot:", response, "\n")
