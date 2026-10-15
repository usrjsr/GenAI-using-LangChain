from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimension=32)
document=[
    "Delhi is the capital of India",
    "Kolkata is capital of West Bengal"
]
result=embedding.embed_documents('Delhi is the capital of India')
print(str(result))