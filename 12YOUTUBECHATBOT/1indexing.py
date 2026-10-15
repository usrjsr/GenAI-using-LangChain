from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

video_id="Gfr50f6ZBvo"
try:
    transcript_list=YouTubeTranscriptApi().fetch(video_id,languages=['en'])
    transcript=" ".join(chunk.text for chunk in transcript_list)
    print(transcript)
except TranscriptsDisabled:
    print("No caption avaialble for this video")

splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks=splitter.create_documents([transcript])
print("Total chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=FAISS.from_documents(chunks,embeddings)
vector_store.save_local("12YOUTUBECHATBOT/faiss_index")
print("Indexing successfull")


