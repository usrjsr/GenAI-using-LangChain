#as the cnt touches the length starts new chunk and it goesss and goess...
#no grammar capturing ..word/sentence maybe be cutdown in between

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader

loader=PyPDFLoader('7DOCUMENT_LOADERS/grades/gradecard_till_sem_4.pdf')
docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,#for overalapping between consecutive chunks(beinfit is for covering/reataining the grammar/context/sentiment)
    separator=','
)

result=splitter.split_documents(docs)
print(result[0])
print(result[1])