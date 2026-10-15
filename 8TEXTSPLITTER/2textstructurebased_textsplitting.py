#firstly tries to split on paragraph then on commas then on words then on characters
#recursive then backtrack to capture the chunk size(kindaa merge sort)
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Artificial Intelligence (AI) is a branch of computer science that focuses on building machines and software capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, understanding language, and recognizing images. AI has become an important part of modern technology and is used in industries such as healthcare, education, banking, transportation, and entertainment.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))
print(chunks)