#two different topic merged together so chunking must be done on the semantic meaning so that the embedding genearted to be better
#sliding window chunking are done as soon as the consectuive chunk deviates from each other and the deciding factor that similarity is less than this threshold then do the split(i.e decided by standard deviation)


from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv



load_dotenv()

text = """
Artificial intelligence is transforming the world.
Machine learning is a subset of artificial intelligence.
It enables computers to learn patterns from data.

Cricket is a popular sport played between two teams.
Each team has eleven players.
The objective is to score more runs than the opponent.

Python is a popular programming language.
It is widely used in data science and web development.
Python has a simple and readable syntax.
"""

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)
    print("-" * 50)

