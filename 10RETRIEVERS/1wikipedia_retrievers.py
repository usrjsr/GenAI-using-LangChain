
import wikipedia

# Set User-Agent BEFORE invoking the retriever
wikipedia.set_user_agent(
    "LangChainLearning/1.0 (student-project)"
)

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "NIT Jamshedpur"
docs = retriever.invoke(query)

for i, doc in enumerate(docs, start=1):
    print(f"\n-------------- Document {i} --------------")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
