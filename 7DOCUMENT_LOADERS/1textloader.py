from langchain_community.document_loaders import TextLoader

loader = TextLoader("7DOCUMENT_LOADERS/cricket.txt", encoding="utf-8")
documents = loader.load()

print(type(documents))
print(type(documents[0]))
print(documents[0].page_content)
print('META DATA:',documents[0].metadata)