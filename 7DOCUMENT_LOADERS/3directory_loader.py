from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader=DirectoryLoader(
    path='7DOCUMENT_LOADERS/grades',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
#docs=loader.load()
docs=loader.lazy_load()

# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)
# print(docs[7].page_content)
# for document in docs:
#     print(document.metadata)

#what if 100 of pdfs then normal load wont work it'll take muchh more time..
#normallyy load() is eager loading means loading all at that time
#so lazy_load() returns generator of document objects...documents are not laoded at once..
# thet're fetched one at a time its needed...
# used when dealing with large documents or files


for document in docs:
    print(document.metadata)