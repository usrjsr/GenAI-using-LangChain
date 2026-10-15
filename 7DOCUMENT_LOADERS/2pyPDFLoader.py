#laod content from pdf file and convert each page into a document object
#not great with scanned pdfs of complex layouts

from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('7DOCUMENT_LOADERS/grades/gradecard_till_sem_4.pdf')
docs=loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)
print(docs[1].page_content)

#for scanned pdf & all read: https://docs.langchain.com/oss/python/integrations/document_loaders
