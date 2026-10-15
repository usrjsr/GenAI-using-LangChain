#uses to load and extract text content from webpage URLs
#uses beautifulsoup under the hood to parse the html and extract visible text(used for blogs,news articles or public websites)
#for javascript heavy pages we use SeleniumURLLoader..loads only static content (whats in the html, not what loads after the page renders)

from langchain_community.document_loaders import WebBaseLoader
url1='https://docs.langchain.com/oss/python/integrations/document_loaders'
url2='https://docs.langchain.com/oss/python/integrations/providers/aws'
loader=WebBaseLoader([url1,url2])

docs=loader.lazy_load()

for documents in docs:
    print(documents.page_content)