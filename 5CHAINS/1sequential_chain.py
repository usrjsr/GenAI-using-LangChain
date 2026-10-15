from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatOpenAI(model='llama-3.3-70b-versatile',temperature=1.5)

template1=PromptTemplate(template="Write a detail report on {topic}",
                    input_variables=["topic"])

template2=PromptTemplate(template="Write a 5 line summary on the following text. /n {text}",
                         input_variables=["text"])
parser=StrOutputParser()

chain =template1|model|parser|template2|model|parser

result=chain.invoke({'topic': 'Unemployement in india'})
chain.get_graph().get_ascii()
print(result)
