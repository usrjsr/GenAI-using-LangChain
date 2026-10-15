from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile',temperature=1.5)

template1=PromptTemplate(template="Write a detail report on {topic}",
                    input_variables=["topic"])

template2=PromptTemplate(template="Write a 5 line summary on the following text. /n {text}",
                         input_variables=["text"])

prompt1=template1.invoke({'topic':'BlackHole'})

result=model.invoke(prompt1)

prompt2=template2.invoke({'text':result.content})

final_result=model.invoke(prompt2)

print(final_result.content)
