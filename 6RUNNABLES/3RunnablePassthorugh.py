#returns input as output
#want to acess the middle model output as next model changes the output so we'll create a parallel chain with RunnablePassthorugh for keeping the same output and other parallel one will goes for other putput
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence,RunnablePassthrough
load_dotenv()

model=ChatOpenAI(model='llama-3.3-70b-versatile',temperature=1.5)

template1=PromptTemplate(template="Write a detail report on {topic}",
                    input_variables=["topic"])

template2=PromptTemplate(template="Write a 5 line summary on the following text. /n {text}",
                         input_variables=["text"])
parser=StrOutputParser()

report_chain=RunnableSequence(template1,model,parser)
parallel_chain=RunnableParallel({
    'report':RunnablePassthrough(),
    'summary': RunnableSequence(template2|model|parser)
})

chain=RunnableSequence(report_chain,parallel_chain)
result=chain.invoke({'topic':'AI'})

print(result['report'], result['summary'])