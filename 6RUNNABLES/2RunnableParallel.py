from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
load_dotenv()


model1=ChatOpenAI(model='llama-3.3-70b-versatile',temperature=1.5)
model2=ChatOpenAI(model='llama-3.3-70b-versatile',temperature=1.5)

prompt1= PromptTemplate(
    template='Generate short notes on the following text {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate 5 short question based on the {text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain=RunnableParallel({
        'notes': RunnableSequence(prompt1|model1|parser),
        'quiz': RunnableSequence(prompt2|model2|parser)
    }
)

result=chain.invoke({'topic':'AI'})
print(result['notes'],result['quiz'])