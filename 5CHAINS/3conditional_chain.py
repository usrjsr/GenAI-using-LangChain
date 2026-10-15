#analzying sentiment of the feedback(replying according to the +ve/-ve feedback

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel,RunnableBranch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative']=Field(description='Give the sentiment of the feedback')

parser2=PydanticOutputParser(pydantic_object=Feedback)
prompt1=PromptTemplate(
    template='Classify the sentiments of the following feeback text into positive or negative: feedback->{feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)


prompt2=PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)
classifier_chain=prompt1|model|parser
# result=classifier_chain.invoke({'feedback':'terrible food'}).sentiment // this is what the above chain does
branched_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model|parser),
    (lambda x:x.sentiment=='negative',prompt2|model|parser),
    RunnableLambda(lambda x: "couldn't find any sentiments")
)

chain=classifier_chain|branched_chain
result=chain.invoke({'feedback':'terrible food'})
print(result)