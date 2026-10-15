from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile',temperature=1.5)
parser=JsonOutputParser()

schema= [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
    ResponseSchema(name='fact_4', description='Fact 4 about the topic')
]
#we can enforce schema but not the data validation whether the returned object is string or int or whatever...this is the thing we cant enforce the data validation(age>18 and all 18 is integer or not ..this type of data validation we cant do even after enfrocong schema so pydantic object is used)
template1=PromptTemplate(
    template="Write four facts on the {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={'format_instructions':parser.get_format_instructions()})


# prompt=template1.invoke({'topic':'NIT Jamshedpur'})
# result= model.invoke(prompt)
# final_result=parser.parse(result.content)
# print(final_result)


#--------------THROUGH CHAINS---------
chain= template1 | model | parser
result=chain.invoke({'topic':'NIT Jamshedpur'})
print(result)