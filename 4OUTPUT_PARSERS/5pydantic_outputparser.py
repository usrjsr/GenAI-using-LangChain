from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import PydanticOutputParser, ResponseSchema, StructuredOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key="",
    temperature=0
)


class Person(BaseModel):
    name: str= Field(description='Name of the Person')
    age: int= Field(description='age of the person', gt=18)
    city: str=Field(description='name of the city')

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template='Generate the name,age and city of a fictinal {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt=template.invoke({'place': 'indian'})
result=model.invoke(prompt)
final_Result=parser.parse(result.content)
print(final_Result)
#chain
chain= template | model | parser
final_Result=chain.invoke({'place': 'indian'})
