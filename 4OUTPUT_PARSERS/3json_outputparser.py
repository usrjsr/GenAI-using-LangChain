from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile',temperature=1.5)
parser=JsonOutputParser()

template1=PromptTemplate(
    template="Write the details on {topic} \n {format_instructions}",
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