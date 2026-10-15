from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
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

prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into a single document\n notes->{notes} and Quiz->{quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()
parallel_chain=RunnableParallel(
    {
        'notes': prompt1|model1|parser,
        'quiz': prompt2|model2|parser
    }
)
#merging the parallel chains
merged_chain=prompt3|model1|parser
chain= parallel_chain|merged_chain

text_file="""
USR JSR
"""
result=chain.invoke({'text': text_file})
chain.get_graph().print_ascii()
print(result)
