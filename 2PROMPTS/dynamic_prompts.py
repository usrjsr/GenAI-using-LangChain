from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
# from prompt_generator import template
from langchain_core.prompts import load_prompt,PromptTemplate


load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile')
st.header('Research Tool')

research_title = st.text_input(
    "📄 Enter Research Paper Title",
    placeholder="e.g. Attention Is All You Need"
)

research_words = st.number_input(
    "📝 Enter Summary Length (in words)",
    min_value=50,
    max_value=5000,
    value=300,
    step=50
)

research_explanation = st.selectbox(
    "🧠 Select Explanation Style",
    [
        "Beginner Friendly",
        "Intermediate",
        "Technical",
        "Mathematical / Research-Level"
    ]
)

template=load_prompt('template.json')

# prompt=template.invoke({
#     'research_title':research_title,
#     'research_words':research_words,
#     'research_explanation':research_explanation}
# )

# if st.button('Summarize'):
#     result = model.invoke(prompt)
#     st.write(result.content)

# ##########CREATING CHAIN###########

if(st.button('Summarize')):
    chain=template | model
    result=chain.invoke({
        'research_title':research_title,
        'research_words':research_words,
        'research_explanation':research_explanation
    })
    st.write(result.content)