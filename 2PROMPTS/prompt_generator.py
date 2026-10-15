from langchain_core.prompts import PromptTemplate

template=PromptTemplate(
    template="""
You are an expert AI Research Paper Summarizer.

Research Paper Title:
"{research_title}"

Your task:
1. Summarize ONLY this research paper.
2. Generate approximately {research_words} words.
3. Explanation style: {research_explanation}
4. Explain:
   - Core Idea
   - Methodology
   - Results
   - Conclusion
5. Use proper headings and bullet points.
6. Do NOT create your own research paper title.
7. Do NOT say 'Since no title was provided'.
8. Provide Image and also flowchart wherever required

Important:
The paper title already exists and must be used exactly as given.
""",
input_variables=['research_title','research_words','research_explanation'],
validate_template=True
)

template.save('template.json')
