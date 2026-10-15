
import json
from langchain_core.prompts import PromptTemplate

with open("12YOUTUBECHATBOT/retrieval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

query = data["query"]
retrieved_docs = data["documents"]

context = "\n\n".join(
    doc["content"] for doc in retrieved_docs
)

prompt_template = PromptTemplate(
    template="""
Answer the question using only the context provided below.

If the answer is not present in the context,
say "I don't know based on the provided transcript."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

augmented_prompt = prompt_template.format(
    context=context,
    question=query
)

with open("12YOUTUBECHATBOT/augmented_prompt.txt", "w", encoding="utf-8") as f:
    f.write(augmented_prompt)

print("Augmentation completed!")
print("\nAugmented Prompt:\n")
print(augmented_prompt)
