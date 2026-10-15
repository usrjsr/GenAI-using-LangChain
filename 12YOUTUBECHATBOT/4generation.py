
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

with open("12YOUTUBECHATBOT/augmented_prompt.txt", "r", encoding="utf-8") as f:
    augmented_prompt = f.read()

model = ChatGoogleGenerativeAI(
    model="models/gemini-3.8-flash",
    temperature=0
)

response = model.invoke(augmented_prompt)

print("\nFinal Answer:\n")
print(response.content)
