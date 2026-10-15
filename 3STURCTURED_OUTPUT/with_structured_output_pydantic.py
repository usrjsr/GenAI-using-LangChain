from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Literal,Optional


load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile',temperature=1.5)

class Review(BaseModel):
    key_themes: list[str]=Field(description='write down all the key themes discussed in the review in a list')
    summary: str=Field(description='A breif summary of the review')
    sentiment: Literal["+ve","-ve"]=Field(description='positive,negative or whther review is neutral')
    pros: Optional[list[str]]=Field(default=None,description='Write down all thr pros inside a list')
    cons: Optional[list[str]]=Field(default=None,description='write down all the cons inside a list')
    name: Optional[str]=Field(default=None,description='Write the name of the reviewer')


structured_model=model.with_structured_output(Review)


result=structured_model.invoke("""I recently bought the NovaSound X200 Wireless Earbuds and I’ve been genuinely impressed so far. The sound quality is crisp with deep bass, and the noise cancellation works surprisingly well for the price. Pairing with my phone took only a few seconds, and the battery easily lasts through an entire workday.

The earbuds are lightweight and comfortable, even after wearing them for hours. I also liked the touch controls, although they can be a little too sensitive sometimes. The charging case feels sturdy and compact enough to carry in a pocket.

Overall, this is a great option for anyone looking for affordable wireless earbuds with premium features. Definitely worth considering if you want solid performance without spending too much money.------""")


print('Key Themes',result.key_themes)
print("Sentiment:",result.sentiment)
print("Summary:",result.summary)
print("Pros:",result.pros)
print("Cons:",result.cons)
print("Name of the reviewer:",result.name)

result=dict(result)

print('Key Themes',result['key_themes'])
print("Sentiment:",result['sentiment'])
print("Summary:",result['summary'])
print("Pros:",result['pros'])
print("Cons:",result['cons'])
print("Name of the reviewer:",result['name'])
