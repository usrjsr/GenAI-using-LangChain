from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Literal,Optional
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key="",
    temperature=0
)

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model=model.with_structured_output(json_schema)


result=structured_model.invoke("""I recently bought the NovaSound X200 Wireless Earbuds and I’ve been genuinely impressed so far. The sound quality is crisp with deep bass, and the noise cancellation works surprisingly well for the price. Pairing with my phone took only a few seconds, and the battery easily lasts through an entire workday.

The earbuds are lightweight and comfortable, even after wearing them for hours. I also liked the touch controls, although they can be a little too sensitive sometimes. The charging case feels sturdy and compact enough to carry in a pocket.

Overall, this is a great option for anyone looking for affordable wireless earbuds with premium features. Definitely worth considering if you want solid performance without spending too much money.------""")


print(result['key_themes'])