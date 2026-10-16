
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["MODELSCOPE_API_KEY"],
    base_url="https://api-inference.modelscope.cn/v1"
)

response = client.chat.completions.create(
    model=os.getenv("MODELSCOPE_MODEL_ID"),
    messages=[
        {"role": "user", "content": "介绍一下AI Agent"}
    ]
)

print(response.choices[0].message.content)
