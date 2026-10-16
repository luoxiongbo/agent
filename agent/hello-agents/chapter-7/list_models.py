import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["MODELSCOPE_API_KEY"],
    base_url="https://api-inference.modelscope.cn/v1"
)

models = client.models.list()

for model in models.data:
    print(model.id)