# -*- encode = utf-8 -*-
# @Time : 2026-10-06 10:50:38
# @Auther : 罗雄波
# @File : company_api_key_test.py
# @Software : PyCharm

import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()


async def main():
    host = os.getenv("LLM_BINDING_HOST").rstrip("/")
    key = os.getenv("LLM_BINDING_API_KEY")

    url = host + "/responses"

    headers = {
        "Authorization": f"Bearer {key.strip()}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "glm-5.3",
        "input": [
            {
                "role": "user",
                "content": "只回答 OK"
            }
        ],
        "reasoning": {
            "effort": "low"
        },
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )

        print("status:", response.status_code)
        print("body:", response.text)


asyncio.run(main())
