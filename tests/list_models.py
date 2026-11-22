# -*- encode = utf-8 -*-
# @Time : 2026-11-22 10:37:03
# @Auther : 罗雄波
# @File : list_models.py
# @Software : PyCharm

import json
import os
import sys
from getpass import getpass
from urllib.request import Request, urlopen
from dotenv import load_dotenv

load_dotenv(override=True)

if __name__ == "__main__":
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://sub2api.sensorsdata.cn")
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key and sys.stdin.isatty():
        api_key = getpass("请粘贴 API key 后按回车（输入不会显示）：").strip()
    if not api_key:
        raise SystemExit("请设置 ANTHROPIC_API_KEY，或在 PowerShell 终端运行脚本后输入 key。")

    url = f"{base_url.rstrip('/').removesuffix('/v1')}/v1/models"
    request = Request(url, headers={"anthropic-version": "2023-06-01"})
    request.add_unredirected_header("x-api-key", api_key)
    print("正在请求网关的模型列表…", file=sys.stderr, flush=True)
    try:
        with urlopen(request, timeout=20) as response:
            models = json.load(response)["data"]
    except OSError as error:
        raise SystemExit(f"查询模型失败：{error}") from None

    if not models:
        print("网关返回了空模型列表。", file=sys.stderr)
    for model in models:
        print(model["id"])
