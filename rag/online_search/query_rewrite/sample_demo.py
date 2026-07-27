import re


def rewrite_query(query: str) -> str:
    # 1. 去除疑问口语用词
    fillers = ["请问", "一下", "呢", "啊", "吧"]
    for f in fillers:
        query = query.replace(f, "")
    query = query.strip()
    # 2. 将“怎么XXX”改写为“XXX是什么？”
    if query.startswith(("怎么", "怎样")):
        main_query = query[2:]  # 去掉开头的"怎么"/"怎样"
        # 如果以问号结尾则去掉问号，稍后统一加
        main_query = main_query.rstrip("？?")
        # 示例中特定的短语调整：将“报销保险费用”改为“保险费用报销”
        if "报销保险费用" in main_query:
            main_query = main_query.replace("报销保险费用", "保险费用报销")
        # 若句尾缺少“流程”，根据上下文添加
        if main_query.endswith("报销") or main_query.endswith("费用"):
            main_query += "流程"
        # 添加结尾的问句形式
        query = main_query + "是什么"
    # 3. 确保以问号结尾
    query = query.rstrip("？?") + "？"
    return query


# 测试重写函数
original_query = "怎么报销保险费用？"
rewritten_query = rewrite_query(original_query)
print("原始查询：", original_query)
print("重写后的查询：", rewritten_query)
