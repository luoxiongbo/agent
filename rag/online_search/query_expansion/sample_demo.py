# 构建一个简单的同义词词典
synonym_dict = {
    "保险理赔": ["保险索赔", "理赔", "索赔流程", "理赔流程"],
    "理赔": ["索赔", "赔付"],
    # 其他词的同义词...
}


def expand_query(query: str) -> [str]:
    expansions = set()
    # 如果查询短语本身在词典中
    if query in synonym_dict:
        expansions.update(synonym_dict[query])
    # 将查询拆分为词（简单按字符，这里假设输入短语本身是一个词或固定短语）
    for term, syns in synonym_dict.items():
        if term in query:
            expansions.update(syns)
    # 加入原始查询本身
    expansions.add(query)
    return list(expansions)


# 测试查询扩展
query = "保险理赔"
expanded_queries = expand_query(query)
print("原始查询：", query)
print("扩展结果：", expanded_queries)
