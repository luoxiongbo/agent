# -*- encode = utf-8 -*-
# @Time : 2026-08-15 17:19:24
# @Auther : 罗雄波
# @File : function_demo_structured_output.py
# @Software : PyCharm

"""
agent 使用工具类型中的 functional 的 demo
结构化(json)输出
"""


def display_cities(cities: list[str], preferences: Optional[str] = None):
    """Provides a list of cities based on the user's search query and preferences.
    Args:
    preferences (str): The user's preferences for the search, like skiing, beac
    cities (list[str]): The list of cities being recommended to the user.
    Returns:
    list[str]: The list of cities being recommended to the user.
    """
    return cities


from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration

model = GenerativeModel("gemini-1.5-flash-001")
display_cities_function = FunctionDeclaration.from_func(display_cities)
tool = Tool(function_declarations=[display_cities_function])
message = "I’d like to take a ski trip with my family but I’m not sure where to go?"
res = model.generate_content(message, tools=[tool])
print(f"Function Name: {res.candidates[0].content.parts[0].function_call.name}")
print(f"Function Args: {res.candidates[0].content.parts[0].function_call.args}")
