from mcp.server import MCPServer


# =========================
# 1. 创建 MCP Server
# =========================

mcp = MCPServer("weather-server")


# =========================
# 2. 暴露 Tool
# =========================

@mcp.tool()
def get_weather(city: str) -> dict:
    """
    Get the current weather of a city.

    Args:
        city: City name, for example Beijing or Shanghai.
    """

    # 为了方便演示，这里不调用真实天气 API
    fake_database = {
        "北京": {
            "temperature": 28,
            "weather": "晴",
        },
        "上海": {
            "temperature": 31,
            "weather": "多云",
        },
        "深圳": {
            "temperature": 33,
            "weather": "阵雨",
        },
    }

    weather = fake_database.get(
        city,
        {
            "temperature": 25,
            "weather": "未知",
        },
    )

    return {
        "city": city,
        **weather,
    }


@mcp.tool()
def calculate(a: float, b: float, operation: str) -> dict:
    """
    Perform a simple calculation.

    Args:
        a: First number.
        b: Second number.
        operation: One of add, subtract, multiply, divide.
    """

    if operation == "add":
        result = a + b

    elif operation == "subtract":
        result = a - b

    elif operation == "multiply":
        result = a * b

    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")

        result = a / b

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return {
        "result": result
    }


# =========================
# 3. 启动 MCP Server
# =========================

if __name__ == "__main__":
    # 默认就是 stdio
    mcp.run()