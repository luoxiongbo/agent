from langgraph.graph


def part1():
    approval = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Do you want to send this email?"
    })

    agent = create_agent(
        model=model,
        tools=[send_email],
        checkpointer=MemorySaver()
    )

    result = agent.invoke(
        Command(resume={"approved": True}),
        config=config
    )


def part2():


def part3():
    # Middleware 有两种风格
    # Node-style Hooks：切面
    # Wrap-style Hooks：包装

    class RequestLoggerMiddleware(AgentMiddleware):

        def before_model(self, state, runtime):
            # 记录当前消息数量
            ...

        def wrap_model_call(self, request, handler):
            # 记录模型和工具信息
            return handler(request)

        def after_model(self, state, runtime):
            # 检查模型是否请求工具
            ...

    agent = create_agent(
        model=model,
        tools=[explain_concept],
        middleware=[RequestLoggerMiddleware()]
    )


def part4():
