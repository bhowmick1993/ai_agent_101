from smolagents import CodeAgent, TransformersModel


model = TransformersModel(
    model_id="Qwen/Qwen3-4B",
    device_map="auto",
    torch_dtype="auto",
    max_new_tokens=1024,
)


agent = CodeAgent(
    tools=[],
    model=model,
)


result = agent.run(
    """
    Calculate how far a vehicle travels in 5 seconds
    if it is moving at 72 km/h.
    Explain your reasoning.
    """
)

print(result)