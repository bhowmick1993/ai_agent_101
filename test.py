from smolagents import TransformersModel

model = TransformersModel(
    model_id="Qwen/Qwen3-4B",
    device_map="auto",
    torch_dtype="auto",
    max_new_tokens=512,
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Explain what an autonomous AI agent is in 3 sentences."
            }
        ],
    }
]

response = model(messages)

print(response.content)