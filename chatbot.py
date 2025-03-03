import os
from openai import OpenAI

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("ChatGPT is ready to converse! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Exiting conversation. Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model_name
    )
    
    reply = response.choices[0].message.content.strip()
    print("ChatGPT:", reply)
    
    messages.append({"role": "assistant", "content": reply})