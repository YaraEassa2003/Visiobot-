import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

client = OpenAI(
    base_url=endpoint,
    api_key=TOKEN,
)
def handle_chat(user_message):
    messages = [
        {"role": "system", "content": "You are VisioBot, an AI assistant that guides users to choose the best data visualization. You cannot make recommendations—only the trained model can do that."},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        messages=messages,
        temperature=0.7,
        top_p=1.0,
        max_tokens=1000,
        model=model_name
    )

    return response.choices[0].message.content.strip()
                                                       