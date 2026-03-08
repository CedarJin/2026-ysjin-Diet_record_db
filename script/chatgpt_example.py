"""
Example script for using the ChatGPT API (OpenAI API).

Make sure you have:
1. Created a .env file with your OPENAI_API_KEY
2. Activated the virtual environment: source .venv/bin/activate
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat_with_gpt(messages, model="gpt-3.5-turbo", temperature=0.7):
    """
    Send messages to ChatGPT and get a response.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
                 Example: [{"role": "user", "content": "Hello!"}]
        model: The model to use (default: gpt-3.5-turbo)
        temperature: Controls randomness (0.0-2.0, default: 0.7)
    
    Returns:
        The assistant's response as a string
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    print("ChatGPT API Example\n")
    
    # Simple conversation
    messages = [
        {"role": "user", "content": "Hello! Can you help me with Python?"}
    ]
    
    response = chat_with_gpt(messages)
    print(f"User: {messages[0]['content']}")
    print(f"Assistant: {response}\n")
    
    # Continue the conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "What is a virtual environment?"})
    
    response = chat_with_gpt(messages)
    print(f"User: {messages[-1]['content']}")
    print(f"Assistant: {response}")
