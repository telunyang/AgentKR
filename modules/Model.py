import os, re, json, random
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv(override=True)
from ollama import chat
from ollama import generate as gen
from ollama import ChatResponse, GenerateResponse


# 選擇模型
'''
api_type: 
    'anthropic', 
    'bedrock', 
    'cerebras', 
    'google', 
    'ollama', 
    'openai', 
    'azure', 
    'deepseek', 
    'cohere', 
    'groq', 
    'mistral', 
    'together'
'''
def get_llm_config():
    return json.dumps([
        {
            "api_type": "ollama",
            "model": "mistral-small3.1:24b",
            "num_predict": -1,
            "num_ctx": 131072,
            "repeat_penalty": 1.1,
            "seed": 42,
            "stream": False,
            "temperature": 0.1,
            "top_k": 20,
        },
        {
            "api_type": "ollama",
            "model": "qwen3:32b",
            "num_predict": -1,
            "num_ctx": 40960,
            "repeat_penalty": 1.1,
            "seed": 42,
            "stream": False,
            "temperature": 0.1,
            "top_k": 20,
        },
        {
            "api_type": "ollama",
            "model": "llama3.3:70b",
            "num_predict": -1,
            "num_ctx": 131072,
            "repeat_penalty": 1.1,
            "seed": 42,
            "stream": False,
            "temperature": 0.1,
            "top_k": 20,
        },
        {
            "api_type": "ollama",
            "model": "deepseek-r1:70b",
            "num_predict": -1,
            "num_ctx": 131072,
            "repeat_penalty": 1.1,
            "seed": 42,
            "stream": False,
            "temperature": 0.1,
            "top_k": 20,
        },
        {
            "api_type": "ollama",
            "model": "llama4:16x17b",
            "num_predict": -1,
            "num_ctx": 10485760,
            "repeat_penalty": 1.1,
            "seed": 42,
            "stream": False,
            "temperature": 0.1,
            "top_k": 20,
        },
        {
            "api_type": "ollama",
            "model": "gpt-oss:20b",
            "num_predict": -1,
            "num_ctx": 131072,
            "repeat_penalty": 1.1,
            "seed": 42,
            "stream": False,
            "temperature": 0.1,
            "top_k": 20,
        },
        {
            "api_type": "ollama",
            "model": "gpt-oss:120b",
            "num_predict": -1,
            "num_ctx": 131072,
            "repeat_penalty": 1.1,
            "seed": 42,
            "stream": False,
            "temperature": 0.1,
            "top_k": 20,
        },
        {
            "api_type": "google",
            "model": "gemini-1.5-flash-8b",
            "api_key": os.environ["GEMINI_API_KEY_03"]
        },
        {
            "api_type": "google",
            "model": "gemini-1.5-flash",
            "api_key": os.environ["GEMINI_API_KEY_04"]
        },
        {
            "api_type": "google",
            "model": "gemini-2.0-flash-lite",
            "api_key": os.environ["GEMINI_API_KEY_04"]
        },
        {
            "api_type": "google",
            "model": "gemini-2.0-flash",
            "api_key": os.environ["GEMINI_API_KEY_03"]
        },
        {
            "api_type": "google",
            "model": "gemini-2.5-flash-lite",
            "api_key": os.environ["GEMINI_API_KEY_04"]
        },
        {
            "api_type": "google",
            "model": "gemini-2.5-flash",
            "api_key": os.environ["GEMINI_API_KEY_01"]
        }
    ])


# 生成內容
def generate(user_prompt: str, model_name='gemini-2.5-flash-lite') -> str:
    if "gemini" in model_name:
        # 讀取 API Key
        sn = random.randint(1, 5)
        GEMINI_API_KEY = os.getenv(f"GEMINI_API_KEY_{sn:02d}")

        # 建立 Client
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
                temperature=0.01,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                seed=42,
                max_output_tokens=4096,
            )
        )

        return response.text
    else:
        # 使用 ollama 的 chat 函數進行對話生成
        # response: ChatResponse = chat(
        response: GenerateResponse = gen(
            model=model_name, 
            system="You are a helpful assistant.", 
            prompt=user_prompt,
            stream=False,
            options={
                "num_predict": 1024,
                "num_ctx": 40960,
                "repeat_penalty": 1.2,
                "presence_penalty": 1.5,
                "frequency_penalty": 1.0,
                "seed": 42,
                "stream": False,
                "temperature": 0.01,
                "top_k": 20,
                "top_p": 0.95,
            },
            think=False, # if you use gpt-oss model, please set think="low","medium" or "high"
        )
        # return response.message.content.strip()
        return str(response.response)
