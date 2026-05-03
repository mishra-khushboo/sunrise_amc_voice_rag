import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3",
    "prompt": "Say hello in one line",
    "stream": False
}

r = requests.post(url, json=payload, timeout=120)

print(r.json()["response"])