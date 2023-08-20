import requests
import json

url = "https://api.openai.com/v1/engines/text-davinci-002/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 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"
}

data = {
    "prompt": "Translate the following English text to Chinese: 'Hello, how are you?'",
    "max_tokens": 60
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())
