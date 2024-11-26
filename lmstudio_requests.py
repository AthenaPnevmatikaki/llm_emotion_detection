import requests
import json

url = "http://localhost:1234/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "llama-3.2-1b-instruct",
    "messages": [
        {"role": "system", "content": """Return the emotion in the given sentences.
         Select between empty, sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, and anger.
         Reply only with a list of the emotions you are most confident that are in the text, their strength and the reasons you selected them.
         Structure the returned list as JSON using: [{"emotion": "detected emotion", "strength": "Low/medium/high", "rationale": "your explanation"]}, ...]"""},
        {"role": "user", "content": "I was bitten by a dog!"}
    ],
    "temperature": 0.2
}

response = requests.post(url, headers=headers, data=json.dumps(data))

# Print the response
response = response.json()
print(response['choices'][0]['message']['content'])
