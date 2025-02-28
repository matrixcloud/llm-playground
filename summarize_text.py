
from libs.client import AiClient

client = AiClient()

with open('./assets/transcript.txt', 'r') as f:
    transcript = f.read()

res = client.ask([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the following text"},
    {"role": "assistant", "content": "Yes."},
    {"role": "user", "content": transcript},
])

print(res)
