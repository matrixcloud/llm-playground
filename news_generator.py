from typing import List

from libs.client import AiClient

client = AiClient()

prompt_role = """You are an assistant for journalists.
    Your task is to write articles, based on the FACTS that are given to you.
    You should respect the instructions: the TONE, the LENGTH, and the style
"""

def assist_journalist(facts: List[str], tone: str, length: int, style: str):
    facts = ",".join(facts)
    prompt = f"{prompt_role}\
        FACTS: {facts} \
        TONE: {tone} \
        LENGTH: {length} words\
        STYLE: {style} \
    "

    return client.ask([{"role": "user", "content": prompt}])

print(assist_journalist(["The sky is blue", "The grass is green"], "informal", 100, "blogpost"))