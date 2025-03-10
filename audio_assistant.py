from typing import List
from libs.client import AiClient
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
ai = AiClient()

prompts = {
    "START": "Classify the intent of the next input. \
             Is it: WRITE_EMAIL, QUESTION, OTHER? Only answer one word.",
    "QUESTION": "If you can answer the question: ANSWER, \
                 if you need more information: MORE, \
                 if you cannot answer: OTHER. Only answer one word.",
    "ANSWER": "Now answer the question",
    "MORE": "Now ask for more information",
    "OTHER": "Now tell me you cannot answer the question or do the action",
    "WRITE_EMAIL": 'If the subject or recipient or message is missing, \
                   answer "MORE". Else if you have all the information, \
                   answer "ACTION_WRITE_EMAIL |\
                   subject:subject, recipient:recipient, message:message".',
}

actions = {
    "ACTION_WRITE_EMAIL": "The mail has been sent. \
    Now tell me the action is done in natural language."
}

def start(input):
    messages = [{"role": "user", "content": prompts["START"]}]
    messages.append({"role": "user", "content": input})
    return disscuss(messages, "")

def disscuss(messages: List[str], last_step: str):
    answer = ai.ask(messages)
    if answer in prompts.keys():
        # A new state is found. Add it to the messages list.
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompts[answer]})
        # Recursively continue moving through the state machine.
        return disscuss(messages, answer)
    elif answer in actions.keys():
        # The new state is an action.
        do_action(answer)
    else:
        # We are in an END state.
        # If we come from MORE, we keep the history of messages.
        # Else we start over
        if last_step != 'MORE':
            messages=[]
        last_step = 'END'
        return answer

def do_action(action):
    print("Doing action " + action)
    return ("I did the action " + action)


def start_chat(file):
    input = ai.transcribe(file)
    print("User said: ", input)
    return start(input)

if __name__ == "__main__":
    gr.Interface(
        fn=start_chat,
        live=True,
        inputs=gr.Audio(sources="microphone", type="filepath"),
        outputs="text",
    ).launch()
