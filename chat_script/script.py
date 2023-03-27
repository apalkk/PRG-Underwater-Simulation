from func import ask,extract_python_code
import os
import openai
import json
openai.api_key = "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" # API KEY
openai.Model.list()


with open("sim.txt","r") as f:
    prompt = f.read()

with open("moves.json","r") as f:
    comm = json.load(f)

# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
chat_history = [
    {
        "role": "system",
        "content": prompt
    }
]

for x in comm:
    y = ask(chat_history,comm[x])
    print(y)
    exec(extract_python_code(y))
import pdb;pdb.set_trace()

