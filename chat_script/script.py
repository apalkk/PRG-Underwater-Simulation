from func import ask,extract_python_code
import math
import numpy
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



#LFS pointer :
"""version https://git-lfs.github.com/spec/v1
oid sha256:4f09a50f14f56f5eaf44da27bedce5c44dafe620834e8a78f1999934ee39996c
size 596"""
