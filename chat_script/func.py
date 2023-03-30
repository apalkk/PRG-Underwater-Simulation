import os
import openai
import re
import math 
import mathutils
import bpy

openai.api_key = "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" # API KEY (expired)

# _________________________________________________________________________
# Function to be used by the script are defined below

def ask(chat_history,prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]


def extract_python_code(content):
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None


"""version https://git-lfs.github.com/spec/v1
oid sha256:3d1fbc19a26281ae3ea43195c82b5217f3394af9f344756dcfe5268423ff256f
size 2632"""
