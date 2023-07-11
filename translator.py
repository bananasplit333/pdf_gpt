import openai
import os
from dotenv import load_dotenv, find_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)   
load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

@retry(wait=wait_random_exponential(min=1, max=69), stop=stop_after_attempt(6))
def translate(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message["content"]

"""
    delimiter = "####"
    user_prompt = f
    You will be given a series of texts that belongs to a document. Please provide tranlsations for each text. 
    The texts will be separated by a ### character.
    I want you to create 5 questions based on the texts. It should test the reader their knowledge of what they just read.
        
        ###{user_prompt}####
    

    print(translate(user_prompt))
"""