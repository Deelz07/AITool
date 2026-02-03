from google import genai
from pydantic import BaseModel
import os
import polars as pl
import logging


def setup_client():
    client = genai.Client(api_key=os.environ.get("GENAI_API_KEY") )
    print(f"Success the client was setup")
    return client

def generate_prompt(preamble:str,instruction:str,data:str):
    return ('\n').join([preamble,instruction,data])

def generate_response(client:genai.Client,prompt,questions:str):

    prompt += questions

    response = client.models.generate_content(model = "gemini-2.5-flash",contents = prompt)

    print(f"Sucess the response was generated")

    return response.text