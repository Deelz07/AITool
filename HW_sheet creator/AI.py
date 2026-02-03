from google import genai
from pydantic import BaseModel
import os
import polars as pl
import logging

#TODO: Separate questions and solutions

def setup_client():
    client = genai.Client(api_key=os.environ.get("GENAI_API_KEY") )
    print(f"Success the client was setup")
    return client


def generate_response(client:genai.Client,prompt,questions:str):

    prompt += questions

    response = client.models.generate_content(model = "gemini-2.5-flash",contents = prompt)

    print(f"Sucess the response was generated")

    return response.text

if __name__ == "__main__":
    client = setup_client()

    prompt = """I have the following text file containing questions.

    THe task is to output the problems and the solutions into ***latex format*** not python.

    Part 1 - Problems only

    1. Remove all solutions for this section
    2. Formatting - 
        - Remove any existing questions number such as 3.xx and instead number it from scratch 
        - Remove any formatting text that arise from copying in
    3. Reformatting - 
        - Format it nicely such that questions are marked with numbers and subpart are marked with letters.
        - If problem statements involve mathematical expression place in the same line of the latex file. (No maths expression in different line unless it is an equation brand new)


    The questions are below:
    """

    path = 'prealgebra_exponents.txt'
    with open(path,'r') as f:
        questions = f.read()
        print('questions were read in from the txt file')
    
        


    output_content = generate_response(client,prompt,questions)

    output_file = "prealgebra.tex"
    with open(output_file,"w") as f:
        f.write(output_content)
        print(f"Success the latex file was written!")



    """
      Part 2 - Solutions section
    1. If a question has solutions output it with the question using similar format to the problem section
    2. If a question does not have solution generate it and label the solution as AI-generated.
    3. Follow the same formatting procedure as the problems section.


    Please keep the problems and solutions part separate from each other.
    """


