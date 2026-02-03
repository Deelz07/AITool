import generate_content as AI
import read_files as rf
from google.genai import types

if __name__ == "__main__":
    path = "input/prealgbera_nt.txt"
    #Setup client
    client = AI.setup_client()

    Pre = "You are a LaTeX expert. Generate a complete, ready-to-compile .tex file."
    Input = """
    You will be required to read through the input data and convert it into latex format.

    1. If there are any solutions remove them.
    2. Remove any existing questions number such as 3.xx and instead number it from scratch 
    3. Remove any excessive formatting from other maths language. E.g. [\color[rgb]{0.35,0.35,0.35}6 = 2+r\] should be converted
        to latex version of that.
    
    4. Format it such that questions are marked with numbers and subpart are marked with letters.

    5. Latex Formatting Rules:
        1. Use the 'amsmath' and 'amsfonts' packages.
        2. Ensure all math is wrapped in $...$ for inline and $$...$$ for display.
        3. Output the RAW content only—no markdown wrappers.
    """

    
    question_file = rf.txtfile_to_string(path)

    prompt = AI.generate_prompt(preamble=Pre,instruction=Input,data=question_file)

    response = client.models.generate_content(
        model = "gemini-2.5-flash",contents = prompt,
        config=types.GenerateContentConfig(
        # This pushes the model toward a cleaner text-based output
        response_mime_type="text/plain", 
        # Lower temperature (0.1 - 0.3) is better for LaTeX 
        # as it makes the formatting more consistent and less 'creative'
        temperature=0.1,
    ))

    output_file = "output/prealntgebra.txt"
    with open(output_file,"w") as f:
        f.write(response.text)
        print(f"Success the latex file was written!")


    

