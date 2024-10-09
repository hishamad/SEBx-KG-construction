from openai import OpenAI
import ast

# Place your OpenAI key in a file called .openai_api_key

api_key = open(".openai_api_key", "r").read()

triplet_prompt = open("triplet_models/triplet_prompt.txt", "r").read()

def triplet_gpt(input_text):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        temperature=0.5,
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": triplet_prompt
            },
            {
                "role": "user",
                "content": 
                f"""
                    Input text: {input_text},
                """
            }
        ]
    )
    result = completion.choices[0].message.content
    result_dict = ast.literal_eval(result)
    return result_dict