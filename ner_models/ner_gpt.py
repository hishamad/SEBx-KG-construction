from openai import OpenAI
import ast

# Place your OpenAI key in a file called .openai_api_key

api_key = open(".openai_api_key", "r").read()

ner_prompt = open("ner_models/NER_prompt.txt", "r").read()

def ner_gpt(input_text):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        temperature=0.5,
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": ner_prompt
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