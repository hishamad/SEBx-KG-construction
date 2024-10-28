import fireworks.client
fireworks.client.api_key = open(".fireworksAI", "r").read().strip()
import ast


def triplet_mixtral(text):
    
    prompt = open("triplet_models/triplet_prompt.txt", "r").read()
    
    response = fireworks.client.ChatCompletion.create(
        temperature=0.5,
        model="accounts/fireworks/models/mixtral-8x22b-instruct", 
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": 
                f"""
                    Input text: {text},
                """
            }
        ]
    )
    
    result = response.choices[0].message.content
    
    try:
        result_dict = ast.literal_eval(result)
        return result_dict
    except:
        return result
