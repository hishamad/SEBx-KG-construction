import fireworks.client
fireworks.client.api_key = open(".fireworksAI", "r").read().strip()


def triplet_llma(text):
    
    prompt = open("triplet_models/triplet_prompt_llama.txt", "r").read()
    
    response = fireworks.client.ChatCompletion.create(
        temperature=0.5,
        model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct", 
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
    
    return result
