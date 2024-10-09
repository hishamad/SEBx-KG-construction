import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Code from here: https://huggingface.co/SciPhi/Triplex
def triplet_triplex(text):
    model = AutoModelForCausalLM.from_pretrained("sciphi/triplex", trust_remote_code=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained("sciphi/triplex", trust_remote_code=True)

    entity_types = [ "Government Body", "Regulatory Body", "PERSON", "Geopolitical Entity", "COMPANY", "PRODUCT", "EVENT", "SECTOR", "ECON_INDICATOR", "CONCEPT"]
    predicates = [ "Has", "Announce", "Operate_In", "Introduce", "Produce", "Control", "Participates_In", "Impact", "Positive_Impact_On", "Negative_Impact_On", "Relate_To", "Is_Member_Of", "Invests_In", "Raise", "Decrease" ]

    input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.
      
        **Entity Types:**
        {entity_types}
        
        **Predicates:**
        {predicates}
        
        **Text:**
        {text}
        """

    message = input_format.format(
                entity_types = json.dumps({"entity_types": entity_types}),
                predicates = json.dumps({"predicates": predicates}),
                text = text)

    messages = [{'role': 'user', 'content': message}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors="pt").to("cuda")
    output = tokenizer.decode(model.generate(input_ids=input_ids, max_length=2048)[0], skip_special_tokens=True)
    return output

