"""
This script runs inference of T0++ on multiple GPUs using model parallelism.
# The minimum requirements to run T0++ (11B parameters) inference are 4 16GB V100 or 2 32GB V100 (or basically, enough GPU memory to fit ~42GB of fp32 parameters).
# (https://github.com/bigscience-workshop/t-zero/blob/master/inference/model_parallelism.py)

"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
import torch
import re
from collections import defaultdict

# List of entity types
ent_types = ["PERS", "LOC", "ORG", "TIME", "PROD.MED", "PROD.DOC"] # I split products into media and doctrines as "products" is too general as a prompt
ent2name = {'PERS': 'person', 'LOC': 'location', 'ORG': 'organization', 'TIME': 'date', 'PROD.MED': 'media', 'PROD.DOC': 'doctrine'}
ent2query = {"PERS": "names of person", "LOC": "names of location", "ORG": "names of organization", 
             "TIME": "dates", "PROD.MED": "names of media", "PROD.DOC": "names of doctrine"}


# Upload model
model_name = "bigscience/T0pp"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded")

model.parallelize()
print("Moved model to GPUs")

# Inference function
def T0_infer(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs)

    answer = (tokenizer.decode(outputs[0], skip_special_tokens=True))

"TO DO: upload datset"

"TO DO: cycle on sentences"
# Import sentence
# Clean sentence: remove at least split words (like "in", "-", "dustrious")
# Regularise spelling
sentence = sentence

# Cycle over prompts
result_tok, result_ent = [], [] # if we are studying sentences one at a time, it is way faster to use a list than a dataframe
for ent, ent_query in ent2query.items():
    prompt = "Input: {}.format(sentence)" + "\n" + "In input, what are the {}? Separate answers with commas".format(ent_query)
    answer = T0_infer(prompt).split(",")

    # Check if token exists and add it to list    
    for tok in answer:
        if tok.lower() in sentence.lower(): # faster than a regex
            result_tok.append(tok)
            result_ent.append(ent)

# Disambiguate 1: choose longest token when token is substring of another token
dup_indices = []
for tok in result_tok:
    dup_indices.extend([i for i, smaller_tok in enumerate(result_tok) if smaller_tok in tok])

dup_indices = set(dup_indices)
result_tok_longest = [tok for ix, tok in result_tok if ix not in dup_indices]
result_ent_longest = [ent for ix, ent in result_ent if ix not in dup_indices]

# Disambiguate 2: disambiguate duplicate entities for a token
tok2ents = defaultdict(list)
for ent, tok in zip(result_tok_longest, result_ent_longest):
    tok2ents[tok].append(ent)

tok2ents_dedup = {}
for tok, ents in tok2ents.items():
    if len(ents) > 1:
        prompt = f"Input: {sentence} \n In input, is \"{token_dup}\" 
        prompt += "".join(f',a {type2name[ent]}' for ent in ents[:-1]) 
        prompt += f"or a {type2name[ents[-1]]}? Give only one answer"

        answer = T0_infer(prompt)

        # Check that answer is actually an entity type, if not we don't keep it
        if answer in ent_types:
            tok2ents_dedup[tok] = answer

    else:
        tok2ents_dedup[tok] = ent
"TO DO: collapse PROD.MED and PROD.DOC into PROD"

"TO DO: add predicted entity to dataframe with all predictions"
