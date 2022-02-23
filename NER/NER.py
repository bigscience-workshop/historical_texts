"""
This script runs inference of T0++ on multiple GPUs using model parallelism.
# The minimum requirements to run T0++ (11B parameters) inference are 4 16GB V100 or 2 32GB V100 (or basically, enough GPU memory to fit ~42GB of fp32 parameters).
# (https://github.com/bigscience-workshop/t-zero/blob/master/inference/model_parallelism.py)
"""

"""TO DO: create version for single GPU"""

######################################################################################################

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, concatenate_datasets
import torch
from collections import defaultdict
import pandas as pd

# List of entity types
ent_types = ["PERS", "LOC", "ORG", "TIME", "PROD.MED", "PROD.DOC"] # I split products into media and doctrines as "products" is too general as a prompt
ent2name = {'PERS': 'person', 'LOC': 'location', 'ORG': 'organization', 'TIME': 'date', 'PROD.MED': 'media', 'PROD.DOC': 'doctrine'}
ent2query = {"PERS": "names of person", "LOC": "names of location", "ORG": "names of organization", 
             "TIME": "dates", "PROD.MED": "names of media", "PROD.DOC": "names of doctrine"}
ent2numb = {'PERS': [4,9], 'LOC': [2,7], 'ORG': [3,8], 'TIME': 11, 'PROD.MED': [5,10], 'PROD.DOC': [5,10]}

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

# Upload dataset
def dataset_upload(lang):
    if lang == "en":
        dataset = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "en", split = "validation")
    elif lang == "de":
        dataset_val = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "de", split = "validation")
        dataset_train = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "de", split = "train")
        dataset = concatenate_datasets(dataset_val, dataset_train)
    elif lang == "fr":
        dataset_val = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "fr", split = "validation")
        dataset_train = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "fr", split = "train")
        dataset = concatenate_datasets(dataset_val, dataset_train)
    return dataset
    print("Dataset loaded")

# Divide dataset into periods of 20 years
"TO DO: split by period"
def dataset_period(dataset):
    date_junction = range(1790, 1951, 20) 


def prediction(dataset):
# Cycle over sentences
    "TO DO: optimize this section"
    for s in range(len(dataset)):

        tokens = dataset['tokens'][s]
        sentence = ' '.join(dataset['tokens'][s])
        
        # List the entities in the sentence
        # Express each entity as a string, not as a list of tokens. This is because the output of T0 will also be a string.
        ents = dataset['NE_COARSE_LIT'][s]
        sent_LOC = []
        sent_ORG = []
        sent_PERS = []
        sent_PROD = []
        sent_TIME = []
    
        for i in range(len(ents)):
            if ents[i] == 2 and i+1 == len(ents):
                sent_LOC.append([tokens[i]])
            if ents[i] == 2 and i+1 <len(ents):
                res = next(x for x, val in enumerate(ents[i+1:]) if val in (0,2,3,4,5,6,8,9,10,11))
                sent_LOC.append(tokens[i:i+res+1])    
        
        for i in range(len(ents)):
            if ents[i] == 3 and i+1 == len(ents):
                sent_ORG.append([tokens[i]])
            if ents[i] == 3 and i+1 <len(ents):
                res = next(x for x, val in enumerate(ents[i+1:]) if val in (0,2,3,4,5,6,7,9,10,11))
                sent_ORG.append(tokens[i:i+res+1])

        for i in range(len(ents)):
            if ents[i] == 4 and i+1 == len(ents):
                sent_PERS.append([tokens[i]])
            if ents[i] == 4 and i+1 <len(ents):
                res = next(x for x, val in enumerate(ents[i+1:]) if val in (0,2,3,4,5,6,7,8,10,11))
                sent_PERS.append(tokens[i:i+res+1])   

        for i in range(len(ents)):
            if ents[i] == 5 and i+1 == len(ents):
                sent_PROD.append([tokens[i]])
            if ents[i] == 5 and i+1 <len(ents):
                res = next(x for x, val in enumerate(ents[i+1:]) if val in (0,2,3,4,5,6,7,8,9,11))
                sent_PROD.append(tokens[i:i+res+1])

        for i in range(len(ents)):
            if ents[i] == 6 and i+1 == len(ents):
                sent_TIME.append([tokens[i]])
            if ents[i] == 6 and i+1 <len(ents):
                res = next(x for x, val in enumerate(ents[i+1:]) if val in (0,2,3,4,5,6,7,8,9,10))
                sent_TIME.append(tokens[i:i+res+1])

        sent_LOC = [' '.join(sub_list) for sub_list in sent_LOC]
        sent_ORG = [' '.join(sub_list) for sub_list in sent_ORG]
        sent_PERS = [' '.join(sub_list) for sub_list in sent_PERS]
        sent_PROD = [' '.join(sub_list) for sub_list in sent_PROD]
        sent_TIME = [' '.join(sub_list) for sub_list in sent_TIME]

        tok2ent_gold = {i : 'LOC' for i in sent_LOC}
        tok2ent_gold.update({i : 'ORG' for i in sent_ORG})
        tok2ent_gold.update({i : 'PERS' for i in sent_PERS})
        tok2ent_gold.update({i : 'PROD' for i in sent_PROD})
        tok2ent_gold.update({i : 'TIME' for i in sent_TIME})   

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
                prompt = f"Input: {sentence} \n In input, is \"{tok}\"" 
                prompt += ''.join(f',a {ent2name[ent]}' for ent in ents[:-1]) 
                prompt += f"or a {ent2name[ents[-1]]}? Give only one answer"

                answer = T0_infer(prompt)

                # Check that answer is actually an entity type, if not we don't keep it
                if answer in ent2name.values():
                    chosen_ent = list(ent2name.keys())[list(ent2name.values()).index(answer)]
                    tok2ents_dedup[tok] = [chosen_ent]

            else:
                tok2ents_dedup[tok] = ent
        
        tok2ents_preds = tok2ents_dedup

        #  Collapse PROD.MED and PROD.DOC into PROD
        for tok, ent in tok2ents_preds.items():
            if ent == 'PROD.MED' or ent == 'PROD.DOC':
                tok2ents_preds[tok] = 'PROD'

    "TO DO: compare predictions with gold standard. Calculate precision, recall and F-score"
    # Compare tok2ents_preds to tok2ent_gold

    # true_positives = 0
    # false_positives = 0
    # false_negatives = 0

    # for tok, ent in tok2ents_preds.items():
    #     if ent == tok2ent_gold[tok]:
    #         true_positives += 1
    # .....

    # precision = true_positives / (true_positives + false_positives)
    # recall = true_positives / (true_positives + false_negatives)
    # F1 = 2 * precision * recall / (precision + recall)

    # Store results
