"""
This script runs inference of T0++ on multiple GPUs using model parallelism.
The model will be splitted across all available devices. Note that this feature is still an experimental feature under 🤗 Transformers.

The minimum requirements to run T0++ (11B parameters) inference are 4 16GB V100 or 2 32GB V100 (or basically, enough GPU memory to fit ~42GB of fp32 parameters).
T0 developers highly advise against performing inference in fp16 as the predictions will be unreliable and don't represent well the performance of the model.
(https://github.com/bigscience-workshop/t-zero/blob/master/inference/README.md)

To manipulate the biggest variants of T0 (11B parameters, like T0++), you will need ~90GB of CPU memory to load the model in memory.
If this is an issue, consider using T0 3B, which is a smaller variant of T0.
To run on T0_3B, set model_name = bigscience/T0_3B

Answers from T0 are exactly matched to words in the input sentence.

"""

######################################################################################################
import json
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, concatenate_datasets
import torch
from collections import defaultdict

# List of entity types
ent_types = ["PERS", "LOC", "ORG", "TIME", "PROD"]
ent2name = {'PERS': 'person', 'LOC': 'location', 'ORG': 'organization', 'TIME': 'date', 'PROD': 'media or doctrine'}
ent2query = {"PERS": "names of person", "LOC": "names of location", "ORG": "names of organization",
             "TIME": "dates", "PROD": "names of media or doctrine"}
ent2numb = {'PERS': [4,9], 'LOC': [2,7], 'ORG': [3,8], 'TIME': 11, 'PROD': [5,10]}
ix2ent = {0: "NONE", 2: 'LOC', 3: 'ORG', 4: 'PERS', 5: 'PROD', 7: 'LOC', 8: 'ORG', 9: 'PERS', 10: 'PROD', 11: 'TIME'}

# Avoid zero-div when calculating precision and recall
epsilon = 1e-10

# Upload model
model_name = "bigscience/T0pp"

# Force a split
split = os.environ.get("SPLIT", "")

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded")

model.parallelize()
print("Moved model to GPUs")


def inference_log(obj):
    with open(f"{__file__[:-3]}_{split}_log.json", "a") as file:
        file.write(json.dumps(obj) + "\n")


# Inference function
def T0_infer(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs)

    answer = (tokenizer.decode(outputs[0], skip_special_tokens=True))
    return answer


# Upload dataset
def dataset_upload(lang):
    if not split:
        if lang == "en":
            dataset = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "en", split="validation")
        elif lang == "de":
            dataset_val = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "de", split="validation")
            dataset_train = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "de", split="train")
            dataset = concatenate_datasets([dataset_val, dataset_train])
        elif lang == "fr":
            dataset_val = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "fr", split="validation")
            dataset_train = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", "fr", split = "train")
            dataset = concatenate_datasets([dataset_val, dataset_train])
    else:
        dataset = load_dataset("bigscience-historical-texts/HIPE2020_sent-split", lang, split=split)
    print("Dataset loaded", split)
    return dataset


# Divide dataset into periods of 20 years
def dataset_period(dataset):
    date_counter = 0
    date_junction = range(1790, 1951, 20)
    period_dict = {i: [] for i in date_junction}

    dataset = dataset.sort("date")
    for item in dataset:
        # Our ranges are [start_year, end_year]
        date_counter = 0
        if item['date'].year > 1949:
            break
        while item['date'].year >= date_junction[date_counter + 1]:
            date_counter += 1

        period_dict[date_junction[date_counter]].append(item)
    return period_dict


def prediction(local_period, period, lang):
    true_positives, false_positives, false_negatives = 0, 0, 0
    # If the datasplit is changed to keep dataset objets, replace the 3 next lines with the commented line
    #for s, (tokens, ents) in enumerate(zip(dataset['tokens'], dataset['NE_COARSE_LIT'])):
    for sentence_obj in local_period:
        tokens = sentence_obj["tokens"]
        ents = sentence_obj["NE_COARSE_LIT"]
        sentence = " ".join(tokens)

        # ---- We store all entities as strings, by types, to confront them to model predictions
        entities = {ent_type: [] for ent_type in ent_types}
        cur_ent, cur_tokens = "NONE", []
        for token, ent in zip(tokens, ents):
            if ent in [1, 6]:
                continue

            # We are faced with a B-PROD/PERS when we alread had a PROD/PERS entity
            if ent == 5 and cur_ent == "PROD" or ent == 4 and cur_ent == "PERS":
                entities[cur_ent].append(cur_tokens)
                cur_tokens = []

            # If we change entity type, we store the current token
            if ix2ent[ent] != cur_ent:
                if cur_ent != "NONE":
                    entities[cur_ent].append(cur_tokens)
                    cur_tokens = []

            if ent != 0:
                cur_tokens.append(token)

            cur_ent = ix2ent[ent]

        ent2gold = {k: [" ".join(v) for v in values] for k, values in entities.items()}
        gold2ent = {v: k for k,values in ent2gold.items() for v in values }

        # ---- For each entity type, we prompt the model
        result_tok, result_ent = [], []
        for ent_type, gold_tokens in entities.items():
            prompt = f"Input: {sentence}\nIn input, what are the {ent2query[ent_type]}? Separate answers with commas"
            answer = T0_infer(prompt).split(",")
            inference_log({"period": period, "lang": lang, "sentence": sentence, "prompt": prompt, "answers": answer, "gold": gold_tokens, "ent_type": ent_type, "dedup": False})

            # Check if token exists and add it to list
            for tok in answer:
                if tok.lower() in sentence.lower():
                    result_tok.append(tok)
                    result_ent.append(ent_type)

        # ---- We remove duplicates and overlapping tokens
        # Disambiguate 1: choose longest token when token is substring of another token
        dup_indices = []
        for tok in result_tok:
            dup_indices.extend([i for i, smaller_tok in enumerate(result_tok) if smaller_tok in tok])

        dup_indices = set(dup_indices)
        result_tok_longest = [tok for ix, tok in enumerate(result_tok) if ix not in dup_indices]
        result_ent_longest = [ent for ix, ent in enumerate(result_ent) if ix not in dup_indices]

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

                answer = T0_infer(prompt, sentence, period, lang, ent2gold[ent_type], ent_type)

                # Check that answer is actually an entity type, if not we don't keep it
                if answer in ent2name.values():
                    chosen_ent = list(ent2name.keys())[list(ent2name.values()).index(answer)]
                    tok2ents_dedup[tok] = chosen_ent
                else:
                    tok2ents_dedup[tok] = None
                inference_log({"period": period, "lang": lang, "sentence": sentence, "prompt": prompt, "answers": answer.split(","), "gold": ent2gold.get(tok2ents_dedup[tok], []), "ent_type": tok2ents_dedup[tok], "dedup": True})
            else:
                tok2ents_dedup[tok] = ents[0]

        # ---- Extract statistics
        for tok, ent in tok2ents_dedup.items():
            # Predicted token does not exist
            if tok not in gold2ent.keys():
                false_negative += 1
            else:
                # Token is correct
                if gold2ent[tok] == ent:
                    true_positive += 1
                # Token predicted is incorrect
                else:
                    false_positive += 1

    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    F1 = 2 * precision * recall / (precision + recall + epsilon)
    return {"precision": precision, "recall": recall, "F1": F1}

# Main
result = {l: {} for l in ["en", "de", "fr"]}
for lang in ["en", "de", "fr"]:
    try:
        dataset = dataset_upload(lang)
    except ValueError:
        continue
    time_splits = dataset_period(dataset)
    for time, time_split in time_splits.items():
        result[lang][time] = prediction(time_split, time, lang)

# Would be better as json dump
with open("result_NER.log", "w+") as f:
    f.write(json.dumps(result))

