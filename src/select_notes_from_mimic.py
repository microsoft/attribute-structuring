import json
import os
import re

from structuring import structure_text

# mimic_dir = "/mnt/hanoverdev/scratch/zelalem/data/mimiciii_notes.json"
# output_dir = "../data/"


# with open(mimic_dir) as infile:
#     mimic_notes = json.load(infile)


# token_text = mimic_notes[list(mimic_notes.keys())[0]]['discharge summary'][0]
# tik_tokenizer = tiktoken.get_encoding('p50k_base')

# # return the number of tokens in a text
# def tiktoken_len(text):
#   tokens = tik_tokenizer.encode(text, disallowed_special = ())
#   return len(tokens)    




def preprocess_notes(mimiciii, tiktoken_len,test_keys, minimum_notes=5, context_length=4000):
    test_notes = {}
    ground_structured = {}
    
    
    for k, v in mimiciii.items():
        if k not in test_keys: #or 'discharge summary' not in v or len(v) < minimum_notes:  # at least a DS plus min_notes 
            continue

        sorted_items = sorted(v.items(), key=lambda x: x[1][1])
        joined_text = ' '.join(item[0] + "\n" + item[1][0] for item in sorted_items if item[0] != 'discharge summary')
        joined_text = re.sub(r"\n+|\[]+", " ", joined_text)

        len_text = tiktoken_len(joined_text)
        ds = v['discharge summary'][0]
        len_ds = tiktoken_len(ds)

        min_context_len = context_length - 400  # select notes with len closer to the full context 
        
        #if (len_text > min_context_len and len_text < context_length) and len_text > len_ds * 1.2: # source note should be longer than the summary
        if True:
            
            #ground_structured[k] = structure_text(v['discharge summary'][0])  # structure to choose notes that have most of the important attributes
            #ground_len = len([w for w, y in ground_structured[k].items() if y.lower() != 'none'])
            
            #if ground_len >= 13:
            test_notes[k] = {"full": joined_text, "summary": v['discharge summary'][0]}

            #break  
        
    return test_notes



# selected_notes = preprocess_notes(mimic_notes)

# with open("../data/selected_mimic_notes.json","w") as outfile:
#     json.dump(selected_notes, outfile, indent=4, sort_keys = True, default = str)
