
import re

from structuring import structure_text


def preprocess_notes(mimiciii, tiktoken_len,test_keys, minimum_notes=5, context_length=4000):
    test_notes = {}
    
    for k, v in mimiciii.items():
        if k not in test_keys:
            continue

        sorted_items = sorted(v.items(), key=lambda x: x[1][1])
        joined_text = ' '.join(item[0] + "\n" + item[1][0] for item in sorted_items if item[0] != 'discharge summary')
        joined_text = re.sub(r"\n+|\[]+", " ", joined_text)

        test_notes[k] = {"full": joined_text, "summary": v['discharge summary'][0]}

            
    return test_notes




