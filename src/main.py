import json
import tiktoken

from select_notes_from_mimic import preprocess_notes
from generate_summaries import generate_summaries
from structure_summaries import structure_ground_gen_summaries
from scoring import score_structured
from scoring import score_full
from metrics import calculate_metric_scores




# return the number of tokens in a text
def tiktoken_len(text):
  tokens = tik_tokenizer.encode(text, disallowed_special = ())
  return len(tokens)  


def main(mimic_notes):
    # Select notes from MIMIC that fit into the MAX_CONTEXT_LEN
    selected_notes = preprocess_notes(mimic_notes, tiktoken_len)
    
    #Generate synthetic summaries given full notes as an input
    generated_summaries = generate_summaries(selected_notes,tiktoken_len)
    
    # Structure the generated and ground summaries 
    structured_summaries =  structure_ground_gen_summaries(generated_summaries, selected_notes)

    # Get scores for each of the structured attributes
    scored_summaries_structured = score_structured(structured_summaries)

    # Score the whole summary without structuring 
    scored_summaries_full = score_full(structured_summaries) 

    # Get the baseline scores(rouge-1, rouge-L, BERTScore)
    metric_scores = calculate_metric_scores(structured_summaries)
    

    return selected_notes, generated_summaries, structured_summaries, scored_summaries_structured, scored_summaries_full, metric_scores
    

if __name__ == "__main__":
   
   mimic_dir = "PATH_TO_MIMIC_DATASET"
   
   with open(mimic_dir) as infile:
    mimic_notes = json.load(infile)

   token_text = mimic_notes[list(mimic_notes.keys())[0]]['discharge summary'][0]
   tik_tokenizer = tiktoken.get_encoding('p50k_base')
   
   selected_notes, generated_summaries, structured_summaries, scored_summaries_structured, scored_summaries_full, metric_scores = main(mimic_notes)    
   