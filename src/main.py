import json
import tiktoken
import os
import numpy as np
import warnings

from select_notes_from_mimic import preprocess_notes
from generate_summaries import generate_summaries
from structure_summaries import structure_ground_gen_summaries
from scoring import score_structured, score_full, normalize_list, get_error_scores2, get_error_scores_full2, get_most_frequent_values
from metrics import calculate_metric_scores

warnings.filterwarnings('ignore')



def tiktoken_len(text):
  # return the number of tokens in a text
  tokens = tik_tokenizer.encode(text, disallowed_special = ())
  return len(tokens)  



def get_correlation_scores(metric_scores,metric_scores_full,gpt_scores_full,gpt_frequent_values,annotations):
    # Return the correlation scores for both unstructured and structured documents
    r1_scores = {x:y['rouge1'] for x,y in metric_scores.items()}
    rL_scores = {x:y['rougeL'] for x,y in metric_scores.items()}
    bertscores = {x:y['bertscore'] for x,y in metric_scores.items()}

    r1_scores_full = {x:y[0][0] for x,y in metric_scores_full.items()}
    rL_scores_full = {x:y[0][1] for x,y in metric_scores_full.items()}
    bertscores_full ={x:y[0][2] for x,y in metric_scores_full.items()}

    gpt_scores_full2 =  {x:np.mean(normalize_list(y,min_val=1, max_val=10)) for x,y in gpt_scores_full.items()}
 
    gpt_user = get_error_scores2(gpt_frequent_values,annotations,True)
    gpt_user_full = get_error_scores_full2(gpt_scores_full2,annotations)
    r1_user = get_error_scores2(r1_scores,annotations)
    r1_user_full = get_error_scores_full2(r1_scores_full,annotations)
    rL_user = get_error_scores2(rL_scores,annotations)
    rL_user_full = get_error_scores_full2(rL_scores_full,annotations)

    bert_user = get_error_scores2(bertscores,annotations)
    bert_user_full = get_error_scores_full2(bertscores_full,annotations)

    

    return gpt_user, gpt_user_full, r1_user, r1_user_full, rL_user, rL_user_full, bert_user, bert_user_full


def generate_structure_score(mimic_notes, test_keys):
    # generate summaries, structure the generated summary, score the summaries

    # Select notes from MIMIC that fit into the MAX_CONTEXT_LEN
    selected_notes = preprocess_notes(mimic_notes, tiktoken_len, test_keys)
    print(f"selected {len(selected_notes)} notes")
    
    #Generate synthetic summaries given full notes as an input
    generated_summaries = generate_summaries(selected_notes,tiktoken_len)
    print(f"generated  {len(generated_summaries)} summaries")
    
    # Structure the generated and ground summaries 
    structured_summaries =  structure_ground_gen_summaries(generated_summaries, selected_notes)
    print(f"structured  {len(structured_summaries)} summaries")
    
    # Get scores for each of the structured attributes (multiple_runs to account for variations)
    scored_summaries_structured = []
    for k,v in structured_summaries.items():
      for _ in range(5):
         scored_summaries_structured.append(score_structured({k:v}))
   
    # Score the whole summary without structuring 
    scored_summaries_full = score_full(structured_summaries) 
    
    return  scored_summaries_structured, scored_summaries_full,structured_summaries
    
def get_baseline_scores(structured_summaries):
   # Get the baseline scores(rouge-1, rouge-L, BERTScore)
    metric_scores = calculate_metric_scores(structured_summaries)
    
    # Get baseline scores (rouge-1, rouge-L, BERTScore) for unstructured inputs
    metric_scores_full = calculate_metric_scores(structured_summaries, True)
    
    return metric_scores, metric_scores_full

def print_output(metric, output_dict):
      # print out calculated metrics

      lst_vals = [v for _,v in output_dict.items()]
      lst_vals.insert(0,metric)
      
      print(' '.join(f"{item:8}" for item in lst_vals))
      


if __name__ == "__main__":
   
   current_dir = os.path.dirname(os.path.realpath(__file__))
  
   parent_dir = os.path.dirname(current_dir)
   data_dir = os.path.join(parent_dir,'data')

   mimic_dir ="MIMIC_DIR" # Replace with a path to your mimic dataset
   

   
   annotations_dir = os.path.join(data_dir,"annotations.json")
   claim_dir = os.path.join(data_dir,"claim_scores.json")
   
   # Read input data
   # MIMIC NOTES
   with open(mimic_dir) as infile:
      mimic_notes = json.load(infile)
     
   # Manual Annotations
   with open(annotations_dir) as infile:
      annotations = infile.read()
      annotations = eval(annotations)

   # Claim-based eval scores
   with open(claim_dir) as infile:
      claim_scores = json.load(infile)  
   
   # document keys with manual annotations
   annotated_keys = [k for k,_ in annotations[0].items()]
   
   # tiktoken tokenizer for contex size
   token_text = next(iter(mimic_notes.values()))['discharge summary'][0]
   tik_tokenizer = tiktoken.get_encoding('p50k_base')
   
   
   scored_summaries_structured, scored_summaries_full,structured_summaries = generate_structure_score(mimic_notes,annotated_keys)   
   baseline_scores, baseline_scores_full = get_baseline_scores(structured_summaries)
   
   gpt_frequent_values = get_most_frequent_values(scored_summaries_structured)
   

   gpt_user, gpt_user_full, r1_user, r1_user_full, rL_user, rL_user_full, bert_user, bert_user_full = get_correlation_scores(baseline_scores, baseline_scores_full,scored_summaries_full, gpt_frequent_values,annotations)
   
   
   print()

   lst_keys = [k for k,_ in gpt_user.items()]
   lst_keys.insert(0,"                   ")
   print(' '.join(f"{item:7} " for item in lst_keys))

   print_output("GPT unstructured ",gpt_user_full)
   print_output("GPT structured   ",gpt_user)
   print_output("R1 unstructured  ", r1_user_full)
   print_output("R1 structured    ",r1_user)
   print_output("RL unstructured  ",rL_user_full)
   print_output("RL structured    ",rL_user)
   print_output("BERT unstructured",bert_user_full)
   print_output("BERT structured  ",bert_user)
   
   claim_full = get_error_scores_full2(claim_scores,annotations)
   print_output("claim-based      ",claim_full)

   