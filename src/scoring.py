

from variables import attr_dict
from variables import prompt_score
from variables import prompt_score_full
from variables import llm

import openai 
import time
import re
from collections import defaultdict
import math
from scipy.stats import pearsonr, kendalltau, spearmanr
import numpy as np

def extract_numbers(text):
    pattern = r"[-+]?\d*\.?\d+"
    matches = re.findall(pattern, text)
    
    # Convert matched strings to numbers (floats or integers)
    numbers = [float(match) if '.' in match else int(match) for match in matches]
    
    return numbers[0]

def score_structured(structured_dict):
    scores = {}

    for instance_id, instance_data in structured_dict.items():
        if instance_id in scores or not instance_data['ground_structured'] or not instance_data['generated_structured']:
            continue

        instance_scores = {}
        for variable, ground_value in instance_data['ground_structured'].items():
            ground_value_str = "".join(ground_value)
            generated_value_str = "".join(instance_data['generated_structured'][variable])

            if ground_value_str.lower() == 'none' or generated_value_str.lower() == 'none':
                instance_scores[variable] = 4 if ground_value_str.lower() == 'none' and generated_value_str.lower() == 'none' else 1
                continue

            cur_prompt = prompt_score.replace('{{variable}}', attr_dict[variable]) \
                                  .replace('{{value1}}', ground_value_str) \
                                  .replace('{{value2}}', generated_value_str)
            
            conversation = [{"role":"system", "content":cur_prompt}]
           
            for _ in range(3):
                try:
                    
                    response = openai.ChatCompletion.create(engine=llm.deployment_name,messages=conversation)
                    response_text = response['choices'][0]['message']['content']
                    parsed_text = extract_numbers(response_text)

                    if float(parsed_text) in [1,2,3,4]:
                        instance_scores[variable] = parsed_text
                        break
                except openai.error.RateLimitError:
                    time.sleep(5)  
                except Exception as e:
                    print(e) 
     
        scores[instance_id] = instance_scores

    return scores


def score_full(unstructured_dict):
    scores = {}

    for instance_id, instance_data in unstructured_dict.items():
        if instance_id in scores or not instance_data['ground_text'] or not instance_data['generated_text']:
            continue

        instance_scores = []
        
        ground_value = instance_data['ground_text']
        generated_value = instance_data['generated_text']

            

        cur_prompt = prompt_score_full.replace('{{value1}}', ground_value) \
                                .replace('{{value2}}', generated_value)
        
        conversation = [{"role":"system", "content":cur_prompt}]
        
        for _ in range(5):
            try:
                #response = chat_completion(params,conversation)
                response = openai.ChatCompletion.create(engine=llm.deployment_name,messages=conversation)
                response_text = response['choices'][0]['message']['content']
                parsed_text = extract_numbers(response_text)
                
                if float(parsed_text) in [1,2,3,4,5,6,7,8,9]:
                    instance_scores.append(parsed_text)
                    
            except openai.error.RateLimitError:
                time.sleep(5)  
            except Exception as e:
                    print(e) 
     
        scores[instance_id] = instance_scores

    return scores

# return the most common score value from multiple runs on a model
def most_frequent_value(sample_input):
  most_frequent_values = {}

  for key, dictionaries in sample_input.items():
      # Create a defaultdict to count occurrences of values for each inner key
      value_counts = defaultdict(lambda: defaultdict(int))

      # Iterate through dictionaries for the current key
      for d in dictionaries:
          for sub_key, sub_value in d.items():
              value_counts[sub_key][sub_value] += 1

      # Find the most frequent value for each inner key
      for sub_key, counts in value_counts.items():
          most_frequent_value = max(counts, key=counts.get)
          if key not in most_frequent_values:
              most_frequent_values[key] = {}
          most_frequent_values[key][sub_key] = most_frequent_value
  return most_frequent_values        




def get_most_frequent_values(multi_run_input):
    sample_input = defaultdict(list)
    for item in multi_run_input:
      for k,v in item.items():
        sample_input[k].append(v)

    return most_frequent_value(sample_input)
def get_values(dict1,dict2):
  common_keys = sorted([k for k in (set(dict1.keys()) & set(dict2.keys())) if k!='author'])
  values1 = [float(dict1[key]) for key in common_keys]
  values2 = [float(dict2[key]) for key in common_keys]
  return values1, values2

def get_values_multiple(dict_list):
    if not dict_list:
        # If the list is empty, return empty lists
        return [], []

    # Extract common keys excluding 'author'
    common_keys = sorted(set.intersection(*(set(d.keys()) for d in dict_list)) - {'author'})

    # Extract values for common keys from each dictionary in the list
    values_list = [[float(d[key]) for key in common_keys] for d in dict_list]

    return values_list

def normalize_list(lst, min_val=1, max_val=4):
    normalized_lst = [round((x - min_val) / (max_val - min_val),2) for x in lst]
    return normalized_lst
def calculate_mae(values1, values2):
    absolute_errors = [abs(v1 - v2) for v1,v2 in zip(values1, values2)]
    #print(absolute_errors)
    mae = sum(absolute_errors) / len(absolute_errors)
    return mae

def calculate_rmse(values1, values2):
    squared_errors = [(v1 - v2)**2 for v1,v2 in zip(values1, values2)]
    mean_squared_error = sum(squared_errors) / len(squared_errors)
    rmse = math.sqrt(mean_squared_error)
    return rmse
def calculate_correlations(values1, values2):
    # Calculate the minimum and maximum values in both sets of values
    min_val_1 = min(values1)
    min_val_2 = min(values2)
    max_val_1 = max(values1)
    max_val_2 = max(values2)

    # Normalize the values using min-max scaling to be between 0 and 1
    values1 = [(val - min_val_1) / (max_val_1 - min_val_1) for val in values1]
    values2 = [(val - min_val_2) / (max_val_2 - min_val_2) for val in values2]

    # print(values1)
    # print(values2)
    # Calculate Pearson correlation
    pearson_corr, _ = pearsonr(values1, values2)

    # Calculate Kendall's Tau correlation
    kendall_tau_corr, _ = kendalltau(values1, values2)

    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(values1, values2)

    return pearson_corr, kendall_tau_corr, spearman_corr

def get_error_scores2(auto_scoring,annotations, normalize = False):
  comparisons = defaultdict(list)
  value1, value2 = [], []
  for k,v in auto_scoring.items():
    
    val1,val2 = get_values(v,annotations[0][k])
    val4 = []
    val4.append(val2)
    for ann in annotations[1:]:
      _,val3 = get_values(v,ann[k])
      val4.append(val3)
      
    val2 = [sum(elements) / len(val4) for elements in zip(*val4)]

    if normalize:
      val1 = normalize_list(val1)
    val2 = normalize_list(val2)
    
    value1.append(np.mean(val1))
    value2.append(np.mean(val2))
  
  
  mae = calculate_mae(value1,value2)
  rmse = calculate_rmse(value1,value2)
  P,K,S = calculate_correlations(value1,value2)

  comparisons['mae'] = float("{:.3f}".format(mae))
  comparisons['rmse'] = float("{:.3f}".format(rmse))
  comparisons['pearson'] = float("{:.3f}".format(P))
  comparisons['kendal'] = float("{:.3f}".format(K))
  comparisons['spearman'] = float("{:.3f}".format(S))
  
  return comparisons  

def get_error_scores_full2(auto_scoring,annotations):
  comparisons = defaultdict(list)
  value1, value2 = [], []
  for k,v in auto_scoring.items():
    for ann in annotations:
      mult_vals = get_values_multiple([ann[k] for ann in annotations])
      #_,val2 = get_values(annotations1[k],annotations1[k])
      #   = get_values(annotation1[k],v)
    average_values = [sum(col) / len(col) for col in zip(*mult_vals)]
    
    
    value1.append(np.mean(normalize_list(average_values)))
    #value1.append(np.mean(average_values))
    value2.append(v)
  
  mae = calculate_mae(value1,value2)
  rmse = calculate_rmse(value1,value2)
  P,K,S = calculate_correlations(value1,value2)

  comparisons['mae'] = float("{:.3f}".format(mae))
  comparisons['rmse'] = float("{:.3f}".format(rmse))
  comparisons['pearson'] = float("{:.3f}".format(P))
  comparisons['kendal'] = float("{:.3f}".format(K))
  comparisons['spearman'] = float("{:.3f}".format(S))
  
  return comparisons  

