

from variables import attr_dict
from variables import prompt_score
from variables import prompt_score_full
from variables import llm

import openai 
import time
import re


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

