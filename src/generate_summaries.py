
from variables import context_query
from variables import llm
import time
import openai


def summarize_text(token_text):
    context_query_mod = {"role": "user", "content": context_query+token_text}
    conversation=[{"role": "system", "content": "You are a helpful assistant."},context_query_mod,]

    response_text  = "[]"
    for _ in range(5):
        try:
            response = openai.ChatCompletion.create(engine=llm.deployment_name,messages=conversation,max_tokens=2000,temperature=0)
            response_text = response['choices'][0]['message']['content']
            return response_text
        except openai.error.RateLimitError:
            time.sleep(5)  
        except Exception as e:
            print(e) 
            continue     
    
    return response_text


def synthetic_summary(full_text, tiktoken_len):
  
  num_tokens = tiktoken_len(full_text)

  # make sure the tokens arent larger than the model context
  while num_tokens > 30000:
    full_text = full_text[:-500]  
    num_tokens = tiktoken_len(full_text)
  generated = summarize_text(full_text)
  return generated

def generate_summaries(test_notes, tiktoken_len):
    # generate summary from the source text

    generated_notes = {}
    for k,v in test_notes.items():
        generated_notes[k] = synthetic_summary(v['full'], tiktoken_len)

    return generated_notes

