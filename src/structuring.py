
import openai 

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

import time

from variables import prompt
from variables import llm 



def structure_text(text):
    for _ in range(3):
        try:
            chain = LLMChain(llm=llm,prompt=prompt)
            out_text = chain.predict_and_parse(text_note = text)
            return out_text
        except openai.error.RateLimitError:
            time.sleep(5)  
        except Exception as e:
            print(e)
            pass   
    return []  


