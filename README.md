# Attribute structuring(AS)

Attribute structuring based GPT scoring of generated text summaries.
Given a ground truth discharge summary and a summary generated by a model, AS uses an ontology to extract important variables, and scores the generated summary using GPT4.  

## Requirements:
1) Access to MIMIC III dataset (https://physionet.org/content/mimiciii/)
2) Manual annotation of selected MIMIC Discharge Summaries.
   data/annotations.json contains annotations for 30 documents by 3 annotators. 
3) Access to GPT APIs. Here we used GPT4 via Microsoft Azure API to structure and score the summaries.

## How to Run:

1) install the libraries in the requirements.txt file
2) add your api_base and api_key to your system environment
3)  modify/set the gpt parameters inside src/variables [OPTIONAL]
4) replace the [MIMIC_DIR] inside src/main with your mimic path
5) -- run the src/main.py script



