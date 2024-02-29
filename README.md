# Attribute structuring(AS)

Attribute structuring based GPT scoring of generated text summaries

Requirements:
1) Access to MIMIC III dataset (https://physionet.org/content/mimiciii/)
2) Manual annotation of selected MIMIC Discharge Summaries
   Here we included annotations for 30 documents by 3 annotators. 
3) Access to GPT APIs. Here we used GPT4 from via Microsoft Azure API

How to Run:
-- install the libraries in the requirements.txt file
-- add your api_base and api_key to your system environment
-- modify/set the gpt parameters inside src/variables [OPTIONAL]
-- replace the [MIMIC_DIR] inside src/main with your mimic path
-- run the src/main.py script

