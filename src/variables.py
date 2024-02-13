from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import  ChatPromptTemplate
from langchain.prompts import  HumanMessagePromptTemplate
from langchain.chat_models import AzureChatOpenAI

import openai
import os 


prompt_score = """You will be given a python dictionary containing a clinical variable as a key and a list containing two values for the variable as values.

Your task is to rate how similar the values are given the variable. Compare value1 and value2 for semantic similarity(similarity in meaning) given the variable and the criteria below. Two values can be very similar in meaning even if they are phrased differently. Also remember that this is a clinical document, take that into account.
When scoring the similarity between two clinical terminologies, assign a value from 1 to 4, where 1 signifies a lack of similarity and 4 indicates identical meanings. Consider the context, clinical relevance, and semantic alignment between the terminologies. If the terminologies convey vastly different meanings, assign a score of 1. A score of 2 is appropriate for terminologies that, while related, represent different concepts or emphasize distinct elements. A score of 3 should be used for terminologies with substantial semantic overlaps but minor differences. Finally, if the terminologies are semantically equivalent and interchangeable, warranting no clinical distinction, assign a score of 4. Ensure that the assessment reflects the degrees of similarity in meaning, irrespective of syntactical differences, context, or minor variances in expression.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Return the similarity score. Return only the score, don't include any other extra text. 

Example:
variable: 
{{variable}}
value1: {{value1}}
value2: {{value2}}

Evaluation Form (scores ONLY):

 """

prompt_score_full = """You will be given two clinical discharge summaries, summary1 and summary2.

Your task is to rate how similar the two summaries are from 1-10. 1 is not similar while 10 is essentially identical. Focus on the following important clinical variables when performing the comparison:
  How similar is the diagnosis, how similar are the goals, how similar is hospital course and history, how similar are the medications administered, how similar are the physical condition diagnoses, how similar are the followup consults and procedures, how similar are the lab tests performed, how similar are the patients discharge status, is the follow-up instructions similar, are there any similar appointments, and instructions
only return the score, dont add any extra text
Here are the inputs discharge summaries:
 summary1: {{value1}}
 summary2: {{value2}}. 
 """

attr_dict = {'ad_diag':'preliminary or working diagnosis given at the time of admission',
'dc_diag':'the list of principal discharge diagnosis or main reason for admission and all additional pertinent diagnoses where applicable',
'main_diag':'diagnosis mostly accountable for the largest portion of the patient"s stay, responsible for the greatest part of the length of stay ',
'history':'a brief summary of initial presentation and diagnostic evaluation',
'physical': 'pertinent physical findings relevant to diagnoses ',
'goals':'goals of care; level of treatment,code status(e.g. curative,life-prolonging palliative, and symptomatic palliative)',
'course':'course in hospital; synotpic,problem-based description of sequential events and respective evaluations, treatments, and prognoses ',
'consults':'hospital consults; description of specialty and/or allied health consults',
'procedures':'procedures in hospital; a list of procedures with key findings and date',
 'ds_med':'a list of all discharge medications with specific description of new, altered, and discontinued medications and rationale for changes',
'lab':'pertinent lab tests and investigative results',
'ds_test':'tests ordered during the hospitalization that are pending at the time of discharge ',
'ds_status':'outcome of care/condition at discharge; sense of the patient health status at discharge includes functional status, and cognitive status',
'followup':'outstanding issues for follow-up and recommendations to a recipient health-care provider during discharge',
'appt':'appointments after discharge including person responsible for scheduling, care provider ',
'instruct':'discharge instructions; list of information/education provided to the patient during discharge',
'author':'main author of the discharge summary or attending clinician'}


key_mapping = {'adm_diag_score':"ad_diag",
'ds_diag_score':"dc_diag",
'main_diag_score':"main_diag",
'hist_score':"history",
'physical_score': "physical",
'goals_score':"goals" ,
'course_score':"course",
'consult_score': "consults",
'proc_score': "procedures",
 'ds_med_score': "ds_med" ,
 'lab_score': "lab",
  'ds_test_score':"ds_test",
  'ds_status_score': "ds_status" ,
  'follow_score': "followup",
  'appt_score': "appt" ,
  'instruct_score':"instruct",
  'author_score': "author"}



ad_diag_schema = ResponseSchema(name = 'ad_diag', description='preliminary or working diagnosis given at the time of admission')
dc_diag_schema = ResponseSchema(name = 'dc_diag', description='the list of principal discharge diagnosis or main reason for admission and all additional pertinent diagnoses where applicable')
main_diag_schema = ResponseSchema(name = 'main_diag', description='diagnosis mostly accountable for the largest portion of the patient"s stay, responsible for the greatest part of the length of stay ')
history_schema = ResponseSchema(name = 'history', description='a brief summary of initial presentation and diagnostic evaluation')
physical_schema = ResponseSchema(name = 'physical', description='pertinent physical findings relevant to diagnoses ')
goals_schema =ResponseSchema(name = 'goals', description='goals of care; level of treatment,code status(e.g. curative,life-prolonging palliative, and symptomatic palliative)') 
course_schema = ResponseSchema(name = 'course', description='course in hospital; synotpic,problem-based description of sequential events and respective evaluations, treatments, and prognoses ')
consults_schema = ResponseSchema(name = 'consults', description='hospital consults; description of specialty and/or allied health consults')
procedures_schema = ResponseSchema(name = 'procedures', description='procedures in hospital; a list of procedures with key findings and date')
ds_med_schema = ResponseSchema(name = 'ds_med', description='a list of all discharge medications with specific description of new, altered, and discontinued medications and rationale for changes')
lab_schema = ResponseSchema(name = 'lab', description='pertinent lab tests and investigative results')
ds_test_schema = ResponseSchema(name = 'ds_test', description='tests ordered during the hospitalization that are pending at the time of discharge ')
ds_status_schema = ResponseSchema(name = 'ds_status', description='outcome of care/condition at discharge; sense of the patient health status at discharge includes functional status, and cognitive status')
followup_schema = ResponseSchema(name = 'followup', description='outstanding issues for follow-up and recommendations to a recipient health-care provider during discharge')
appt_schema = ResponseSchema(name = 'appt', description='appointments after discharge including person responsible for scheduling, care provider ')
instruct_schema = ResponseSchema(name = 'instruct', description='discharge instructions; list of information/education provided to the patient during discharge')
author_schema = ResponseSchema(name = 'author', description='main author of the discharge summary or attending clinician')

response_schemas_2 = [ad_diag_schema,dc_diag_schema, main_diag_schema, history_schema,physical_schema, goals_schema, course_schema, consults_schema, procedures_schema, ds_med_schema, lab_schema, ds_test_schema, ds_status_schema, followup_schema, appt_schema,instruct_schema,author_schema]
output_parser_2 = StructuredOutputParser.from_response_schemas(response_schemas_2)
format_instructions_2 = output_parser_2.get_format_instructions()

template_string = """You are an expert in information extraction and structuring from clinical notes.  Given a clinical note, create a structured output. For a given variable, if you can not determine/find a value, return NONE. Dont add any extra text, just the structured value. Here is the note: {text_note}. {format_instructions}
"""
prompt = ChatPromptTemplate(messages=[HumanMessagePromptTemplate.from_template(template_string)],
                            input_variables=['text_note'],
                            partial_variables={"format_instructions":format_instructions_2},
                            output_parser=output_parser_2
)



context_query = """ You are tasked with generating a high quality clinical discharge summary for the provided input text {token_text}. The summary has to be very relevant to the input document and cover the most important aspects of the input. Follow the following steps:
  Step 1: Generate the most common sections that usually appear in a clinical discharge summary.
  Step 1.5: Not all the sections from Step 1 will have content. See which of those sections have contents from the {token_text}. Remove the sections that don't have information.
  Step 2 : Use the following concepts to generate appropriate content. These are not Sections. Concepts:
    patient information and service type, diagnosis given at the time of admission,  brief summary of initial presentation and diagnostic evaluation, pertinent physical findings relevant to diagnoses, goals of care; level of treatment,code status(e.g. curative,life-prolonging palliative, and symptomatic palliative), course in hospital; synotpic,problem-based description of sequential events and respective evaluations, treatments, and prognoses, hospital consults; description of specialty and/or allied health consults, procedures in hospital; a list of procedures with key findings and date, principal discharge diagnosis or main reason for admission and all additional pertinent diagnoses where applicable,  discharge medications with specific description of new, altered, and discontinued medications and rationale for changes,  lab tests and investigative results, tests ordered during the hospitalization that are pending at the time of discharge, outcome of care/condition at discharge; sense of the patient health status at discharge includes functional status, and cognitive status, outstanding issues for follow-up and recommendations to a recipient health-care provider during discharge, appointments after discharge including person responsible for scheduling, care provider, discharge instructions; list of information/education provided to the patient during discharge, main author of the discharge summary or attending clinician. 

   Step 3: Put the generated content in a coherent order. Format in such a way that the dishcarge summary has an excellent coherence, fluency, and consistency. Remember to use the sections you generated in Step 1, don't make the concepts in Step 2 as a section, they are just suggestive concepts not sections. 

   Step 4: Remove sections with no information. Dont put 'not specified' or 'not mentioned' or 'none specified' in a section. Just remove everything for that section including the section header.
   Step 5: Return the final discharge summary with all the remaining sections that have contents. Remember to remove sections with no information
     
    Context : 
          """

api_key = os.environ.get("OPENAI_API_KEY")
api_base = os.environ.get("OPENAI_API_BASE")
api_params = {'api_key': api_key, 'api_type': "azure", "api_base":api_base, "api_version":"2023-03-15-preview"}
model_params  = {"model":"gpt-4-32k-0314", "temperature":0.1, "max_tokens":2000, "top_p":1}

openai.api_type= api_params['api_type']
openai.api_version= api_params['api_version']
openai.api_base= api_params['api_base']
openai.api_key = api_params['api_key']

llm = AzureChatOpenAI(deployment_name='gpt-4-32k-0314', openai_api_version=api_params['api_version'], temperature=0.1,max_tokens=2000)
  