
from structuring import structure_text


def structure_ground_gen_summaries(generated_summaries, ground_summaries):
    # structure both the generated and ground summaries to extract attributes
    structured_summaries = {}
    for k,v in generated_summaries.items():
        gen_structured= structure_text(v)  
        ground_structured = structure_text(ground_summaries[k]['summary'])  

        structured_summaries[k] = {"ground_text":ground_summaries[k]['summary'],"ground_structured":ground_structured,"generated_text":v,"generated_structured":gen_structured}  

    return structured_summaries

