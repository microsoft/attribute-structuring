import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from bert_score import score as bert_score
from bert_score import BERTScorer

from collections import defaultdict
import json

def normalize_list2(lst, min_val=-1, max_val=1):
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst  


def calculate_metric_scores(ds_scores2,full_eval=False):
  scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'])
  bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)

  metric_scores = defaultdict(list)
  
  for k2, v2 in ds_scores2.items():

    if full_eval:
      preds = [v2['generated_text']]
      target = [v2['ground_text']]
      rouge =  scorer.score(target[0], preds[0])
      rouge_1 = rouge['rouge1'][2]
      rouge_l = rouge['rougeL'][2]
      P, R, F1 = bert_score(preds, target, lang="en")
      metric_scores[k2].append((rouge_1,rouge_l,F1.mean()))
      
    else:
      preds = [v2['generated_structured']]
      targets = [v2['ground_structured']]
      keys = sorted([key for key in preds[0]])
      pred_vals = [preds[0][key] for key in keys]
      target_vals = [targets[0][key] for key in keys]

      predicted = {}
      pred_r1, pred_rL, bertscore = {}, {},{}
      
      _,_,bertscore = bertscorer.score(pred_vals, target_vals)
      
      bertscore = normalize_list2(bertscore.numpy())
      bertscore_dict = {key:val for key,val in zip(keys,bertscore)}
      
      for kk in preds[0]:
        pred_r1[kk] = scorer.score(targets[0][kk], preds[0][kk])['rouge1'][2]
        pred_rL[kk] = scorer.score(targets[0][kk], preds[0][kk])['rougeL'][2]
        
        
      predicted['rouge1'] = pred_r1  
      predicted['rougeL'] = pred_rL
      predicted['bertscore'] = bertscore_dict
      metric_scores[k2] = predicted
      
  return metric_scores     

