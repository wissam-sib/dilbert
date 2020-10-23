BERT_PATH = 'bert_output_folder/'
DILBERT_PATH = 'dilbert_output_folder/'
ALBERT_PATH = "albert_output_folder/"
DILALBERT_PATH = 'dilalbert_output_folder'
OPEN_SQUAD_PATH = 'formatted_open_squad/open_squad.pkl'
WIKI_INDEX_PATH = 'formatted_open_squad/indexes/paragraphs_indexing'
USED_DEVICE = "cpu" #or "cpu"

import pickle
from pyserini.search import SimpleSearcher

import argparse
import json
import os
import time

from transformers import BertTokenizer, BertForQuestionAnswering, AlbertTokenizer, AlbertForQuestionAnswering

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from dilalbert import DilAlbert
from dilbert import DilBert
import numpy as np
import torch
from squad_tools import build_squad_input, evaluate
from datetime import datetime
from transformers.data.metrics.squad_metrics import _get_best_indexes, get_final_text, _compute_softmax
import collections

tokenizer = BertTokenizer.from_pretrained(DILBERT_PATH, do_lower_case=True)
model = DilBert.from_pretrained(DILBERT_PATH)

device = USED_DEVICE
print("Device = ", device)
model.to(torch.device(device))

n_questions = 100
n_paragraphs = 100

with open(OPEN_SQUAD_PATH, 'rb') as f1:
    squad1_for_orqa = pickle.load(f1)
    
question_load_par = squad1_for_orqa["questions"][0]

searcher = SimpleSearcher(WIKI_INDEX_PATH)
searcher.set_bm25()
searcher.unset_rm3()

hits = searcher.search(question_load_par, k=n_paragraphs)

paragraphs = [hit.raw for hit in hits]

def build_squad_input(question: str, contexts: [str]) -> dict:
    global_id = 0
    bert_input = {
        'data': [{
            'title': "time_test",
            'paragraphs': []  # Going to fill this one with the pages and question
        }],
        'version': "2.0"
    }

    for i in range(len(contexts)):
        context = contexts[i]
        global_id += 1
        id_ = global_id
        element = {
            'context': context,
            'qas': [{'id': id_,
                     'question': question,
                     'is_impossible': True,
                     'answers': [{'text': "", 'answer_start': 0}]
                    }]
        }
        bert_input['data'][0]['paragraphs'].append(element)
    return bert_input

bert_input = build_squad_input("[SEP]", paragraphs)
squad_processor = SquadV2Processor()

examples = squad_processor._create_examples(bert_input["data"], "dev")
features, dataset = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=357,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    return_dataset="pt",
    threads=1,
)

total_time_passages = 0
preprocessed_items = []
eval_dataloader = DataLoader(dataset, batch_size=1)
for batch in eval_dataloader:
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0][:,3:],
            "attention_mask": batch[1][:,3:],
            "token_type_ids": batch[2][:,3:],
        }
        example_indices = batch[3]
        start = time.time()
        outputs = model.process_A(**inputs)
        total_time_passages = total_time_passages + time.time() - start
    # 2034MiB - 183 examples
    preprocessed_items.append(outputs)
    
print("NI P DilBert :", total_time_passages)

total_time_questions = 0
total_time_questions_paragraphs_pairs = 0
for question in squad1_for_orqa["questions"][:n_questions]:
    input_ids = torch.tensor([tokenizer.encode(question)], device=device)
    attention_mask = torch.tensor([[1]*input_ids.shape[1]], device=device)
    token_type_ids = torch.tensor([[0]*input_ids.shape[1]], device=device)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    start = time.time()
    preprocessed_question = model.process_A(**inputs)
    total_time_questions = total_time_questions + time.time() - start
    for preprocessed_paragraph in preprocessed_items:
        with torch.no_grad():
            start = time.time()
            outputs = model.process_B(preprocessed_question, preprocessed_paragraph)
            total_time_questions_paragraphs_pairs = total_time_questions_paragraphs_pairs + time.time() - start

    
print("NI Q DilBert :", total_time_questions)
print("I Q-P DilBert :", total_time_questions_paragraphs_pairs)
total_dilbert = total_time_passages+total_time_questions+total_time_questions_paragraphs_pairs
print("Total Dilbert : ", total_dilbert)

tokenizer = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
model = BertForQuestionAnswering.from_pretrained(BERT_PATH)
model.to(torch.device(device))

total_time_questions_paragraphs_pairs_bert = 0
eval_dataloader = DataLoader(dataset, batch_size=1)
for question in squad1_for_orqa["questions"][:n_questions]:
    input_ids = torch.tensor([tokenizer.encode(question)], device=device)
    attention_mask = torch.tensor([[1]*input_ids.shape[1]], device=device)
    token_type_ids = torch.tensor([[0]*input_ids.shape[1]], device=device)
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": torch.cat((input_ids,batch[0][:,3:]),1),
                "attention_mask": torch.cat((attention_mask,batch[1][:,3:]),1),
                "token_type_ids": torch.cat((token_type_ids,batch[2][:,3:]),1),
            }
        example_indices = batch[3]
        start = time.time()
        model(**inputs)
        total_time_questions_paragraphs_pairs_bert = total_time_questions_paragraphs_pairs_bert + time.time() - start

print("I Q-P Bert: ", total_time_questions_paragraphs_pairs_bert)
total_bert = total_time_questions_paragraphs_pairs_bert
print("Total Bert : ", total_bert)
print("Speedup Dilbert : ", total_bert/total_dilbert)


tokenizer = AlbertTokenizer.from_pretrained(DILALBERT_PATH, do_lower_case=True)
model = DilAlbert.from_pretrained(DILALBERT_PATH)
model.to(torch.device(device))

features, dataset = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=357,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    return_dataset="pt",
    threads=1,
)

total_time_passages = 0
preprocessed_items = []
inputs_par = []
eval_dataloader = DataLoader(dataset, batch_size=1)
for batch in eval_dataloader:
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0][:,3:],
            "attention_mask": batch[1][:,3:],
            "token_type_ids": batch[2][:,3:],
        }
        inputs_par.append(inputs)
        example_indices = batch[3]
        start = time.time()
        outputs = model.process_A(**inputs)
        total_time_passages = total_time_passages + time.time() - start
    # 2034MiB - 183 examples
    preprocessed_items.append(outputs)
    
print("NI P DilAlbert :", total_time_passages)

total_time_questions = 0
total_time_questions_paragraphs_pairs = 0
for question in squad1_for_orqa["questions"][:n_questions]:
    input_ids = torch.tensor([tokenizer.encode(question)], device=device)
    attention_mask = torch.tensor([[1]*input_ids.shape[1]], device=device)
    token_type_ids = torch.tensor([[0]*input_ids.shape[1]], device=device)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    start = time.time()
    preprocessed_question = model.process_A(**inputs)
    total_time_questions = total_time_questions + time.time() - start
    for i,preprocessed_paragraph in enumerate(preprocessed_items):
        with torch.no_grad():            
            inputs = {
                "input_ids": torch.cat((input_ids,inputs_par[i]["input_ids"]),1),
                "attention_mask": torch.cat((attention_mask,inputs_par[i]["attention_mask"]),1),
                "token_type_ids": torch.cat((token_type_ids,inputs_par[i]["token_type_ids"]),1),
            }
            start = time.time()
            outputs = model.process_B(preprocessed_question, preprocessed_paragraph,**inputs)
            total_time_questions_paragraphs_pairs = total_time_questions_paragraphs_pairs + time.time() - start


print("NI Q DilAlbert :", total_time_questions)
print("I Q-P DilAlbert :", total_time_questions_paragraphs_pairs)
total_dilalbert = total_time_passages+total_time_questions+total_time_questions_paragraphs_pairs
print("Total DilAlbert : ", total_dilalbert)

tokenizer = AlbertTokenizer.from_pretrained(ALBERT_PATH, do_lower_case=True)
model = AlbertForQuestionAnswering.from_pretrained(ALBERT_PATH)
model.to(torch.device(device))

total_time_questions_paragraphs_pairs_albert = 0
eval_dataloader = DataLoader(dataset, batch_size=1)
for question in squad1_for_orqa["questions"][:n_questions]:
    input_ids = torch.tensor([tokenizer.encode(question)], device=device)
    attention_mask = torch.tensor([[1]*input_ids.shape[1]], device=device)
    token_type_ids = torch.tensor([[0]*input_ids.shape[1]], device=device)
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": torch.cat((input_ids,batch[0][:,3:]),1),
                "attention_mask": torch.cat((attention_mask,batch[1][:,3:]),1),
                "token_type_ids": torch.cat((token_type_ids,batch[2][:,3:]),1),
            }
        example_indices = batch[3]
        start = time.time()
        model(**inputs)
        total_time_questions_paragraphs_pairs_albert = total_time_questions_paragraphs_pairs_albert + time.time() - start

print("I Q-P Albert: ", total_time_questions_paragraphs_pairs_albert)
total_albert = total_time_questions_paragraphs_pairs_albert
print("Total Albert : ", total_albert)
print("Speedup DilAlbert : ", total_albert/total_dilalbert)
