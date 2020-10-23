PATH_TO_WIKI_INDEX = 'formatted_open_squad/indexes/paragraphs_indexing'
PATH_TO_DILBERT = 'dilbert_output_folder/'
DEVICE_COMP = "cuda" #or "cpu"

import pickle
from pyserini.search import SimpleSearcher

import argparse
import json
import os
import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import squad_convert_examples_to_features
from transformers import BertTokenizer
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

import numpy as np
import torch

from squad_tools import build_squad_input, evaluate
from datetime import datetime
from dilalbert import DilAlbert
from dilbert import DilBert
from transformers.data.metrics.squad_metrics import _get_best_indexes, get_final_text, _compute_softmax
import collections

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def process_one_question(features, dataset, model, tokenizer, examples, device, use_ir_score=False, mu=0.0,
                                                 ir_scores=None):
    all_results = []
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=12)

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output
            if (use_ir_score):
                ir_scores_seq = np.ones(len(start_logits)) * ir_scores[eval_feature.example_index]
                start_logits = list(np.array(start_logits) * (1 - mu) + mu * ir_scores_seq)
                end_logits = list(np.array(end_logits) * (1 - mu) + mu * ir_scores_seq)
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    prefix = ""
    output_dir = "./tmp_dir"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    curr_date_str = str(time.time())
        
    output_prediction_file = os.path.join(output_dir, curr_date_str + "_predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, curr_date_str + "_nbest_predictions_{}.json".format(prefix))
    output_null_log_odds_file = os.path.join(output_dir, curr_date_str + "_null_odds_{}.json".format(prefix))

    compute_predictions_logits_all(
        examples,
        features,
        all_results,
        20,    # 20 args.n_best_size,
        384,    # args.max_answer_length,
        True,    # args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,    # args.verbose_logging,
        False,    # args.version_2_with_negative,
        0.0,    # args.null_score_diff_threshold,
        tokenizer,
    )

    predictions = json.load(
        open(os.path.join(output_dir, curr_date_str + "_nbest_predictions_{}.json".format(prefix)), 'r'))
    
    if os.path.exists(output_prediction_file):
        os.remove(output_prediction_file)
    if os.path.exists(output_nbest_file):
        os.remove(output_nbest_file)
    if os.path.exists(output_null_log_odds_file):
        os.remove(output_null_log_odds_file)
    
    return all_results, predictions


def compute_predictions_logits_all(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,
):
    """This is a function from the transformer library modified to work on the multi-passage setting"""
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(    # pylint: disable=invalid-name
        "PrelimPrediction", ["example_index", "feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    prelim_predictions = []

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000    # large and positive
        min_null_feature_index = 0    # the paragraph slice with min null score
        null_start_logit = 0    # the start logit at the slice with min null score
        null_end_logit = 0    # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            example_index=example_index,
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(    # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        example = all_examples[pred.example_index]
        features = example_index_to_features[pred.example_index]
        feature = features[pred.feature_index]

        if pred.start_index > 0:    # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # tok_text = " ".join(tok_tokens)
            #
            # # De-tokenize WordPieces that have been split off.
            # tok_text = tok_text.replace(" ##", "")
            # tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
    # if we didn't include the empty option in the n-best, include it
    if version_2_with_negative:
        if "" not in seen_predictions:
            nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

        # In very rare edge cases we could only have single null prediction.
        # So we just create a nonce prediction in this case to avoid failure.
        if len(nbest) == 1:
            nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not version_2_with_negative:
        all_predictions["0"] = nbest_json[0]["text"]    # all_predictions[example.qas_id] = nbest_json[0]["text"] same below..
    else:
        # predict "" iff the null score - the score of best non-null > threshold
        score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
        scores_diff_json[example.qas_id] = score_diff
        if score_diff > null_score_diff_threshold:
            all_predictions["0"] = ""
        else:
            all_predictions[example.qas_id] = best_non_null_entry.text
    all_nbest_json["0"] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, prelim_predictions

class QAengine():    
    def __init__(self):
        self.searcher = SimpleSearcher(PATH_TO_WIKI_INDEX)
        self.searcher.set_bm25()
        self.searcher.unset_rm3()
        self.processor = SquadV2Processor()
        self.k = 29
        self.mu = 0.5
        self.use_ir_score = True
        self.tokenizer = BertTokenizer.from_pretrained(PATH_TO_DILBERT, do_lower_case=True)
        self.model = DilBert.from_pretrained(PATH_TO_DILBERT)
        self.device = DEVICE_COMP 
        self.model.to(torch.device(self.device))
    
    def answer(self,question):
        hits = self.searcher.search(question, k=self.k)
        ir_scores = []
        paragraphs = []
        for j in range(len(hits)):
            passage = hits[j].raw
            ir_scores.append(hits[j].score)            
            paragraphs.append(passage)
        input_ = build_squad_input(question, paragraphs)
        examples = self.processor._create_examples(input_["data"], "dev")
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            threads=1,
        )
        all_results, predictions = process_one_question(features, dataset, self.model, self.tokenizer, examples, self.device, self.use_ir_score, self.mu,ir_scores)
        
        scores = np.array([(p['start_logit'] + p['end_logit']) for p in predictions['0']])
        texts = [p['text'] for p in predictions['0']]

        predicted_p_indexes_all = scores.argsort()[::-1].argsort()
        iterator_idx = 0
        is_empty = True
        predicted_p_index = 0
        while is_empty and iterator_idx < len(predicted_p_indexes_all):
            predicted_p_index = predicted_p_indexes_all[iterator_idx]
            is_empty = texts[predicted_p_index] == "empty"
            iterator_idx += 1

        predicted_answer = texts[predicted_p_index]
        return predicted_answer
        
my_qa_engine = QAengine()
        
while True:
    question = input("Hello, ask me a question (enter 'quit' to exit): ") 
    if question == "quit":
        break
    predicted_answer = my_qa_engine.answer(question)
    print("Found Answer : ",predicted_answer)