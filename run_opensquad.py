"""
The evaluation script for the Open Domain Question Answering Task on OpenSQuAD

"""

import pickle
from pyserini.search import SimpleSearcher

import argparse
import json
import os

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForQuestionAnswering, AlbertTokenizer, AlbertForQuestionAnswering
from transformers import squad_convert_examples_to_features

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
import numpy as np
import torch
from squad_tools import build_squad_input, evaluate
from datetime import datetime
from dilalbert import DilAlbert
from dilbert import DilBert
from transformers.data.metrics.squad_metrics import _get_best_indexes, get_final_text, _compute_softmax
import collections

curr_date_str = str(int((datetime.now().timestamp())))


def to_list(tensor):
	return tensor.detach().cpu().tolist()


def run_benchmark(tokenizer, model, small_portion: bool, device: str = 'cuda', k: int = 10, mu: float = None,
									use_ir_score: bool = False):
	"""Main Benchmark function.
	"""

	# initializing pyserini's searcher
	searcher = SimpleSearcher('formatted_open_squad/indexes/paragraphs_indexing')
	searcher.set_bm25()
	searcher.unset_rm3()

	# loading squad
	processor = SquadV2Processor()
	counter = 0
	model.to(torch.device(device))
	squad_dataset = json.load(open("SQuAD_1_1/dev-v1.1.json", 'r'))['data']
	with open('formatted_open_squad/open_squad.pkl', 'rb') as f1:
		squad1_for_orqa = pickle.load(f1)

	ans_predictions = dict()

	if small_portion:
		np.random.seed(42)
		id_examples = np.random.permutation(len(squad1_for_orqa['questions']))[:100]
	else:
		id_examples = np.arange(len(squad1_for_orqa['questions']))

	# Main loop : evaluation IR and ODQA
	for i in id_examples:
		print(i)
		curr_question = squad1_for_orqa['questions'][i]
		curr_answer = squad1_for_orqa['answers'][i]
		print('Question : ', curr_question)
		print('Answer : ', curr_answer)

		is_in = False
		hits = searcher.search(squad1_for_orqa['questions'][i], k=k)
		ir_scores = []
		paragraphs = []
		for j in range(len(hits)):
			passage = hits[j].raw
			ir_scores.append(hits[j].score)
			is_in = is_in or (squad1_for_orqa['answers'][i] in passage)
			paragraphs.append(passage)
		if is_in:
			counter += 1
		input_ = build_squad_input(curr_question, paragraphs)
		examples = processor._create_examples(input_["data"], "dev")
		features, dataset = squad_convert_examples_to_features(
			examples=examples,
			tokenizer=tokenizer,
			max_seq_length=384,
			doc_stride=128,
			max_query_length=64,
			is_training=False,
			return_dataset="pt",
			threads=1,
		)
		if use_ir_score:
			all_results, predictions = process_one_question(features, dataset, model, tokenizer, examples, device, True, mu,
																											ir_scores)
		else:
			all_results, predictions = process_one_question(features, dataset, model, tokenizer, examples, device)

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

		ans_predictions[squad1_for_orqa['ids'][i]] = texts[predicted_p_index]
		print('Predicted Answer : ', texts[predicted_p_index])

	evaluation = evaluate(squad_dataset, ans_predictions, ignore_missing_qids=True)

	em = evaluation['exact_match']
	f1 = evaluation['f1']

	write_in_result_file("Running evaluation on " + str(len(ans_predictions)) + " predictions")
	write_in_result_file(f"exact_match: {em}, f1: {f1}")

	print("IR : ", counter / len(id_examples))

	write_in_result_file(f"IR : {counter / len(id_examples)}")

	print(f"exact_match: {em}, f1: {f1}")


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

	output_prediction_file = os.path.join(output_dir, curr_date_str + "_predictions_{}.json".format(prefix))
	output_nbest_file = os.path.join(output_dir, curr_date_str + "_nbest_predictions_{}.json".format(prefix))
	output_null_log_odds_file = os.path.join(output_dir, curr_date_str + "_null_odds_{}.json".format(prefix))

	compute_predictions_logits_all(
		examples,
		features,
		all_results,
		20,	# 20 args.n_best_size,
		384,	# args.max_answer_length,
		True,	# args.do_lower_case,
		output_prediction_file,
		output_nbest_file,
		output_null_log_odds_file,
		False,	# args.verbose_logging,
		False,	# args.version_2_with_negative,
		0.0,	# args.null_score_diff_threshold,
		tokenizer,
	)

	predictions = json.load(
		open(os.path.join(output_dir, curr_date_str + "_nbest_predictions_{}.json".format(prefix)), 'r'))
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

	_PrelimPrediction = collections.namedtuple(	# pylint: disable=invalid-name
		"PrelimPrediction", ["example_index", "feature_index", "start_index", "end_index", "start_logit", "end_logit"]
	)

	all_predictions = collections.OrderedDict()
	all_nbest_json = collections.OrderedDict()
	scores_diff_json = collections.OrderedDict()

	prelim_predictions = []

	for (example_index, example) in enumerate(all_examples):
		features = example_index_to_features[example_index]

		# keep track of the minimum score of null start+end of position 0
		score_null = 1000000	# large and positive
		min_null_feature_index = 0	# the paragraph slice with min null score
		null_start_logit = 0	# the start logit at the slice with min null score
		null_end_logit = 0	# the end logit at the slice with min null score
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

		_NbestPrediction = collections.namedtuple(	# pylint: disable=invalid-name
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

		if pred.start_index > 0:	# this is a non-null prediction
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
		all_predictions["0"] = nbest_json[0]["text"]	# all_predictions[example.qas_id] = nbest_json[0]["text"] same below..
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


def write_in_result_file(string):
	with open("bench_results.txt", "a") as f:
		f.write(string + "\n")


if __name__ == '__main__':

	parser = argparse.ArgumentParser("Final Benchmark")
	parser.add_argument("--bert_path", default='bert_output_folder', type=str, help="Path to trained bert")
	parser.add_argument('--dilbert_path', default='dilbert_output_folder', type=str, help="Path to trained dilbert")
	parser.add_argument("--albert_path", default='albert_output_folder', type=str, help="Path to trained albert")
	parser.add_argument('--dilalbert_path', default='dilalbert_output_folder', type=str, help="Path to trained dilalbert")
	parser.add_argument('-', "--device", default='cuda', type=str, help="Whether to use gpu or cpu")
	args = parser.parse_args()

	use_albert_bench = [False, True]

	k_bench = [29, 100]

	use_dil_bench = [False, True]
	device_bench = args.device
	mu_bench = 0.5
	use_ir_score_bench = [True, False]

	small_portion_bench = False

	for use_albert in use_albert_bench:
		for use_dil in use_dil_bench:
			for k in k_bench:
				for use_ir_score in use_ir_score_bench:

					if use_albert:
						if use_dil:
							write_in_result_file("DilAlbert")
						else:
							write_in_result_file("Albert")
					else:
						if use_dil:
							write_in_result_file("Dilbert")
						else:
							write_in_result_file("Bert")

					write_in_result_file("k = " + str(k))

					if use_ir_score:
						write_in_result_file('Using IR score with mu = ' + str(mu_bench))
					else:
						write_in_result_file('Not using IR score')

					if use_albert:
						if not use_dil:
							tokenizer = AlbertTokenizer.from_pretrained(args.albert_path, do_lower_case=True)
							model = AlbertForQuestionAnswering.from_pretrained(args.albert_path)
						else:
							tokenizer = AlbertTokenizer.from_pretrained(args.dilalbert_path, do_lower_case=True)
							model = DilAlbert.from_pretrained(args.dilalbert_path)
					else:
						if not use_dil:
							tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
							model = BertForQuestionAnswering.from_pretrained(args.bert_path)
						else:
							tokenizer = BertTokenizer.from_pretrained(args.dilbert_path, do_lower_case=True)
							model = DilBert.from_pretrained(args.dilbert_path)

					run_benchmark(tokenizer, model, small_portion_bench, device_bench, k, mu_bench, use_ir_score)