# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on MSMARCO. The decode layer is to extract keywords not a span."""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnsweringKeywords
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from MSMarco.text_input import rich_tokenize
from multiprocessing import Pool
from functools import partial
from collections import OrderedDict
from nltk import sent_tokenize, word_tokenize

#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


with open('../Vocab/stop_word_dictionary', 'r') as f:
    stopword_word2id = json.load(f)
#print(stopword_word2id)

fliter_words = ['the', 'a', 'of', 'and', ',', '.', 'is', 'are', 'you', 'i', 'to']

class SquadExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self,
               qas_id,
               passage_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               answer_words=None,
               is_select=0):
    self.qas_id = qas_id
    self.passage_id = passage_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.answer_words = answer_words
    self.is_select = is_select


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               tokens,
               input_ids,
               input_mask,
               answer_vector=None,
               is_select=0):
    self.unique_id = unique_id
    self.example_index = example_index
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.answer_vector = answer_vector
    self.is_select = is_select



def read_msmarco_examples(input_file, is_training):
    """Read a MS-Marco json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        lines = reader.readlines()

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for line in lines:
        info = line.split('\t')
        if len(info) == 7:
            uid, p_id, title, context, query, answer, is_select = line.split('\t')

            if int(is_select) == 0 and is_training:
                continue
            print(is_select)
            paragraph_text = context
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            qas_id = uid
            question_text = query
            start_position = None
            end_position = None
            orig_answer_text = None

            example = SquadExample(
                qas_id=qas_id,
                passage_id=p_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=context,
                answer_words=answer,
                is_select=is_select)
        else:
            print(len(info))

        examples.append(example)
    return examples


def convert_examples_to_features(example, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training
                                 ):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000


    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    for (i, token) in enumerate(example.doc_tokens):

        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    answer = tokenizer.tokenize(example.answer_words)
    answer_vector = np.zeros(max_seq_length)
    for i, token in enumerate(all_doc_tokens):
        if token in answer:
            answer_vector[len(query_tokens) + 2 + i] = 1.0
        if token in fliter_words:
            answer_vector[len(query_tokens) + 2 + i] = 0.0
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    if len(all_doc_tokens) > max_tokens_for_doc:
        example_doc_tokens = all_doc_tokens[:max_tokens_for_doc]
        answer_vector = answer_vector[:max_tokens_for_doc]
    tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + all_doc_tokens + ["[SEP]"]
    # The -3 accounts for [CLS], [SEP] and [SEP]


    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length


    return (unique_id, tokens, input_ids, input_mask, answer_vector, example.is_select, example.qas_id)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def write_predictions(all_features, all_results, output_prediction_file, remove_duplication=True):
    """Write final predictions to the json file."""
    def find_keywords(tokens, probs, remove_duplication=True):
        key_words = []
        flag = tokens.index('[SEP]')
        for i in range(flag, len(tokens)):
            if probs[i] == 1:
                if tokens[i] in stopword_word2id:
                    continue
                if tokens[i] in key_words and remove_duplication:
                    continue
                else:
                    key_words.append(tokens[i])

        return key_words

    with open(output_prediction_file, "w") as writer:
        for i, feature in enumerate(all_features):
            writer.writelines(json.dumps({
                feature.unique_id:find_keywords(feature.tokens, all_results[i])}) + "\n")




def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir", default='record', type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default='MSMarco/data/train_ranker_small.tsv', type=str, help="MSMARCO json for training. E.g., train-v2.1.json")
    parser.add_argument("--predict_file", default='MSMarco/data/dev_answerable.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=384, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_find_words", default=False, action='store_true', help="Whether to find keywords.")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=64, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=5, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict and not args.do_find_words:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if not os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #raise ValueError("Output directory () already exists and is not empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = read_msmarco_examples(
            input_file=args.train_file, is_training=True)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForQuestionAnsweringKeywords.from_pretrained(args.bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0
    if args.do_train:
        cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}'.format(
            args.bert_model, str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        train_features = None

        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            pool = Pool(processes=4)
            partial_convert_examples_to_features = partial(convert_examples_to_features,
                                                           tokenizer=tokenizer,
                                                           max_seq_length=args.max_seq_length,
                                                           doc_stride=args.doc_stride,
                                                           max_query_length=args.max_query_length,
                                                           is_training=True)
            results = pool.map(partial_convert_examples_to_features, train_examples)
            train_features = []

            for id, result in enumerate(results):
                if result == None:
                    continue
                feature = InputFeatures(
                    unique_id=1000000000 + id,
                    example_index=id,
                    tokens=result[1],
                    input_ids=result[2],
                    input_mask=result[3],
                    answer_vector=result[4],
                    is_select=result[5])
                train_features.append(feature)
            #(unique_id, doc_span_index, tokens, token_to_orig_map, token_is_max_context,
            #input_ids, input_mask, segment_ids, start_position, end_position, example.is_select, example.qas_id)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
                logger.info("***** Running training *****")
                logger.info("  Num orig examples = %d", len(train_examples))
                logger.info("  Num split examples = %d", len(train_features))
                logger.info("  Batch size = %d", args.train_batch_size)
                logger.info("  Num steps = %d", num_train_steps)
                all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
                #all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
                all_answer_vectors = torch.tensor([f.answer_vector for f in train_features], dtype=torch.long)
                all_is_select = torch.tensor([int(f.is_select) for f in train_features], dtype=torch.long)
                train_data = TensorDataset(all_input_ids, all_input_mask,
                                           all_answer_vectors, all_is_select)
                if args.local_rank == -1:
                    train_sampler = RandomSampler(train_data)
                else:
                    train_sampler = DistributedSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

                model.train()

                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", mininterval=3)):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                    input_ids, input_mask, answer_vectors, is_selects = batch
                    #print(is_selects)

                    loss = model(input_ids, None, input_mask, answer_vectors, is_selects)
                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # modify learning rate with special warm up BERT uses
                        lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model_keywords.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(args.output_dir, "pytorch_model_keywords.bin")
    model_state_dict = torch.load(output_model_file)
    model = BertForQuestionAnsweringKeywords.from_pretrained(args.bert_model, state_dict=model_state_dict)
    model.to(device)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_msmarco_examples(
            input_file=args.predict_file, is_training=False)
        pool = Pool(processes=10)
        partial_convert_examples_to_features = partial(convert_examples_to_features,
                                                       tokenizer=tokenizer,
                                                       max_seq_length=args.max_seq_length,
                                                       doc_stride=args.doc_stride,
                                                       max_query_length=args.max_query_length,
                                                       is_training=False)
        results = pool.map(partial_convert_examples_to_features, eval_examples)
        eval_features = []
        for id, result in enumerate(results):
            if result == None:
                continue
            feature = InputFeatures(
                unique_id=1000000000 + id,
                example_index=id,
                tokens=result[1],
                input_ids=result[2],
                input_mask=result[3],
                answer_vector=result[4],
                is_select=result[5])
            eval_features.append(feature)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)


        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_answer_vectors = torch.tensor([f.answer_vector for f in eval_features], dtype=torch.long)
        all_is_select = torch.tensor([f.is_select for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_answer_vectors, all_is_select, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        all_right_rate = 0.
        all_wrong_rate = 0.
        zero_vector = 0
        for input_ids, input_mask, answer_vectors, is_select, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
            #print(input_ids.size(), answer_vector.size())
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            with torch.no_grad():
                batch_logits, batch_no_answer_logits = model(input_ids, None, input_mask)
            for i, example_index in enumerate(example_indices):
                logits = batch_logits[i].detach().cpu().tolist()
                answer_vector = answer_vectors[i].detach().cpu().tolist()

                right = 0.
                wrong = 0.
                total = 0.
                for k, logit in enumerate(logits):
                    if answer_vector[k] == 1:
                        total += 1.
                        if logit >= 0.5:
                            right += 1.
                        else:
                            wrong += 1.
                right_rate = right / (total + float(1e-5))
                wrong_rate = wrong / (total + float(1e-5))
                if total < 1.:
                    zero_vector += 1
                    continue
                all_right_rate += right_rate
                all_wrong_rate += wrong_rate
        logger.info(" All right rate is = %f", all_right_rate/len(all_input_ids-zero_vector))
        logger.info(" All wrong rate is = %f", all_wrong_rate/len(all_input_ids-zero_vector))
        logger.info(" Zero vector rate is = %f", zero_vector / len(all_input_ids))

    if args.do_find_words and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_msmarco_examples(
            input_file=args.predict_file, is_training=False)
        pool = Pool(processes=10)
        partial_convert_examples_to_features = partial(convert_examples_to_features,
                                                       tokenizer=tokenizer,
                                                       max_seq_length=args.max_seq_length,
                                                       doc_stride=args.doc_stride,
                                                       max_query_length=args.max_query_length,
                                                       is_training=False)
        results = pool.map(partial_convert_examples_to_features, eval_examples)
        eval_features = []
        for id, result in enumerate(results):
            if result == None:
                continue
            feature = InputFeatures(
                unique_id=result[-1],
                example_index=id,
                tokens=result[1],
                input_ids=result[2],
                input_mask=result[3],
                answer_vector=result[4],
                is_select=result[5])
            eval_features.append(feature)
        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)


        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_answer_vectors = torch.tensor([f.answer_vector for f in eval_features], dtype=torch.long)
        all_is_select = torch.tensor([f.is_select for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_answer_vectors, all_is_select, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        key_words_prob = []
        for input_ids, input_mask, answer_vectors, is_select, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
            #print(input_ids.size(), answer_vector.size())
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            with torch.no_grad():
                batch_logits, batch_no_answer_logits = model(input_ids, None, input_mask)
            for i, example_index in enumerate(example_indices):
                feature = eval_features[example_index]

                logits = batch_logits[i].detach().cpu().tolist()
                ids = input_ids[i].detach().cpu().tolist()
                #print(ids)
                #print(tokenizer.vocab)
                words = tokenizer.convert_ids_to_tokens(ids)
                #print(feature.tokens)

                logit = []
                for i, word in enumerate(feature.tokens):
                    logit.append(logits[i])
                #print(feature.tokens)
                #print(logit)
                key_words_prob.append({'id':feature.unique_id, 'words':feature.tokens, 'logit':logit})
                #print(logits)


        output_prediction_file = os.path.join(args.output_dir, "words_predictions.json")
        with open(output_prediction_file, 'w') as f:
            json.dump(key_words_prob, f)





if __name__ == "__main__":
    main()
