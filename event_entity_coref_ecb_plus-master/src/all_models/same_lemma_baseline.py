import os
import gc
import sys
import json
import numpy as np
from scipy.spatial.distance import cosine
# for pack in os.listdir("src"):
#     sys.path.append(os.path.join("src", pack))
from nltk.corpus import wordnet as wn
sys.path.append("/export/home/workspace/EventCoref/event_entity_coref_ecb_plus-master/src/shared/")
import torch
import _pickle as cPickle
import logging
import argparse
from classes import *
from model_utils import *
from predict_model_wenpeng import run_conll_scorer
from transformers.tokenization_bert import BertTokenizer
# from transformers.optimization import AdamW
from transformers.modeling_bert import BertModel#RobertaForSequenceClassification

parser = argparse.ArgumentParser(description='Run same lemma baseline')

parser.add_argument('--config_path', type=str,
                    help=' The path configuration json file')
parser.add_argument('--out_dir', type=str,
                    help=' The directory to the output folder')

args = parser.parse_args()

# Loads json configuration file
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

# Saves json configuration file in the experiment's folder
with open(os.path.join(args.out_dir,'lemma_baseline_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

# from classes import *
from model_utils import *
from eval_utils import *


def load_word2vec():
    word2vec = {}

    print("==> loading 300d word2vec")

    f=open('/export/home/Dataset/word2vec_words_300d.txt', 'r')#glove.6B.300d.txt, word2vec_words_300d.txt, glove.840B.300d.txt
    co = 0
    for line in f:
        l = line.split()
        word2vec[l[0]] = list(map(float, l[1:]))
        co+=1
        if co % 50000 == 0:
            print('loading w2v size:', co)
        # if co % 10000 == 0:
        #     break
    print("==> word2vec is loaded")
    return word2vec


def get_clusters_by_head_lemma(mentions, is_event):
    '''
    Given a list of mentions, this function clusters mentions that share the same head lemma.
    :param mentions: list of Mention objects (can be event or entity mentions)
    :param is_event: whether the function clusters event or entity mentions.
    :return: list of Cluster objects
    '''
    mentions_by_head_lemma = {}
    clusters = []

    for mention in mentions:
        # print('mention:', mention, mention.mention_head_lemma)
        if mention.mention_head_lemma not in mentions_by_head_lemma:
            mentions_by_head_lemma[mention.mention_head_lemma] = []
        mentions_by_head_lemma[mention.mention_head_lemma].append(mention)
    # exit(0)
    for head_lemma, mentions in mentions_by_head_lemma.items():
        cluster = Cluster(is_event=is_event)
        for mention in mentions:
            cluster.mentions[mention.mention_id] = mention
        clusters.append(cluster)

    return clusters

def sent_2_emb(wordlist, word2vec):
    emb_list = []
    for word in wordlist:
        emb = word2vec.get(word, None)
        if emb is not None:
            emb_list.append(emb)
    if len(emb_list) > 0:
        arr = np.array(emb_list)
        return np.sum(arr, axis=0)
    else:
        return None#np.array([0.0]*300)

def wordsimi_wordnet(word1, word2):
    # print('word1:', word1, 'wn.synsets(word1):', wn.synsets(word1))
    word1_syn = wn.synsets(word1)
    word2_syn = wn.synsets(word2)
    if len(word1_syn) == 0 or len(word2_syn) ==0:
        return 0.0
    else:
        word1 = word1_syn[0]
        word2 = word2_syn[0]
        simi = word1.wup_similarity(word2)
        if simi is None:
            return 0.0
        else:
            return simi

def trigger_BERT_rep(bert_model, tokenizer, sentence, trigger_str):
    sentence_wordlist = sentence.split()
    trigger_len = len(trigger_str.split())
    trigger_index = sentence_wordlist.index(trigger_str)
    left_wordlist = sentence_wordlist[:trigger_index]
    right_wordlist = sentence_wordlist[trigger_index+trigger_len:]
    left_context = ' '.join(left_wordlist)
    right_context = ' '.join(right_wordlist)

    left_tokenized = tokenizer.tokenize('[CLS] ' + left_context)
    trigger_tokenized = tokenizer.tokenize(trigger_str)
    right_tokenized = tokenizer.tokenize(right_context + ' [SEP]')

    left_boundary_token_position = len(left_tokenized)
    right_boundary_token_position = len(left_tokenized+trigger_tokenized)-1
    tokenized_text = left_tokenized+trigger_tokenized+right_tokenized

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    input_ids = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        # last_hidden_states = model(input_ids, segments_tensors)[0][0]
        outputs = bert_model(input_ids, segments_tensors)
    last_hidden_states = outputs[0] #(batch, maxlen, hidden_size)
    left_token_rep = last_hidden_states[:, left_boundary_token_position,:]
    right_token_rep = last_hidden_states[:, right_boundary_token_position,:]
    trigger_rep_concate = torch.cat([left_token_rep, right_token_rep], dim=0) #(2, hidden)
    trigger_rep = torch.mean(trigger_rep_concate, dim=0) #hidden
    return trigger_rep

def get_clusters_by_head_lemma_wenpeng(topic, mentions, word2vec, bert_model, tokenizer, is_event):
    '''
    Given a list of mentions, this function clusters mentions that share the same head lemma.
    :param mentions: list of Mention objects (can be event or entity mentions)
    :param is_event: whether the function clusters event or entity mentions.
    :return: list of Cluster objects
    '''
    mentions_by_head_lemma = {}
    clusters = []

    same_lemma_error=0
    same_lemma_error_after=0
    diff_lemma_error=0
    diff_lemma_error_after=0
    list_of_list_mention=[]
    list_of_list_mention.append([mentions[0]])
    for mention_i in mentions[1:]:
        insert=False
        vec_i = word2vec.get(mention_i.mention_head_lemma)
        mention_i_arg1 = mention_i.arg0[0] if mention_i.arg0 is not None else ''
        mention_i_arg2 = mention_i.arg1[0] if mention_i.arg1 is not None else ''
        mention_i_amtmp = mention_i.amtmp[0] if mention_i.amtmp is not None else ''
        mention_i_amloc = mention_i.amloc[0] if mention_i.amloc is not None else ''
        mention_i_str = mention_i.mention_str
        mention_i_full_str = ' '.join([mention_i_arg1, mention_i_str, mention_i_arg2])#, mention_i_amtmp, mention_i_amloc])
        mention_i_triggerStr_emb = sent_2_emb(mention_i_str.lower().split(), word2vec)
        mention_i_full_str_emb = sent_2_emb(mention_i_full_str.lower().split(), word2vec)
        # print('mention_i gold_tag:', mention_i.gold_tag)

        mention_i_doc_id = mention_i.doc_id
        mention_i_sen_id = mention_i.sent_id
        mention_i_sen = topic.docs[mention_i_doc_id].sentences[mention_i_sen_id].get_raw_sentence()
        # tokenized_text = tokenizer.tokenize('[CLS] ' + mention_i_sen + ' [SEP]')
        mention_i_bert_rep = trigger_BERT_rep(bert_model, tokenizer, mention_i_sen, mention_i_str)
        print('mention_i_bert_rep:', mention_i_bert_rep)
        exit(0)
        for list_id, mention_list in enumerate(list_of_list_mention):
            mention_list_score = 0.0
            for mention_j in mention_list:
                vec_j = word2vec.get(mention_j.mention_head_lemma)
                mention_j_arg1 = mention_j.arg0[0] if mention_j.arg0 is not None else ''
                mention_j_arg2 = mention_j.arg1[0] if mention_j.arg1 is not None else ''
                mention_j_amtmp = mention_j.amtmp[0] if mention_j.amtmp is not None else ''
                mention_j_amloc = mention_j.amloc[0] if mention_j.amloc is not None else ''
                mention_j_str = mention_j.mention_str
                mention_j_full_str = ' '.join([mention_j_arg1, mention_j_str, mention_j_arg2])#, mention_j_amtmp, mention_j_amloc])
                mention_j_triggerStr_emb = sent_2_emb(mention_j_str.lower().split(), word2vec)
                mention_j_full_str_emb = sent_2_emb(mention_j_full_str.lower().split(), word2vec)
                # print('mention_j gold_tag:', mention_j.gold_tag)

                '''four types of cosine'''
                wn_cos = wordsimi_wordnet(mention_i.mention_head_lemma, mention_j.mention_head_lemma)
                if vec_i is not None and vec_j is not None:
                    lemma_cos = 1.0-cosine(vec_i, vec_j)
                else:
                    lemma_cos = 0.0
                if mention_i_triggerStr_emb is not None and mention_j_triggerStr_emb is not None:
                    trigger_cos = 1.0-cosine(mention_i_triggerStr_emb, mention_j_triggerStr_emb)
                else:
                    trigger_cos = 0.0
                if mention_i_full_str_emb is not None and mention_j_full_str_emb is not None:
                    full_mention_cos = 1.0-cosine(mention_i_full_str_emb, mention_j_full_str_emb)
                else:
                    full_mention_cos = 0.0

                '''start algorithm'''
                if mention_i.mention_head_lemma == mention_j.mention_head_lemma:
                    mention_list_score+=1
                elif wn_cos==1.0:
                    mention_list_score+=1
                else:
                    mention_list_score+= max(lemma_cos, trigger_cos)

            mention_list_score/=len(mention_list)
            if mention_list_score > 0.7:

                for mention_k in list_of_list_mention[list_id]:
                    if mention_i.gold_tag != mention_k.gold_tag:
                        print('mention_i:', mention_i)
                        for mention_m in list_of_list_mention[list_id]:
                            print('.........mention_m:', mention_m)
                        break
                list_of_list_mention[list_id].append(mention_i)
                insert=True
                break

        if not insert:
            list_of_list_mention.append([mention_i])


    print('same_lemma_error:', same_lemma_error, 'diff_lemma_error:', diff_lemma_error)
    print('same_lemma_error_after:', same_lemma_error_after, 'diff_lemma_error_after:', diff_lemma_error_after)

    # for head_lemma, mentions in mentions_by_head_lemma.items():
    # for head_lemma, mentions in new_mentions_by_head_lemma.items():
    for mentions in list_of_list_mention:
        cluster = Cluster(is_event=is_event)
        for mention in mentions:
            cluster.mentions[mention.mention_id] = mention
        clusters.append(cluster)

    return clusters, same_lemma_error-same_lemma_error_after, diff_lemma_error-diff_lemma_error_after

def merge_all_topics(test_set):
    '''
    Merges all topics and sub-topics to a single topic
    :param test_set: a Corpus object represents the test set
    :return: a topics dictionary contains a single topic
    '''
    new_topics = {}
    new_topics['all'] = Topic('all')
    topics_keys = test_set.topics.keys()
    for topic_id in topics_keys:
        topic = test_set.topics[topic_id]
        new_topics['all'].docs.update(topic.docs)
    return new_topics


def run_same_lemmma_baseline(test_set):
    '''
    Runs the head lemma baseline and writes its predicted clusters.
    :param test_set: A Corpus object representing the test set.
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    bert_model.eval()
    word2vec = load_word2vec()
    topics_counter = 0
    if config_dict["merge_sub_topics_to_topics"]:
        topics = merge_sub_topics_to_topics(test_set)
    elif config_dict["run_on_all_topics"]:
        topics = merge_all_topics(test_set)
    elif config_dict["load_predicted_topics"]:
        topics = load_predicted_topics(test_set,config_dict)
    else:
        topics = test_set.topics
    topics_keys = topics.keys()
    print('topic size:', len(topics_keys))
    decrease_same_lemma=0
    decrease_diff_lemma=0
    for topic_id in topics_keys:
        topic = topics[topic_id]
        topics_counter += 1

        event_mentions, entity_mentions = topic_to_mention_list(topic, is_gold=config_dict["test_use_gold_mentions"])

        event_clusters, decrease_same_lemma_i,  decrease_diff_lemma_i= get_clusters_by_head_lemma_wenpeng(topic, event_mentions, word2vec, bert_model, tokenizer, is_event=True)
        entity_clusters = get_clusters_by_head_lemma(entity_mentions,  is_event=False)

        decrease_same_lemma+=decrease_same_lemma_i
        decrease_diff_lemma+=decrease_diff_lemma_i

        if config_dict["eval_mode"] == 1:
            event_clusters = separate_clusters_to_sub_topics(event_clusters, is_event=True)
            entity_clusters = separate_clusters_to_sub_topics(entity_clusters, is_event=False)

        with open(os.path.join(args.out_dir,'entity_clusters.txt'), 'a') as entity_file_obj:
            write_clusters_to_file(entity_clusters, entity_file_obj, topic_id)

        with open(os.path.join(args.out_dir, 'event_clusters.txt'), 'a') as event_file_obj:
            write_clusters_to_file(event_clusters, event_file_obj, topic_id)
        '''remove parameter: remove_singletons'''
        set_coref_chain_to_mentions(event_clusters, is_event=True,
                                    is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)
        set_coref_chain_to_mentions(entity_clusters, is_event=False,
                                    is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)
    write_event_coref_results(test_set, args.out_dir, config_dict)
    write_entity_coref_results(test_set, args.out_dir, config_dict)
    print('decrease_same_lemma:', decrease_same_lemma, 'decrease_diff_lemma:', decrease_diff_lemma)
    run_conll_scorer(config_dict)

def main():
    '''
    This script loads the test set, runs the head lemma baseline and writes
    its predicted clusters.
    '''
    logger.info('Loading test data...')
    with open(config_dict["test_path"], 'rb') as f:
        test_data = cPickle.load(f)

    logger.info('Test data have been loaded.')

    logger.info('Running same lemma baseline...')
    run_same_lemmma_baseline(test_data)
    logger.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    main()


'''
python same_lemma_baseline.py --config_path ../../lemma_baseline_config.json --out_dir wenpeng/

0.6: 77.26
0.6, 0.22: 77.51%; change 8/13
'''
