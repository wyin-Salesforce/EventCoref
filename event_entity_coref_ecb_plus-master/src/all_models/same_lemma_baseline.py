import os
import gc
import sys
import json
from scipy.spatial.distance import cosine
# for pack in os.listdir("src"):
#     sys.path.append(os.path.join("src", pack))

sys.path.append("/export/home/workspace/EventCoref/event_entity_coref_ecb_plus-master/src/shared/")

import _pickle as cPickle
import logging
import argparse
from classes import *
from model_utils import *
from predict_model_wenpeng import run_conll_scorer

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

def get_clusters_by_head_lemma_wenpeng(mentions, word2vec, is_event):
    '''
    Given a list of mentions, this function clusters mentions that share the same head lemma.
    :param mentions: list of Mention objects (can be event or entity mentions)
    :param is_event: whether the function clusters event or entity mentions.
    :return: list of Cluster objects
    '''
    mentions_by_head_lemma = {}
    clusters = []


    list_of_list_mention=[]
    list_of_list_mention.append([mentions[0]])
    for mention_i in mentions[1:]:
        insert=False
        vec_i = word2vec.get(mention_i.mention_head_lemma)
        for list_id, mention_list in enumerate(list_of_list_mention):
            for mention_j in mention_list:
                if mention_i.mention_head_lemma == mention_j.mention_head_lemma:
                    '''put in this list'''
                    list_of_list_mention[list_id].append(mention_i)
                    insert=True
                    break
                else:
                    '''add extra beyong lemma matching'''
                    vec_j = word2vec.get(mention_j.mention_head_lemma)
                    if vec_i is not None and vec_j is not None:
                        cos = 1.0-cosine(vec_i, vec_j)
                    else:
                        cos = 0.0
                    if cos > 0.4:
                        list_of_list_mention[list_id].append(mention_i)
                        insert=True
                        break


            if insert:
                break
        if not insert:
            list_of_list_mention.append([mention_i])



    # for head_lemma, mentions in mentions_by_head_lemma.items():
    # for head_lemma, mentions in new_mentions_by_head_lemma.items():
    for mentions in list_of_list_mention:
        cluster = Cluster(is_event=is_event)
        for mention in mentions:
            cluster.mentions[mention.mention_id] = mention
        clusters.append(cluster)

    return clusters

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

    for topic_id in topics_keys:
        topic = topics[topic_id]
        topics_counter += 1

        event_mentions, entity_mentions = topic_to_mention_list(topic, is_gold=config_dict["test_use_gold_mentions"])

        event_clusters = get_clusters_by_head_lemma_wenpeng(event_mentions, word2vec,  is_event=True)
        entity_clusters = get_clusters_by_head_lemma(entity_mentions,  is_event=False)

        if config_dict["eval_mode"] == 1:
            event_clusters = separate_clusters_to_sub_topics(event_clusters, is_event=True)
            entity_clusters = separate_clusters_to_sub_topics(entity_clusters, is_event=False)

        with open(os.path.join(args.out_dir,'entity_clusters.txt'), 'a') as entity_file_obj:
            write_clusters_to_file(entity_clusters, entity_file_obj, topic_id)

        with open(os.path.join(args.out_dir, 'event_clusters.txt'), 'a') as event_file_obj:
            write_clusters_to_file(event_clusters, event_file_obj, topic_id)

        # set_coref_chain_to_mentions(event_clusters, is_event=True,
        #                             is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True
        #                             ,remove_singletons=config_dict["remove_singletons"])
        # set_coref_chain_to_mentions(entity_clusters, is_event=False,
        #                             is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True
        #                             ,remove_singletons=config_dict["remove_singletons"])
        '''remove parameter: remove_singletons'''
        set_coref_chain_to_mentions(event_clusters, is_event=True,
                                    is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)
        set_coref_chain_to_mentions(entity_clusters, is_event=False,
                                    is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)
    write_event_coref_results(test_set, args.out_dir, config_dict)
    write_entity_coref_results(test_set, args.out_dir, config_dict)

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

0.7: 77.07
'''
