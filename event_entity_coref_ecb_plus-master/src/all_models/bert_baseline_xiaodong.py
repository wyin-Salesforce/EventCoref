# cluster + bert
import os
import gc
import sys
import json
import subprocess

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

import _pickle as cPickle
import logging
import argparse
from classes import *
from model_utils import *

import torch
from transformers import *

parser = argparse.ArgumentParser(description='Run bert baseline')

parser.add_argument('--config_path', type=str,
                    help=' The path configuration json file')
parser.add_argument('--out_dir', type=str,
                    help=' The directory to the output folder')

args = parser.parse_args()

# Loads json configuration file
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

# Saves json configuration file in the experiment's folder
with open(os.path.join(args.out_dir,'bert_baseline_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

from classes import *
from model_utils import *
from eval_utils import *
import operator


def mention_pair_scorer(mention1, mention2, alpha, cos):
    if mention1.mention_head_lemma == mention2.mention_head_lemma:
        score = 1 - alpha + alpha * cos(mention1.bert, mention2.bert)
    else:
        score = alpha * cos(mention1.bert, mention2.bert)
        # score = cos(mention1.bert, mention2.bert)
    return score

def merge_cluster(clusters, cluster_pairs, epoch, topics_counter,
          topics_num, threshold, alpha, is_event):
    '''
    Merges cluster pairs in agglomerative manner till it reaches a pre-defined threshold. In each step, the function merges
    cluster pair with the highest score, and updates the candidate cluster pairs according to the
    current merge.
    Note that all Cluster objects in clusters should have the same type (event or entity but
    not both of them).
    other_clusters are fixed during merges and should have the opposite type
    i.e. if clusters are event clusters, so other_clusters will be the entity clusters.

    :param clusters: a list of Cluster objects of the same type (event/entity)
    :param cluster_pairs: a list of the cluster pairs (tuples)
    :param epoch: current epoch (relevant to training)
    :param topics_counter: current topic number
    :param topics_num: total number of topics
    :param threshold: merging threshold
    :param is_event: True if clusters are event clusters and false if they are entity clusters
    '''
    print('Initialize cluster pairs scores... ')
    logging.info('Initialize cluster pairs scores... ')
    # initializes the pairs-scores dict
    pairs_dict = {}
    mode = 'event' if is_event else 'entity'
    # init the scores (that the model assigns to the pairs)
    cos = torch.nn.CosineSimilarity(dim=0)
    # alpha = config_dict["alpha"]

    for cluster_pair in cluster_pairs:
        mention_pairs = cluster_pair_to_mention_pair(cluster_pair)
        mention_score = 0.0
        for mention1, mention2 in mention_pairs:
            mention_score += mention_pair_scorer(mention1, mention2, alpha, cos).data.cpu().numpy()
        pairs_dict[cluster_pair] = (mention_score / len(mention_pairs))

    while True:
        # finds max pair (break if we can't find one  - max score < threshold)
        if len(pairs_dict) < 2:
            print('Less the 2 clusters had left, stop merging!')
            logging.info('Less the 2 clusters had left, stop merging!')
            break
        # print(pairs_dict)
        (max_pair, max_score) = max(pairs_dict.items(), key=operator.itemgetter(1))
        # max_pair, max_score = key_with_max_val(pairs_dict)

        if max_score > threshold:
            print('epoch {} topic {}/{} - merge {} clusters with score {} clusters : {} {}'.format(
                epoch, topics_counter, topics_num, mode, str(max_score), str(max_pair[0]),
                str(max_pair[1])))
            logging.info('epoch {} topic {}/{} - merge {} clusters with score {} clusters : {} {}'.format(
                epoch, topics_counter, topics_num, mode, str(max_score), str(max_pair[0]),
                str(max_pair[1])))

            cluster_i = max_pair[0]
            cluster_j = max_pair[1]
            new_cluster = Cluster(is_event)
            new_cluster.mentions.update(cluster_j.mentions)
            new_cluster.mentions.update(cluster_i.mentions)

            keys_pairs_dict = list(pairs_dict.keys())
            for pair in keys_pairs_dict:
                cluster_pair = (pair[0], pair[1])
                if cluster_i in cluster_pair or cluster_j in cluster_pair:
                    del pairs_dict[pair]

            clusters.remove(cluster_i)
            clusters.remove(cluster_j)
            clusters.append(new_cluster)

            new_pairs = []
            for cluster in clusters:
                if cluster != new_cluster:
                    new_pairs.append((cluster, new_cluster))

            # create scores for the new pairs

            for pair in new_pairs:
                mention_pairs = cluster_pair_to_mention_pair(pair)
                mention_score = 0.0
                for mention1, mention2 in mention_pairs:
                    mention_score += mention_pair_scorer(mention1, mention2, alpha, cos).data.cpu().numpy()
                pairs_dict[pair] = (mention_score / len(mention_pairs))

        else:
            print('Max score = {} is lower than threshold = {},'
                  ' stopped merging!'.format(max_score, threshold))
            logging.info('Max score = {} is lower than threshold = {},' \
                         ' stopped merging!'.format(max_score, threshold))
            break


def test_models(test_set, write_clusters, out_dir, isProcessed, alpha, threshold):
    '''
    Runs the inference procedure for both event and entity models calculates the B-cubed
    score of their predictions.
    :param test_set: Corpus object containing the test documents.
    :param write_clusters: whether to write predicted clusters to file (for analysis purpose)
    :param out_dir: output files directory
    chains predicted by an external (WD) entity coreference system.
    :return: B-cubed scores for the predicted event and entity clusters
    '''

    global clusters_count
    clusters_count = 1
    event_errors = []
    entity_errors = []
    all_event_clusters = []
    all_entity_clusters = []

    if isProcessed:
        infile = open('bert_baseline/processed_topics', 'rb')
        topics = cPickle.load(infile)
    else:
        if config_dict["load_predicted_topics"]:
            topics = load_predicted_topics(test_set, config_dict) # use the predicted sub-topics
        else:
            topics = test_set.topics # use the gold sub-topics
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertForPreTraining.from_pretrained('bert-large-uncased', output_hidden_states=True)

        # tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        # model = RobertaModel.from_pretrained('roberta-large-mnli')
        model.eval()

    topics_num = len(topics.keys())
    topics_counter = 0
    topics_keys = topics.keys()
    epoch = 0 #
    all_event_mentions = []
    all_entity_mentions = []

    with torch.no_grad():
        for topic_id in topics_keys:
            topic = topics[topic_id]
            topics_counter += 1

            logging.info('=========================================================================')
            logging.info('Topic {}:'.format(topic_id))
            print('Topic {}:'.format(topic_id))

            event_mentions, entity_mentions = topic_to_mention_list(topic,
                                                                    is_gold=config_dict["test_use_gold_mentions"])

            all_event_mentions.extend(event_mentions)
            all_entity_mentions.extend(entity_mentions)

            # create span rep for both entity and event mentions
            if not isProcessed:
                for event in event_mentions:
                    doc_id = event.doc_id
                    sen_id = event.sent_id
                    sen = topic.docs[doc_id].sentences[sen_id]
                    # if not sen.is_bert:
                    get_bert_vec(model, tokenizer, sen)
                    event_head = int(event.get_head_index())
                    # print("event head idx: ", event_head)
                    # print("token mapping: ", sen.token_mapping[event_head])
                    if len(sen.token_mapping[event_head]) == 1:
                        event.bert = sen.bert[sen.token_mapping[event_head][0]]
                    else:
                        # avg_vec = torch.zeros([768], dtype=torch.float)
                        avg_vec = torch.zeros([1024], dtype=torch.float)
                        for idx in sen.token_mapping[event_head]:
                            avg_vec += sen.bert[idx]
                        avg_vec /= len(sen.token_mapping[event_head])
                        event.bert = avg_vec

            print('number of event mentions : {}'.format(len(event_mentions)))
            print('number of entity mentions : {}'.format(len(entity_mentions)))
            logging.info('number of event mentions : {}'.format(len(event_mentions)))
            logging.info('number of entity mentions : {}'.format(len(entity_mentions)))
            topic.event_mentions = event_mentions
            topic.entity_mentions = entity_mentions

            # initialize within-document entity clusters with the output of within-document system
            # wd_entity_clusters = init_entity_wd_clusters(entity_mentions, doc_to_entity_mentions)

            # topic_entity_clusters = []
            # for doc_id, clusters in wd_entity_clusters.items():
            #     topic_entity_clusters.extend(clusters)

            # initialize event clusters as singletons
            topic_event_clusters = init_cd(event_mentions, is_event=True)

            # init cluster representation
            # update_lexical_vectors(topic_entity_clusters, cd_entity_model, device,
            #                        is_event=False, requires_grad=False)
            # update_lexical_vectors(topic_event_clusters, cd_event_model, device,
            #                        is_event=True, requires_grad=False)

            # entity_th = config_dict["entity_merge_threshold"]
            event_th = config_dict["event_merge_threshold"]

            # for i in range(1,config_dict["merge_iters"]+1):
            #     print('Iteration number {}'.format(i))
            #     logging.info('Iteration number {}'.format(i))

                # Merge entities
                # print('Merge entity clusters...')
                # logging.info('Merge entity clusters...')
                # test_model(clusters=topic_entity_clusters, other_clusters=topic_event_clusters,
                #            model=cd_entity_model, device=device, topic_docs=topic.docs,is_event=False,epoch=epoch,
                #            topics_counter=topics_counter, topics_num=topics_num,
                #            threshold=entity_th,
                #            use_args_feats=config_dict["use_args_feats"],
                #            use_binary_feats=config_dict["use_binary_feats"])

                # Merge events
            print('Merge event clusters...')
            logging.info('Merge event clusters...')
                # test_model(clusters=topic_event_clusters, other_clusters=topic_entity_clusters,
                #            model=cd_event_model,device=device, topic_docs=topic.docs, is_event=True,epoch=epoch,
                #            topics_counter=topics_counter, topics_num=topics_num,
                #            threshold=event_th,
                #            use_args_feats=config_dict["use_args_feats"],
                #            use_binary_feats=config_dict["use_binary_feats"])
            cluster_pairs, _ = generate_cluster_pairs(topic_event_clusters, is_train=False)
            merge_cluster(topic_event_clusters, cluster_pairs, epoch, topics_counter, topics_num, threshold, alpha, True)

            set_coref_chain_to_mentions(topic_event_clusters, is_event=True,
                                        is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)
            # set_coref_chain_to_mentions(topic_entity_clusters, is_event=False,
            #                             is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)

            if write_clusters:
                # Save for analysis
                all_event_clusters.extend(topic_event_clusters)
                # all_entity_clusters.extend(topic_entity_clusters)

                # with open(os.path.join(out_dir, 'entity_clusters.txt'), 'a') as entity_file_obj:
                #     write_clusters_to_file(topic_entity_clusters, entity_file_obj, topic_id)
                #     entity_errors.extend(collect_errors(topic_entity_clusters, topic_event_clusters, topic.docs,
                #                                         is_event=False))

                with open(os.path.join(out_dir, 'event_clusters.txt'), 'a') as event_file_obj:
                    write_clusters_to_file(topic_event_clusters, event_file_obj, topic_id)
                    # event_errors.extend(collect_errors(topic_event_clusters, topic_entity_clusters, topic.docs,
                    #                                    is_event=True))

        if write_clusters:
            write_event_coref_results(test_set, out_dir, config_dict)
            # write_entity_coref_results(test_set, out_dir, config_dict)
            sample_errors(event_errors, os.path.join(out_dir,'event_errors'))
            # sample_errors(entity_errors, os.path.join(out_dir,'entity_errors'))

    with open('bert_baseline/processed_topics', 'wb') as f:
        cPickle.dump(topics, f)

    if config_dict["test_use_gold_mentions"]:
        event_predicted_lst = [event.cd_coref_chain for event in all_event_mentions]
        true_labels = [event.gold_tag for event in all_event_mentions]
        true_clusters_set = set(true_labels)

        labels_mapping = {}
        for label in true_clusters_set:
            labels_mapping[label] = len(labels_mapping)

        event_gold_lst = [labels_mapping[label] for label in true_labels]
        event_r, event_p, event_b3_f1 = bcubed(event_gold_lst, event_predicted_lst)

        # entity_predicted_lst = [entity.cd_coref_chain for entity in all_entity_mentions]
        # true_labels = [entity.gold_tag for entity in all_entity_mentions]
        true_clusters_set = set(true_labels)

        labels_mapping = {}
        for label in true_clusters_set:
            labels_mapping[label] = len(labels_mapping)

        # entity_gold_lst = [labels_mapping[label] for label in true_labels]
        # entity_r, entity_p, entity_b3_f1 = bcubed(entity_gold_lst, entity_predicted_lst)

        return event_b3_f1

    else:
        print('Using predicted mentions, can not calculate CoNLL F1')
        logging.info('Using predicted mentions, can not calculate CoNLL F1')
        return 0,0

def get_bert_vec(model, tokenizer, sentence):

    # Encode text
    tokenized_text = tokenizer.tokenize('[CLS] ' + sentence.get_raw_sentence() + ' [SEP]')
    # tokenized_text = tokenizer.tokenize('<s> ' + sentence.get_raw_sentence() + ' </s>')

    # print(sentence.get_raw_sentence())
    # print(tokenized_text)

    mapping_tokenized = []
    i = 1
    for token_obj in sentence.get_tokens():
        token = token_obj.token.lower()
        # token = token_obj.token

        # print(token, tokenized_text[i])
        # if token == tokenized_text[i].replace('Ġ', ''):
        '''when the sentence word is the same with the wordpiece word'''
        if token == tokenized_text[i]:
            mapping_tokenized.append([i])
            i += 1
            continue
        # if token.startswith(tokenized_text[i].replace('Ġ', '')):
        if token.startswith(tokenized_text[i]):
            span = [i]
            # surface = tokenized_text[i].replace('Ġ', '')
            surface = tokenized_text[i]
            while 1:
                i += 1
                surface += tokenized_text[i].replace('##', '')
                span.append(i)
                # print(token, surface)
                if token == surface:
                    break
                else:
                    if token.startswith(surface):
                        continue
                    else:
                        print("Cannot match token: ", token)
            mapping_tokenized.append(span)
            i += 1
    # print(mapping_tokenized)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    input_ids = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        # last_hidden_states = model(input_ids, segments_tensors)[0][0]
        last_hidden_states = model(input_ids, segments_tensors)
    # print(last_hidden_states[0].size())
    # print(last_hidden_states[2][-1][0].size())
    sentence.add_bert(last_hidden_states[2][-1][0], mapping_tokenized)
    # print(sentence.bert.size())

    return


def run_bert_baseline(test_set):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForPreTraining.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model.eval()

    topics_counter = 0
    if config_dict["load_predicted_topics"]:
        topics = load_predicted_topics(test_set, config_dict)
    else:
        topics = test_set.topics
    topics_keys = topics.keys()

    for topic_id in topics_keys:
        topic = topics[topic_id]
        topics_counter += 1

        event_mentions, entity_mentions = topic_to_mention_list(topic, is_gold=config_dict["test_use_gold_mentions"])
        # print(topic.docs.sentences[0].tokens)

        for event in event_mentions:
            doc_id = event.doc_id
            sen_id = event.sent_id
            sen = topic.docs[doc_id].sentences[sen_id]
            # if not sen.is_bert:
            get_bert_vec(model, tokenizer, sen)
            event_head = int(event.get_head_index())
            print("event head idx: ", event_head)
            print("token mapping: ", sen.token_mapping[event_head])
            if len(sen.token_mapping[event_head]) == 1:
                event.bert = sen.bert[sen.token_mapping[event_head][0]]
            else:
                avg_vec = torch.zeros([1, 768], dtype=torch.float)
                for idx in sen.token_mapping[event_head]:
                    avg_vec += sen.bert[idx]
                avg_vec /= len(sen.token_mapping[event_head])
                event.bert = avg_vec



            # print(len(last_hidden_states.tolist()[0]))
            # break
        # print(event_mentions)
        # break
        # event_clusters = get_clusters_by_head_lemma(event_mentions, is_event=True)
        # entity_clusters = get_clusters_by_head_lemma(entity_mentions, is_event=False)

    #     if config_dict["eval_mode"] == 1:
    #         event_clusters = separate_clusters_to_sub_topics(event_clusters, is_event=True)
    #         entity_clusters = separate_clusters_to_sub_topics(entity_clusters, is_event=False)
    #
    #     with open(os.path.join(args.out_dir,'entity_clusters.txt'), 'a') as entity_file_obj:
    #         write_clusters_to_file(entity_clusters, entity_file_obj, topic_id)
    #
    #     with open(os.path.join(args.out_dir, 'event_clusters.txt'), 'a') as event_file_obj:
    #         write_clusters_to_file(event_clusters, event_file_obj, topic_id)
    #
    #     set_coref_chain_to_mentions(event_clusters, is_event=True,
    #                                 is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)
    #                                 # ,remove_singletons=config_dict["remove_singletons"])
    #     set_coref_chain_to_mentions(entity_clusters, is_event=False,
    #                                 is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)
    #                                 # ,remove_singletons=config_dict["remove_singletons"])
    #
    # write_event_coref_results(test_set, args.out_dir, config_dict)
    # write_entity_coref_results(test_set, args.out_dir, config_dict)

def read_conll_f1(filename):
    '''
    This function reads the results of the CoNLL scorer , extracts the F1 measures of the MUS,
    B-cubed and the CEAF-e and calculates CoNLL F1 score.
    :param filename: a file stores the scorer's results.
    :return: the CoNLL F1
    '''
    f1_list = []
    with open(filename, "r") as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find('F1:') != -1:
                f1_list.append(float(new_line.split(': ')[-1][:-1]))

    muc_f1 = f1_list[1]
    bcued_f1 = f1_list[3]
    ceafe_f1 = f1_list[7]

    return (muc_f1 + bcued_f1 + ceafe_f1)/float(3)


def run_conll_scorer():
    if config_dict["test_use_gold_mentions"]:
        event_response_filename = os.path.join(args.out_dir, 'CD_test_event_mention_based.response_conll')
        # entity_response_filename = os.path.join(args.out_dir, 'CD_test_entity_mention_based.response_conll')
    else:
        event_response_filename = os.path.join(args.out_dir, 'CD_test_event_span_based.response_conll')
        # entity_response_filename = os.path.join(args.out_dir, 'CD_test_entity_span_based.response_conll')

    event_conll_file = os.path.join(args.out_dir,'event_scorer_cd_out.txt')
    # entity_conll_file = os.path.join(args.out_dir,'entity_scorer_cd_out.txt')

    event_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
            (config_dict["event_gold_file_path"], event_response_filename, event_conll_file))

    # entity_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
    #         (config_dict["entity_gold_file_path"], entity_response_filename, entity_conll_file))

    processes = []
    print('Run scorer command for cross-document event coreference')
    processes.append(subprocess.Popen(event_scorer_command, shell=True))

    # print('Run scorer command for cross-document entity coreference')
    # processes.append(subprocess.Popen(entity_scorer_command, shell=True))

    while processes:
        status = processes[0].poll()
        if status is not None:
            processes.pop(0)

    print ('Running scorers has been done.')
    print ('Save results...')

    scores_file = open(os.path.join(args.out_dir, 'conll_f1_scores.txt'), 'w')

    event_f1 = read_conll_f1(event_conll_file)
    # entity_f1 = read_conll_f1(entity_conll_file)
    scores_file.write('Event CoNLL F1: {}\n'.format(event_f1))
    # scores_file.write('Entity CoNLL F1: {}\n'.format(entity_f1))

    scores_file.close()
    return event_f1


def main():
    '''
    This script loads the test set, runs the head lemma baseline and writes
    its predicted clusters.
    '''
    logger.info('Loading test data...')
    with open(config_dict["test_path"], 'rb') as f:
        test_data = cPickle.load(f)

    logger.info('Test data have been loaded.')

    logger.info('Running bert baseline...')
    # run_bert_baseline(test_data)
    alpha = [0.6]
    threshold = [0.75]
    # alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # threshold = [0.7, 0.75, 0.8]
    # outfile = open("bert_baseline/parameter_result_0_without_alpha.out", 'a')
    # outfile = open("bert_baseline/parameter_result.out", 'a')


    for i in alpha:
        for j in threshold:
            print("alpha: {}, threshold: {}".format(i,j))
            test_models(test_data, write_clusters=True, out_dir=args.out_dir, isProcessed=False, alpha=i, threshold=j)
            score = run_conll_scorer()
            print("alpha: {}, threshold: {}, score: {}\n".format(i,j,score))
            # outfile.write("alpha: {}, threshold: {}, score: {}\n".format(i,j,score))
    logger.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    main()
