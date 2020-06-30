
import csv
from scipy.spatial.distance import cosine

import en_core_web_sm
nlp = en_core_web_sm.load()

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

def preprocess():
    word2vec = load_word2vec()
    filename = '/export/home/Dataset/EventCoref/test.only.tsv'
    cluster_36 = []
    cluster_37 = []
    cluster_38 = []
    cluster_39 = []
    cluster_40 = []
    cluster_41 = []
    cluster_42 = []
    cluster_43 = []
    cluster_44 = []
    cluster_45 = []

    # list_of_eventlist = []
    new_eventlist = []
    doc_clusters = set()
    with open(filename) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            if len(row.get('Cluster ID'))>0:
                doc_cluster_id = int(row.get('Document').split('_')[0])
                row['doc_cluster_id'] = doc_cluster_id
                doc_clusters.add(doc_cluster_id)
                new_eventlist.append(row)
                # print(row)
                # exit(0)
            else:
                if len(new_eventlist) > 0: # we count one cluster if at least one event there
                    if doc_cluster_id == 36:
                        cluster_36.append(new_eventlist)
                    if doc_cluster_id == 37:
                        cluster_37.append(new_eventlist)
                    if doc_cluster_id == 38:
                        cluster_38.append(new_eventlist)
                    if doc_cluster_id == 39:
                        cluster_39.append(new_eventlist)
                    if doc_cluster_id == 40:
                        cluster_40.append(new_eventlist)
                    if doc_cluster_id == 41:
                        cluster_41.append(new_eventlist)
                    if doc_cluster_id == 42:
                        cluster_42.append(new_eventlist)
                    if doc_cluster_id == 43:
                        cluster_43.append(new_eventlist)
                    if doc_cluster_id == 44:
                        cluster_44.append(new_eventlist)
                    if doc_cluster_id == 45:
                        cluster_45.append(new_eventlist)

                    # list_of_eventlist.append(new_eventlist)
                new_eventlist = []

    tsvfile.close()
    # print('cluster size:', len(list_of_eventlist))
    # for eventlit in list_of_eventlist:
    #     print('size in each cluster:', len(eventlit))

    print('doc_clusters:', doc_clusters)

    all_clusters = []
    all_clusters.append(cluster_36)
    all_clusters.append(cluster_37)
    all_clusters.append(cluster_38)
    all_clusters.append(cluster_39)
    all_clusters.append(cluster_40)
    all_clusters.append(cluster_41)
    all_clusters.append(cluster_42)
    all_clusters.append(cluster_43)
    all_clusters.append(cluster_44)
    all_clusters.append(cluster_45)

    overall_f1 = 0.0
    for cluster_i in all_clusters:
        # print('cluster_i:', cluster_i)
        # exit(0)
        f1 = compute_f1(cluster_i, word2vec)
        overall_f1+=f1
    mean_f1 = overall_f1/len(all_clusters)
    print('mean_f1:', mean_f1)

def compute_f1(list_of_chain, word2vec):

    print('doc cluster has chain size:', len(list_of_chain), [len(chain) for chain in list_of_chain])
    gold_list = []
    pred_list = []
    for i, chain_i in enumerate(list_of_chain):
        for j, event_j in enumerate(chain_i):
            #iter again
            for k, chain_k in enumerate(list_of_chain):
                for m, event_m in enumerate(chain_k):
                    if i==k:
                        gold_list.append(1)
                    else:
                        gold_list.append(0)
                    '''computer prediction between two events'''
                    if i==k and j==m: #the same events:
                        pred_list.append(1)
                    else:
                        # event_1 = nlp(event_j.get('Event'))
                        # for word in event_1:
                        #     lemma_1 = word.lemma_
                        #     break
                        lemma_1 = event_j.get('Event').lower()
                        # event_2 = nlp(event_m.get('Event'))
                        # for word in event_2:
                        #     lemma_2 = word.lemma_
                        #     break
                        lemma_2 = event_m.get('Event').lower()
                        # common_substring = longestSubstringFinder(lemma_1, lemma_2)
                        # if len(common_substring)/len(lemma_1) > 0.3 or len(common_substring)/len(lemma_2) > 0.3:
                        #     pred_list.append(1)
                        # else:
                        #     pred_list.append(0)
                        vec_1 = word2vec.get(lemma_1)
                        vec_2 = word2vec.get(lemma_2)
                        if vec_1 is not None and vec_2 is not None:
                            cos = 1.0-cosine(vec_1, vec_2)
                        else:
                            cos = 0.0

                        if lemma_1 == lemma_2:
                            pred_list.append(1)
                        else:
                            if cos>0.3:
                                pred_list.append(1)
                            else:
                                pred_list.append(0)


    assert len(gold_list) == len(pred_list)

    overlap = 0
    for i in range(len(gold_list)):
        if gold_list[i]==1 and pred_list[i]==1:
            overlap+=1
    precision = overlap/sum(pred_list)
    recall = overlap/sum(gold_list)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    print('current f1:', f1)
    return f1

def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer


if __name__ == "__main__":
    preprocess()
    # longestSubstringFinder('Confirms', 'confirmed')

'''
    lemma matching: 41%
'''
