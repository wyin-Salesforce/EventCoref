
import csv
from scipy.spatial.distance import cosine
import numpy as np
import en_core_web_sm
import nltk

from nltk.corpus import wordnet as wn
nlp = en_core_web_sm.load()
nltk.download('wordnet')

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
        return np.array([0.0]*300)

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

                trigger_strings = nlp(row.get('Event'))
                lemma_list = []
                for word in trigger_strings:
                    lemma = word.lemma_
                    lemma_list.append(lemma)
                row['lemma'] = ' '.join(lemma_list)


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

                        # SRL_1 = event_j.get('SRL output')
                        # SRL_2 = event_m.get('SRL output')
                        # print('SRL_1:', SRL_1)
                        # print('SRL_2:', SRL_2)
                        # exit(0)
                        lemma_1 = event_j.get('lemma')
                        trigger_1 = event_j.get('Event').lower()
                        lemma_2 = event_m.get('lemma')
                        trigger_2 = event_m.get('Event').lower()
                        # vec_1 = sent_2_emb(trigger_1.split(), word2vec)
                        # vec_2 = sent_2_emb(trigger_2.split(), word2vec)
                        # vec_1 = sent_2_emb(lemma_1.split(), word2vec)
                        # vec_2 = sent_2_emb(lemma_2.split(), word2vec)
                        # if vec_1 is not None and vec_2 is not None:
                        #     cos = 1.0-cosine(vec_1, vec_2)
                        # else:
                        #     cos = 0.0
                        cos = wordsimi_wordnet(lemma_1.split()[0], lemma_2.split()[0])
                        '''assign a score'''
                        if lemma_1 == lemma_2:
                            pred_list.append(1)
                            # if i != k: #if lemma the same, but different chain
                            #     print('same lemma, different chains, gold:', False)
                            #     print('event_j:', event_j)
                            #     print('event_m:', event_m)
                        else:
                            if cos>0.5:
                                pred_list.append(1)
                                # if i !=k:
                                #     print('different lemma, high similarity ', cos, ', gold: ', i==k )
                                #     print('event_j:', event_j)
                                #     print('event_m:', event_m)
                            else:
                                pred_list.append(0)
                                # if i ==k:
                                #     print('different lemma, low similarity ', cos, ', gold: ', i==k)
                                #     print('event_j:', event_j)
                                #     print('event_m:', event_m)


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

if __name__ == "__main__":
    preprocess()
    # longestSubstringFinder('Confirms', 'confirmed')
    # print(wordsimi_wordnet('confirms', 'confirmed'))

'''
    lemma matching: 41%
    cosine >0.3: 43%
    cosine >0.4: 49.40%
    cosine >0.5: 49.02%
    vector comes from trigger sentence: 54.58%
    vector comes from lemma sentence: 56.74%

    next:
    1, each event is embedding sum up of all components
'''
