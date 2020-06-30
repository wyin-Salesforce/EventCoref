
import csv


import en_core_web_sm
nlp = en_core_web_sm.load()

def preprocess():
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
    print('cluster size:', len(list_of_eventlist))
    for eventlit in list_of_eventlist:
        print('size in each cluster:', len(eventlit))

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
        f1 = compute_f1(cluster_i)
        overall_f1+=f1
    mean_f1 = overall_f1/len(all_clusters)
    print('mean_f1:', mean_f1)

def compute_f1(list_of_chain):
    gold_list = []
    pred_list = []
    for i, chain_i in list_of_chain:
        for j, event_j in chain_i:
            #iter again
            for k, chain_k in list_of_chain:
                for m, event_m in chain_k:
                    if i==k:
                        gold_list.append(1)
                    else:
                        gold_list.append(0)
                    '''computer prediction between two events'''
                    if i==k and j==m: #the same events:
                        pred_list.append(1)
                    else:
                        event_1 = nlp(event_j.get('Event'))
                        for word in event_1:
                            lemma_1 = word.lemma_
                            break
                        event_2 = nlp(event_m.get('Event'))
                        for word in event_2:
                            lemma_2 = word.lemma_
                            break
                        if lemma_1 == lemma_2:
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




if __name__ == "__main__":
    preprocess()
