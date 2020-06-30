
import csv



def preprocess():
    filename = '/export/home/Dataset/EventCoref/test.only.tsv'

    list_of_eventlist = []
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
                    list_of_eventlist.append(new_eventlist)
                new_eventlist = []

    tsvfile.close()
    print('cluster size:', len(list_of_eventlist))
    for eventlit in list_of_eventlist:
        print('size in each cluster:', len(eventlit))

    print('doc_clusters:', doc_clusters)


if __name__ == "__main__":
    preprocess()
