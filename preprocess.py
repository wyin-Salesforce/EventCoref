
import csv



def preprocess():
    filename = '/export/home/Dataset/EventCoref/test.only.tsv'

    list_of_eventlist = []
    new_eventlist = []
    with open(filename) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            if len(row.get('Cluster ID'))>0:
                new_eventlist.append(row)
                # print(row)
                # exit(0)
            else:
                if len(new_eventlist) > 1: # we count one cluster if at least one event there
                    list_of_eventlist.append(new_eventlist)
                new_eventlist = []

    tsvfile.close()
    print('cluster size:', len(list_of_eventlist))
    for eventlit in list_of_eventlist:
        print('size in each cluster:', len(eventlit))


if __name__ == "__main__":
    preprocess()
