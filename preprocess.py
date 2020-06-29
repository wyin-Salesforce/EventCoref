
import csv



def preprocess():
    filename = '/export/home/Dataset/EventCoref/gold_mention_in_cluster.tsv'


    with open(filename) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            if len(row.strip())>0:
                print(row)
                exit(0)


if __name__ == "__main__":
    preprocess()
