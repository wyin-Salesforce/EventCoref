

# import _pickle as cPickle
import pickle

with open('/export/home/Dataset/EventCoref/processed/full_swirl_ecb/test_data', 'rb') as f:
    # test_data = cPickle.load(f)
    test_data = pickle.loads(f)
