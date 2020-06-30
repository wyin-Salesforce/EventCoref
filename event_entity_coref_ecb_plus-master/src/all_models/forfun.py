

import _pickle as cPickle
# import pickle
from ../shared/classes import *

with open('/export/home/Dataset/EventCoref/processed/full_swirl_ecb/test_data', 'rb') as f:
    test_data = cPickle.load(f)
