import pickle
import os

data_dir = '../MTL-AQA_dataset_release/Ready_2_Use/updated_data'
with open(os.path.join(data_dir, 'final_captions_dict_updated.pkl'), 'rb') as f:
    caps = pickle.load(f, encoding='latin1')
sample = caps.get((1,1), [])
print('Sample updated caption for (1,1):', sample[:10])
print('Total unique words in vocab:', len(caps))
print('First few keys:', list(caps.keys())[:3])
