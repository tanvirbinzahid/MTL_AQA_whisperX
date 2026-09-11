import pickle
import re
import os
import json
import shutil

data_dir = '../MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data'
srt_dir = '../MTL-AQA_dataset_release/srt_captions'
output_dir = '../MTL-AQA_dataset_release/Ready_2_Use/updated_data'
os.makedirs(output_dir, exist_ok=True)

# Load original data
with open(os.path.join(data_dir, 'final_annotations_dict.pkl'), 'rb') as f:
    annotations = pickle.load(f, encoding='latin1')
with open(os.path.join(data_dir, 'final_captions_dict.pkl'), 'rb') as f:
    captions_dict = pickle.load(f, encoding='latin1')
with open(os.path.join(data_dir, 'vocab.json'), 'r') as f:
    vocab = json.load(f)

def extract_srt_text(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.findall(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.*?)(?=\n\d+\n|\Z)', content, re.DOTALL | re.IGNORECASE)
    text = ' '.join([block.strip() for block in blocks if block.strip()])
    return re.sub(r'\s+', ' ', text.lower().strip())[:500]

updated_annotations = annotations.copy()
updated_captions = captions_dict.copy()
missing = []
updated_count = 0
for vid in annotations.keys():
    # vid is tuple like (1,1); use first element for SRT match, e.g., "01.srt" for all (1,*)
    dive_id = f"{vid[0]:02d}"
    srt_path = os.path.join(srt_dir, f"{dive_id}.srt")
    if os.path.exists(srt_path):
        new_text = extract_srt_text(srt_path)
        updated_annotations[vid]['text'] = new_text
        words = new_text.split()[:20]
        updated_captions[vid] = words
        print(f"Updated {vid} with {dive_id}.srt: {new_text[:50]}...")
        updated_count += 1
    else:
        missing.append(vid)

# Rebuild vocab from train split
with open(os.path.join(data_dir, 'train_split_0.pkl'), 'rb') as f:
    train_split = pickle.load(f, encoding='latin1')
train_captions = [updated_captions.get(vid, []) for vid in train_split]
new_vocab = {'word2idx': {'<pad>': 0, '<unk>': 1}, 'idx2word': {0: '<pad>', 1: '<unk>'}, 'vocab_size': 2}
idx = 2
for cap in train_captions:
    for word in cap:
        if word not in new_vocab['word2idx']:
            new_vocab['word2idx'][word] = idx
            new_vocab['idx2word'][idx] = word
            idx += 1
new_vocab['vocab_size'] = idx

# Save
with open(os.path.join(output_dir, 'final_annotations_dict_updated.pkl'), 'wb') as f:
    pickle.dump(updated_annotations, f, protocol=2)
with open(os.path.join(output_dir, 'final_captions_dict_updated.pkl'), 'wb') as f:
    pickle.dump(updated_captions, f, protocol=2)
with open(os.path.join(output_dir, 'vocab_updated.json'), 'w') as f:
    json.dump(new_vocab, f)
shutil.copy(os.path.join(data_dir, 'train_split_0.pkl'), os.path.join(output_dir, 'train_split_0.pkl'))
shutil.copy(os.path.join(data_dir, 'test_split_0.pkl'), os.path.join(output_dir, 'test_split_0.pkl'))

print(f"\nProcessed: Updated {updated_count} videos (from SRTs). Total entries: {len(annotations)}. Missing SRTs for: {len(missing)} dives.")
if missing:
    print(f"Sample missing: {missing[:5]}...")
