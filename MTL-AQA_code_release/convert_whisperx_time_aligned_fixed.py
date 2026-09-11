import os, re, json, pickle
from collections import Counter

SRTDIR = "../MTL-AQA_dataset_release/Raw_Annotations/whisper_srt"
ANNODIR = "../MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data"
ANNOTATIONPKL = os.path.join(ANNODIR, "final_annotations_dict.pkl")
OUTPUTPKL = os.path.join(ANNODIR, "final_captions_dict.pkl")
OUTPUTVOCAB = os.path.join(ANNODIR, "vocab.json")

def parse_srt(srtfile):
    with open(srtfile, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    captions, buf = [], []
    for line in lines:
        line = line.strip()
        if not line or line.isdigit() or '-->' in line:
            if buf:
                captions.append(' '.join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        captions.append(' '.join(buf))
    full = ' '.join(captions)
    full = re.sub(r'[^\w\s]', '', full.lower())
    full = re.sub(r'\s+', ' ', full).strip()
    return full.split() if full else []

with open(ANNOTATIONPKL, 'rb') as f:
    annotations = pickle.load(f)

keys_by_vid = {}
for k in annotations.keys():
    keys_by_vid.setdefault(k[0], []).append(k)

captionsdict = {}
allwords = []
missing = []

for vid, keys in sorted(keys_by_vid.items()):
    srtfile = os.path.join(SRTDIR, f"{vid:02d}.srt")
    if not os.path.exists(srtfile) or os.path.getsize(srtfile) == 0:
        missing.append(f"{vid:02d}")
        words = []
    else:
        words = parse_srt(srtfile)
    cap = ['<START>'] + words + ['<END>']
    for key in keys:
        captionsdict[key] = cap
    allwords.extend(words)

wordfreq = Counter(allwords)
vocabwords = ['<PAD>', '<START>', '<END>', '<UNK>'] + sorted([w for w, c in wordfreq.items() if c >= 2])
word_to_ix = {w:i for i,w in enumerate(vocabwords)}
ix_to_word = {str(i):w for i,w in enumerate(vocabwords)}

print("ANNOTATIONS", len(annotations))
print("CAPTION_SAMPLES", len(captionsdict))
print("MISSING_VIDS", missing)
print("TOTAL_WORDS", len(allwords))
print("UNIQUE_WORDS", len(wordfreq))
print("VOCAB_SIZE", len(vocabwords))
print("SAMPLE_CAP", next(iter(captionsdict.values()))[:15])

backup_pkl = OUTPUTPKL + ".backup_from_whisper"
backup_vocab = OUTPUTVOCAB + ".backup_from_whisper"
if os.path.exists(OUTPUTPKL):
    os.replace(OUTPUTPKL, backup_pkl)
if os.path.exists(OUTPUTVOCAB):
    os.replace(OUTPUTVOCAB, backup_vocab)

with open(OUTPUTPKL, 'wb') as f:
    pickle.dump(captionsdict, f)
with open(OUTPUTVOCAB, 'w') as f:
    json.dump({'word_to_ix': word_to_ix, 'ix_to_word': ix_to_word}, f, indent=2)

print("SAVED", OUTPUTPKL)
print("SAVED", OUTPUTVOCAB)