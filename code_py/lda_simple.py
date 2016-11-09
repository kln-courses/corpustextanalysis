__author__ = 'kln-courses'
import io, glob, os, re, unicodedata
wd = "/home/kln"
datapath = "/home/kln/Desktop/plaintext"
###import unicode plain text
os.chdir(datapath)
contents = []
regex = re.compile('[,\.!?0-9]')# filter with regex
for file in glob.glob("*.txt"):
    with io.open(file,'r',encoding='utf8') as f:
       text = f.read()
       text = unicodedata.normalize('NFKD', text).encode('ascii','ignore')# normalize unicode
       contents.append(regex.sub('', text))
os.chdir(wd)
# import stopword list
filename = 'stopwords_eng.txt'
with io.open(filename,'r',encoding='utf8') as f:
    text = f.read()
stoplist = set(text.split())
# tokenize and case fold
contents_tok = [[w for w in doc.lower().split() if w not in stoplist] for doc in contents]
# chunk documents in n chuncks
n = 100
from gensim.utils import chunkize
contents_chunk = []
for doc in contents_tok:
    clen = len(doc)/n
    for c in chunkize(doc,clen):
         contents_chunk.append(c)
# extract raw frequencies
from gensim import corpora, models
from collections import defaultdict
import numpy as np
# compute word freq
frequency = defaultdict(int)
for chunk in contents_chunk:
    for token in chunk:
        frequency[token] += 1
freq = [val for val in frequency.values()]
# prune bottum (mn) and top (mx)
mn = 1
mx = np.percentile(freq, 98)
contents_chunk = [[token for token in chunk if frequency[token] > mn and frequency[token] <= mx] for chunk in contents_chunk]
## lemmatize using with NLTK using POS tags from WordNet
from nltk.corpus import wordnet
# change from treebank to wordnet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN# assume noun as baseline
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.tag import pos_tag
for i, _ in enumerate(contents_chunk):# loop over chunks
    tmp = pos_tag(contents_chunk[i])
    for ii, _ in enumerate(tmp):# loop over tokens
        contents_chunk[i][ii] = wordnet_lemmatizer.lemmatize(tmp[ii][0],get_wordnet_pos(tmp[ii][1]))
## simple LDA model
# bag-of-words
dictionary = corpora.Dictionary(contents_chunk)
corpus = [dictionary.doc2bow(chunk) for chunk in contents_chunk]
# for reproducibility
import numpy as np
fixed_seed = 23
np.random.seed(fixed_seed)
# train model on k topics
k = 20
mdl = models.LdaModel(corpus, id2word=dictionary, num_topics=k, chunksize=3125, passes=25, update_every=0, alpha=None, eta=None, decay=0.5, distributed=False)
# print topics
for i in range(0,k):
    print 'Topic', i+1
    print(mdl.show_topic(i))
    print('-----')
