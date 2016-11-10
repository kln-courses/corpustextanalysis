__author__ = 'kln-courses'
from gensim import corpora, models, similarities
from gensim.models import ldamodel
#from itertools import izip
from collections import defaultdict
import io, codecs, os, glob, math, re
import numpy as np
import unicodedata
stopwords = [i.strip() for i in codecs.open('python/stopextend.txt','r','utf8').readlines() if i[0] != "#" and i != ""]
def gen_topics(corpus, dictionary, k):
    # Build LDA model using the above corpus    
    np.random.seed(23)
    lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=k)
    corpus_lda = lda[corpus]
    
    # group topics with similar words    
    tops = set(lda.show_topics(k))
    top_cl = []
    for l in tops:
        top = []
        for t in str(l).split(" + "):# tuples have no split attribute
            top.append((t.split("*")[0], t.split("*")[1]))
        top_cl.append(top)        
        
    # generate word topics
    top_w = []
    for i in top_cl:
        top_w.append(":".join([j[1] for j in i]))

    return lda, corpus_lda, top_cl, top_w

####################################################################### 
# Read textfile, build dictionary and bag-of-words corpus
os.chdir("/home/kln/corpora/kjv_books")
docs = []
regex = re.compile('[{}:,\.!?0-9]')
for file in glob.glob("*.txt"):
    with io.open(file,'r',encoding='utf8') as f:
       doc = f.read()
       #text = text.rstrip('\n')
       doc = doc.replace('\n',' ')
       doc = unicodedata.normalize('NFKD', doc).encode('ascii','ignore')# normalize unicode chars
       doc = regex.sub('', doc)
       docs.append(doc.rstrip())
os.chdir("/home/kln")
texts = [[w for w in doc.lower().split() if w not in stopwords]
             for doc in docs]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda, corpus_lda, top_cl, top_w = gen_topics(corpus, dictionary, 10)

for i in top_w:
    print i

for i in top_cl:
    print i    