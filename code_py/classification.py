__author__ = 'kln-courses'
## import unicode vanilla
import io, glob, os, re, unicodedata
# normalize: punctuation, numeric, character
def vanilla_folder(datapath):
    os.chdir(datapath)
    docs = []
    regex = re.compile('["<>(),\.!?0-9]')
    for file in glob.glob("*.txt"):
        with io.open(file,'r',encoding = 'utf8') as f:
            text = f.read()
            text = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
            docs.append(regex.sub('', text))
    return docs

datapath = '/home/kln/corpora/kjv_books'
docs = vanilla_folder(datapath)

import pandas as pd
import numpy as np
metadata = pd.read_csv('/home/kln/corpora/kjv_metadata.csv')
class_id = metadata['class'].tolist()
class_u, class_int = np.unique(class_id, return_inverse = True)

from sklearn.feature_extraction.text import CountVectorizer
countvect = CountVectorizer()
vectspc = countvect.fit_transform(docs)
vectspc.shape

# index value of a word in the vocabulary
countvect.vocabulary_.get(u'god')
countvect.vocabulary_.get(u'woman')

# build vector space model
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
vectspc_tfidf = tfidf_transformer.fit_transform(vectspc)
vectspc_tfidf.shape

# train naive bayes classfier
from sklearn.naive_bayes import MultinomialNB
nb_class = MultinomialNB().fit(vectspc_tfidf, class_id)
# classifier training performance
predicted = nb_class.predict(vectspc_tfidf)
np.mean(predicted == class_id)

# svm for comparison
from sklearn.linear_model import SGDClassifier
svm_class = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(vectspc_tfidf, class_id)
predicted = svm_class.predict(vectspc_tfidf)
np.mean(predicted == class_id)

# unseen data
newdoc = vanilla_folder('/home/kln/corpora/kjv_test')
vectspc_new = countvect.transform(newdoc)
vectspc_tfidf_new = tfidf_transformer.transform(vectspc_new)
vectspc_tfidf_new.shape
print nb_class.predict(vectspc_tfidf_new)
print svm_class.predict(vectspc_tfidf_new)

## pipeline for nb and svm classfier for performance metrics
from sklearn.pipeline import Pipeline
docs_test = docs
#nb
nb_class = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
_ = nb_class.fit(docs, class_int)
predicted = nb_class.predict(docs_test)
np.mean(predicted == class_int)
# svm
svm_class = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
_ = svm_class.fit(docs, class_int)
predicted = svm_class.predict(docs_test)
np.mean(predicted == class_int)
# metrics
from sklearn import metrics
print metrics.classification_report(class_int, predicted,target_names=class_u)
print metrics.confusion_matrix(class_int, predicted)
