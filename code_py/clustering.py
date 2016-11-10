__author__ = 'kln-courses'
## import unicode vanilla
import io, glob, os, re, unicodedata
# normalize: punctuation, numeric, character
def vanilla_folder(datapath):
    os.chdir(datapath)
    docs = []
    regex = re.compile('[,\.!?0-9]')# filter with regex
    for file in glob.glob("*.txt"):
        with io.open(file,'r',encoding = 'utf8') as f:
            text = f.read()
            text = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
            docs.append(regex.sub('', text))
    return docs

datapath = '/home/kln/corpora/kjv_books'
docs = vanilla_folder(datapath)

## kmeans partioning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# build vector space
vectorizer = TfidfVectorizer(stop_words='english')
vectspc = vectorizer.fit_transform(docs)

# train model
k = 5
mdl = KMeans(n_clusters = k, init='k-means++', max_iter=100, n_init=1, random_state = 1234)
mdl.fit(vectspc)

print("Top features per cluster:")
order_centroids = mdl.cluster_centers_.argsort()[:, ::-1]
features = vectorizer.get_feature_names()
for i in range(k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % features[ind],
    print

## add 2d projection for visualization
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
# build vector space and transform to sparse matrix
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])
vectspc = pipeline.fit_transform(docs).todense()
# train model
k = 2
mdl = KMeans(n_clusters = k, init='k-means++', max_iter=100, n_init=1, random_state = 1234)
mdl.fit(vectspc)
# projection
pca = PCA(n_components = 2).fit(vectspc)
data_proj = pca.transform(vectspc)
centers_proj = pca.transform(mdl.cluster_centers_)
# plot
plt.scatter(data_proj[:,0], data_proj[:,1], c = mdl.labels_)
plt.hold(True)
plt.scatter(centers_proj[:,0], centers_proj[:,1],
            marker='x', s=200, linewidths=3, c='r')
plt.show()
