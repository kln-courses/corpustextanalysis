__author__ = 'kln-courses'
import pandas as pd
### import labMT's dictionary from PLOSOne
url = 'http://www.plosone.org/article/fetchSingleRepresentation.action?uri=info:doi/10.1371/journal.pone.0026752.s001'
labmt = pd.read_csv(url, skiprows=2, sep='\t', index_col=0)
# raw scores
dictionary = labmt.happiness_average.to_dict()
# zero center
labmtbar = labmt.happiness_average.mean()
dictionary_0 = (labmt.happiness_average - labmtbar).to_dict()
# function for string input
def sent_scr(string):
    tokens = string.split()
    #return sum([dictionary.get(token.lower(), 0.0) for token in tokens])
    return sum([dictionary_0.get(token.lower(), 0.0) for token in tokens])
    #return sum([dictionary.get(token.lower(), 0.0) for token in tokens]) / len(tokens)
    #return sum([dictionary_0.get(token.lower(), 0.0) for token in tokens]) / len(tokens)
    #return sum([dictionary.get(token.lower(), 0.0) for token in tokens]) / len(tokens)# for comparison

### example
#sent_scr('nigger')
#print sent_scr('some hate')
#sent_scr('some love and some hate')
## bag of words problem
#sent_scr('love to hate') == sent_scr('hate to love')

sent = 'Did Crooked Hillary help disgusting (check out sex tape and past) Alicia M become a U.S. citizen so she could use her in the debate?'
print sent_scr(sent.lower())

sent_tok = sent.split()


scr = []
for i in range(0,len(sent_tok)):
    scr.append(sent_scr(sent_tok[i]))

zip(sent_tok,scr)
