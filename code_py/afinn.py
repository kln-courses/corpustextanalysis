from afinn import Afinn
afinn = Afinn()
afinn.score('This is utterly excellent!')


afinn = Afinn(language='da')
afinn.score('Hvis ikke det er det mest afskyelige flueknepper')

from afinn import Afinn
from nltk.corpus import gutenberg
import textwrap
afinn = Afinn()
sentences = (" ".join(wordlist) for wordlist in gutenberg.sents('austen-sense.txt'))
scored_sentences = ((afinn.score(sent), sent) for sent in sentences)
sorted_sentences = sorted(scored_sentences)
print("\n".join(textwrap.wrap(sorted_sentences[0][1], 70)))



from afinn import Afinn
afinn = Afinn(language='da')
bh_str = 'Røvhul! Dumme svin! Du behandler mig som lort! Nu kan du rende og hoppe jeg vil ikke tale med dig nogensinde igen'
afinn.score(bh_str)
