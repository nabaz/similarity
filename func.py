from sklearn.feature_extraction.text import TfidfVectorizer
import csv
file = './synopsis.csv'
synopese =['The minute-by-minute story of the first attempt to bring down the World Trade Center, with eyewitness accounts from the people who were there. At 12:18 p.m. on February 26, 1993, an explosion rocked the giant World Trade Center in New York City, and America discovered how vulnerable it was to terrorism.',
 'An explosion rocked the World Trade Center in February 1993 and witnesses describe their reactions to the terrorist attack.']


import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]
try:
    with open(file, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        ppl = []
        gracenot = []
        for row in csv_reader:
            # import pdb; pdb.set_trace()
            ppl.append(row[1])
            gracenot.append(row[2])
except:
    print("woops")


file = open("result.txt", "w")

for p, g in zip(ppl, gracenot):
    file.write(str(cosine_sim(p, g)))
    file.write("\n")