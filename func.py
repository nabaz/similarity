from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import nltk, string

nltk.download('punkt') # if necessary...

syn_file = './synopsis.csv'
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
    with open(syn_file, encoding='utf-8') as csv_file:
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
