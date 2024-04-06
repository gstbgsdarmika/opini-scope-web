import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

class TextPreprocessor:
    def __init__(self, normalisasi_path='models/machine/preprocessing/normalisasi.csv', stopword_path='models/machine/preprocessing/stopword.csv'):
        self.normalized_word_dict = self.load_normalization(normalisasi_path)
        self.stopwords = self.load_stopwords(stopword_path)
        self.stemmer = self.create_stemmer()

    def load_normalization(self, path):
        normalized_word = pd.read_csv(path, encoding='latin1')
        normalized_word_dict = {}
        for index, row in normalized_word.iterrows():
            if row[0] not in normalized_word_dict:
                normalized_word_dict[row[0]] = row[1]
        return normalized_word_dict

    def load_stopwords(self, path):
        txt_stopword = pd.read_csv(path, header=None, names=["stopwords"])
        additional_stopwords = list(txt_stopword["stopwords"][0].split(' '))
        factory = StopWordRemoverFactory()
        stopword_sastrawi = factory.get_stop_words()
        stopword = stopword_sastrawi + additional_stopwords + ["petrus", "sech", "bulakparen", "dcs", "mug","apa", "dkk", "kek","bla","nihhh","nyinyir",
                    "background2","nya", "klik", "nih", "wah", "bd","cie", "wahh", "gtgt", "wkwkw", "grgr", "thun", "dong", "mkmk","gp","brengkelan","woi",
                    "twit", "iii", "08alian", "wkwkwkwk", "wkwk","wkwkwk", "ah", "ampnbsp", "bawaslu", "hihihi", "hihi", "eh", "ng","dl","do","kwkwkwkk",
                    "ltpgtampnbspltpgt", "dancukkk", "yach", "kepl", "wow","kretek", "woww", "smpn", "hmmmm", "hehe", "oooiii","onana","kjaernett",
                    "hahaha", "ppp", "nek", "rang", "tuh", "pls", "otw", "pas","haha", "ha", "hahahahaha", "hahahasenget","wakakakakak","wkwkwkw",
                    "xixixixi", "hehehehee", "nder", "aduuuhhh", "lah","lah", "deh", "si", "kan", "njirrrr", "huehehee","yoongi","sulli","bjir",
                    "hehehe", "yahh", "yah", "loh", "elo", "gw", "didkgkl","sih", "lu", "yeyeye", "dlllllllllll", "se","yoon","de","ruu","apeeeeee",
                    "pisss", "yo", "kok", "nge", "wkwkkw", "dah", "wahhh", "apa", "btw", "kwkwkwkwk", "nahh", "nah", "iya"]
        return stopword

    def create_stemmer(self):
        factory = StemmerFactory()
        return factory.create_stemmer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"@\S+", "", text)
        text = re.sub(r'(@[^\s]+|http\S+|#\w+|<.*?>)', '', text)
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.encode('ascii', 'replace').decode('ascii')
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())

    def tokenize_text(self, text):
        regexp = RegexpTokenizer(r'\w+|\$[0-9]+|\S+')
        tokens = regexp.tokenize(text)
        return tokens

    def normalized_term(self, document):
        return [self.normalized_word_dict[term] if term in self.normalized_word_dict else term for term in document]

    def filter_stopwords(self, text):
        return [word for word in text if word not in self.stopwords]

    def stemmed_wrapper(self, term):
        return self.stemmer.stem(term)

    def stemming_text(self, text):
        return [self.stemmed_wrapper(term) for term in text]

    def preprocess_text(self, text):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        normalized_tokens = self.normalized_term(tokens)
        filtered_tokens = self.filter_stopwords(normalized_tokens)
        stemmed_tokens = self.stemming_text(filtered_tokens)

        # Gabungkan hasil stemmed_tokens menjadi satu kalimat
        processed_text = ' '.join(stemmed_tokens)
        return processed_text