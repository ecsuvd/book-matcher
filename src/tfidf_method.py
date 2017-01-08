"""
Method 1 - TFIDF

TFIDF (term frequency and inverse term frequency) for the most frequent words in the texts.
Values are inspired from these links:
https://gist.github.com/himzzz/4105717
http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
"""
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.book import FreqDist
import math
import numpy

class TFIDF_Method:
    # Basic text cleaning function
    def clean_text(self, text, language):
        lowercase_text = [word.lower() for word in text if word.isalpha()]
        stop = set(stopwords.words(language))
        text_without_stopwords = [word for word in lowercase_text if word not in stop]
        porter = PorterStemmer()
        return [porter.stem(w) for w in text_without_stopwords]

    # Collect the most frequent N words in a (cleaned) text
    def most_freq_words(self, text, number):
        word_freq = FreqDist(text)
        words_counts = word_freq.most_common(number)
        words = [pair[0] for pair in words_counts]
        return words

    def tf(self, word, book):
        return (book.count(word) / float(len(book)))

    def n_docs_containing(self, word, booklist):
        return sum(1 for book in booklist if book.count(word) > 0)

    # Formula is from the "Data Science for Business" (2013). It is equal to scilearn smooth idf=false
    def idf(self, word, booklist):
        return (1 + math.log(len(booklist) / self.n_docs_containing(word, booklist)))

    def tfidf(self, word, book, booklist):
        return self.tf(word, book) * self.idf(word, booklist)

    def create_vector(self, nltk_books):
		# Cleaning texts
        cleaned_books = []
        for book in nltk_books:
            cleaned_books.append(self.clean_text(book, "english"))

		# List of 100 most frequent words in each text:
        freq_words = []
        for book in cleaned_books:
            freq_words.append(self.most_freq_words(book, 100))

		# List of all words from frequent word list:
        all_words = []
        for words in freq_words:
            for word in words:
                if word not in all_words:
                    all_words.append(word)
		
        # TFIDF vector matrix for the texts
        tfidf_vector = numpy.zeros((len(cleaned_books), len(all_words)))
        j = 0
        for book in cleaned_books:
            i = 0
            for word in all_words:
                tfidf_vector[j][i] = self.tfidf(word, book, cleaned_books)
                i = i + 1
            j = j + 1

        return tfidf_vector
