"""
Method 2 - Collocations into vector
"""
import io
from contextlib import redirect_stdout
from nltk import bigrams
import numpy

class Collocation_Method:
    # Function to store nltk collocation output to a list:
    def collocation_list(self, tokenized_text, num, window_size):
        f = io.StringIO()
        with redirect_stdout(f):
            tokenized_text.collocations(num=num, window_size=window_size)
            collocation = f.getvalue()
            collocation = collocation.replace("\n", " ")
            collocation = collocation.strip()
            return collocation.split("; ")

    # Function to return a list of bigram words
    def text_bigrams(self, tokenized_text):
        book_bigrams = list(bigrams(tokenized_text))
        bigrams_combined = [element[0]+" "+element[1] for element in book_bigrams]
        return bigrams_combined

    def create_vector(self, nltk_books):

        # Getting collocations from all books and storing them into lists
        all_book_collocations = []
        one_list_collocations = []
        for book in nltk_books:
            all_book_collocations.append(self.collocation_list(book, 30, 4))
            one_list_collocations = one_list_collocations + self.collocation_list(book, 30, 4)

		# Creating a matrix of normalized collocation frequencies of the books
        collocation_tf_vector = numpy.zeros((len(nltk_books), len(one_list_collocations)))

        i = 0
        for book in nltk_books:
            bi = self.text_bigrams(book)
            collocation_tf_vector[i] = [bi.count(word)/len(bi) for word in one_list_collocations]
            i = i + 1

        return collocation_tf_vector
