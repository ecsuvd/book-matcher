"""
Book matcher which loads books and returns most similar books using 2 different methods.
"""
import nltk
import os
from os import listdir
from nltk import word_tokenize
from scipy import spatial
import heapq
import src.tfidf_method
import src.collocation_method
import numpy

class Book_Matcher:
    def get_text(self, path):
        fileptr = open(path)
        print("Opening " + path)
        return fileptr.read()

    def text_nltk(self, raw):
        tokens = word_tokenize(raw)
        return nltk.Text(tokens)

    def closest_texts(self, vector, text_titles):
        max3 = heapq.nlargest(4, enumerate(vector), key=lambda x: x[1])
        indexes = [x[0] for x in max3[1:4]]
        return [(text_titles[i], vector[i]) for i in indexes]
        
    def get_closest_books(self, calc_method):
        # Preparing texts for the analysis
        book_dir = "books/"

        books = []
        book_titles = []
        for book in os.listdir(book_dir):
            book_titles.append(book)
            books.append(self.get_text(book_dir + book))

       # Converting into nltk text format
        nltk_books = []
        for book in books:
            nltk_books.append(self.text_nltk(book))

        # Getting text vectors
        if calc_method == "collocation":
            mymethod = src.collocation_method.Collocation_Method()
        elif calc_method == "tfidf":
            mymethod = src.tfidf_method.TFIDF_Method()

        vectors = mymethod.create_vector(nltk_books)

        # Calculating cosine similarity of the texts
        cosines_matrix = numpy.zeros((len(nltk_books), len(nltk_books)))
        for i in range(len(nltk_books)):
            for j in range(len(nltk_books)):
                cosines_matrix[i][j] = 1 - spatial.distance.cosine(vectors[i], vectors[j])

        # Finding the most similar 3 texts for each text
        closest3 = {}
        for i, item in enumerate(cosines_matrix):
            closest3[book_titles[i]] = self.closest_texts(cosines_matrix[i], book_titles)

        return closest3
