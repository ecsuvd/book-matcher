#!/usr/local/bin/python3
"""
Command-line starter for score matching text analysis of books
"""
import sys
import src.book_matcher

def main(argv):
    if len(argv) > 0:
        if argv[0] == "collocation":
            calc_method = "collocation"
        elif argv[0] == "tfidf":
            calc_method = "tfidf"
        else:
            print("Error: method is not included.")
            sys.exit(1)
        book_matcher = src.book_matcher.Book_Matcher()
        closest_match_list = book_matcher.get_closest_books(calc_method)
        print(closest_match_list)
    else:
        print("Error: no arguments provided")
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[1:])
