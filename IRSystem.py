#!/usr/bin/env python
import json
import math
import os
import re
import sys

from PorterStemmer import PorterStemmer
from collections import defaultdict, Counter


class IRSystem:

    def __init__(self):
        # For holding the data - initialized in read_data()
        self.titles = []    # list of titles
        self.docs = []      # list of contents (each content is list of str)
        self.vocab = []     # list of vocabulary words
        # For the text pre-processing.
        self.alphanum = re.compile('[^a-zA-Z0-9]')
        self.p = PorterStemmer()


    def get_uniq_words(self):
        uniq = set()
        for doc in self.docs:
            for word in doc:
                uniq.add(word)
        return uniq


    def __read_raw_data(self, dirname):
        print("Stemming Documents...")

        titles = []
        docs = []
        os.mkdir('%s/stemmed' % dirname)
        title_pattern = re.compile('(.*) \d+\.txt')

        # make sure we're only getting the files we actually want
        filenames = []
        for filename in os.listdir('%s/raw' % dirname):
            if filename.endswith(".txt") and not filename.startswith("."):
                filenames.append(filename)

        for i, filename in enumerate(filenames):
            title = title_pattern.search(filename).group(1)
            print("    Doc %d of %d: %s" % (i+1, len(filenames), title))
            titles.append(title)
            contents = []
            f = open('%s/raw/%s' % (dirname, filename), 'r', encoding="utf-8")
            of = open('%s/stemmed/%s.txt' % (dirname, title), 'w', encoding="utf-8")
            for line in f:
                # make sure everything is lower case
                line = line.lower()
                # split on whitespace
                line = [xx.strip() for xx in line.split()]
                # remove non alphanumeric characters
                line = [self.alphanum.sub('', xx) for xx in line]
                # remove any words that are now empty
                line = [xx for xx in line if xx != '']
                # stem words
                line = [self.p.stem(xx) for xx in line]
                # add to the document's conents
                contents.extend(line)
                if len(line) > 0:
                    of.write(" ".join(line))
                    of.write('\n')
            f.close()
            of.close()
            docs.append(contents)
        return titles, docs


    def __read_stemmed_data(self, dirname):
        print("Already stemmed!")
        titles = []
        docs = []

        # make sure we're only getting the files we actually want
        filenames = []
        for filename in os.listdir('%s/stemmed' % dirname):
            if filename.endswith(".txt") and not filename.startswith("."):
                filenames.append(filename)

        if len(filenames) != 60:
            msg = "There are not 60 documents in ../data/RiderHaggard/stemmed/\n"
            msg += "Remove ../data/RiderHaggard/stemmed/ directory and re-run."
            raise Exception(msg)

        for i, filename in enumerate(filenames):
            title = filename.split('.')[0]
            titles.append(title)
            contents = []
            f = open('%s/stemmed/%s' % (dirname, filename), 'r', encoding="utf-8")
            for line in f:
                # split on whitespace
                line = [xx.strip() for xx in line.split()]
                # add to the document's conents
                contents.extend(line)
            f.close()
            docs.append(contents)

        return titles, docs


    def read_data(self, dirname):
        """
        Given the location of the 'data' directory, reads in the documents to
        be indexed.
        """
        # NOTE: We cache stemmed documents for speed
        #       (i.e. write to files in new 'stemmed/' dir).

        print("Reading in documents...")
        # dict mapping file names to list of "words" (tokens)
        filenames = os.listdir(dirname)
        subdirs = os.listdir(dirname)
        if 'stemmed' in subdirs:
            titles, docs = self.__read_stemmed_data(dirname)
        else:
            titles, docs = self.__read_raw_data(dirname)

        # Sort document alphabetically by title to ensure we have the proper
        # document indices when referring to them.
        ordering = [idx for idx, title in sorted(enumerate(titles),
            key = lambda xx : xx[1])]

        self.titles = []
        self.docs = []
        numdocs = len(docs)
        for d in range(numdocs):
            self.titles.append(titles[ordering[d]])
            self.docs.append(docs[ordering[d]])

        # Get the vocabulary.
        self.vocab = [xx for xx in self.get_uniq_words()]

#--------------------------------------------------------------------------#
# CODE TO UPDATE STARTS HERE                                               #
#--------------------------------------------------------------------------#

    def index(self):
        """
        Build an index of the documents.
        """
        print("Indexing...")
        # ------------------------------------------------------------------
        # TODO: Create an inverted, positional index.
        #       Granted this may not be a linked list as in a proper
        #       implementation.
        #       This index should allow easy access to both 
        #       1) the documents in which a particular word is contained, and 
        #       2) for every document, the positions of that word in the document 
        #       Some helpful instance variables:
        #         * self.docs = List of documents
        #         * self.titles = List of titles

        inv_index = defaultdict(set)  # map word to {doc_1: [index, index]}, {doc_2; [index, ...]}
            # i.e. format of : {word: [{doc: [idx_list]}, {doc2: [idx_list]}]

        # Generate inverted index here

        for word in self.vocab:
            inv_index[word] = {}      # each word gets its own sub-dictionary
        
        for doc, title in zip(self.docs, self.titles):  
            for word in self.vocab:
                inv_index[word][title] = []        # {word: [dict_for_title, dict_for_title2, ...], word2: ...}  and then dict_for_doct1 = {docid: [idx, idx, idx]}
            for idx, word in enumerate(doc):
                inv_index[word][title].append(idx)

        self.inv_index = inv_index

        # ------------------------------------------------------------------

        # turn self.docs into a map from ID to bag of words
        id_to_bag_of_words = {}
        for d, doc in enumerate(self.docs):
            bag_of_words = set(doc)
            id_to_bag_of_words[d] = bag_of_words
        self.docs = id_to_bag_of_words


    def get_word_positions(self, word, doc):
        """
        Given a word and a document, use the inverted index to return
        the positions of the specified word in the specified document.
        """
        # ------------------------------------------------------------------
        # TODO: return the list of positions for a word in a document.

        # doc is already a doc id

        title = self.titles[doc]
        positions = self.inv_index[word][title]     # this is a list

        return positions
        # ------------------------------------------------------------------


    def get_posting(self, word):
        """
        Given a word, this returns the list of document indices (sorted) in
        which the word occurs.
        """
        # ------------------------------------------------------------------
        # TODO: return the list of postings for a word.
        posting = [idx for idx, title in enumerate(self.titles) if len(self.inv_index[word][title]) != 0]

        return posting
        # ------------------------------------------------------------------


    def get_posting_unstemmed(self, word):
        """
        Given a word, this *stems* the word and then calls get_posting on the
        stemmed word to get its postings list. You should *not* need to change
        this function. It is needed for submission.
        """
        word = self.p.stem(word)
        return self.get_posting(word)



    def boolean_retrieve(self, query):
        """
        Given a query in the form of a list of *stemmed* words, this returns
        the list of documents in which *all* of those words occur (ie an AND
        query).
        Return an empty list if the query does not return any documents.
        """
        # ------------------------------------------------------------------
        # TODO: Implement Boolean retrieval. You will want to use your
        #       inverted index that you created in index().
        # Right now this just returns all the possible documents!

        # ELISSA NOTES:
        # the query is a list of stemmed words
        # for each word in the query, i want to get its posting (the list of doc_id's it appears in)
        # and i want all of that in a list, i.e. a list of word postings for each word, i.e. a list of lists

        all_the_postings = [set(self.get_posting(word)) for word in query]
        docs = set.intersection(*all_the_postings)      # list of doc indices

        # a = {1,2,3}
        # b = {2,3,4}
        # c = {7,8,9}

        # x = [a,b,c]

        # y = [1,2,3,3]
        # set(y)

        # set.intersection(*x)

        # ------------------------------------------------------------------

        return sorted(docs)   # sorted doesn't actually matter
            # but actually super helpful bc will turn my set back into a list


    def phrase_retrieve(self, query):
        """
        Given a query in the form of an ordered list of *stemmed* words, this 
        returns the list of documents in which *all* of those words occur, and 
        in the specified order. 
        Return an empty list if the query does not return any documents. 
        """
        # ------------------------------------------------------------------
        # TODO: Implement Phrase Query retrieval (ie. return the documents 
        #       that don't just contain the words, but contain them in the 
        #       correct order) You will want to use the inverted index 
        #       that you created in index(), and may also consider using
        #       boolean_retrieve. 
        #       NOTE that you no longer have access to the original documents
        #       in self.docs because it is now a map from doc IDs to set
        #       of unique words in the original document.
        # Right now this just returns all possible documents!

        docs = []

        # goal is to get docs that match the query in both vocabulary *and* word order
        # so first i'm just gonna narrow down to the docs with the right words/right vocab:

        docs_with_right_words = self.boolean_retrieve(query)    # returns list of doc indices


        # now, of the docs with the right vocab, i want to narrow down to the ones with the right word order
        # will do this by tracking position distance from a starter position (the position of the first word in the query)
            # basically am asking: if the first word in the query appears in place 15, does the second word in the query appear in place 16? and the third in 17? 
        # perf_match is a bool that says, "This doc is a perfect query match until proven otherwise". as i check each word in the query for a doc,
            # if the word is in the right place (still incrementing by 1), then perf_match stays happy. if at any point a word isn't where it should be,
            # perf_match becomes sad and breaks - moves on to the next doc.

        for doc in docs_with_right_words:
            title = self.titles[doc]
            pos_list = []                           # will be list of [pos_lists] for each word in query

            for word in query: 
                pos_list.append(self.get_word_positions(word, doc))
            
            if len(pos_list) == 1:  # if i only 
                docs.append(doc)
                break

            perf_match = True                       # perf_match starts off happy
            
            for pos in pos_list[0]:                 # for the position list for the first word
                for i in range(1, len(query)):      # iterate through the next words in query
                    if (pos + i) in pos_list[i]:    # if starter pos + new positions are in the pos_list:
                        perf_match = True           # perf_match stays happy
                    else:
                        perf_match = False          # perf_match is sad
                        break
                if perf_match == True: 
                    docs.append(doc)

        # ------------------------------------------------------------------

        return sorted(docs)   # sorted doesn't actually matter


    def compute_tfidf(self):
        # -------------------------------------------------------------------
        # TODO: Compute and store TF-IDF values for words and documents.
        #       Recall that you can make use of:
        #         * self.vocab: a list of all distinct (stemmed) words
        #         * self.docs: a list of lists, where the i-th document is
        #                   self.docs[i] => ['word1', 'word2', ..., 'wordN']
        #       NOTE that you probably do *not* want to store a value for every
        #       word-document pair, but rather just for those pairs where a
        #       word actually occurs in the document.
        print("Calculating tf-idf...")
        # self.tfidf = defaultdict(Counter)       # should be {doc_id: {word:score, word:score...}, doc_id: {word:score, word:score}...}
        # self.tf = defaultdict(Counter)          # should be {doc_id: {word:tf, word:tf...}, doc_id: {word:tf, word:tf}...}
        #                                         # self.docs = {doc_id: [words, words, words], doc_id: [words, words, words]...}

        # first get the word's tf for each doc
        for d in range(len(self.docs)):     # for each doc(id) in the list (which is a map of doc id's to set of unique words)
            for word in self.docs[d]:       # for each word is in the doc_id's set of unique words:
                self.tf[d][word] += 1       # word freq in doc
            
        for word in self.vocab:
            idf = math.log10(len(self.docs)/len(self.get_posting(word)))        # idf is (tot docs)/(num docs the word appears in)
            # then update the tfidf dict 
            for d in range(len(self.docs)):
                try:
                    self.tfidf[d][word] = (1 + math.log10(self.tf[d][word])) * idf
                except ValueError:
                    self.tfidf[d][word] = 0
                

        # ------------------------------------------------------------------


    def get_tfidf(self, word, document):
        # ------------------------------------------------------------------
        # TODO: Return the tf-idf weigthing for the given word (string) and
        #       document index.
        tfidf = self.tfidf[document][word]
        # ------------------------------------------------------------------
        return tfidf


    def get_tfidf_unstemmed(self, word, document):
        """
        This function gets the TF-IDF of an *unstemmed* word in a document.
        Stems the word and then calls get_tfidf. You should *not* need to
        change this interface, but it is necessary for submission.
        """
        word = self.p.stem(word)
        return self.get_tfidf(word, document)


    def rank_retrieve(self, query):
        """
        Given a query (a list of words), return a rank-ordered list of
        documents (by ID) and score for the query.
        """
        k=10
        scores = [0.0 for xx in range(len(self.titles))]
        # ------------------------------------------------------------------
        # TODO: Implement cosine similarity between a document and a list of
        #       query words.

        # Right now, this code simply gets the score by taking the Jaccard
        # similarity between the query and every document.

        # cos_sim = query_vec . doc_vec / len(doc_vec)


        words_in_query = set()                  # this is all the words/the vocab in the query
        for word in query:
            words_in_query.add(word)

        query_vec = {}      # this is a dict of loc frequencies (the tf_)
        for word in query:
            word_count = query.count(word)      # how many times this word appears in the query
            query_vec[word] = 1 + math.log10(word_count)    # each word's tf   ... note there are no zeroes since this is only for the words already in the query (everything else *is* 0)


        for d, words_in_doc in self.docs.items():
            doc_vec = {}                    # there's a doc_vec for each doc of form {word: tfidf}
            qd_words = words_in_query.intersection(words_in_doc)   # the words in the query that also appear in the doc
            for word in qd_words:
                doc_vec[word] = self.get_tfidf(word, d)     # where 'd' is a document
            sq_len_d = 0
            for word in words_in_doc:
                sq_len_d += self.get_tfidf(word, d)**2       # the length of document vector is the sqrt of sum of squares of tfidf's of each elem

            len_d = math.sqrt(sq_len_d)
            doc_vec = {word:v / len_d for word, v in doc_vec.items()}
            scores[d] = 0
            q_words = query_vec.keys()
            for word in q_words:
                if word in doc_vec:
                    scores[d] += query_vec[word] * doc_vec[word]        # dot_prod of the query and doc


        # ------------------------------------------------------------------

        ranking = [idx for idx, sim in sorted(enumerate(scores),
            key = lambda xx : xx[1], reverse = True)]
        results = []
        for i in range(10):
            results.append((ranking[i], scores[ranking[i]]))
        return results

#--------------------------------------------------------------------------#
# CODE TO UPDATE ENDS HERE                                                 #
#--------------------------------------------------------------------------#


    def process_query(self, query_str):
        """
        Given a query string, process it and return the list of lowercase,
        alphanumeric, stemmed words in the string.
        """
        # make sure everything is lower case
        query = query_str.lower()
        # split on whitespace
        query = query.split()
        # remove non alphanumeric characters
        query = [self.alphanum.sub('', xx) for xx in query]
        # stem words
        query = [self.p.stem(xx) for xx in query]
        return query

    def query_retrieve(self, query_str):
        """
        Given a string, process and then return the list of matching documents
        found by boolean_retrieve().
        """
        query = self.process_query(query_str)
        return self.boolean_retrieve(query)

    def phrase_query_retrieve(self, query_str):
        """
        Given a string, process and then return the list of matching documents
        found by phrase_retrieve().
        """
        query = self.process_query(query_str)
        return self.phrase_retrieve(query)

    def query_rank(self, query_str):
        """
        Given a string, process and then return the list of the top matching
        documents, rank-ordered.
        """
        query = self.process_query(query_str)
        return self.rank_retrieve(query)


def run_tests(irsys):
    print("===== Running tests =====")

    ff = open('../data/queries.txt')
    questions = [xx.strip() for xx in ff.readlines()]
    ff.close()
    ff = open('../data/solutions.txt')
    solutions = [xx.strip() for xx in ff.readlines()]
    ff.close()

    epsilon = 1e-4
    for part in range(6):
        points = 0
        num_correct = 0
        num_total = 0

        prob = questions[part]
        soln = json.loads(solutions[part])

        if part == 0:   # inverted index test
            print("Inverted Index Test")
            queries = prob.split("; ")
            queries = [xx.split(", ") for xx in queries]
            queries = [(xx[0], int(xx[1])) for xx in queries]
            for i, (word, doc) in enumerate(queries):
                num_total += 1
                guess = irsys.get_word_positions(word, doc)
                if sorted(guess) == soln[i]:
                    num_correct += 1

        if part == 1:     # get postings test
            print("Get Postings Test")
            words = prob.split(", ")
            for i, word in enumerate(words):
                num_total += 1
                posting = irsys.get_posting_unstemmed(word)
                if posting == soln[i]:
                    num_correct += 1

        elif part == 2:   # boolean retrieval test
            print("Boolean Retrieval Test")
            queries = prob.split(", ")
            for i, query in enumerate(queries):
                num_total += 1
                guess = irsys.query_retrieve(query)
                if set(guess) == set(soln[i]):
                    num_correct += 1

        elif part == 3: # phrase query test
            print("Phrase Query Retrieval")
            queries = prob.split(", ")
            for i, query in enumerate(queries):
                num_total += 1
                guess = irsys.phrase_query_retrieve(query)
                if set(guess) == set(soln[i]):
                    num_correct += 1

        elif part == 4:   # tfidf test
            print("TF-IDF Test")
            queries = prob.split("; ")
            queries = [xx.split(", ") for xx in queries]
            queries = [(xx[0], int(xx[1])) for xx in queries]
            for i, (word, doc) in enumerate(queries):
                num_total += 1
                guess = irsys.get_tfidf_unstemmed(word, doc)
                if guess >= float(soln[i]) - epsilon and \
                        guess <= float(soln[i]) + epsilon:
                    num_correct += 1

        elif part == 5:   # cosine similarity test
            print("Cosine Similarity Test")
            queries = prob.split(", ")
            for i, query in enumerate(queries):
                num_total += 1
                ranked = irsys.query_rank(query)
                top_rank = ranked[0]
                if top_rank[0] == soln[i][0]:
                    if top_rank[1] >= float(soln[i][1]) - epsilon and \
                            top_rank[1] <= float(soln[i][1]) + epsilon:
                        num_correct += 1

        feedback = "%d/%d Correct. Accuracy: %f" % \
                (num_correct, num_total, float(num_correct)/num_total)

        if part == 1:
            if num_correct == num_total:
                points = 2
            elif num_correct >= 0.5 * num_total:
                points = 1
            else:
                points = 0
        elif part == 2:
            if num_correct == num_total:
                points = 1
            else:
                points = 0
        else:
            if num_correct == num_total:
                points = 3
            elif num_correct > 0.75 * num_total:
                points = 2
            elif num_correct > 0:
                points = 1
            else:
                points = 0

        print("    Score: %d Feedback: %s" % (points, feedback))


def main(args):
    irsys = IRSystem()
    irsys.read_data('../data/RiderHaggard')
    irsys.index()
    irsys.compute_tfidf()

    if len(args) == 0:
        run_tests(irsys)
    else:
        query = " ".join(args)
        print("Best matching documents to '%s':" % query)
        results = irsys.query_rank(query)
        for docId, score in results:
            print("%s: %e" % (irsys.titles[docId], score))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
