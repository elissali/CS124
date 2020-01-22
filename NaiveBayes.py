import sys
import getopt
import os
import math
import operator
from collections import defaultdict

class NaiveBayes:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.dev and self.test. 
        """
        def __init__(self):
            self.train = []
            self.dev = []
            self.test = []

    class Example:
        """Represents a document with a label. klass is 'aid' or 'not' by convention.
             words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """NaiveBayes initialization"""
        self.FILTER_STOP_WORDS = False
        self.USE_BIGRAMS = False
        self.BEST_MODEL = False
        self.stopList = set(self.readFile('data/english.stop'))
        #TODO: add other data structures needed in classify() and/or addExample() below
        
        self.count_class = defaultdict(int)     # {(word_i, klass): # occurrences}
        self.count_per_class = defaultdict(int) # {klass: # words}      count number of words per class 
        self.n_docs = defaultdict(int)          # {klass: # docs}       count total number of docs (per class) 
        self.vocab = set()                      # {word}             unique words in corpus

    #############################################################################
    # TODO TODO TODO TODO TODO 
    # Implement the Multinomial Naive Bayes classifier with add-1 smoothing
    # If the FILTER_STOP_WORDS flag is true, you must remove stop words
    # If the USE_BIGRAMS flag is true, your methods must use bigram features instead of the usual 
    # bag-of-words (unigrams)
    # If either of the FILTER_STOP_WORDS or USE_BIGRAMS flags is on, the other is meant to be off. 
    # Hint: Use filterStopWords(words) defined below
    # Hint: Remember to add start and end tokens in the bigram implementation
    # Hint: When doing add-1 smoothing with bigrams, V = # unique bigrams in data. 

    def classify(self, words):
        """ TODO
            'words' is a list of words to classify. Return 'aid' or 'not' classification.
        """

        total_docs = sum(self.n_docs.values())

        # words is like a sentence

        # calculate probabilities
        prior_aid = self.n_docs['aid'] / total_docs
        prior_not = self.n_docs['not'] / total_docs


        # initialize. These will be the cumulative log probs
        # then i want to add the individual word probs to them
        cumprob_aid = math.log(prior_aid)
        cumprob_not = math.log(prior_not)

        # for the no bigrams case:
        if self.USE_BIGRAMS == False:
            for word in words:
                if word not in self.vocab:
                    continue
                else:
                    aid_word_count = self.count_class[(word, 'aid')]                # num occurences of word in 'aid'
                    all_aid_words = self.count_per_class['aid']                    # num 'aid' words
                    cumprob_aid += math.log((aid_word_count + 1)/(all_aid_words + len(self.vocab)))
                    
                    not_word_count = self.count_class[(word, 'not')]
                    all_not_words = self.count_per_class['not']
                    cumprob_not += math.log((not_word_count + 1)/(all_not_words + len(self.vocab)))
                    
        elif self.USE_BIGRAMS == True: 
            words = ["<s>"] + words + ["</s"]       # add start/end signs
            
            for i in range(1, len(words)):
                window = (words[i - 1], words[i])
                if window not in self.vocab:
                    continue
                if window in self.vocab:
                    aid_word_count = self.count_class[(window, 'aid')]
                    all_aid_words = self.count_per_class['aid']
                    cumprob_aid += math.log((aid_word_count + 1)/(all_aid_words + len(self.vocab)))
                    
                    not_word_count = self.count_class[(window, 'not')]
                    all_not_words = self.count_per_class['not']
                    cumprob_not += math.log((not_word_count + 1)/(all_not_words + len(self.vocab)))


        if cumprob_not > cumprob_aid: return 'not'
        return 'aid'
    

    def addExample(self, klass, words):
        """
         * TODO
         * Train your model on an example document with label klass ('aid' or 'not') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier 
         * in the NaiveBayes class.
         * Returns nothing
        """

        self.n_docs[klass] += 1     # for each example, if new klass, add it to self.n_docs
                                    # and add to the n_docs counter 
                                    # (since each example is a new doc)
        if self.FILTER_STOP_WORDS == True:
            words = self.filterStopWords(words) 
        
        if self.USE_BIGRAMS == False:
            for word in words:
                self.vocab.add(word)                        # add it to the vocab
                self.count_class[(word, klass)] += 1    # add occurrence of this word to count_class
                self.count_per_class[klass] += 1        # add occurrence of this word to count_per_class

        if self.USE_BIGRAMS == True:
            words = ["<s>"] + words + ["</s"]       # add start/end signs
            for i in range(1, len(words)):
                window = (words[i - 1], words[i])
                self.vocab.add(window)
                self.count_class[(window, klass)] += 1
                self.count_per_class[klass] += 1


    # END TODO (Modify code beyond here with caution)
    #############################################################################
    
    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here, 
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName,encoding="utf8")
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents)) 
        return result

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def buildSplit(self,include_test=True):
    
        split = self.TrainSplit()
        datasets = ['train','dev']
        if include_test:
            datasets.append('test')
        for dataset in datasets:
            for klass in ['aid','not']:
                dataFile = os.path.join('data',dataset,klass + '.txt')
                with open(dataFile,'r', encoding="utf8") as f:
                    docs = [line.rstrip('\n') for line in f]
                    for doc in docs:
                        example = self.Example()
                        example.words = doc.split()
                        example.klass = klass
                        if dataset == 'train':
                            split.train.append(example)
                        elif dataset == 'dev':
                            split.dev.append(example)
                        else:
                            split.test.append(example)
        return split


    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered
    
def evaluate(FILTER_STOP_WORDS,USE_BIGRAMS):
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.USE_BIGRAMS = USE_BIGRAMS
    split = classifier.buildSplit(include_test=False)
   
    for example in split.train:
        classifier.addExample(example.klass,example.words)

    train_accuracy = calculate_accuracy(split.train,classifier)
    dev_accuracy = calculate_accuracy(split.dev,classifier)

    print('Train Accuracy: {}'.format(train_accuracy))
    print('Dev Accuracy: {}'.format(dev_accuracy))


def calculate_accuracy(dataset,classifier):
    acc = 0.0
    if len(dataset) == 0:
        return 0.0
    else:
        for example in dataset:
            guess = classifier.classify(example.words)
            if example.klass == guess:
                acc += 1.0
        return acc / len(dataset)

        
def main():
    FILTER_STOP_WORDS = False
    USE_BIGRAMS = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fb')
    if ('-f','') in options:
      FILTER_STOP_WORDS = True
    elif ('-b','') in options:
      USE_BIGRAMS = True

    evaluate(FILTER_STOP_WORDS,USE_BIGRAMS)

if __name__ == "__main__":
        main()
