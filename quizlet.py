"""
CS124 PA5: Quizlet // Stanford, Winter 2020
by @lcruzalb, with assistance from @jchen437
"""
import sys
import getopt
import os
import math
import operator
import random
from collections import defaultdict
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize

import math

#############################################################################
###                    CS124 Homework 5: Quizlet!                         ###
#############################################################################

# ---------------------------------------------------------------------------
# Part 1: Synonyms                                                          #
# ----------------                                                          #
# You will implement 4 functions:                                           #
#   cosine_similarity, euclidean_distance, find_synonym, part1_written      #
# ---------------------------------------------------------------------------

def cosine_similarity(v1, v2):
    '''
    Calculates and returns the cosine similarity between vectors v1 and v2

    Arguments:
        v1, v2 (numpy vectors): vectors

    Returns:
        cosine_sim (float): the cosine similarity between v1, v2
    '''
    cosine_sim = 0
    #########################################################
    ## TODO: calculate cosine similarity between v1, v2    ##
    #########################################################

    len_v1 = math.sqrt(np.dot(v1, v1))
    len_v2 = math.sqrt(np.dot(v2, v2))

    cosine_sim = np.dot(v1, v2)/(len_v1 * len_v2)

    #########################################################
    ## End TODO                                            ##
    #########################################################
    return cosine_sim   


def euclidean_distance(v1, v2):
    '''
    Calculates and returns the euclidean distance between v1 and v2

    Arguments:
        v1, v2 (numpy vectors): vectors

    Returns:
        euclidean_dist (float): the euclidean distance between v1, v2
    '''
    euclidean_dist = 0
    #########################################################
    ## TODO: calculate euclidean distance between v1, v2   ##
    #########################################################

    diff_vec = v1 - v2
    euclidean_dist = math.sqrt(np.dot(diff_vec, diff_vec))

    #########################################################
    ## End TODO                                           ##
    #########################################################
    return euclidean_dist                 

def find_synonym(word, choices, embeddings, comparison_metric):
    '''
    Answer a multiple choice synonym question! Namely, given a word w 
    and list of candidate answers, find the word that is most similar to w.
    Similarity will be determined by either euclidean distance or cosine
    similarity, depending on what is passed in as the comparison_metric.

    Arguments:
        word (string): word
        choices (list of strings): list of candidate answers
        embeddings (map): map of words (strings) to their embeddings (np vectors)
        comparison_metric (string): either 'euc_dist' or 'cosine_sim'. 
            This indicates which metric to use - either euclidean distance or cosine similarity.
            With euclidean distance, we want the word with the lowest euclidean distance.
            With cosine similarity, we want the word with the highest cosine similarity.

    Returns:
        answer (string): the word in choices most similar to the given word
    '''
    answer = None
    #########################################################
    ## TODO: find synonym                                  ##
    #########################################################

    # given a word and [4 choices], find the synonym.

    best_choice = None

    if comparison_metric == 'cosine_sim':
        best_score = 0                      # want the highest similarity score
        for choice in choices:
            word_vec = embeddings[word]
            choice_vec = embeddings[choice]
            score = cosine_similarity(word_vec, choice_vec)
            if score > best_score:
                best_score = score
                best_choice = choice

    elif comparison_metric == 'euc_dist':
        best_score = 100000                 # want the lowest distance score
        for choice in choices:
            word_vec = embeddings[word]
            choice_vec = embeddings[choice]
            score = euclidean_distance(word_vec, choice_vec)
            if score < best_score:
                best_score = score
                best_choice = choice
    
    answer = best_choice

    #########################################################
    ## End TODO                                            ##
    ######################################################### 
    return answer

def part1_written():
    '''
    Finding synonyms using cosine similarity on word embeddings does fairly well!
    However, it's not perfect. In particular, you should see that it gets the last
    synonym quiz question wrong (the true answer would be positive):

    30. What is a synonym for sanguine?
        a) pessimistic
        b) unsure
        c) sad
        d) positive

    What word does it choose instead? In 1-2 sentences, explain why you think 
    it got the question wrong.
    '''
    #########################################################
    ## TODO: replace string with your answer               ##
    ######################################################### 
    answer =  "The right answer is (d), positive, but it chooses 'pessimistic' even though that means the opposite... this is probably because the words are often used in the same context/with the same surrounding words, so their word embeddings look very similar. GloVe embedding similarity represents words that appear in the same context/closer together, so words that aren't technically synonyms but which are used in the same context can have high similarity. 'Sanguine' might often be used in a similar context as 'pessimistic' in what the GloVe embeddings were trained on (e.g. maybe all the training texts talked about sanguine in the context of antonyms so sanguine often appeared with its antonym)." 
    
    #########################################################
    ## End TODO                                            ##
    ######################################################### 
    return answer


# ---------------------------------------------------------------------------
# Part 2: Analogies                                                         #
# -----------------                                                         #
# You will implement 1 function: find_analogy_word                          #
# ---------------------------------------------------------------------------

def find_analogy_word(a, b, aa, choices, embeddings):
    '''
    Find the word bb that completes the analogy: a:b -> aa:bb
    A classic example would be: man:king -> woman:queen

    Note: use cosine similarity as your similarity metric

    Arguments:
        a, b, aa (strings): words in the analogy described above
        choices (list): list of strings for possible answer
        embeddings (map): map of words (strings) to their embeddings (np vectors)

    Returns:
        answer: the word bb that completes the analogy
    '''
    answer = None
    #########################################################
    ## TODO: analogy                                       ##
    #########################################################

    a_vec = embeddings[a]
    b_vec = embeddings[b]
    aa_vec = embeddings[aa]

    # this is my ideal vector (the perfect vector for the analogy)
    # e.g. vector('king') - vector('man') + vector('woman')
        # for a:b --> aa:bb
    bb_vec = b_vec - a_vec + aa_vec

    best_choice = None
    best_score = 0

    # want to find the best match for bb_vec from my choices
    for choice in choices:
        choice_vec = embeddings[choice]
        score = cosine_similarity(bb_vec, choice_vec)       
        if score > best_score:
            best_score = score
            best_choice = choice
    
    answer = best_choice

    #########################################################
    ## End TODO                                            ##
    #########################################################
    return answer


# ---------------------------------------------------------------------------
# Part 3: Sentence Similarity                                               #
# ---------------------------                                               #
# You will implement 2 functions: get_embedding and get_similarity          #
# ---------------------------------------------------------------------------

def get_embedding(s, embeddings, use_POS=False, POS_weights=None):
    '''
    Returns vector embedding for a given sentence.

    Hint:
    - to get all the words in the sentence, you can use nltk's `word_tokenize` function
        >>> list_of_words = word_tokenize(sentence_string)
    - to get part of speech tags for words in a sentence, you can use `nltk.pos_tag`
        >>> tagged_tokens = nltk.pos_tag(list_of_words)
    - you can read more here: https://www.nltk.org/book/ch05.html

    Arguments:
        s (string): sentence
        embeddings (map): map of words (strings) to their embeddings (np vectors)
        use_POS (boolean): flag indicating whether to use POS weightings when
            calculating the sentence embedding
        POS_weights (map): map of part of speech tags (strings) to their weights (floats),
            it is only to be used if the use_POS flag is true

    Returns:
        embed (np vector): vector embedding of sentence s
    '''
    embed = np.zeros(embeddings.vector_size)
    #########################################################
    ## TODO: get embedding                                 ##
    #########################################################

    # 1. tokenize the sentence and get list of words

    sentence = word_tokenize(s)

    sentence_embedding = embed

    # 2. if simple sum: just add up all the vectors for each sentence

    if use_POS == False:
        for word in sentence:
            if word in embeddings:              # some of the words aren't in the embedding dict
                word_vec = embeddings[word]     # so if not included, just skip it
                sentence_embedding += word_vec
    
    # 3. if not simple sum... 

    if use_POS == True:
        # a. need to get the POS tag for each word:

        tagged_tokens = nltk.pos_tag(sentence)  # this is a list of tuples of (word, tag)

        # b. then need to get the weight for each word based on its POS tag + weighted sum:

        for tagged_token in tagged_tokens:
            word = tagged_token[0]
            tag = tagged_token[1]
            if word in embeddings:              # again, excluding words that aren't in the embedding dict
                if tag in POS_weights:          # also exclude tags that aren't in the tag dict
                    word_vec = embeddings[word]
                    weight = POS_weights[tag]
                    sentence_embedding += weight * word_vec


    #########################################################
    ## End TODO                                            ##
    #########################################################
    return sentence_embedding

def get_similarity(s1, s2, embeddings, use_POS, POS_weights=None):
    '''
    Given 2 sentences and the embeddings dictionary, convert the sentences
    into sentence embeddings and return the cosine similarity between them.

    Arguments:
        s1, s2 (strings): sentences
        embeddings (map): map of words (strings) to their embeddings (np vectors)
        use_POS (boolean): flag indicating whether to use POS weightings when
            calculating the sentence embedding
        POS_weights (map): map of part of speech tags (strings) to their weights (floats),
            it is only to be used if the use_POS flag is true

    Returns:
        similarity (float): cosine similarity of the two sentence embeddings
    '''
    similarity = 0
    #########################################################
    ## TODO: compute similarity                            ##
    #########################################################

    # 1. get the sentence embeddings for each sentence

    s1_embed = get_embedding(s1, embeddings, use_POS, POS_weights)
    s2_embed = get_embedding(s2, embeddings, use_POS, POS_weights)

    # 2. get the cosine_similarity for them

    similarity = cosine_similarity(s1_embed, s2_embed)

    #########################################################
    ## End TODO                                            ##
    #########################################################
    return similarity


# ---------------------------------------------------------------------------
# Part 4: Exploration                                                       #
# ---------------------------                                               #
# You will implement 2 functions: occupation_exploration & part4_written    #
# ---------------------------------------------------------------------------

def occupation_exploration(occupations, embeddings):
    '''
    Given a list of occupations, return the 5 occupations that are closest
    to 'man', and the 5 closest to 'woman', using cosine similarity between
    corresponding word embeddings as a measure of similarity.

    Arguments:
        occupations (list): list of occupations (strings)
        embeddings (map): map of words (strings) to their embeddings (np vectors)

    Returns:
        top_man_occs (list): list of 5 occupations (strings) closest to 'man'
        top_woman_occs (list): list of 5 occuptions (strings) closest to 'woman'
            note: both lists should be sorted, with the occupation with highest
                  cosine similarity first in the list
    '''
    top_man_occs = []
    top_woman_occs = []
    #########################################################
    ## TODO: get 5 occupations closest to 'man' & 'woman'  ##
    #########################################################

    # 1. top 5 occupations with highest cosine_sim to "man"

    man_vec = embeddings['man']
    all_man_occs = []
    for occupation in occupations:
        occ_vec = embeddings[occupation]
        occ_score = cosine_similarity(man_vec, occ_vec)
        all_man_occs.append((occupation, occ_score))        # append tuple: (job, score)
    sorted_man_occs = sorted(all_man_occs, key = lambda x: x[1], reverse = True)    # sort by the score
    top_man_occs = [x[0] for x in sorted_man_occs][:5]      # only want the first 5, and only want the job (not the score)

    # 2. top 5 occupations with highest cosine_sim to "woman"

    woman_vec = embeddings['woman']
    all_woman_occs = []
    for occupation in occupations:
        occ_vec = embeddings[occupation]
        occ_score = cosine_similarity(woman_vec, occ_vec)
        all_woman_occs.append((occupation, occ_score))
    sorted_woman_occs = sorted(all_woman_occs, key = lambda x: x[1], reverse = True)
    top_woman_occs = [x[0] for x in sorted_woman_occs][:5]

    #########################################################
    ## End TODO                                            ##
    #########################################################
    return top_man_occs, top_woman_occs

def part4_written():
    '''
    Take a look at what occupations you found are closest to 'man' and
    closest to 'woman'. Do you notice anything curious? In 1-2 sentences,
    describe what you find, and why you think this occurs.
    '''
    #########################################################
    ## TODO: replace string with your answer               ##
    ######################################################### 
    answer = "Jobs close to 'woman' tend to be caretaker-type roles (teacher, nurse, maid) whereas the male roles are maybe more gender-neutral (with the exception of 'warrior'). This is to do with gender bias in the data; the data that these embeddings were trained on have 'woman' appearing with 'nurse', 'teacher' etc. associating women with more old-fashioned gender stereotypes."
    #########################################################
    ## End TODO                                            ##
    ######################################################### 
    return answer

# ------------------------- Do not modify code below --------------------------------

# Functions to run each section
def part1(embeddings, synonym_qs, print_q=False):
    '''
    Calculates accuracy on part 1
    '''
    print ('Part 1: Synonyms!')
    print ('-----------------')

    acc_euc_dist = get_synonym_acc('euc_dist', embeddings, synonym_qs, print_q)
    acc_cosine_sim = get_synonym_acc('cosine_sim', embeddings, synonym_qs, print_q)

    print ('accuracy using euclidean distance: %.5f' % acc_euc_dist)
    print ('accuracy using cosine similarity : %.5f' % acc_cosine_sim)
    
    # sanity check they answered written - this is just a heuristic
    written_ans = part1_written()
    if 'TODO' in written_ans:
        print ('Part 1 written answer contains TODO, did you answer it?')

    print (' ')
    return acc_euc_dist, acc_cosine_sim

def get_synonym_acc(comparison_metric, embeddings, synonym_qs, print_q=False):
    '''
    Helper function to compute synonym answering accuracy
    '''
    if print_q:
        metric_str = 'cosine similarity' if comparison_metric == 'cosine_sim' else 'euclidean distance'
        print ('Answering part 1 using %s as the comparison metric...' % metric_str)

    n_correct = 0
    for i, (w, choices, answer) in enumerate(synonym_qs):
        ans = find_synonym(w, choices, embeddings, comparison_metric)
        if ans == answer: n_correct += 1

        if print_q:
            print ('%d. What is a synonym for %s?' % (i+1, w))
            a, b, c, d = choices[0], choices[1], choices[2], choices[3]
            print ('    a) %s\n    b) %s\n    c) %s\n    d) %s' % (a, b, c, d))
            print ('you answered: %s \n' % ans)

    acc = n_correct / len(synonym_qs)
    return acc

def part2(embeddings, analogy_qs, print_q=False):
    '''
    Calculates accuracy on part 2.
    '''
    print ('Part 2: Analogies!')
    print ('------------------')

    n_correct = 0
    for i, (tup, choices) in enumerate(analogy_qs):
        a, b, aa, true_bb = tup
        ans = find_analogy_word(a, b, aa, choices, embeddings)
        if ans == true_bb: n_correct += 1

        if print_q:
            print ('%d. %s is to %s as %s is to ___?' % (i+1, a, b, aa))
            print ('    a) %s\n    b) %s\n    c) %s\n    d) %s' % tuple(choices))
            print ('You answered: %s\n' % ans)

    acc = n_correct / len(analogy_qs)
    print ('accuracy: %.5f' % acc)
    print (' ')
    return acc

def part3(embeddings, sentence_sim_qs, print_q=False):
    '''
    Calculates accuracy of part 3.
    '''
    print ('Part 3: Sentence similarity!')
    print ('----------------------------')

    acc_base = get_sentence_sim_accuracy(embeddings, sentence_sim_qs, use_POS=False, print_q=print_q)
    acc_POS = get_sentence_sim_accuracy(embeddings, sentence_sim_qs, use_POS=True, print_q=print_q)

    print ('accuracy (regular): %.5f' % acc_base)
    print ('accuracy with POS weighting: %.5f' % acc_POS)
    print (' ')
    return acc_base, acc_POS

def get_sentence_sim_accuracy(embeddings, sentence_sim_qs, use_POS, print_q=False):
    '''
    Helper function to compute sentence similarity classification accuracy.
    '''
    THRESHOLD = 0.95
    POS_weights = load_pos_weights_map() if use_POS else None

    if print_q:
        type_str = 'with POS weighting' if use_POS else 'regular'
        print ('Answering part 3 (%s)...' % type_str)

    n_correct = 0
    for i, (label, s1, s2) in enumerate(sentence_sim_qs):
        sim = get_similarity(s1, s2, embeddings, use_POS, POS_weights)
        pred = 1 if sim > THRESHOLD else 0
        if pred == label: n_correct += 1

        if print_q:
            print ('%d. True/False: the following two sentences are semantically similar:' % (i+1))
            print ('     1. %s' % s1)
            print ('     2. %s' % s2)
            print ('You answered: %r\n' % (True if pred == 1 else False))

    acc = n_correct / len(sentence_sim_qs)
    return acc

def part4(embeddings):
    '''
    Runs part 4 functions
    '''
    print ('Part 4: Exploration!')
    print ('--------------------')

    occupations = load_occupations_list()
    top_man_occs, top_woman_occs = occupation_exploration(occupations, embeddings)
    
    print ('occupations closest to "man" - you answered:')
    for i, occ in enumerate(top_man_occs):
        print (' %d. %s' % (i+1, occ))
    print ('occupations closest to "woman" - you answered:')
    for i, occ in enumerate(top_woman_occs):
        print (' %d. %s' % (i+1, occ))

    # sanity check they answered written - this is just a heuristic
    written_ans = part4_written()
    if 'TODO' in written_ans:
        print ('Part 4 written answer contains TODO, did you answer it?')
    print (' ')
    return top_man_occs, top_woman_occs


# Helper functions to load questions
def load_synonym_qs(filename):
    '''
    input line:
        word    c1,c2,c3,c4     answer

    returns list of tuples, each of the form:
        (word, [c1, c2, c3, c4], answer)
    '''
    synonym_qs = []
    with open(filename) as f:
        f.readline()    # skip header
        for line in f:
            word, choices_str, ans = line.strip().split('\t')
            choices = [c.strip() for c in choices_str.split(',')]
            synonym_qs.append((word.strip(), choices, ans.strip()))
    return synonym_qs

def load_analogy_qs(filename):
    '''
    input line:
        a,b,aa,bb   c1,c2,c3,c4

    returns list of tuples, each of the form:
        (a, b, aa, bb)  // for analogy a:b --> aa:bb
    '''
    analogy_qs = []
    with open(filename) as f:
        f.readline()    # skip header
        for line in f:
            toks, choices_str = line.strip().split('\t')
            analogy_words = tuple(toks.strip().split(','))          # (a, b, aa, bb)
            choices = [c.strip() for c in choices_str.split(',')]   # [c1, c2, c3, c4]
            analogy_qs.append((analogy_words, choices))
    return analogy_qs

def load_sentence_sim_qs(filename):
    '''
    input line:
        label   s1  s2
    
    returns list of tuples, each of the form:
        (label, s1, s2)
    '''
    samples = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            label_str, s1, s2 = line.split('\t')
            label = int(label_str)
            samples.append((label, s1.strip(), s2.strip()))
    return samples

def load_pos_weights_map():
    '''
    Helper that loads the POS tag weights for part 3
    '''
    d = {}
    with open("data/pos_weights.txt") as f:
        for line in f:
           pos, weight = line.split()
           d[pos] = float(weight)
    return d

def load_occupations_list():
    '''
    Helper that loads the list of occupations for part 4
    '''
    occupations = []
    with open("data/occupations.txt") as f:
        for line in f:
            occupations.append(line.strip())
    return occupations

def main():
    (options, args) = getopt.getopt(sys.argv[1: ], '1234p')

    # load embeddings
    embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove50_4k.txt", binary=False)

    # load questions
    root_dir = 'data/dev/'
    synonym_qs = load_synonym_qs(root_dir + 'synonyms.csv')
    analogy_qs = load_analogy_qs(root_dir + 'analogies.csv')
    sentence_sim_qs = load_sentence_sim_qs(root_dir + 'sentences.csv')

    # if user specifies p, we'll print out the quiz questions
    PRINT_Q = False
    if ('-p', '') in options:
        PRINT_Q = True

    # if user specifies section (1-4), only run that section
    if ('-1', '') in options:
        part1(embeddings, synonym_qs, PRINT_Q)

    elif ('-2', '') in options:
        part2(embeddings, analogy_qs, PRINT_Q)

    elif ('-3', '') in options:
        part3(embeddings, sentence_sim_qs, PRINT_Q)

    elif ('-4', '') in options:
        part4(embeddings)

    # otherwise, run all 4 sections
    else:
        part1(embeddings, synonym_qs, PRINT_Q)
        part2(embeddings, analogy_qs, PRINT_Q)
        part3(embeddings, sentence_sim_qs, PRINT_Q)
        part4(embeddings)

if __name__ == "__main__":
        main()
