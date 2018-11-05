#! /usr/bin/python
# -*- coding: utf-8 -*-


"""Rank sentences based on cosine similarity and a query."""


from argparse import ArgumentParser
import numpy as np


def get_sentences(file_path):
    """Return a list of sentences from a file."""
    with open(file_path, encoding='utf-8') as hfile:
        return hfile.read().splitlines()


def get_top_k_words(sentences, k):
    """Return the k most frequent words as a list."""
    word_frequencies = {} # define frequency dictionary
    words = [word.lower() for s in sentences for word in s.split()] # get words and make them lower case
    for word in words: # count each word occurence
        if word_frequencies.get(word):
            word_frequencies[word] += 1
        else:
            word_frequencies[word] = 1
    top_k = sorted(word_frequencies.items(), key=lambda f: f[1], reverse=True)[:k] # sort words by their frequencies and take k most frequent of them
    return [top[0] for top in top_k] # return just words


def encode(sentence, vocabulary):
    """Return a vector encoding the sentence."""
    vector = np.zeros(len(vocabulary)) # define null vector
    words = [word.lower() for word in sentence.split()] # make every word lowercase
    for w in words: # match each word and its corresponding dimention in vector
        for i, v in enumerate(vocabulary):
            if v == w:
                vector[i] += 1
    return vector

def norm(v):
    """Return vector norm"""
    temp = 0
    for i in v:
        temp += i ** 2
    return temp ** (1/2)

def cosine_sim(u, v):
    """Return the cosine similarity of u and v."""
    norm_u = norm(u) # calculate norm of vector u
    norm_v = norm(v) # calculate norm of vector v
    if not norm_u or not norm_v: # if one of the vectors is null vector - return 0
        return 0
    dot = 0
    for i, val in enumerate(u): # calculate dot product of each element
        dot += val * v[i]
    sim = dot / (norm_u * norm_v) # divide dot product by norm multiplication
    return int(sim * 10000) / 10000 # fix floats and round them

def get_top_l_sentences(sentences, query, vocabulary, l):
    """
    For every sentence in "sentences", calculate the similarity to the query.
    Sort the sentences by their similarities to the query.

    Return the top-l most similar sentences as a list of tuples of the form
    (similarity, sentence).
    """
    encoded_query = encode(query, vocabulary) # encode query sentence
    similarities = [(cosine_sim(encode(s, vocabulary), encoded_query), s) for s in sentences] # calculate cosine similarity between each sentence and given query
    return sorted(similarities, key=lambda s: s[0], reverse=True)[:l] # sort the most similar sentences and get tom l of them


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('INPUT_FILE', help='An input file containing sentences, one per line')
    arg_parser.add_argument('QUERY', help='The query sentence')
    arg_parser.add_argument('-k', type=int, default=1000,
                            help='How many of the most frequent words to consider')
    arg_parser.add_argument('-l', type=int, default=10, help='How many sentences to return')
    args = arg_parser.parse_args()
    
    sentences = get_sentences(args.INPUT_FILE)
    top_k_words = get_top_k_words(sentences, args.k)
    query = args.QUERY.lower()

    print('using vocabulary: {}\n'.format(top_k_words))
    print('using query: {}\n'.format(query))

    # suppress numpy's "divide by 0" warning.
    # this is fine since we consider a zero-vector to be dissimilar to other vectors
    with np.errstate(invalid='ignore'):
        result = get_top_l_sentences(sentences, query, top_k_words, args.l)

    print('result:')
    for sim, sentence in result:
        print('{:.5f}\t{}'.format(sim, sentence))


if __name__ == '__main__':
    main()
