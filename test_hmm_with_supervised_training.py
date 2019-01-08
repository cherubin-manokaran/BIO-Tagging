from corpus import Document, NPChunkCorpus, NPChunkUnlabeledCorpus
from hmm import HMM
from evaluator import compute_cm as ConfusionMatrix
from unittest import TestCase, main
from random import shuffle, seed
import sys

def accuracy(classifier, test, verbose=sys.stderr):
    correct = [pred == label for x in test for pred,label in zip(classifier.classify(x),x.label)]
    if verbose:
        #print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
        print("%.2d%% " % ((100 * sum(correct) / len(correct)) if len(correct) != 0 else 0))
    return (float(sum(correct)) / len(correct)) if len(correct) != 0 else 0

class WordsAndPOS(Document):
    def features(self):
        return self.data

class Words(Document):
    def features(self):
        return [w for w, pos in self.data]

class POS(Document):
    def features(self):
        return[pos for w, pos in self.data]

class HMMTest(TestCase):
    u"""Tests for the HMM sequence labeler."""

    def split_np_chunk_corpus(self, document_class):
        """Split the yelp review corpus into training, dev, and test sets"""
        sentences = NPChunkCorpus('np_chunking_wsj', document_class=document_class)
        seed(1)
        shuffle(sentences)
        return (sentences[:8936], sentences[8936:])

    def test_np_chunk_words_and_pos(self):
        """predicting sequences using baseline feature"""
        train, test = self.split_np_chunk_corpus(WordsAndPOS)
        classifier = HMM()
        classifier.train(train)
        results = ConfusionMatrix(classifier, test)
        _, _, _, _ = results.print_out()
        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_np_chunk_words(self):
        """predicting sequences using baseline feature"""
        train, test = self.split_np_chunk_corpus(Words)
        classifier = HMM()
        classifier.train(train)
        results = ConfusionMatrix(classifier, test)
        _, _, _, _ = results.print_out()
        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_np_chunk_pos(self):
        """predicting sequences using baseline feature"""
        train, test = self.split_np_chunk_corpus(POS)
        classifier = HMM()
        classifier.train(train)
        results = ConfusionMatrix(classifier, test)
        _, _, _, _ = results.print_out()
        self.assertGreater(accuracy(classifier, test), 0.55)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)


