"""
Simple Tokenizer for text processing
"""
from collections import Counter


class SimpleTokenizer:
    """Simple tokenizer for text processing"""
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.word_to_index = {}
        self.index_to_word = {}

    def fit_on_texts(self, texts):
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        # Create vocabulary with most common words
        most_common = word_counts.most_common(self.max_features - 1)

        # Reserve index 0 for padding
        self.word_to_index = {'<PAD>': 0}
        self.index_to_word = {0: '<PAD>'}

        for i, (word, _) in enumerate(most_common, 1):
            self.word_to_index[word] = i
            self.index_to_word[i] = word

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = text.lower().split()
            sequence = [self.word_to_index.get(word, 0) for word in words]
            sequences.append(sequence)
        return sequences
