import numpy as np

class OneHotEncoder:
    def __init__(self, sentences):
        self.sentences = sentences
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.build_vocab()

    def _preprocess(self, text):        
        return text.lower().strip().split()

    def build_vocab(self):
        idx = len(self.vocab)
        for sentence in self.sentences:
            tokens = self._preprocess(sentence)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = idx 
                    self.idx_to_word[idx] = token
                    idx += 1
        
        print("\nVocabulary:")
        for word, i in self.vocab.items():
            print(f"{word}: {i}")

    def get_one_hot_vector(self, word):
        vector = np.zeros(len(self.vocab), dtype=int)
        index = self.vocab.get(word.lower(), self.vocab["<UNK>"])
        vector[index] = 1
        return vector

    def encode_sentence(self, sentence, max_len=None):
        tokens = self._preprocess(sentence)
        vectors = []
        
        if max_len:
            tokens = tokens[:max_len]
            while len(tokens) < max_len:
                tokens.append("<PAD>")

        for token in tokens:
            vectors.append(self.get_one_hot_vector(token))
            
        return np.array(vectors)


if __name__ == "__main__":
    sentences = [
        "I love NLP",
        "NLP is fun"
    ]

    encoder = OneHotEncoder(sentences)

    print("\n[Step 4] Sentence Encoding (with Padding):")
    test_sentence = "I love fun"

    encoded_matrix = encoder.encode_sentence(test_sentence, max_len=5)
    
    print(f"  Input: '{test_sentence}' (Padded to length 5)")
    print(f"  Matrix Shape: {encoded_matrix.shape} (Words x Vocab Size)")
    print("  One-hot Matrix:")
    for row in encoded_matrix:
        print(f"    {row.tolist()}")
