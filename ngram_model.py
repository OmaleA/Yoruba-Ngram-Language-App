class YorubaNgram(object):
    """A language model that uses n-grams to make probabilistic predictions.

    The model is built by:
    1. Preprocessing the input data by adding special tokens for start, end, and unknown words.
    2. Calculating probabilities for each n-gram with the option to apply smoothing techniques.

    This class provides functions to assess perplexity on a dataset and to generate text based on the model.

    Attributes:
        training_sentences (list of str): Sentences used to train the model.
        gram_order (int): The order of n-grams (e.g., 1 for unigram).
        smoothing_factor (int): The multiplier for Laplace smoothing (default is 1).

    """


    def __init__(self, train_data=None, n=None, laplace=1):
        self.n = n
        self.laplace = laplace
        if train_data is not None:
          self.tokens = preprocess(train_data, n)
          self.vocab  = nltk.FreqDist(self.tokens)
          self.model  = self._create_ngram()
          self.masks  = list(reversed(list(product((0,1), repeat=n))))

        else:
          self.model = {}
          self.vocab = {}
          self.masks = []

    def _laplace_smooth(self):
        """Smooth the frequency distribution of n-grams using Laplace's method.

        Returns:
            dict: Each n-gram mapped to its smoothed probability.

        """
        vocab_size = len(self.vocab)

        n_grams = nltk.ngrams(self.tokens, self.n)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens, self.n-1)
        m_vocab = nltk.FreqDist(m_grams)

        def smoothed_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return (n_count + self.laplace) / (m_count + self.laplace * vocab_size)

        return { n_gram: smoothed_count(n_gram, count) for n_gram, count in n_vocab.items() }

    def _create_ngram(self):
        """Establish a probability distribution for the training data vocabulary.

        Returns:
            A dictionary associating each n-gram with its probability.

        """
        if self.n == 1:
            num_tokens = len(self.tokens)
            return { (unigram,): count / num_tokens for unigram, count in self.vocab.items() }
        else:
            return self._laplace_smooth()

    def _adjust_unknown(self, ngram):
        """Adjust an n-gram to align with a known variant in the model if necessary.

        It cycles through permutations of the n-gram, replacing tokens with <UNK>
        based on a bitmask, until a recognized permutation is identified.

        Returns:
            The adjusted n-gram with <UNK> inserted where the model lacks information.

        """

        mask = lambda ngram, bitmask: tuple((token if flag == 1 else "<UNK>" for token,flag in zip(ngram, bitmask)))

        ngram = (ngram,) if type(ngram) is str else ngram
        for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
            if possible_known in self.model:
                return possible_known

    def perplexity(self, test_data):
        """Calculate the perplexity of the language model on a test dataset.

        Parameters:
            evaluation_data (list of str): Sentences extracted from the test dataset.

        Returns:
            float: The computed perplexity score for the model.

        """

        test_tokens = preprocess(test_data, self.n)
        test_ngrams = nltk.ngrams(test_tokens, self.n)
        N = len(test_tokens)

        known_ngrams  = (self._adjust_unknown(ngram) for ngram in test_ngrams)
        probabilities = [self.model[ngram] for ngram in known_ngrams]

        return math.exp((-1/N) * sum(map(math.log, probabilities)))

    def _high_probability(self, prev, i, without=[]):
        """Determine the most probable subsequent token based on prior context.

        Selects the next token to maximize sentence probability, avoiding specified exclusions.
        If no valid candidates are available, the end token is returned with certainty.

        Parameters:
            preceding_tokens (tuple of str): The context tokens preceding the new token.
            selection_rank (int): Rank of the candidate to select for variety.
            exclude_tokens (list of str): Tokens to omit from consideration.

        Returns:
            tuple: The chosen token and its probability.

        """
        blacklist  = ["<UNK>"] + without
        candidates = ((ngram[-1],prob) for ngram,prob in self.model.items() if ngram[:-1]==prev)
        candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return ("</s>", 1)
        else:
            return candidates[0 if prev != () and prev[-1] != "<s>" else i]

    def generate_sentences(self, num, min_len=12, max_len=24):
        """Generate a specified number of sentences using the model.

        Sentences are constructed by iteratively selecting the most probable next token,
        starting and ending with predefined tokens, and avoiding repetitions.

        Parameters:
            sentence_count (int): The quantity of sentences to create.
            shortest_length (int): The minimum sentence length.
            longest_length (int): The maximum sentence length.

        Yields:
            tuple: The constructed sentence and its probability in log-space.

        """
        for i in range(num):
            sent, prob = ["<s>"] * (self.n - 1), 1
            while not sent[-1].endswith("</s>"):
                prev = () if self.n == 1 else tuple(sent[-(self.n-1):])
                blacklist = sent + (["</s>"] if len(sent) < min_len else [])
                next_token, next_prob = self._high_probability(prev, i, without=blacklist)
                sent.append(next_token)
                prob *= next_prob

                if len(sent) >= max_len + self.n - 1:
                    sent.append("</s>")

            yield ' '.join(sent[self.n-1:]), -1/math.log(prob)

    def save_to_file(self, file_path):
        """Save the model to a file using pickle.

        Args:
            file_path (str): The path to save the model.

        Returns:
            None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {file_path}")

    def load_from_file(cls, file_path):
        """Load the model from a file using pickle.

        Args:
            file_path (str): The path to load the model from.

        Returns:
            LanguageModel: The loaded model.
        """
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_path}")
        return model

    def Predict(self, inputPredict):
        """Predict the next word and provide top 5 high probability suggestions, excluding <UNK>.

        Parameters:
          inputPredict (str): The input word or sequence of words to predict from.

        Returns:
          list: A list of the top 5 high probability suggestions, excluding <UNK>.
        """
        input_tokens = inputPredict.split()
        if len(input_tokens) < self.n - 1:
            return "Input must have at least {} words for the {}-gram model.".format(self.n - 1, self.n)

        # Adjust for the n-gram model
        prev = tuple(input_tokens[-(self.n - 1):]) if self.n > 1 else ()

        # Get candidates for the next word, excluding <UNK>
        candidates = [(ngram[-1], prob) for ngram, prob in self.model.items() if ngram[:-1] == prev and ngram[-1] != "<UNK>"]

        # Sort candidates by probability
        sorted_candidates = sorted(candidates, key=lambda item: item[1], reverse=True)

        # Return top 5 predictions, excluding <UNK>
        return sorted_candidates[:10]

