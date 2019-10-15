from collections import Counter
from collections import defaultdict

class BytePairEncoder:

    """
    Arguments
    ---------
    sents_or_vocab2count : list of str or {str:int}
        If the input is sentences, the form of input should be list of str (or like)
        If the input is count of vocabulary, the form of input should be {vocab:count}
    num_merge : int
        The number of units.
    verbose : Boolean
        If True, it shows progress
    method : str
        Choice of ['origin', 'fast']
    """
    def __init__(self, sents_or_vocab2count, num_merge=10, verbose=True, method='origin'):

        if num_merge <= 0:
            raise ValueError('num_merge should be positive integer')

        self.num_merge = num_merge
        self.units = {}
        self.max_length = -1
        self.verbose = verbose
        self.method = method

        self._train(sents_or_vocab2count)

    def _train(self, sents_or_vocab2count):
        if self.verbose:
            print('begin vocabulary scanning', end='', flush=True)

        if hasattr(sents_or_vocab2count, 'get'):
            vocab2count = sents_or_vocab2count
        else:
            vocab2count = sent_to_vocabs(sents_or_vocab2count)

        if self.verbose:
            print('\rterminated vocabulary scanning', flush=True)

        if self.method == 'origin':
            self.units, self.max_length = train(vocab2count, self.num_merge, self.verbose)
        else:
            self.units, self.max_length = train_fast(vocab2count, self.num_merge, self.verbose)

    def tokenize(self, s):
        return ' '.join([self._tokenize(w) for w in s.split()])

    def _tokenize(self, w):
        def initialize(w):
            w += '_'
            subwords = []
            n = len(w)
            for b in range(n):
                for e in range(b+1, min(n, b+self.max_length)+1):
                    subword = w[b:e]
                    if not subword in self.units:
                        continue
                    subwords.append((subword, b, e, e-b))
            return subwords

        def longest_match(subwords):
            matched = []
            subwords = sorted(subwords, key=lambda x:(-x[3], x[1]))
            while subwords:
                s, b, e, l = subwords.pop(0) # str, begin, end, length
                matched.append((s, b, e, l))
                removals = []
                for i, (_, b_, e_, _) in enumerate(subwords):
                    if (b_ < e and b < e_) or (b_ < e and e_ > b):
                        removals.append(i)
                for i in reversed(removals):
                    del subwords[i]
            return sorted(matched, key=lambda x:x[1])

        subwords = initialize(w)
        subwords = longest_match(subwords)
        subwords = ' '.join([s for s, _, _, _ in subwords])
        return subwords

    def save(self, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('n_iters={}\n'.format(self.n_iters))
            f.write('max_length={}\n'.format(self.max_length))
            for unit, frequency in sorted(self.units.items(), key=lambda x:(-x[1], -len(x[0]))):
                f.write('{}\t{}\n'.format(unit, frequency))

    def load(self, fname):
        with open(fname, encoding='utf-8') as f:
            try:
                self.n_iters = int(next(f).strip().split('=')[1])
                self.max_length = int(next(f).strip().split('=')[1])
            except Exception as e:
                print(e)

            self.units = {}
            for row in f:
                try:
                    unit, frequency = row.strip().split('\t')
                    self.units[unit] = int(frequency)
                except Exception as e:
                    print('BPE load exception: {}'.format(str(e)))
                    break

def sent_to_vocabs(sents):
    return Counter((eojeol.replace('_', '') for sent in sents for eojeol in sent.split() if eojeol))

def train(vocabs, n_iters, verbose):
    def get_stats(vocabs):
        pairs = defaultdict(int)
        for word, freq in vocabs.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i],symbols[i+1])] += freq
        return pairs

    def merge_vocab(pair, v_in):
        v_out = {}
        bigram = ' '.join(pair)
        replacer = ''.join(pair)
        for word, freq in v_in.items():
            w_out = word.replace(bigram, replacer)
            v_out[w_out] = freq
        return v_out

    vocabs = {' '.join(w)+' _':c for w, c in vocabs.items()}
    for i in range(n_iters):
        pairs = get_stats(vocabs)
        if not pairs:
            break
        best, frequency = sorted(pairs.items(), key=lambda x:(-x[1], x[0]))[0]
        vocabs = merge_vocab(best, vocabs)
        if verbose:
            print('merged {} / {} : {}'.format(i+1, n_iters, best))

    units = {}
    for word, freq in vocabs.items():
        for unit in word.split():
            units[unit] = units.get(unit, 0) + freq
    max_length = max((len(w) for w in units))
    return units, max_length

def merge(vocab, bi):
    before = ' '.join(bi)
    after = ''.join(bi)
    return vocab.replace(before, after)

def to_bi(word):
    word = word.split()
    n = len(word)
    return [tuple(word[i:i+2]) for i in range(n-1)]

def indexing(vocab2count):
    def initialize(vocab2count):
        return {' '.join(vocab)+' _':count for vocab, count in vocab2count.items()}

    uni2vocab = defaultdict(lambda: set())
    bi2vocab = defaultdict(lambda: set())
    bi2count = defaultdict(int)
    vocab2count = initialize(vocab2count)
    for vocab, count in vocab2count.items():
        for bi in to_bi(vocab):
            bi2vocab[bi].add(vocab)
            bi2count[bi] += count
        for uni in vocab.split():
            uni2vocab[uni].add(vocab)
    sum_count = lambda vocabs: sum(vocab2count[v] for v in vocabs)
    return uni2vocab, bi2vocab, bi2count, vocab2count

def unitify(vocab2count):
    units = defaultdict(int)
    for vocab, count in vocab2count.items():
        for unit in vocab.split():
            units[unit] += count
    return dict(units)

def train_fast(vocab2count, num_merge, verbose):
    uni2vocab, bi2vocab, bi2count, vocab2count = indexing(vocab2count)

    for i in range(num_merge):
        bi_merge, _ = sorted(bi2count.items(), key=lambda x:(-x[1], x[0]))[0]
        if verbose:
            print('merged {} / {} : {}'.format(i+1, num_merge, bi_merge))

        # find removals and updates
        removal = []
        update = []
        for vocab_before in bi2vocab[bi_merge]:
            vocab_after = merge(vocab_before, bi_merge)
            count = vocab2count.pop(vocab_before)
            vocab2count[vocab_after] = count
            for old_bi in to_bi(vocab_before):
                removal.append((old_bi, vocab_before, count))
            for new_bi in to_bi(vocab_after):
                update.append((new_bi, vocab_after, count))

        # remove and decrease count
        for old_bi, vocab_before, count in removal:
            bi2count[old_bi] -= count
            bi2vocab[old_bi].discard(vocab_before)

        # update and increase count
        for new_bi, vocab_after, count in update:
            bi2count[new_bi] += count
            bi2vocab[new_bi].add(vocab_after)

        # remove useless values
        for old_bi, _, _ in removal:
            if bi2count[old_bi] == 0:
                bi2count.pop(old_bi)
            if not bi2vocab[old_bi]:
                bi2vocab.pop(old_bi)

    units = unitify(vocab2count)
    max_length = max(len(unit) for unit in units)
    return units, max_length
