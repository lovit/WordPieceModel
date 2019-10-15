from collections import Counter
from collections import defaultdict

class BytePairEncoder:

    def __init__(self, sents_or_vocab2count, n_iters=10, verbose=True, method='origin'):
        self.n_iters = n_iters if n_iters > 0 else 10
        self.units = {}
        self.max_length = 0
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

        self.units, self.max_length = train(vocab2count, self.n_iters, self.verbose)

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
    for i in range(n_iters + 1):
        pairs = get_stats(vocabs)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocabs = merge_vocab(best, vocabs)
        if verbose and i % 100 == 99:
            print('\rtraining bpe {} / {}'.format(i+1, n_iters), end='', flush=True)
    if verbose:
        print('\rtraining bpe was done{}'.format(' '*40))

    units = {}
    for word, freq in vocabs.items():
        for unit in word.split():
            units[unit] = units.get(unit, 0) + freq
    max_length = max((len(w) for w in units))
    return units, max_length