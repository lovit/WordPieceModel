class BytePairEncoder:
    
    def __init__(self, num_units=10):
        self.num_units = num_units if num_units > 0 else 10
        self.units = {}
        self.max_length = 0
        
    def train(self, sents):
        def to_subwords(s):
            s = s.replace('_', '') + '_'
            n = len(s)
            return [s[b:b+r] for b in range(n) for r in range(1, n+1) if b+r <= n]

        def counting(sents):
            from collections import Counter
            return Counter((subword for sent in sents for eojeol in sent.split() for subword in to_subwords(eojeol) if eojeol))

        counter = counting(sents)
        a_syllables = {subword:freq for subword, freq in counter.items() if len(subword) == 1}
        self.units = dict(
            sorted(
                filter(lambda x:len(x[0]) > 1, counter.items()), 
                key=lambda x:(-x[1], -len(x[0]), x[0]))
            [:max(0, self.num_units - len(a_syllables))]
        )
        self.units.update(a_syllables)
        self.max_length = max((len(w) for w in self.units))
    
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
            f.write('num_units={}\n'.format(self.num_units))
            f.write('max_length={}\n'.format(self.max_length))
            for unit, frequency in sorted(self.units.items(), key=lambda x:(-x[1], -len(x[0]))):
                f.write('{}\t{}\n'.format(unit, frequency))
                
    def load(self, fname):
        with open(fname, encoding='utf-8') as f:
            try:
                self.num_units = int(next(f).strip().split('=')[1])
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