"""Lab 03 - Chaotic Text Generator: n-gram sampling vs a supervised POS-HMM (no UI code here).

HMM = hidden states (part-of-speech tags) + transition probabilities P(tag_t | tag_{t-1})
      + emission probabilities P(word_t | tag_t).
Generation: sample a hidden tag sequence first, then sample one word per tag.
Tagging  : Viterbi finds the most probable tag sequence for a given sentence.

Enhancements over the provided version (same public API, same numbers for a given seed):
  * Sampler          - pre-sorted keys and cached cumulative weights per temperature. rng.choices is
                       called with the exact same cumulative weights, so a seed gives the same text,
                       but a batch of 500 sentences is several times faster.
  * Log tables       - Viterbi reads pre-computed log-probabilities instead of recomputing them.
  * viterbi_detail   - tags + hidden states + out-of-vocabulary flags + per-word confidence
                       (forward-backward posterior) + log-probability of the best path.
  * Suffix guessing  - optional model for unknown words: P(tag | last letters) learned from rare
                       training words (guide, section 5).  Off by default so the guide's numbers hold.
  * SentenceJudge    - per-sentence verdicts (valid pattern, plausible subject-verb, novel/copied),
                       shared by the metrics and by the application table.
  * generate_batch   - one entry point used by the app and the evaluation for all three models.
"""
import csv
import math
import random
import re
from collections import Counter, defaultdict
from itertools import accumulate

BOS, EOS = "<s>", "</s>"
MODEL_NAMES = {"hmm": "HMM, plain states", "hmm_split": "HMM, role-split states", "ngram": "Trigram",
               "hmm2": "HMM, second order"}
# reading order of the states, used by the graphs (grammar order instead of alphabetical)
PLAIN_ORDER = ["DET", "ADJ", "NOUN", "VERB", "PREP", "PUNCT"]
SPLIT_ORDER = ["DET_S", "ADJ_S", "NOUN_S", "VERB", "DET_O", "ADJ_O", "NOUN_O", "PREP",
               "DET_P", "ADJ_P", "NOUN_P", "PUNCT"]
FAILURES = {"structure": "invalid tag pattern", "meaning": "valid pattern, odd subject-verb pair",
            "memorisation": "copied from the training set", "none": "no failure"}


def reshape(counter, temperature):
    """{key: probability} after the temperature reshaping count ** (1/T)."""
    w = {k: c ** (1.0 / temperature) for k, c in counter.items() if c > 0}
    z = sum(w.values()) or 1.0
    return {k: v / z for k, v in w.items()}


def ordered(states, order):
    return [s for s in order if s in states] + sorted(s for s in states if s not in order)


# ------------------------------------------------------------------ data
def load_tagged(path="data/corpus.csv"):
    """Return {doc_id: [(words, tags), ...]}."""
    docs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            words, tags = row["sentence"].split(), row["tags"].split()
            assert len(words) == len(tags), "word/tag length mismatch"
            docs[int(row["doc_id"])].append((words, tags))
    return dict(docs)


def split_by_document(docs, test_fraction=0.2):
    ids = sorted(docs)
    cut = int(len(ids) * (1 - test_fraction))
    return ([x for i in ids[:cut] for x in docs[i]], [x for i in ids[cut:] for x in docs[i]])


_TOKEN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*|[^\sa-z0-9]")


def tokenize(text):
    """Lower-case and split off punctuation: 'The cats sleep.' -> ['the', 'cats', 'sleep', '.']"""
    return _TOKEN.findall((text or "").lower())


# ------------------------------------------------------------------ sampling helpers
class Sampler:
    """Weighted sampler over a fixed Counter.  Keys are sorted once (=> reproducible for a seed);
    cumulative weights c ** (1/T) are cached per temperature."""

    __slots__ = ("keys", "counts", "_cum")

    def __init__(self, counter):
        items = sorted(counter.items())
        self.keys = [k for k, _ in items]
        self.counts = [c for _, c in items]
        self._cum = {}

    def sample(self, rng, temperature):
        cum = self._cum.get(temperature)
        if cum is None:
            cum = self._cum[temperature] = list(accumulate(c ** (1.0 / temperature) for c in self.counts))
        return rng.choices(self.keys, cum_weights=cum, k=1)[0]


def sample_from(counter, temperature, rng):
    """Sample a key of `counter`; weights c ** (1/T). T < 1 -> safer, T > 1 -> more chaotic.
    Kept for backward compatibility; the models use cached Samplers (identical results)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return Sampler(counter).sample(rng, temperature)


def _logsumexp(xs):
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


# ------------------------------------------------------------------ state splitting
def refine(tags):
    """Split every state by its ROLE in the sentence: S = before the verb, O = after the verb,
    P = inside the prepositional phrase.  DET NOUN VERB DET NOUN PREP DET NOUN PUNCT becomes
    DET_S NOUN_S VERB DET_O NOUN_O PREP DET_P NOUN_P PUNCT.  A first-order HMM cannot know where it
    is in the sentence from the coarse tag alone; richer states give it that memory."""
    out, role = [], "S"
    for t in tags:
        if t == "VERB":
            out.append(t); role = "O"
        elif t == "PREP":
            out.append(t); role = "P"
        elif t == "PUNCT":
            out.append(t)
        else:
            out.append(f"{t}_{role}")
    return out


def coarse(states):
    """DET_S -> DET (back to the plain part-of-speech tag)."""
    return [s.split("_")[0] for s in states]


# ------------------------------------------------------------------ unknown words: suffix model
class SuffixGuesser:
    """P(tag | suffix) estimated on RARE training tokens (rare words behave like unseen ones).
    For an unknown word the emission score is P(tag | suffix) / P(tag)  (Bayes, up to a constant
    that is the same for every tag at that position and therefore does not change Viterbi)."""

    def __init__(self, tagged_sentences, max_suffix=3, rare_max=10, min_count=5, k=0.01):
        word_freq = Counter(w for ws, _ in tagged_sentences for w in ws)
        tokens = [(w, t) for ws, ts in tagged_sentences for w, t in zip(ws, ts)]
        rare = [(w, t) for w, t in tokens if word_freq[w] <= rare_max]
        if len(rare) < 50:                       # tiny synthetic vocabulary: no rare words, use all
            rare = tokens
        self.max_suffix, self.min_count, self.k = max_suffix, min_count, k
        self.tags = sorted({t for _, t in tokens})
        prior = Counter(t for _, t in tokens)
        self.log_prior = {t: math.log(prior[t] / len(tokens)) for t in self.tags}
        self.table = defaultdict(Counter)
        for w, t in rare:
            for key in self._keys(w):
                self.table[key][t] += 1

    def _keys(self, w):
        if any(ch.isdigit() for ch in w):
            yield "<num>"
        if not any(ch.isalnum() for ch in w):
            yield "<sym>"
        for n in range(1, self.max_suffix + 1):
            if len(w) > n:
                yield "-" + w[-n:]
        yield ""                                   # empty suffix = distribution of rare words

    def log_scores(self, word):
        """{tag: log P(tag|suffix) - log P(tag)} using the most specific key seen often enough."""
        keys = list(self._keys(word))
        specials = [k for k in keys if k.startswith("<")]
        suffixes = sorted((k for k in keys if k.startswith("-")), key=len, reverse=True)
        for key in specials + suffixes + [""]:
            c = self.table.get(key)
            if c and sum(c.values()) >= self.min_count:
                tot = sum(c.values()) + self.k * len(self.tags)
                return {t: math.log((c[t] + self.k) / tot) - self.log_prior[t] for t in self.tags}
        return {t: 0.0 for t in self.tags}


# ------------------------------------------------------------------ HMM
class HMM:
    def __init__(self, tagged_sentences, k=0.01, split_states=False, unknown="smooth"):
        """unknown = 'smooth' (add-k emission, as in the guide) or 'suffix' (SuffixGuesser)."""
        if unknown not in ("smooth", "suffix"):
            raise ValueError("unknown must be 'smooth' or 'suffix'")
        self.k, self.split_states, self.unknown = k, split_states, unknown
        self.start, self.trans, self.emit = Counter(), defaultdict(Counter), defaultdict(Counter)
        for words, tags in tagged_sentences:
            tags = refine(tags) if split_states else tags
            self.start[tags[0]] += 1
            for i, (w, t) in enumerate(zip(words, tags)):
                self.emit[t][w] += 1
                if i > 0:
                    self.trans[tags[i - 1]][t] += 1
        self.tags = sorted(self.emit)
        self.vocab = {w for c in self.emit.values() for w in c}
        self.V = len(self.vocab) + 1                      # +1 for the unknown-word slot
        self.tag_totals = {t: sum(c.values()) for t, c in self.emit.items()}

        # samplers (generation) and log tables (decoding), built once
        self._s_start = Sampler(self.start)
        self._s_trans = {t: Sampler(c) for t, c in self.trans.items() if c}
        self._s_emit = {t: Sampler(c) for t, c in self.emit.items()}
        self.log_start = {t: math.log(self.p_start(t)) for t in self.tags}
        self.log_trans = {a: {b: math.log(self.p_trans(a, b)) for b in self.tags} for a in self.tags}
        self._log_unk = {t: math.log(self.k / (self.tag_totals[t] + self.k * self.V)) for t in self.tags}
        self._log_emit = {t: {w: math.log(self.p_emit(t, w)) for w in self.vocab} for t in self.tags}
        self.suffix = None
        if unknown == "suffix":
            self.suffix = SuffixGuesser(tagged_sentences)          # coarse tags, before splitting
            self._coarse_of = {t: t.split("_")[0] for t in self.tags}

    # ---- generation
    def sample_with_states(self, rng, temperature=1.0, max_len=14):
        """Sample hidden states first, then one word per state.
        Returns (words, states); states are refined (DET_S ...) when split_states is on."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        tag = self._s_start.sample(rng, temperature)
        states = [tag]
        while tag != "PUNCT" and len(states) < max_len and tag in self._s_trans:
            tag = self._s_trans[tag].sample(rng, temperature)
            states.append(tag)
        return [self._s_emit[t].sample(rng, temperature) for t in states], states

    def sample_sentence(self, rng, temperature=1.0, max_len=14):
        """(words, coarse tags).  Uses the random stream exactly like sample_with_states."""
        words, states = self.sample_with_states(rng, temperature, max_len)
        return words, coarse(states)

    # ---- probabilities (smoothed) used by Viterbi
    def p_trans(self, a, b):
        c = self.trans.get(a, Counter())
        return (c[b] + self.k) / (sum(c.values()) + self.k * len(self.tags))

    def p_start(self, t):
        return (self.start[t] + self.k) / (sum(self.start.values()) + self.k * len(self.tags))

    def p_emit(self, t, w):
        return (self.emit[t][w] + self.k) / (self.tag_totals[t] + self.k * self.V)

    def log_emit(self, t, w):
        lp = self._log_emit[t].get(w)
        if lp is not None:
            return lp
        if self.suffix is not None:                       # unknown word, suffix model
            return self._log_unk[t] + self.suffix.log_scores(w).get(self._coarse_of[t], 0.0)
        return self._log_unk[t]

    # ---- decoding
    def _emissions(self, words):
        cache = {}
        out = []
        for w in words:
            if w not in cache:
                cache[w] = {t: self.log_emit(t, w) for t in self.tags}
            out.append(cache[w])
        return out

    def viterbi_states(self, words):
        """Most probable hidden-state sequence and its log-probability (log-space DP)."""
        if not words:
            return [], 0.0
        T, LT = self.tags, self.log_trans
        em = self._emissions(words)
        score = {t: self.log_start[t] + em[0][t] for t in T}
        back = []
        for e in em[1:]:
            col, bp = {}, {}
            for t in T:
                best_prev = max(T, key=lambda p: score[p] + LT[p][t])
                col[t] = score[best_prev] + LT[best_prev][t] + e[t]
                bp[t] = best_prev
            score = col; back.append(bp)
        last = max(T, key=lambda t: score[t])
        path = [last]
        for bp in reversed(back):
            path.append(bp[path[-1]])
        return path[::-1], score[last]

    def viterbi(self, words):
        """Most probable tag sequence; returns COARSE tags."""
        return coarse(self.viterbi_states(words)[0])

    def posteriors(self, words):
        """Forward-backward: list of {coarse tag: P(tag at position i | sentence)}."""
        if not words:
            return []
        T, LT = self.tags, self.log_trans
        em = self._emissions(words)
        fwd = [{t: self.log_start[t] + em[0][t] for t in T}]
        for e in em[1:]:
            prev = fwd[-1]
            fwd.append({t: _logsumexp([prev[p] + LT[p][t] for p in T]) + e[t] for t in T})
        bwd = [{t: 0.0 for t in T}]
        for e in reversed(em[1:]):
            nxt = bwd[0]
            bwd.insert(0, {t: _logsumexp([LT[t][n] + e[n] + nxt[n] for n in T]) for t in T})
        z = _logsumexp(list(fwd[-1].values()))
        out = []
        for f, b in zip(fwd, bwd):
            d = defaultdict(float)
            for t in T:
                d[t.split("_")[0]] += math.exp(f[t] + b[t] - z)
            out.append(dict(d))
        return out

    def viterbi_detail(self, words):
        """Everything the tagger UI needs, for any input (empty, unknown words...)."""
        states, logp = self.viterbi_states(words)
        tags = coarse(states)
        post = self.posteriors(words)
        return {"words": list(words), "tags": tags, "states": states, "logprob": logp,
                "oov": [w not in self.vocab for w in words],
                "confidence": [p.get(t, 0.0) for p, t in zip(post, tags)],
                "posteriors": post}

    def transition_matrix(self):
        """Row-normalised transition counts (for the heat-map)."""
        rows = []
        for a in self.tags:
            tot = sum(self.trans[a].values()) or 1
            rows.append([self.trans[a][b] / tot for b in self.tags])
        return self.tags, rows

    def row_order(self):
        return ordered(self.tags, SPLIT_ORDER if self.split_states else PLAIN_ORDER)

    def step_distributions(self, words, states, temperature=1.0):
        """For each position: the distribution the next state was drawn from (after temperature),
        and the probability of the emitted word.  Used to draw the state-path graph."""
        steps = []
        for i, (w, s) in enumerate(zip(words, states)):
            branches = reshape(self.start if i == 0 else self.trans.get(states[i - 1], Counter()), temperature)
            emit = reshape(self.emit.get(s, Counter()), temperature).get(w, 0.0)
            steps.append({"word": w, "state": s, "branches": branches, "p_emit": emit})
        return steps


# ------------------------------------------------------------------ extension: second-order HMM
class SecondOrderHMM:
    """Extension (guide, section 5): the next tag depends on the TWO previous tags,
    P(tag_t | tag_{t-2}, tag_{t-1}).  Emissions are the plain HMM's.  An unseen pair context falls
    back to the first-order transition.  Compare it with role-split states: a pair such as
    (DET, NOUN) still does not say whether the noun is the subject or the object."""

    split_states = False

    def __init__(self, tagged_sentences, k=0.01):
        self.k = k
        self.first = HMM(tagged_sentences, k=k)
        self.tags, self.vocab = self.first.tags, self.first.vocab
        self.trans2 = defaultdict(Counter)
        for _, tags in tagged_sentences:
            p = [BOS, BOS] + tags
            for i in range(2, len(p)):
                self.trans2[(p[i - 2], p[i - 1])][p[i]] += 1
        self._s2 = {ctx: Sampler(c) for ctx, c in self.trans2.items()}
        n = len(self.tags)
        self._log2 = {ctx: {t: math.log((c[t] + k) / (sum(c.values()) + k * n)) for t in self.tags}
                      for ctx, c in self.trans2.items()}

    def row_order(self):
        return ordered(self.tags, PLAIN_ORDER)

    def sample_with_states(self, rng, temperature=1.0, max_len=14):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        ctx, states = (BOS, BOS), []
        while len(states) < max_len:
            s = self._s2.get(ctx)
            if s is None:                                   # unseen pair: first-order back-off
                s = self.first._s_trans.get(states[-1]) if states else None
                if s is None:
                    break
            tag = s.sample(rng, temperature)
            states.append(tag)
            ctx = (ctx[1], tag)
            if tag == "PUNCT":
                break
        return [self.first._s_emit[t].sample(rng, temperature) for t in states], states

    def sample_sentence(self, rng, temperature=1.0, max_len=14):
        return self.sample_with_states(rng, temperature, max_len)

    def log_trans2(self, a, b, c):
        row = self._log2.get((a, b))
        if row is not None:
            return row[c]
        return self.first.log_trans[b][c] if b in self.first.log_trans else self.first.log_start[c]

    def viterbi_states(self, words):
        """Viterbi over pair states (previous tag, tag)."""
        if not words:
            return [], 0.0
        T = self.tags
        em = self.first._emissions(words)
        score = {(BOS, t): self.log_trans2(BOS, BOS, t) + em[0][t] for t in T}
        back = []
        for e in em[1:]:
            col, bp = {}, {}
            prevs = list(score)
            for b in T:
                cands = [p for p in prevs if p[1] == b]
                if not cands:
                    continue
                for c in T:
                    best = max(cands, key=lambda p: score[p] + self.log_trans2(p[0], b, c))
                    col[(b, c)] = score[best] + self.log_trans2(best[0], b, c) + e[c]
                    bp[(b, c)] = best
            score = col; back.append(bp)
        last = max(score, key=score.get)
        path = [last]
        for bp in reversed(back):
            path.append(bp[path[-1]])
        return [p[1] for p in reversed(path)], score[last]

    def viterbi(self, words):
        return self.viterbi_states(words)[0]

    def posteriors(self, words):
        """Forward-backward over pair states; returns {tag: P(tag at i | sentence)} per position."""
        if not words:
            return []
        T = self.tags
        em = self.first._emissions(words)
        fwd = [{(BOS, t): self.log_trans2(BOS, BOS, t) + em[0][t] for t in T}]
        for e in em[1:]:
            prev, col = fwd[-1], {}
            for b in T:
                cands = [p for p in prev if p[1] == b]
                for c in T:
                    col[(b, c)] = _logsumexp([prev[p] + self.log_trans2(p[0], b, c) for p in cands]) + e[c]
            fwd.append(col)
        bwd = [{k: 0.0 for k in fwd[-1]}]
        for i in range(len(words) - 1, 0, -1):
            nxt, e, col = bwd[0], em[i], {}
            for (a, b) in fwd[i - 1]:
                col[(a, b)] = _logsumexp([self.log_trans2(a, b, c) + e[c] + nxt[(b, c)] for c in T])
            bwd.insert(0, col)
        z = _logsumexp(list(fwd[-1].values()))
        out = []
        for f, b in zip(fwd, bwd):
            d = defaultdict(float)
            for key in f:
                d[key[1]] += math.exp(f[key] + b[key] - z)
            out.append(dict(d))
        return out

    def viterbi_detail(self, words):
        tags, logp = self.viterbi_states(words)
        post = self.posteriors(words)
        return {"words": list(words), "tags": tags, "states": tags, "logprob": logp,
                "oov": [w not in self.vocab for w in words],
                "confidence": [p.get(t, 0.0) for p, t in zip(post, tags)], "posteriors": post}

    def step_distributions(self, words, states, temperature=1.0):
        steps, ctx = [], (BOS, BOS)
        for w, s in zip(words, states):
            c = self.trans2.get(ctx) or self.first.trans.get(ctx[1], Counter())
            steps.append({"word": w, "state": s, "branches": reshape(c, temperature),
                          "p_emit": reshape(self.first.emit.get(s, Counter()), temperature).get(w, 0.0)})
            ctx = (ctx[1], s)
        return steps


# ------------------------------------------------------------------ n-gram generator (re-uses Lab 02 ideas)
class NGramGenerator:
    """Trigram counts with fall-back to bigram / unigram when a context is unseen."""

    def __init__(self, sentences):
        self.tri, self.bi, self.uni = defaultdict(Counter), defaultdict(Counter), Counter()
        for s in sentences:
            p = [BOS, BOS] + s + [EOS]
            for i in range(2, len(p)):
                self.tri[(p[i - 2], p[i - 1])][p[i]] += 1
                self.bi[(p[i - 1],)][p[i]] += 1
                self.uni[p[i]] += 1
        self.vocab = {w for s in sentences for w in s}
        self._s_tri = {k: Sampler(c) for k, c in self.tri.items()}
        self._s_bi = {k: Sampler(c) for k, c in self.bi.items()}
        self._s_uni = Sampler(self.uni)

    def next_counts(self, history):
        h = [BOS, BOS] + history
        for table, ctx in ((self.tri, tuple(h[-2:])), (self.bi, tuple(h[-1:]))):
            if ctx in table:
                return table[ctx]
        return self.uni

    def _next_sampler(self, history):
        h = [BOS, BOS] + history
        s = self._s_tri.get(tuple(h[-2:])) or self._s_bi.get(tuple(h[-1:]))
        return s or self._s_uni

    def sample_sentence(self, rng, temperature=1.0, max_len=14, prefix=None):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        words = list(prefix or [])
        while len(words) < max_len:
            w = self._next_sampler(words).sample(rng, temperature)
            if w == EOS:
                break
            words.append(w)
        return words


# ------------------------------------------------------------------ evaluation helpers
def sentence_features(words, tags):
    """(tag pattern, (first noun, first verb) pair) of one tagged sentence."""
    noun = next((w for w, t in zip(words, tags) if t == "NOUN"), None)
    verb = next((w for w, t in zip(words, tags) if t == "VERB"), None)
    return " ".join(tags), (noun, verb)


class SentenceJudge:
    """Per-sentence verdicts relative to the training set (the four metrics, one sentence at a time)."""

    def __init__(self, train_sentences):
        self.train_text = {" ".join(w) for w, _ in train_sentences}
        self.patterns = Counter(" ".join(t) for _, t in train_sentences)
        self.pairs = {sentence_features(w, t)[1] for w, t in train_sentences}

    def judge(self, words, tags):
        pattern, pair = sentence_features(words, tags)
        valid, plausible = pattern in self.patterns, pair in self.pairs
        novel = " ".join(words) not in self.train_text
        failure = ("structure" if not valid else "memorisation" if not novel
                   else "meaning" if not plausible else "none")       # guide, section 4 step 6
        return {"pattern": pattern, "valid": valid, "plausible": plausible, "novel": novel,
                "subject": pair[0], "verb": pair[1], "failure": failure}


def generation_metrics(samples, train_sentences, tagger, judge=None):
    """samples: list of (words, tags_or_None). Tags are computed with Viterbi when missing."""
    judge = judge or SentenceJudge(train_sentences)
    valid = novel = plausible = 0
    bigrams = []
    for words, tags in samples:
        tags = tags or tagger.viterbi(words)
        v = judge.judge(words, tags)
        valid += v["valid"]; novel += v["novel"]; plausible += v["plausible"]
        bigrams += list(zip(words, words[1:]))
    n = max(len(samples), 1)
    return {"valid_tag_pattern": valid / n, "novel_sentences": novel / n,
            "plausible_subject_verb": plausible / n,
            "distinct_2": len(set(bigrams)) / max(len(bigrams), 1)}


def oov_rate(words, vocab):
    return sum(w not in vocab for w in words) / max(len(words), 1)


# ------------------------------------------------------------------ one entry point for all models
class Lab:
    """Trains every generator once and serves batches, metrics, tagging and the data inspection."""

    def __init__(self, path="data/corpus.csv"):
        self.docs = load_tagged(path)
        self.train, self.test = split_by_document(self.docs)
        self.hmm = HMM(self.train)
        self.hmm_split = HMM(self.train, split_states=True)
        self.hmm_suffix = HMM(self.train, unknown="suffix")
        self.hmm_split_suffix = HMM(self.train, split_states=True, unknown="suffix")
        self.hmm2 = SecondOrderHMM(self.train)
        self.ngram = NGramGenerator([w for w, _ in self.train])
        self.judge = SentenceJudge(self.train)
        self.vocab = self.hmm.vocab

    def model(self, name):
        return {"hmm": self.hmm, "hmm_split": self.hmm_split, "hmm2": self.hmm2}.get(name)

    def tagger(self, split_states=False, suffix=False, second_order=False):
        if second_order:
            return self.hmm2
        return {(False, False): self.hmm, (True, False): self.hmm_split,
                (False, True): self.hmm_suffix, (True, True): self.hmm_split_suffix}[(split_states, suffix)]

    def generate_batch(self, model, n=500, temperature=1.0, seed=42, prefix=None):
        """List of dicts (one per sentence) with words, tags, hidden states and verdicts.
        Seeds and taggers follow src/evaluate.py, so (seed 42, n 500) reproduces the guide."""
        rng = random.Random(int(seed))
        rows = []
        for _ in range(int(n)):
            if model == "ngram":
                words = self.ngram.sample_sentence(rng, temperature, prefix=prefix)
                tags = self.hmm_split.viterbi(words)          # tagger used by the evaluation script
                states = None
            else:
                words, states = self.model(model).sample_with_states(rng, temperature)
                tags = coarse(states)
            rows.append({"words": words, "tags": tags, "states": states or tags,
                         **self.judge.judge(words, tags)})
        return rows

    def explain(self, model, row, temperature):
        """Data for the state-path graph of one generated sentence."""
        if model == "ngram":                                  # no hidden states: show the Viterbi tags
            return {"rows": ordered(self.hmm.tags, PLAIN_ORDER),
                    "steps": [{"word": w, "state": t, "branches": None, "p_emit": None}
                              for w, t in zip(row["words"], row["tags"])]}
        m = self.model(model)
        return {"rows": m.row_order(), "steps": m.step_distributions(row["words"], row["states"], temperature)}

    def explain_tagging(self, tagger, detail):
        """Graph data for a tagged sentence: the Viterbi path and the transitions out of it (T = 1)."""
        steps = tagger.step_distributions(detail["words"], detail["states"], 1.0)
        for st, c in zip(steps, detail["confidence"]):
            st["confidence"] = c
        return {"rows": tagger.row_order(), "steps": steps}

    @staticmethod
    def batch_metrics(rows):
        n = max(len(rows), 1)
        bigrams = [b for r in rows for b in zip(r["words"], r["words"][1:])]
        return {"valid_tag_pattern": sum(r["valid"] for r in rows) / n,
                "plausible_subject_verb": sum(r["plausible"] for r in rows) / n,
                "novel_sentences": sum(r["novel"] for r in rows) / n,
                "distinct_2": len(set(bigrams)) / max(len(bigrams), 1)}

    def temperature_sweep(self, temperatures, n=500, seed=42):
        """[{model, temperature, metric...}] for the comparison plot."""
        out = []
        for model in MODEL_NAMES:
            for t in temperatures:
                m = self.batch_metrics(self.generate_batch(model, n, t, seed))
                out.append({"model": model, "temperature": t, **m})
        return out

    def profile(self, rows):
        """Everything the Data view tab draws, for one generated batch, with the training set as reference."""
        gen_tags = Counter(t for r in rows for t in r["tags"])
        tr_tags = Counter(t for _, ts in self.train for t in ts)
        words_by_tag = defaultdict(Counter)
        for r in rows:
            for w, t in zip(r["words"], r["tags"]):
                words_by_tag[t][w] += 1
        pairs = Counter((r["subject"], r["verb"]) for r in rows if r["subject"] and r["verb"])
        return {
            "n": len(rows),
            "tags": {"generated": gen_tags, "training": tr_tags},
            "lengths": {"generated": Counter(len(r["words"]) for r in rows),
                        "training": Counter(len(w) for w, _ in self.train)},
            "patterns": {"generated": Counter(r["pattern"] for r in rows),
                         "training": Counter(" ".join(t) for _, t in self.train)},
            "words_by_tag": words_by_tag,
            "pairs": [(s, v, c, (s, v) in self.judge.pairs) for (s, v), c in pairs.most_common()],
            "position": Counter((i + 1, t) for r in rows for i, t in enumerate(r["tags"])),
            "failures": Counter(r["failure"] for r in rows),
            "vocab_used": len({w for r in rows for w in r["words"]}),
        }

    # ---- data inspection (guide, section 4 step 1)
    def inspect(self):
        patterns = Counter(" ".join(t) for _, t in self.train)
        watch = Counter(t for ws, ts in self.train for w, t in zip(ws, ts) if w == "watch")
        watch_examples = {}
        for ws, ts in self.train:
            for w, t in zip(ws, ts):
                if w == "watch" and t not in watch_examples:
                    watch_examples[t] = " ".join(ws)
        after_noun = Counter(b for _, ts in self.train for a, b in zip(ts, ts[1:]) if a == "NOUN")
        tot = sum(after_noun.values())
        return {"patterns": patterns, "watch": watch, "watch_examples": watch_examples,
                "after_noun": {k: v / tot for k, v in after_noun.most_common()},
                "n_train": len(self.train), "n_test": len(self.test), "vocab": len(self.vocab)}

    def viterbi_accuracy(self, tagger):
        ok = tot = w_ok = w_tot = 0
        for words, gold in self.test:
            pred = tagger.viterbi(words)
            for w, g, p in zip(words, gold, pred):
                tot += 1; ok += g == p
                if w == "watch":
                    w_tot += 1; w_ok += g == p
        return ok / tot, w_ok, w_tot


if __name__ == "__main__":
    lab = Lab()
    rng = random.Random(7)
    print("HMM    :", " ".join(lab.hmm.sample_sentence(rng)[0]))
    print("n-gram :", " ".join(lab.ngram.sample_sentence(rng)))
