"""Run:  python -m pytest -q   (from the project folder)"""
import random

import pytest

from src.pipeline import Lab, Sampler, sample_from, tokenize


@pytest.fixture(scope="module")
def lab():
    return Lab()


def test_split_by_document(lab):
    assert (len(lab.train), len(lab.test)) == (960, 240)


@pytest.mark.parametrize("model,valid,plausible,novel,d2", [
    ("hmm", 0.162, 0.148, 0.998, 0.198),
    ("hmm_split", 1.0, 0.606, 0.994, 0.148),
    ("ngram", 0.57, 0.76, 0.912, 0.105),
])
def test_guide_metrics_at_T1(lab, model, valid, plausible, novel, d2):
    m = Lab.batch_metrics(lab.generate_batch(model, 500, 1.0, 42))
    assert round(m["valid_tag_pattern"], 3) == valid
    assert round(m["plausible_subject_verb"], 3) == plausible
    assert round(m["novel_sentences"], 3) == novel
    assert round(m["distinct_2"], 3) == d2


def test_cached_sampler_matches_reference(lab):
    """The cached sampler draws exactly what the original sorted-weights code drew."""
    counter = lab.hmm.emit["NOUN"]
    for t in (0.5, 1.0, 2.5):
        r1, r2 = random.Random(1), random.Random(1)
        ref = []
        for _ in range(200):
            items = sorted(counter.items())
            ref.append(r1.choices([k for k, _ in items], weights=[c ** (1 / t) for _, c in items], k=1)[0])
        s = Sampler(counter)
        assert [s.sample(r2, t) for _ in range(200)] == ref


def test_viterbi_perfect_on_synthetic_test(lab):
    for words, gold in lab.test:
        assert lab.hmm.viterbi(words) == gold


def test_watch_is_ambiguous(lab):
    assert lab.hmm.viterbi("the girls watch the birds .".split())[2] == "VERB"
    assert lab.hmm.viterbi("the farmers find the watch .".split())[4] == "NOUN"


@pytest.mark.parametrize("text", ["", "!!!", "zebra", "The Zebras WATCH the moon, quickly!!", "x" * 3000])
def test_tagger_never_crashes(lab, text):
    for split in (False, True):
        for suffix in (False, True):
            d = lab.tagger(split, suffix).viterbi_detail(tokenize(text))
            assert len(d["tags"]) == len(d["words"]) == len(d["oov"]) == len(d["confidence"])
            assert all(0.0 <= c <= 1.0 + 1e-9 for c in d["confidence"])


def test_posteriors_sum_to_one(lab):
    for p in lab.hmm_split.posteriors("the cats eat the zebra .".split()):
        assert abs(sum(p.values()) - 1) < 1e-9


def test_tokenize():
    assert tokenize("The cats sleep.") == ["the", "cats", "sleep", "."]


def test_bad_temperature(lab):
    with pytest.raises(ValueError):
        lab.hmm.sample_sentence(random.Random(0), 0)
    with pytest.raises(ValueError):
        sample_from({"a": 1}, 0, random.Random(0))


def test_second_order_hmm(lab):
    m = Lab.batch_metrics(lab.generate_batch("hmm2", 500, 1.0, 42))
    assert round(m["valid_tag_pattern"], 3) == 0.2          # far below role-split states: pairs do not encode role
    assert lab.viterbi_accuracy(lab.hmm2)[0] == 1.0


def test_failure_types(lab):
    rows = lab.generate_batch("ngram", 500, 0.5, 42)
    for r in rows:
        expected = ("structure" if not r["valid"] else "memorisation" if not r["novel"]
                    else "meaning" if not r["plausible"] else "none")
        assert r["failure"] == expected
    assert any(r["failure"] == "memorisation" for r in rows)


@pytest.mark.parametrize("model", ["hmm", "hmm_split", "hmm2", "ngram"])
def test_graph_data(lab, model):
    row = lab.generate_batch(model, 1, 1.0, 3)[0]
    g = lab.explain(model, row, 1.0)
    assert [s["word"] for s in g["steps"]] == row["words"]
    for s in g["steps"]:
        assert s["state"] in g["rows"]
        if s["branches"] is not None:
            assert abs(sum(s["branches"].values()) - 1) < 1e-9 and s["state"] in s["branches"]
