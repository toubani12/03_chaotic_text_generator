# Lab 03: Chaotic Text Generator

A test bench for text generators. It samples sentences from a Hidden Markov Model over part-of-speech tags (plain, role-split or second-order states) or from a trigram model, draws the hidden state path behind each sentence, scores each batch with four metrics, labels every failure, and tags any sentence you type with Viterbi. Guide: `Lab03_Chaotic_Text_Generator_Guide.pdf` (in the parent folder).

> The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined relative to this corpus and do not measure the quality of real English.

## Run (uv)

```
uv sync                                        # creates .venv from pyproject.toml / uv.lock (Python 3.12)
uv run streamlit run app.py
uv run python -m src.evaluate                  # synthetic corpus -> results/metrics.json, results/hmm_results.png
uv run python -m src.evaluate_real             # real text (UD English EWT, downloaded on first run) -> results/ewt_results.json
uv run pytest -q                               # 21 checks, including the guide's numbers at T = 1.0
uv run python -m report.build_report           # one-page summary -> report/report.html (+ PDF if Playwright is installed)
```

Without uv: `pip install -r requirements.txt` and the same commands without `uv run`. Run every command from this folder.

## The application

| Area | What it does |
|---|---|
| Control panel (sidebar) | Generator (plain HMM, role-split HMM, trigram, second-order HMM), temperature 0.3 to 3.0, seed, 10 to 500 sentences, optional trigram prompt, Generate, Download CSV |
| Stage | The state-path graph of one generated sentence: the hidden states it went through, the branches the model could have taken with their probabilities at the chosen temperature, and the word each state emitted. Below: failure type, the four batch metrics, the notice, and buttons to step through sentences or jump to a structure, meaning or memorisation failure |
| Sentences | One row per sentence: each word with its hidden state as a coloured tag chip, valid/invalid mark, subject-verb verdict and failure type (structure, meaning, memorisation); copied sentences get a white edge. Filters by failure type, chips or plain text |
| Data view | The batch profiled against the training set: failure mix, tag share, sentence length, tags by word position (heat-map), most frequent tag patterns, words generated per tag, subject-verb pairs |
| Tagger | Viterbi tag and confidence under each word, a warning and dashed outline for unknown words, and the Viterbi path drawn as a graph. Plain, role-split or second-order states; optional suffix guessing. Any input is accepted without crashing |
| Compare models | All generators at the same settings, and each metric plotted against temperature |
| Transitions | Heat-map of the learned transition matrix, plain or role-split states |
| Real text | The real-data table, the step 7 analysis and the most frequent tagging errors |
| Lab answers | Every question of the guide answered with live numbers: data inspection (watch, the 16 patterns), prediction, metrics and why, temperature choice, failure classification, real text, what state splitting changed, extensions, limits |

With seed 42 and 500 sentences at T = 1.0 the app shows the guide's values: 16.2%, 100% and 57.0% valid tag pattern.

## Changes to the provided code

* `src/pipeline.py` keeps the same API and gives the same output for a given seed (checked by `tests/`):
  * cached samplers (sorted keys and cumulative weights per temperature) and pre-computed log tables for Viterbi, so a 500-sentence batch takes milliseconds;
  * `viterbi_detail`: tags, hidden states, unknown-word flags, per-word confidence (forward-backward) and the path log-probability;
  * `SuffixGuesser` (guide, section 5): for unknown words, P(tag | last 1 to 3 letters) learned on rare training words. Off by default;
  * `SentenceJudge`: per-sentence verdicts shared by the metrics and the table, including the memorisation check;
  * `Lab`: trains everything once and serves batches, metrics, taggers and the temperature sweep.
* `SecondOrderHMM` (guide, section 5): P(tag | two previous tags), as a generator and a tagger. It reaches only 20.0% valid tag patterns at T = 1.0: the pair (DET, NOUN) is the same for a subject and an object, so a longer history does not replace the role.
* `src/evaluate.py` also reports the second-order HMM; `src/evaluate_real.py` adds the suffix-guessing row and the data sizes.
* `app.py` and `ui/` hold the interface (`ui/graph.py` draws the state-path graph); `.streamlit/config.toml` sets the dark theme.
* `report/build_report.py` writes the one-page report from the same numbers (`report/Lab03_report.pdf`). Put your repository and application URLs at the top of the script and run it again.

### Real-text results (UD English EWT test set)

| Model | Accuracy | Known words | Unknown words | Whole sentence |
|---|---|---|---|---|
| Most-frequent tag | 86.2% | 90.8% | 28.6% | 31.9% |
| HMM + Viterbi | 88.0% | 92.9% | 28.2% | 40.4% |
| HMM + Viterbi + suffix guessing | 91.2% | 93.3% | 64.3% | 49.0% |

The guide lists 88.1% for HMM + Viterbi. The provided code also gives 88.0% on the current treebank download, so the 0.1-point gap comes from an update to the upstream data. Suffix guessing more than doubles accuracy on unknown words.

## Report

`report/Rapport_TP03_TOUBANI.pdf` (and the editable `.docx`) is the full lab report, in French, with screenshots. `report/Lab03_report.pdf` is the one-page summary.

## Deployment (Streamlit Community Cloud)

Push this folder to GitHub with `requirements.txt`, `data/corpus.csv`, `results/`, `ui/`, `src/` and `.streamlit/`. On share.streamlit.io choose the repository and `app.py`. Do not commit `data/ewt/` (17 MB, already in `.gitignore`).

## Data

* `data/corpus.csv` is synthetic, generated by a small hand-written grammar (seed 42).
* Universal Dependencies English EWT: licence CC BY-SA 4.0, https://github.com/UniversalDependencies/UD_English-EWT
