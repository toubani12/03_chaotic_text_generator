"""Builds the one-page report (deliverable D) from the live models and results/ewt_results.json.
Run from the project folder:   uv run python -m report.build_report      -> report/report.html
Print it to PDF from a browser (A4, margins none), or let the script do it when Playwright is installed."""
import json
import sys
from html import escape as esc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import MODEL_NAMES, Lab  # noqa: E402

REPO_URL = "https://github.com/toubani12/chaotic-text-generator"
APP_URL = "https://<your-app>.streamlit.app"                      # fill in after deploying


def pct(x):
    return f"{x * 100:.1f}%"


def build():
    lab = Lab(str(ROOT / "data" / "corpus.csv"))
    t1 = {m: Lab.batch_metrics(lab.generate_batch(m, 500, 1.0, 42)) for m in MODEL_NAMES}
    sw = {(m, t): Lab.batch_metrics(lab.generate_batch(m, 500, t, 42)) for m in MODEL_NAMES for t in (0.5, 1.0, 1.5, 2.5)}
    real = json.loads((ROOT / "results" / "ewt_results.json").read_text())
    after = lab.inspect()["after_noun"]
    good = next(r for r in lab.generate_batch("hmm_split", 500, 1.0, 42) if r["failure"] == "none")
    bad = next(r for r in lab.generate_batch("hmm", 500, 1.0, 42) if r["failure"] == "structure")
    copy = next(r for r in lab.generate_batch("ngram", 500, 0.5, 42) if r["failure"] == "memorisation")
    p, s, g, h2 = t1["hmm"], t1["hmm_split"], t1["ngram"], t1["hmm2"]
    rb, rh = real["models"]["most-frequent tag"], real["models"]["HMM + Viterbi"]
    rs = real["models"].get("HMM + Viterbi + suffix")

    metric_rows = "".join(
        f'<tr{" class=ext" if m == "hmm2" else ""}><td>{esc(MODEL_NAMES[m])}{" (extension)" if m == "hmm2" else ""}</td>'
        f'<td>{pct(t1[m]["valid_tag_pattern"])}</td><td>{pct(t1[m]["plausible_subject_verb"])}</td>'
        f'<td>{pct(t1[m]["novel_sentences"])}</td><td>{t1[m]["distinct_2"]:.3f}</td></tr>' for m in MODEL_NAMES)
    real_rows = (f'<tr><td>HMM + Viterbi, synthetic corpus</td><td>100%</td><td>100%</td><td>no unknown words</td><td>100%</td></tr>'
                 f'<tr><td>Most-frequent tag, EWT</td><td>{pct(rb["accuracy"])}</td><td>{pct(rb["known_words"])}</td>'
                 f'<td>{pct(rb["unknown_words"])}</td><td>{pct(rb["whole_sentence"])}</td></tr>'
                 f'<tr><td>HMM + Viterbi, EWT</td><td>{pct(rh["accuracy"])}</td><td>{pct(rh["known_words"])}</td>'
                 f'<td>{pct(rh["unknown_words"])}</td><td>{pct(rh["whole_sentence"])}</td></tr>')
    if rs:
        real_rows += (f'<tr class=ext><td>HMM + suffix guessing, EWT (extension)</td><td>{pct(rs["accuracy"])}</td>'
                      f'<td>{pct(rs["known_words"])}</td><td>{pct(rs["unknown_words"])}</td><td>{pct(rs["whole_sentence"])}</td></tr>')
    info = real.get("data", {})

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Lab 03 report</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet">
<style>
@page {{ size: A4; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: 'DM Sans', Helvetica, Arial, sans-serif; color: #1a1a1a; font-size: 8.7pt; line-height: 1.38; }}
.page {{ width: 210mm; height: 297mm; padding: 11mm 13mm 9mm; overflow: hidden; }}
header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1.5px solid #111; padding-bottom: 2.2mm; }}
h1 {{ font-size: 15pt; margin: 0; font-weight: 600; letter-spacing: -.01em; }}
.meta {{ text-align: right; color: #555; font-size: 7.8pt; line-height: 1.4; }}
.meta a {{ color: #1a1a1a; }}
h2 {{ font-size: 9.6pt; font-weight: 600; margin: 3.2mm 0 1mm; }}
p {{ margin: 0 0 1.4mm; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0 7mm; }}
table {{ width: 100%; border-collapse: collapse; margin: 1mm 0 1.6mm; font-size: 8pt; }}
th {{ text-align: right; font-weight: 500; color: #555; border-bottom: 1px solid #222; padding: .8mm 1.2mm; }}
td {{ text-align: right; padding: .75mm 1.2mm; border-bottom: .5px solid #d6d6d6; }}
th:first-child, td:first-child {{ text-align: left; }}
tr.ext td {{ color: #666; }}
ul {{ margin: 0 0 1.4mm 4mm; padding: 0; }} li {{ margin-bottom: .7mm; }}
.ex {{ border-left: 2px solid #111; padding: .3mm 0 .3mm 2.5mm; margin: 1.2mm 0; }}
.ex q {{ font-weight: 500; quotes: "\\201C" "\\201D"; }}
.tag {{ display: inline-block; font-size: 7pt; border: 1px solid #111; border-radius: 6px; padding: 0 1.4mm; margin-right: 1mm; }}
footer {{ margin-top: 2mm; color: #777; font-size: 7pt; border-top: .5px solid #ccc; padding-top: 1.2mm; }}
</style></head><body><div class="page">
<header><div><h1>Lab 03: Chaotic Text Generator</h1>
<div style="color:#555">Badr Eddine Toubani, Master SDIA, ENSET</div></div>
<div class="meta">Repository: <a href="{REPO_URL}">{esc(REPO_URL)}</a><br>Application: <a href="{APP_URL}">{esc(APP_URL)}</a></div></header>

<h2>Problem and solution</h2>
<p>A generated sentence has no reference answer, so a text generator cannot be judged from a few examples. We built a
web test bench that samples hundreds of sentences from a part-of-speech HMM (plain, role-split or second-order states)
or from a trigram model, shows the hidden tag path behind each sentence, and scores every batch with four metrics. A
Viterbi tagger recovers the tags of any typed sentence. The same HMM is then tested on real English (UD English EWT)
to see how far the synthetic results transfer.</p>

<div class="cols"><div>
<h2>Metrics at T = 1.0 (500 sentences, seed 42)</h2>
<table><tr><th>Model</th><th>Valid pattern</th><th>Plausible S-V</th><th>Novel</th><th>Distinct-2</th></tr>{metric_rows}</table>
<p><b>Plain HMM, {pct(p["valid_tag_pattern"])}.</b> A first-order model only knows the previous tag. After a NOUN it moves
to VERB ({after.get("VERB", 0):.2f}), PREP ({after.get("PREP", 0):.2f}) or PUNCT ({after.get("PUNCT", 0):.2f}) whatever
the role of the noun, so a subject can end the sentence before the verb and an object can take a second verb.</p>
<p><b>Role-split HMM, {pct(s["valid_tag_pattern"])}.</b> NOUN_S is always followed by VERB, NOUN_O by PREP or PUNCT and
NOUN_P by PUNCT: every path the model can sample is one of the 16 training patterns. The second-order HMM reaches only
{pct(h2["valid_tag_pattern"])}, because the pair (DET, NOUN) is the same for a subject and an object.</p>
</div><div>
<h2>Choice of temperature</h2>
<ul><li><b>Writing toy: role-split HMM at T = 1.5.</b> Structure stays at {pct(sw[("hmm_split", 1.5)]["valid_tag_pattern"])}
while distinct-2 rises to {sw[("hmm_split", 1.5)]["distinct_2"]:.3f} (from {sw[("hmm_split", 1.0)]["distinct_2"]:.3f} at
T = 1.0); surprising word pairs are acceptable in a toy.</li>
<li><b>Autocomplete: trigram at T = 0.5.</b> It is the only model that continues the user's words, and a low temperature
gives the most predictable continuation ({pct(sw[("ngram", 0.5)]["valid_tag_pattern"])} valid patterns, against
{pct(sw[("ngram", 2.5)]["valid_tag_pattern"])} at T = 2.5).</li></ul>
<h2>Two generated sentences</h2>
<div class="ex"><span class="tag">good</span><q>{esc(" ".join(good["words"]))}</q> (role-split HMM, T = 1.0): valid
pattern, subject-verb pair seen in training, not copied.</div>
<div class="ex"><span class="tag">bad: structure</span><q>{esc(" ".join(bad["words"]))}</q> (plain HMM, T = 1.0): the tag path
{esc(" ".join(bad["tags"]))} is not a sentence. Cause: after the NOUN the model chose PUNCT (probability
{after.get("PUNCT", 0):.2f}) without knowing that no verb had been produced yet.</div>
<p style="color:#555">Other failure types seen: meaning, a valid pattern with an odd pair ({pct(1 - s["plausible_subject_verb"])}
of role-split sentences); memorisation, <q>{esc(" ".join(copy["words"]))}</q> copied by the trigram at T = 0.5.</p>
</div></div>

<h2>What state splitting changed, and what it did not</h2>
<p>Splitting NOUN, ADJ and DET by role gives the first-order model the memory it lacked: where it is in the sentence.
Valid tag patterns rise from {pct(p["valid_tag_pattern"])} to {pct(s["valid_tag_pattern"])} and plausible subject-verb
pairs from {pct(p["plausible_subject_verb"])} to {pct(s["plausible_subject_verb"])}, because subject nouns are now drawn
only from subject positions. It did not add meaning: nouns and verbs are still drawn independently, so the trigram stays
better on plausibility ({pct(g["plausible_subject_verb"])}); diversity drops (distinct-2 {s["distinct_2"]:.3f} against
{p["distinct_2"]:.3f}); tagging accuracy was already 100%; and the states are written by hand for this grammar.</p>

<h2>Synthetic corpus against real text (UD English EWT, {info.get("train_sentences", 12544):,} training and
{info.get("test_sentences", 2077):,} test sentences)</h2>
<table><tr><th>Model</th><th>Accuracy</th><th>Known words</th><th>Unknown words</th><th>Whole sentence</th></tr>{real_rows}</table>
<p>The synthetic grammar is unambiguous once the previous tag is known and every test word is in the 61-word vocabulary,
so Viterbi is perfect. On real text the HMM gains only {(rh["accuracy"] - rb["accuracy"]) * 100:.1f} points over the
most-frequent-tag baseline. Unknown words ({pct(rh["unknown_share_of_tokens"])} of tokens) are tagged correctly only
{pct(rh["unknown_words"])} of the time: the smoothed emission k / (count + k·V) is largest for rare tags, so unknown proper
nouns become X or INTJ. With {info.get("distinct_tag_sequences", 10359):,} distinct tag sequences, the valid-pattern
metric has no meaning on real text.{f" Guessing unknown words from their suffix raises unknown-word accuracy to {pct(rs['unknown_words'])}." if rs else ""}
The guide lists 88.1% for the HMM; the current treebank download gives {pct(rh["accuracy"])} with the unchanged code.</p>
<footer>The corpus is synthetic and the grammar is tiny; metrics such as valid tag pattern are defined relative to this
corpus and do not measure the quality of real English. All numbers are produced by the application and by
src/evaluate.py and src/evaluate_real.py.</footer>
</div></body></html>"""
    out = ROOT / "report" / "report.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page()
            page.goto(out.as_uri())
            page.wait_for_timeout(800)
            page.pdf(path=str(ROOT / "report" / "Lab03_report.pdf"), format="A4", print_background=True)
            b.close()
        print("wrote", ROOT / "report" / "Lab03_report.pdf")
    except Exception as exc:                                        # Playwright is optional
        print("PDF skipped (open report.html in a browser and print to PDF):", type(exc).__name__)


if __name__ == "__main__":
    build()
