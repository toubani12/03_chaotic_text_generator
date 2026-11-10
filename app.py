"""Lab 03 - Chaotic Text Generator: a test bench for text generators.   Run:  uv run streamlit run app.py"""
import json
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.pipeline import MODEL_NAMES, PLAIN_ORDER, SPLIT_ORDER, Lab, ordered, tokenize
from ui import components as ui
from ui.graph import state_graph
from ui.styles import CSS

ROOT = Path(__file__).parent
st.set_page_config(page_title="Chaotic Text Generator", page_icon="◯", layout="wide",
                   initial_sidebar_state="auto")
st.markdown(CSS, unsafe_allow_html=True)

ORDER = list(MODEL_NAMES.values())
GREYS = ["#9a9a9a", "#f4f4f4", "#5e5e5e", "#c8c8c8"]
DASHES = [[6, 4], [1, 0], [2, 3], [10, 3, 2, 3]]
SWEEP_TEMPS = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
GUIDE_TEMPS = [0.5, 1.0, 1.5, 2.5]
METRIC_NAMES = {"valid_tag_pattern": "Valid tag pattern", "plausible_subject_verb": "Plausible subject-verb",
                "novel_sentences": "Novel sentences", "distinct_2": "Distinct-2"}
METRIC_COLS = [("valid_tag_pattern", "Valid tag pattern", ui.pct),
               ("plausible_subject_verb", "Plausible subject-verb", ui.pct),
               ("novel_sentences", "Novel sentences", ui.pct),
               ("distinct_2", "Distinct-2", lambda v: f"{v:.3f}")]
ALL_MAX = {k: "max" for k, _, _ in METRIC_COLS}


@st.cache_resource
def get_lab():
    return Lab(str(ROOT / "data" / "corpus.csv"))


@st.cache_data(show_spinner=False)
def run_batch(model, n, temperature, seed, prefix=()):
    return get_lab().generate_batch(model, n, temperature, seed, list(prefix) or None)


@st.cache_data(show_spinner=False)
def run_sweep(n, seed):
    return get_lab().temperature_sweep(SWEEP_TEMPS, n, seed)


@st.cache_data(show_spinner=False)
def viterbi_scores():
    lab = get_lab()
    return {name: lab.viterbi_accuracy(m) for name, m in
            (("HMM, plain states", lab.hmm), ("HMM, role-split states", lab.hmm_split),
             ("HMM, second order", lab.hmm2))}


def chart_theme(chart):
    return (chart.configure(background="transparent", font="DM Sans, Helvetica, Arial, sans-serif")
            .configure_view(stroke=None)
            .configure_axis(labelColor="#8e8e8e", titleColor="#8e8e8e", gridColor="#232323",
                            domainColor="#333", tickColor="#333", labelFontSize=12, titleFontSize=12,
                            titleFontWeight=400)
            .configure_legend(labelColor="#cfcfcf", titleColor="#8e8e8e", labelFontSize=12, titleFontWeight=400,
                              orient="top", symbolType="stroke", symbolStrokeWidth=2, symbolSize=260))


def html_table(headers, rows, cls=""):
    head = "".join(f"<th>{ui.esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl-wrap {cls}"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


lab = get_lab()

# ------------------------------------------------------------------ sidebar: control panel
with st.sidebar:
    st.html('<div class="brand"><div class="brand-mark"></div><div class="brand-name">Chaotic</div></div>')
    st.html('<div class="side-label">Generator</div>')
    model = st.radio("Generator", list(MODEL_NAMES), format_func=MODEL_NAMES.get, index=1,
                     label_visibility="collapsed",
                     help="The second-order HMM is an extension from section 5 of the guide.")
    st.html('<div class="side-label">Sampling</div>')
    temperature = st.slider("Temperature", 0.3, 3.0, 1.0, 0.1,
                            help="Sampling weights are count^(1/T). Below 1 is safer, above 1 is more chaotic.")
    c1, c2 = st.columns(2)
    seed = int(c1.number_input("Random seed", 0, 100_000, 42, step=1))
    n = int(c2.number_input("Sentences", 10, 500, 500, step=10))
    prefix_text = ""
    if model == "ngram":
        prefix_text = st.text_input("Start with (optional)", "", placeholder="e.g. the old",
                                    help="The trigram continues these words. Only training words are accepted.")
    prefix = tuple(tokenize(prefix_text))
    unknown_prefix = [w for w in prefix if w not in lab.vocab]
    if unknown_prefix:
        st.html(ui.warn(f"Not in the vocabulary: <b>{ui.esc(', '.join(unknown_prefix))}</b>. "
                        "The prompt is ignored until every word is a training word."))
        prefix = ()
    generate = st.button("Generate", type="primary", use_container_width=True)

params = {"model": model, "n": n, "temperature": round(float(temperature), 2), "seed": seed, "prefix": prefix}
if generate or "batch" not in st.session_state or "failure" not in st.session_state.batch["rows"][0]:
    st.session_state.batch = {**params, "rows": run_batch(model, n, params["temperature"], seed, prefix)}
    st.session_state.featured = 0
    st.session_state.jump_msg = None
batch = st.session_state.batch
rows = batch["rows"]
metrics = Lab.batch_metrics(rows)

df = pd.DataFrame([{
    "id": i + 1, "sentence": " ".join(r["words"]), "tags": " ".join(r["tags"]),
    "hidden_states": " ".join(r["states"]), "valid_tag_pattern": r["valid"],
    "plausible_subject_verb": r["plausible"], "novel": r["novel"], "failure_type": r["failure"],
    "model": MODEL_NAMES[batch["model"]], "temperature": batch["temperature"], "seed": batch["seed"]}
    for i, r in enumerate(rows)])

with st.sidebar:
    st.html(f'<div class="side-card"><h4>Export this batch</h4><p>{len(rows)} sentences with their words, '
            f'tags, hidden states, verdicts and failure type, as CSV.</p></div>')
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), use_container_width=True,
                       file_name=f"{batch['model']}_T{batch['temperature']}_seed{batch['seed']}_n{len(rows)}.csv",
                       mime="text/csv")

# ------------------------------------------------------------------ stage: the state-path graph
idx = st.session_state.get("featured", 0) % len(rows)
featured = rows[idx]
key = (tuple(sorted((k, v) for k, v in batch.items() if k != "rows")), idx)
animate = st.session_state.get("rendered") != key
st.session_state.rendered = key

graph = lab.explain(batch["model"], featured, batch["temperature"])
if batch["model"] == "ngram":
    caption = ("The trigram has no hidden states: the path shows the tags that Viterbi recovers for its sentence. "
               "Columns are words, rows are tags.")
else:
    caption = (f"Columns are words, rows are hidden states. The white path is the state sequence the model sampled; "
               f"grey branches are the other states it could have moved to. Numbers on the path are transition "
               f"probabilities, numbers under the words are P(word | state), both at T = {batch['temperature']}.")
flags = ["valid tag pattern" if featured["valid"] else "invalid tag pattern",
         "plausible subject-verb" if featured["plausible"] else "odd subject-verb pair",
         "new sentence" if featured["novel"] else "copied from the training set"]
st.html(ui.topbar(MODEL_NAMES[batch["model"]], f'T = {batch["temperature"]}, seed {batch["seed"]}'))
st.markdown(ui.stage(state_graph(graph["rows"], graph["steps"], uid="stage", animate=animate), caption,
                     " ".join(featured["words"]), f"Sentence {idx + 1} of {len(rows)}: " + ", ".join(flags),
                     featured["failure"], metrics), unsafe_allow_html=True)      # st.html would strip the SVG

JUMP = "Jump to a failure"


def jump_to_failure():
    choice = st.session_state.jump
    st.session_state.jump = JUMP
    kind = choice.split()[0].lower()
    hits = [i for i, r in enumerate(st.session_state.batch["rows"]) if r["failure"] == kind]
    cur = st.session_state.get("featured", 0) % len(st.session_state.batch["rows"])
    if hits:
        st.session_state.featured = next((i for i in hits if i > cur), hits[0])
        st.session_state.jump_msg = None
    else:
        st.session_state.jump_msg = f"No {kind} failure in this batch."


b1, b2, b3 = st.columns([1, 1, 1])
if b1.button("Previous sentence", use_container_width=True, disabled=idx == 0):
    st.session_state.featured, st.session_state.jump_msg = idx - 1, None
    st.rerun()
b2.selectbox("Jump to", [JUMP, "Structure failure", "Meaning failure", "Memorisation failure"], key="jump",
             on_change=jump_to_failure, label_visibility="collapsed")
if st.session_state.get("jump_msg"):
    b2.caption(st.session_state.jump_msg)
if b3.button("Next sentence", use_container_width=True):
    st.session_state.featured, st.session_state.jump_msg = idx + 1, None
    st.rerun()
if {k: batch[k] for k in params} != params:
    st.html('<div class="stale">Settings changed. Press Generate to apply them.</div>')

tab_rows, tab_data, tab_tag, tab_cmp, tab_mat, tab_real, tab_ans = st.tabs(
    ["Sentences", "Data view", "Tagger", "Compare models", "Transitions", "Real text", "Lab answers"])

# ------------------------------------------------------------------ sentences
with tab_rows:
    mix = Counter(r["failure"] for r in rows)
    st.html('<h3 class="sec">Generated batch</h3><p class="sub">One row per sentence: the words, the hidden '
            'states that produced them, whether the tag pattern is one of the 16 seen in training, and the failure '
            'type of the guide: structure (invalid pattern), meaning (valid pattern, odd subject-verb pair) or '
            'memorisation (copied from the training set, also marked with a white edge).</p>')
    view = st.segmented_control(
        "Show", ["All", "Valid", "Structure", "Meaning", "Memorisation"], default="All",
        format_func=lambda v: v if v in ("All", "Valid") else f"{v} ({mix.get(v.lower(), 0)})",
        label_visibility="collapsed") or "All"
    keep = {"All": lambda r: True, "Valid": lambda r: r["valid"],
            "Structure": lambda r: r["failure"] == "structure", "Meaning": lambda r: r["failure"] == "meaning",
            "Memorisation": lambda r: r["failure"] == "memorisation"}[view]
    shown = [(i, r) for i, r in enumerate(rows) if keep(r)]
    v1, v2 = st.columns([3, 1])
    v1.caption(f"{len(shown)} of {len(rows)} sentences. Failure mix: " + ", ".join(
        f"{k} {mix.get(k, 0) / len(rows):.1%}" for k in ("structure", "meaning", "memorisation", "none")))
    display = v2.segmented_control("Display", ["Tag chips", "Plain text"], default="Tag chips",
                                   label_visibility="collapsed") or "Tag chips"
    label = "Hidden states" if batch["model"] != "ngram" else "Viterbi tags (role-split HMM)"
    if display == "Tag chips":
        used = ordered({s_ for r in rows for s_ in r["states"]}, list(dict.fromkeys(SPLIT_ORDER + PLAIN_ORDER)))
        st.html(ui.tag_legend(used))
    PAGE = 50
    pages = max(1, -(-len(shown) // PAGE))
    if pages > 1:
        pg_ = st.segmented_control(
            "Rows", list(range(pages)), default=0, key=f"page-{view}-{len(shown)}",
            format_func=lambda k: f"{k * PAGE + 1}-{min((k + 1) * PAGE, len(shown))}",
            label_visibility="collapsed")
        pg_ = pg_ or 0
    else:
        pg_ = 0
    st.html(ui.sentence_table(shown[pg_ * PAGE:(pg_ + 1) * PAGE], label, chips=display == "Tag chips"))

# ------------------------------------------------------------------ data view: the generated batch, profiled
with tab_data:
    prof = lab.profile(rows)
    N = prof["n"]
    st.html(f'<h3 class="sec">The generated batch, profiled</h3><p class="sub">{N} sentences from '
            f'{ui.esc(MODEL_NAMES[batch["model"]])} at T = {batch["temperature"]}, seed {batch["seed"]}, against the '
            f'{len(lab.train)} training sentences. Grey markers on the bars show the training share. '
            f'{prof["vocab_used"]} of the {len(lab.vocab)} vocabulary words appear in the batch.</p>')

    fm = prof["failures"]
    st.html('<h3 class="sec">Failure mix</h3>' + ui.stack_bar(
        [("Structure", fm.get("structure", 0) / N, "structure"), ("Meaning", fm.get("meaning", 0) / N, "meaning"),
         ("Memorisation", fm.get("memorisation", 0) / N, "memorisation"), ("No failure", fm.get("none", 0) / N, "none")]))

    gt, tt = prof["tags"]["generated"], prof["tags"]["training"]
    g_tot, t_tot = sum(gt.values()) or 1, sum(tt.values()) or 1
    tag_order = [t for t in PLAIN_ORDER if t in gt or t in tt]
    top = max(max(gt[t] / g_tot, tt[t] / t_tot) for t in tag_order)
    tag_rows = [[ui.tag_chip(t), ui.inline_bar(gt[t] / g_tot, top, ui.tag_color(t), ui.pct(gt[t] / g_tot), tt[t] / t_tot),
                 f'<span class="muted-s">{ui.pct(tt[t] / t_tot)}</span>', f"{gt[t]:,}"] for t in tag_order]
    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.html('<h3 class="sec">Tag share</h3>' + ui.visual_table(["Tag", "Generated", "Training", "Tokens"], tag_rows))
    with d2:
        st.html('<h3 class="sec">Sentence length</h3>')
        lg, lt = prof["lengths"]["generated"], prof["lengths"]["training"]
        lens = sorted(set(lg) | set(lt))
        ldf = pd.DataFrame([{"length": L, "set": "Generated", "share": lg[L] / N} for L in lens] +
                           [{"length": L, "set": "Training", "share": lt[L] / len(lab.train)} for L in lens])
        lchart = alt.Chart(ldf).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("length:O", title="Words per sentence", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("set:N", sort=["Generated", "Training"]),
            y=alt.Y("share:Q", title="Share of sentences", axis=alt.Axis(format=".0%", tickCount=4)),
            color=alt.Color("set:N", scale=alt.Scale(domain=["Generated", "Training"], range=["#f0f0f0", "#4d4d4d"]),
                            title=None, legend=alt.Legend(symbolType="square", symbolSize=120)),
            tooltip=["set", "length", alt.Tooltip("share:Q", format=".1%")])
        st.altair_chart(chart_theme(lchart.properties(height=250)), use_container_width=True)

    st.html('<h3 class="sec">Tags by position</h3><p class="sub">Share of the generated sentences that have each '
            'tag at each word position. A grammatical batch shows the same staircase as the training data.</p>')
    pos = prof["position"]
    max_pos = max((p_ for p_, _ in pos), default=1)
    pdf = pd.DataFrame([{"position": p_, "tag": t, "share": pos[(p_, t)] / N}
                        for p_ in range(1, max_pos + 1) for t in tag_order if pos[(p_, t)]])
    if not pdf.empty:
        enc = dict(x=alt.X("position:O", title="Word position", axis=alt.Axis(labelAngle=0)),
                   y=alt.Y("tag:N", sort=tag_order, title=None))
        heat = alt.Chart(pdf).mark_rect(cornerRadius=3, stroke="#151515", strokeWidth=2).encode(
            **enc, color=alt.Color("tag:N", scale=alt.Scale(domain=tag_order, range=[ui.tag_color(t) for t in tag_order]),
                                   legend=None),
            opacity=alt.Opacity("share:Q", scale=alt.Scale(domain=[0, 1], range=[0.12, 1]), legend=None),
            tooltip=["tag", "position", alt.Tooltip("share:Q", format=".1%")])
        txt = alt.Chart(pdf[pdf.share >= 0.05]).mark_text(fontSize=10.5).encode(
            **enc, text=alt.Text("share:Q", format=".0%"),
            color=alt.condition("datum.share > 0.5", alt.value("#111"), alt.value("#d8d8d8")))
        st.altair_chart(chart_theme((heat + txt).properties(height=240)), use_container_width=True)

    st.html('<h3 class="sec">Tag patterns</h3><p class="sub">The most frequent tag sequences of the batch. A pattern '
            'is valid when it is one of the 16 sequences of the training set.</p>')
    gp, tp = prof["patterns"]["generated"], prof["patterns"]["training"]
    tp_tot = sum(tp.values())
    top_p = gp.most_common(15)
    pmax = max([c / N for _, c in top_p] + [tp[p_] / tp_tot for p_, _ in top_p] + [0.01])
    pat_rows = [[f'<span class="muted-s">{k + 1}</span>', '<span class="legend" style="margin:0">' +
                 "".join(ui.tag_chip(t) for t in p_.split()) + "</span>", ui.mark(p_ in tp),
                 ui.inline_bar(c / N, pmax, "#e6e6e6", ui.pct(c / N), tp[p_] / tp_tot if p_ in tp else None),
                 f'<span class="muted-s">{ui.pct(tp[p_] / tp_tot) if p_ in tp else "never"}</span>']
                for k, (p_, c) in enumerate(top_p)]
    rest = N - sum(c for _, c in top_p)
    st.html(ui.visual_table(["#", "Tag pattern", "In training", "Generated", "Training"], pat_rows))
    st.caption(f"{len(gp)} distinct patterns in the batch"
               + (f"; the other {len(gp) - len(top_p)} cover {rest} sentences." if rest else "."))

    d3, d4 = st.columns(2, gap="large")
    with d3:
        wb = prof["words_by_tag"]
        w_rows = [[ui.tag_chip(t), '<div class="wchips">' + "".join(
            f'<span>{ui.esc(w)}<em>{c}</em></span>' for w, c in wb[t].most_common(8)) + "</div>", str(len(wb[t]))]
            for t in tag_order if wb.get(t)]
        st.html('<h3 class="sec">Most generated words per tag</h3>' +
                ui.visual_table(["Tag", "Words (count)", "Distinct"], w_rows))
    with d4:
        pr = prof["pairs"][:12]
        cmax = max([c for _, _, c, _ in pr] + [1])
        pair_rows = [[ui.esc(s_), ui.esc(v_), ui.inline_bar(c, cmax, "#86afd6", str(c)),
                      ui.mark(seen, "seen", "never seen")] for s_, v_, c, seen in pr]
        st.html('<h3 class="sec">Subject-verb pairs</h3>' +
                ui.visual_table(["Subject", "Verb", "Sentences", "In training"], pair_rows))

# ------------------------------------------------------------------ tagger
with tab_tag:
    st.html('<h3 class="sec">Tag a sentence</h3><p class="sub">Type any sentence. Viterbi picks the most '
            'probable tag for each word; the bar shows how sure the model is (forward-backward posterior). '
            'Words outside the 61-word vocabulary are tagged from context, plus their ending when suffix '
            'guessing is on.</p>')
    examples = ["the girls watch the old birds near the garden .", "the farmers find the watch behind the river .",
                "the sleepy zebras chase the moon quickly ."]
    pick = st.pills("Examples", examples, label_visibility="collapsed")
    text = st.text_input("Sentence", value=pick or examples[0], max_chars=400,
                         placeholder="Type a sentence, e.g. the cats eat the bread .", label_visibility="collapsed")
    o1, o2 = st.columns([1.3, 1])
    tagger_kind = o1.segmented_control("Tagger", ["Plain states", "Role-split states", "Second order"],
                                       default="Plain states", label_visibility="collapsed") or "Plain states"
    use_suffix = o2.toggle("Guess unknown words from their suffix", value=False,
                           disabled=tagger_kind == "Second order",
                           help="Extension from the guide: P(tag | last letters) learned from the training words.")
    words = tokenize(text)
    if not words:
        st.html('<div class="empty">Type a sentence above to see its tags.</div>')
    else:
        try:
            tagger = lab.tagger(tagger_kind == "Role-split states", use_suffix, tagger_kind == "Second order")
            detail = tagger.viterbi_detail(words)
            oov = [w for w, o in zip(words, detail["oov"]) if o]
            if oov:
                uniq = list(dict.fromkeys(oov))
                how = ("context and the word ending" if use_suffix and tagger_kind != "Second order"
                       else "the surrounding tags only")
                st.html(ui.warn(f"{len(uniq)} word{'s' if len(uniq) > 1 else ''} not in the training vocabulary: "
                                f"<b>{ui.esc(', '.join(uniq[:12]))}{' ...' if len(uniq) > 12 else ''}</b>. "
                                f"Their tags are guessed from {how}, so treat them with care."))
            st.html(ui.tokens(detail, tagger_kind == "Role-split states"))
            st.caption(f"Tag sequence: {' '.join(detail['tags'])}. Log-probability of the best path: "
                       f"{detail['logprob']:.2f}")
            st.html('<h3 class="sec">Viterbi path</h3><p class="sub">The best path through the states. Grey '
                    'branches are the transitions the model allows out of each chosen state; for watch, only '
                    'the previous tag decides between NOUN and VERB.</p>')
            g = lab.explain_tagging(tagger, detail)
            st.markdown(state_graph(g["rows"], g["steps"], uid="tagger", oov=detail["oov"]), unsafe_allow_html=True)
            if len(words) > 24:
                st.caption("The graph shows the first 24 words.")
        except Exception as exc:                                  # never crash on user input
            st.html(ui.warn(f"This sentence could not be tagged ({ui.esc(type(exc).__name__)}). "
                            "Try shorter input made of words and punctuation."))

# ------------------------------------------------------------------ compare
with tab_cmp:
    st.html(f'<h3 class="sec">Every generator, same settings</h3><p class="sub">{n} sentences each at '
            f'T = {params["temperature"]}, seed {seed}. The best value in each column is in bold.</p>')
    cmp_rows = [(MODEL_NAMES[m], Lab.batch_metrics(run_batch(m, n, params["temperature"], seed))) for m in MODEL_NAMES]
    st.html(ui.metric_table(cmp_rows, METRIC_COLS, best=ALL_MAX))

    st.html('<h3 class="sec">What temperature does to each model</h3>')
    metric = st.segmented_control("Metric", list(METRIC_NAMES), format_func=METRIC_NAMES.get,
                                  default="valid_tag_pattern", label_visibility="collapsed") or "valid_tag_pattern"
    with st.spinner("Sampling every temperature..."):
        sweep = pd.DataFrame(run_sweep(n, seed))
    sweep["Model"] = sweep["model"].map(MODEL_NAMES)
    is_pct = metric != "distinct_2"
    y = alt.Y(f"{metric}:Q", title=METRIC_NAMES[metric],
              axis=alt.Axis(format=".0%" if is_pct else ".2f", tickCount=5),
              scale=alt.Scale(domain=[0, 1.02]) if is_pct else alt.Scale(zero=False))
    base = alt.Chart(sweep).encode(
        x=alt.X("temperature:Q", title="Temperature", scale=alt.Scale(domain=[0.3, 3.0], nice=False)),
        y=y,
        color=alt.Color("Model:N", sort=ORDER, scale=alt.Scale(domain=ORDER, range=GREYS), title=None),
        strokeDash=alt.StrokeDash("Model:N", sort=ORDER, scale=alt.Scale(domain=ORDER, range=DASHES), title=None),
        tooltip=["Model", alt.Tooltip("temperature:Q", title="T"),
                 alt.Tooltip(f"{metric}:Q", title=METRIC_NAMES[metric], format=".1%" if is_pct else ".3f")])
    chart = base.mark_line(strokeWidth=2) + base.mark_point(size=40, filled=True)
    rule = alt.Chart(pd.DataFrame({"t": [params["temperature"]]})).mark_rule(color="#444", strokeDash=[2, 3]).encode(x="t:Q")
    st.altair_chart(chart_theme((chart + rule).properties(height=340)), use_container_width=True)
    st.caption("The dotted line marks the temperature selected in the control panel.")

# ------------------------------------------------------------------ transitions
with tab_mat:
    st.html('<h3 class="sec">Learned transitions</h3><p class="sub">P(next state | state), counted on the 960 '
            'training sentences. With plain states a NOUN does not know whether it is the subject or the object; '
            'the role-split states remove that ambiguity.</p>')
    which = st.segmented_control("States", ["Plain states", "Role-split states"], default="Plain states",
                                 label_visibility="collapsed") or "Plain states"
    m = lab.hmm_split if which == "Role-split states" else lab.hmm
    tags, mat = m.transition_matrix()
    order = m.row_order()
    heat = pd.DataFrame([{"from": a, "to": b, "p": mat[i][j]} for i, a in enumerate(tags) for j, b in enumerate(tags)])
    enc = dict(x=alt.X("to:N", sort=order, title="Next state", axis=alt.Axis(labelAngle=-40)),
               y=alt.Y("from:N", sort=order, title="Current state"))
    cells = alt.Chart(heat).mark_rect(cornerRadius=3, stroke="#151515", strokeWidth=2).encode(
        **enc, color=alt.Color("p:Q", scale=alt.Scale(domain=[0, 1], range=["#191919", "#f4f4f4"]), legend=None),
        tooltip=["from", "to", alt.Tooltip("p:Q", format=".2f")])
    labels = alt.Chart(heat[heat.p > 0.005]).mark_text(fontSize=11).encode(
        **enc, text=alt.Text("p:Q", format=".2f"),
        color=alt.condition("datum.p > 0.55", alt.value("#111"), alt.value("#d0d0d0")))
    side = 420 if len(tags) <= 6 else 620
    st.altair_chart(chart_theme((cells + labels).properties(height=side)), use_container_width=True)

# ------------------------------------------------------------------ real text
real_path = ROOT / "results" / "ewt_results.json"
real = json.loads(real_path.read_text()) if real_path.exists() else None
with tab_real:
    st.html('<h3 class="sec">The same model on real text</h3><p class="sub">Universal Dependencies English EWT '
            '(web text, 17 tags, CC BY-SA 4.0): 12,544 training sentences, 2,077 test sentences. On the synthetic '
            'corpus Viterbi is 100% correct; on real text unseen words pull it down.</p>')
    if real is None:
        st.html('<div class="empty">No results yet. Run <code>python -m src.evaluate_real</code> once, '
                'then reload this page.</div>')
    else:
        nice = {"most-frequent tag": "Most-frequent tag (baseline)", "HMM + Viterbi": "HMM + Viterbi",
                "HMM + Viterbi + suffix": "HMM + Viterbi + suffix guessing"}
        rows_real = [("Synthetic corpus, HMM + Viterbi", {"accuracy": 1.0, "known_words": 1.0,
                                                          "unknown_words": float("nan"), "whole_sentence": 1.0})]
        rows_real += [(nice.get(k, k), v) for k, v in real["models"].items()]
        fmt = lambda v: "n/a" if v != v else ui.pct(v)        # noqa: E731  (nan -> n/a)
        st.html(ui.metric_table(rows_real, [
            ("accuracy", "Accuracy", fmt), ("known_words", "Known words", fmt),
            ("unknown_words", "Unknown words", fmt), ("whole_sentence", "Whole sentence correct", fmt)]))
        hmm_r, base_r = real["models"]["HMM + Viterbi"], real["models"]["most-frequent tag"]
        sfx = real["models"].get("HMM + Viterbi + suffix")
        info = real.get("data", {})
        conf = {(g, p): c for g, p, c in real["confusions"]}
        st.html(f'''<div class="qa"><h4>Why real text is harder</h4><ul>
<li>The HMM gains only {(hmm_r["accuracy"] - base_r["accuracy"]) * 100:.1f} points over the most-frequent-tag baseline
({ui.pct(hmm_r["accuracy"])} against {ui.pct(base_r["accuracy"])}) and gets {ui.pct(hmm_r["whole_sentence"])} of whole
sentences right, against 100% on the synthetic corpus.</li>
<li>Unknown words are {ui.pct(hmm_r["unknown_share_of_tokens"])} of the test tokens and only
{ui.pct(hmm_r["unknown_words"])} of them are tagged correctly. The smoothed emission of an unseen word is
k / (tag count + k·V), largest for rare tags, so errors go to rare tags: PROPN tagged X {conf.get(("PROPN", "X"), "?")}
times, PROPN tagged INTJ {conf.get(("PROPN", "INTJ"), "?")} times.</li>
<li>The training set has {info.get("distinct_tag_sequences", 10359):,} distinct tag sequences for
{info.get("train_sentences", 12544):,} sentences, against 16 in the synthetic grammar, so the valid tag pattern
metric cannot be used on real text.</li>
{f'<li>Extension: guessing unknown words from their last letters lifts unknown-word accuracy to {ui.pct(sfx["unknown_words"])} and overall accuracy to {ui.pct(sfx["accuracy"])}.</li>' if sfx else ''}
</ul></div>''')
        st.html('<h3 class="sec">Most frequent HMM errors</h3><p class="sub">Gold tag, then predicted tag.</p>')
        top = real["confusions"][:8]
        st.html(ui.bars([(f"{g} as {p}", c) for g, p, c in top], top[0][2]))

# ------------------------------------------------------------------ lab answers (every question of the guide)
with tab_ans:
    insp = lab.inspect()
    t1 = {m: Lab.batch_metrics(run_batch(m, 500, 1.0, 42)) for m in MODEL_NAMES}
    sw = {(r["model"], r["temperature"]): r for r in run_sweep(500, 42)}
    vit = viterbi_scores()
    an = insp["after_noun"]
    st.html('<h3 class="sec">Answers to the lab questions</h3><p class="sub">Every number below is computed live '
            'by this application with the guide\'s settings (500 sentences, seed 42), so it matches '
            '<code>python -m src.evaluate</code>.</p>')

    # step 1
    ex_rows = [[ui.esc(" ".join(w)), f'<span class="muted-s">{ui.esc(" ".join(t))}</span>'] for w, t in lab.train[:10]]
    pat_rows = [[ui.esc(p), str(c)] for p, c in insp["patterns"].most_common()]
    we = insp["watch_examples"]
    st.html(f'''<div class="qa"><h4>1. Inspect the data</h4>
<p>The training set has <b>{insp["n_train"]}</b> sentences, the test set {insp["n_test"]} (split by document), the
vocabulary {insp["vocab"]} words, and there are only <b>{len(insp["patterns"])}</b> distinct tag sequences.
<i>watch</i> is a VERB {insp["watch"].get("VERB", 0)} times and a NOUN {insp["watch"].get("NOUN", 0)} times in training,
for example "{ui.esc(we.get("VERB", ""))}" (VERB) and "{ui.esc(we.get("NOUN", ""))}" (NOUN). Viterbi tags all
{vit["HMM, plain states"][2]} test occurrences correctly ({vit["HMM, plain states"][1]}/{vit["HMM, plain states"][2]}),
because the previous tag (DET or a NOUN) decides.</p>
{html_table(["First 10 training sentences", "Tags"], ex_rows)}
{html_table(["The 16 tag sequences of the training set", "Sentences"], pat_rows)}</div>''')

    # step 2
    st.html(f'''<div class="qa"><h4>2. Prediction, written before running</h4>
<p>The role-split HMM should have the highest valid-tag-pattern rate: its states carry the role of each noun, so after
NOUN_S only VERB can follow and after NOUN_O only PREP or PUNCT. The plain HMM forgets the role after every NOUN. The
trigram sees two words of context, which covers short phrases but not the joins between them. Result: confirmed,
{ui.pct(t1["hmm_split"]["valid_tag_pattern"])} against {ui.pct(t1["ngram"]["valid_tag_pattern"])} (trigram) and
{ui.pct(t1["hmm"]["valid_tag_pattern"])} (plain).</p></div>''')

    # steps 3-4
    st.html(f'''<div class="qa"><h4>3 and 4. Metrics at T = 1.0</h4>
{ui.metric_table([(MODEL_NAMES[m], t1[m]) for m in MODEL_NAMES], METRIC_COLS, best=ALL_MAX)}
<p><b>Why the plain HMM scores low.</b> After a NOUN it moves to VERB with probability {an.get("VERB", 0):.2f}, PREP
{an.get("PREP", 0):.2f} or PUNCT {an.get("PUNCT", 0):.2f}, whatever the role of the noun. A subject noun can then end the
sentence before any verb, and an object noun can be followed by a second verb. A sentence has to win several of these
draws in a row, so only {ui.pct(t1["hmm"]["valid_tag_pattern"])} of the sampled tag paths are one of the 16 patterns.</p>
<p><b>Why the role-split HMM scores 100%.</b> NOUN_S is followed by VERB with probability 1, NOUN_O by PREP or PUNCT,
NOUN_P by PUNCT. Every path the model can sample is one of the training patterns. Viterbi accuracy on the 240 test
sentences is {vit["HMM, plain states"][0]:.0%} for both models.</p></div>''')

    # step 5
    tr = [[ui.esc(MODEL_NAMES[m])] + [f'{ui.pct(sw[(m, t)]["valid_tag_pattern"])} <span class="muted-s">/ '
                                       f'{sw[(m, t)]["distinct_2"]:.3f}</span>' for t in GUIDE_TEMPS] for m in MODEL_NAMES]
    s15, n05 = sw[("hmm_split", 1.5)], sw[("ngram", 0.5)]
    st.html(f'''<div class="qa"><h4>5. Vary the temperature</h4>
{html_table(["Valid tag pattern / distinct-2"] + [f"T = {t}" for t in GUIDE_TEMPS], tr)}
<p>For the trigram, valid patterns fall from {ui.pct(sw[("ngram", 0.5)]["valid_tag_pattern"])} at T = 0.5 to
{ui.pct(sw[("ngram", 2.5)]["valid_tag_pattern"])} at T = 2.5 while distinct-2 rises from
{sw[("ngram", 0.5)]["distinct_2"]:.3f} to {sw[("ngram", 2.5)]["distinct_2"]:.3f}.</p>
<ul><li><b>Writing toy: role-split HMM at T = 1.5.</b> Structure stays at {ui.pct(s15["valid_tag_pattern"])} while
diversity rises (distinct-2 {s15["distinct_2"]:.3f} against {sw[("hmm_split", 1.0)]["distinct_2"]:.3f} at T = 1.0) and
{ui.pct(s15["novel_sentences"])} of the sentences are new; odd meanings are acceptable in a toy.</li>
<li><b>Autocomplete: trigram at T = 0.5.</b> It is the only model that continues the user's words, and low temperature
gives the most predictable continuation ({ui.pct(n05["valid_tag_pattern"])} valid patterns,
{ui.pct(n05["plausible_subject_verb"])} plausible pairs); repeating frequent training phrases is acceptable for a
suggestion.</li></ul></div>''')

    # step 6
    def first(model, temp, kind, k=2, seed_=42):
        return [r for r in run_batch(model, 500, temp, seed_) if r["failure"] == kind][:k]
    cause = {"structure": "First-order memory: after a NOUN the plain HMM does not know if the noun is the subject or "
                          "the object, so it ends or continues the sentence in the wrong place.",
             "meaning": "States encode grammar, not meaning: the noun and the verb are drawn independently from "
                        "their states, so nothing links them.",
             "memorisation": "At low temperature the trigram follows the most frequent word paths of the training "
                             "set and reproduces a whole sentence."}
    fr = []
    for (mname, temp, kind) in (("hmm", 1.0, "structure"), ("hmm_split", 1.0, "meaning"), ("ngram", 0.5, "memorisation")):
        for r in first(mname, temp, kind):
            fr.append([ui.esc(" ".join(r["words"])), f"{ui.esc(MODEL_NAMES[mname])}, T = {temp}",
                       ui.failure_chip(kind), ui.esc(cause[kind])])
    failures_file = ROOT / "results" / "failures.txt"
    listed = [ln.strip()[2:] for ln in failures_file.read_text().splitlines() if ln.strip().startswith("- ")] \
        if failures_file.exists() else []
    lr = []
    for s_ in listed:
        w = s_.split()
        v = lab.judge.judge(w, lab.hmm_split.viterbi(w))
        lr.append([ui.esc(s_), ui.failure_chip(v["failure"]), ui.esc(cause.get(v["failure"], ""))])
    mix_rows = []
    for m in MODEL_NAMES:
        c = Counter(r["failure"] for r in run_batch(m, 500, 1.0, 42))
        mix_rows.append([ui.esc(MODEL_NAMES[m])] + [ui.pct(c.get(k, 0) / 500) for k in ("structure", "meaning", "memorisation", "none")])
    good = next(r for r in run_batch("hmm_split", 500, 1.0, 42) if r["failure"] == "none")
    st.html(f'''<div class="qa"><h4>6. Classify failures</h4>
<p>The sentences of <code>results/failures.txt</code>, labelled:</p>
{html_table(["Sentence", "Failure type", "Cause"], lr)}
<p>Examples of each type from the generated batches:</p>
{html_table(["Sentence", "Model", "Failure type", "Cause"], fr)}
<p>Share of each failure type over 500 sentences at T = 1.0:</p>
{html_table(["Model", "Structure", "Meaning", "Memorisation", "No failure"], mix_rows)}
<p>A good sentence for the report: "{ui.esc(" ".join(good["words"]))}" (role-split HMM, T = 1.0): valid pattern,
seen subject-verb pair, not in the training set.</p></div>''')

    # step 7 + splitting paragraph
    if real:
        h, b = real["models"]["HMM + Viterbi"], real["models"]["most-frequent tag"]
        st.html(f'''<div class="qa"><h4>7. Real text</h4>
<p>On the synthetic corpus Viterbi is 100% correct; on English EWT the same code reaches {ui.pct(h["accuracy"])}
({ui.pct(h["known_words"])} on known words, {ui.pct(h["unknown_words"])} on unknown words, {ui.pct(h["whole_sentence"])}
of whole sentences), only {(h["accuracy"] - b["accuracy"]) * 100:.1f} points above the most-frequent-tag baseline. The
synthetic grammar is unambiguous once the previous tag is known and every test word is in the vocabulary; real text
has a large vocabulary, {ui.pct(h["unknown_share_of_tokens"])} unseen tokens, and almost every sentence has its own tag
sequence. The details are in the Real text tab. The guide lists 88.1% for the HMM; the current download of the
treebank gives {ui.pct(h["accuracy"])} with the unchanged code, so the gap comes from an update of the data.</p></div>''')
    p_, s_ = t1["hmm"], t1["hmm_split"]
    st.html(f'''<div class="qa"><h4>What state splitting changed, and what it did not change</h4>
<p>Splitting NOUN, ADJ and DET by role gives the first-order model the one piece of memory it lacked: where it is in
the sentence. Valid tag patterns rise from {ui.pct(p_["valid_tag_pattern"])} to {ui.pct(s_["valid_tag_pattern"])} and
plausible subject-verb pairs from {ui.pct(p_["plausible_subject_verb"])} to {ui.pct(s_["plausible_subject_verb"])}, because
subject nouns are now drawn only from subject positions. It did not add meaning: {ui.pct(1 - s_["plausible_subject_verb"])}
of the sentences still pair a noun and a verb never seen together, and the trigram stays better on that metric
({ui.pct(t1["ngram"]["plausible_subject_verb"])}). Diversity drops (distinct-2 {s_["distinct_2"]:.3f} against
{p_["distinct_2"]:.3f}), tagging accuracy is unchanged at 100%, and the states are written by hand for this grammar:
a relative clause, or real text, would need new ones.</p></div>''')

    # extensions + limits
    sfx_txt = (f'Suffix guessing for unknown words: {ui.pct(real["models"]["HMM + Viterbi + suffix"]["unknown_words"])} '
               f'on unknown EWT words against {ui.pct(real["models"]["HMM + Viterbi"]["unknown_words"])}.'
               if real and "HMM + Viterbi + suffix" in real["models"] else "Suffix guessing for unknown words.")
    st.html(f'''<div class="qa"><h4>Extensions implemented</h4><ul>
<li>Second-order HMM (generator and tagger): {ui.pct(t1["hmm2"]["valid_tag_pattern"])} valid tag patterns at T = 1.0
against {ui.pct(s_["valid_tag_pattern"])} for role-split states. Two previous tags are not enough, because the pair
(DET, NOUN) is the same for a subject and an object: the useful memory is the role, not a longer history.</li>
<li>{sfx_txt}</li>
<li>Memorisation detector: the "copied" mark and the memorisation filter in the Sentences tab.</li>
<li>Transition heat-map (Transitions tab) and the state-path graph on the main stage.</li></ul></div>''')
    st.html('''<div class="qa"><h4>Limits and ethics</h4><ul>
<li>The data is synthetic; valid tag pattern and plausible subject-verb are defined relative to its grammar and do not
measure the quality of real English.</li>
<li>100% tagging accuracy holds because the synthetic grammar is unambiguous once the previous tag is known.</li>
<li>The real-data test uses one lower-cased English web treebank; it does not carry over to other languages or
domains, and a first-order HMM is not a state-of-the-art tagger.</li>
<li>A generator can reproduce its training data: never train one on private messages you do not own.</li>
<li>Generated text is fluent but has no meaning or intent; do not present it as written by a person.</li>
<li>Negative result: the role-split HMM scores 100% on structure but lower than the trigram on subject-verb
plausibility.</li></ul></div>''')
