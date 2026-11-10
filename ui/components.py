"""Small HTML builders.  Every value from the model or the user goes through esc()."""
from html import escape as esc

NOTICE = ("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined "
          "relative to this corpus and do not measure the quality of real English.")


def pct(x):
    return f"{x * 100:.1f}%"


# One muted hue per part of speech: readable on the graphite background, quiet enough for long tables.
TAG_COLORS = {"DET": "#9aa3ad", "ADJ": "#d2b27a", "NOUN": "#86afd6", "VERB": "#f2f2f2", "PREP": "#93c4a3",
              "PUNCT": "#777777"}
EXTRA_COLORS = ["#c99bc4", "#d69a86", "#9ac9c9", "#b7b78a", "#a3a3d6", "#d6c086", "#86d6b0", "#c48f8f",
                "#8fa8c4", "#b0b0b0", "#c4a68f"]


def tag_color(tag):
    base = tag.split("_")[0]
    if base in TAG_COLORS:
        return TAG_COLORS[base]
    return EXTRA_COLORS[sum(map(ord, base)) % len(EXTRA_COLORS)]


def tag_chip(state):
    """DET_S -> chip 'DET' with a small role suffix 'S'."""
    base, _, role = state.partition("_")
    suffix = f'<small>{esc(role)}</small>' if role else ""
    return f'<span class="tg" style="--c:{tag_color(state)}">{esc(base)}{suffix}</span>'


def interlinear(words, states):
    """Each word with its tag chip underneath."""
    return '<span class="il">' + "".join(
        f'<span class="ilw"><b>{esc(w)}</b>{tag_chip(s)}</span>' for w, s in zip(words, states)) + "</span>"


def tag_legend(tags):
    return '<div class="legend">' + "".join(tag_chip(t) for t in tags) + "</div>"


def topbar(model_label, right_text):
    return (f'<div class="topbar"><div class="info-dot" title="{esc(NOTICE)}">!</div>'
            f'<div class="model">{esc(model_label)}</div><div class="right">{esc(right_text)}</div></div>')


FAILURE_LABEL = {"structure": "Structure", "meaning": "Meaning", "memorisation": "Memorisation", "none": "None"}


def metric_pill(metrics):
    cells = "".join(f"<div><strong>{v}</strong><span>{esc(k)}</span></div>" for k, v in (
        ("Valid tag pattern", pct(metrics["valid_tag_pattern"])),
        ("Plausible subject-verb", pct(metrics["plausible_subject_verb"])),
        ("Novel sentences", pct(metrics["novel_sentences"])),
        ("Distinct-2", f'{metrics["distinct_2"]:.3f}')))
    return f'<div class="pill" role="group" aria-label="Metrics for this batch">{cells}</div>'


def stage(graph_html, caption, sentence, verdict, failure, metrics):
    chip = (f'<span class="fail f-{failure}">{FAILURE_LABEL[failure]} failure</span>' if failure != "none"
            else '<span class="fail f-none">No failure</span>')
    return (f'<div class="hero">{graph_html}<div class="graph-cap">{esc(caption)}</div>'
            f'<div class="verdict">{chip}<span>{esc(verdict)}</span></div>'
            f'{metric_pill(metrics)}'
            f'<div class="notice"><div class="info-dot">i</div><div>{esc(NOTICE)}</div></div></div>')


def mark(ok, yes="valid", no="invalid"):
    return f'<span class="mark {"ok" if ok else "no"}">{yes if ok else no}</span>'


def failure_chip(failure):
    if failure == "none":
        return '<span class="tag-new">none</span>'
    return f'<span class="fail f-{failure}">{FAILURE_LABEL[failure].lower()}</span>'


def sentence_table(rows, show_states_label="Hidden states", chips=True):
    body = []
    for i, r in rows:
        if chips:
            sent = f'<td class="s">{interlinear(r["words"], r["states"])}</td>'
        else:
            sent = (f'<td class="s">{esc(" ".join(r["words"]))}</td>'
                    f'<td class="states">{esc(" ".join(r["states"]))}</td>')
        body.append(
            f'<tr class="{"" if r["valid"] else "bad"} {"" if r["novel"] else "copied"}"><td class="n">{i + 1}</td>{sent}'
            f'<td>{mark(r["valid"])}</td>'
            f'<td>{mark(r["plausible"], "plausible", "odd pair")}</td>'
            f'<td>{failure_chip(r["failure"])}{"" if r["novel"] else "<br><span class=tag-copy style=margin-top:4px>copied</span>"}</td></tr>')
    if not body:
        return '<div class="empty">No sentence in this batch matches the filter.</div>'
    head = (f"<th>Sentence and {esc(show_states_label.lower())}</th>" if chips
            else f"<th>Sentence</th><th>{esc(show_states_label)}</th>")
    return ('<div class="tbl-wrap"><table class="tbl fit"><thead><tr><th>#</th>' + head +
            '<th>Pattern</th><th>Subject-verb</th>'
            '<th>Failure</th></tr></thead><tbody>' + "".join(body) + "</tbody></table></div>")


def visual_table(headers, rows, cls=""):
    """rows: list of lists of ready HTML cells."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl-wrap vt {cls}"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def inline_bar(value, max_value, color="#e6e6e6", label=None, ghost=None):
    """A thin bar with its value; `ghost` draws a reference marker (e.g. the training share)."""
    w = 0 if max_value <= 0 else value / max_value * 100
    g = "" if ghost is None or max_value <= 0 else f'<i style="left:{min(ghost / max_value, 1) * 100:.1f}%"></i>'
    return (f'<div class="ib"><div class="ib-track"><div style="width:{w:.1f}%;background:{color}"></div>{g}</div>'
            f'<span>{esc(label if label is not None else str(value))}</span></div>')


def stack_bar(parts):
    """parts: [(label, share, css_class)] -> one horizontal stacked bar with a legend."""
    segs = "".join(f'<div class="sb-{c}" style="width:{p * 100:.2f}%" title="{esc(l)} {p:.1%}"></div>'
                   for l, p, c in parts if p > 0)
    legend = "".join(f'<span><i class="sb-{c}"></i>{esc(l)} {p:.1%}</span>' for l, p, c in parts)
    return f'<div class="sb">{segs}</div><div class="sb-legend">{legend}</div>'


def metric_table(rows, columns, best=None):
    """rows: [(label, {key: value})], columns: [(key, header, fmt)]; best: {key: 'max'|'min'}."""
    best = best or {}
    winners = {}
    for key, how in best.items():
        vals = [r[key] for _, r in rows]
        winners[key] = max(vals) if how == "max" else min(vals)
    head = "".join(f'<th class="num">{esc(h)}</th>' for _, h, _ in columns)
    body = ""
    for label, r in rows:
        cells = ""
        for key, _, fmt in columns:
            cls = "num best" if key in winners and abs(r[key] - winners[key]) < 1e-12 else "num"
            cells += f'<td class="{cls}">{fmt(r[key])}</td>'
        body += f"<tr><td>{esc(label)}</td>{cells}</tr>"
    return (f'<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Model</th>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def tokens(detail, show_states):
    out = []
    for w, t, s, oov, c in zip(detail["words"], detail["tags"], detail["states"], detail["oov"], detail["confidence"]):
        state = f'<div class="state">{esc(s)}</div>' if show_states else ""
        out.append(
            f'<div class="tok {"oov" if oov else ""}"><div class="word">{esc(w)}</div><div class="tag">{tag_chip(t)}</div>'
            f'{state}<div class="bar"><div style="width:{c * 100:.0f}%"></div></div>'
            f'<div class="conf">{"not in vocabulary, " if oov else ""}{c * 100:.0f}% sure</div></div>')
    return '<div class="tokens">' + "".join(out) + "</div>"


def warn(text):
    return f'<div class="warn" role="alert"><div class="info-dot">!</div><div>{text}</div></div>'


def bars(items, max_value):
    rows = "".join(f'<div class="lbl">{esc(lbl)}</div><div class="track"><div style="width:{v / max_value * 100:.1f}%">'
                   f'</div></div><div class="val">{v}</div>' for lbl, v in items)
    return f'<div class="bars">{rows}</div>'
