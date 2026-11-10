"""State-path graph (an HMM trellis drawn as SVG).

Columns are word positions, rows are hidden states.  The white path is the state sequence that
produced the sentence; grey branches are the other states the model could have moved to at each
step, drawn with a weight proportional to their probability.  A dashed drop line links each state
to the word it emitted.
"""
from html import escape as esc

from ui.components import tag_color

LABEL_W, START_W, COL_W, RIGHT = 92, 54, 96, 24


def _fmt(p):
    return "1" if p >= 0.995 else f"{p:.2f}".lstrip("0")


def _curve(x0, y0, x1, y1):
    dx = (x1 - x0) * 0.5
    return f"C {x0 + dx:.1f} {y0:.1f}, {x1 - dx:.1f} {y1:.1f}, {x1:.1f} {y1:.1f}"


def _short(w, n=12):
    return w if len(w) <= n else w[: n - 1] + "…"


def state_graph(rows, steps, uid="g", animate=False, oov=None, max_cols=24):
    """rows: state names top to bottom.  steps: [{word, state, branches|None, p_emit|None, confidence?}]."""
    steps = steps[:max_cols]
    if not steps:
        return ""
    oov = (oov or [False] * len(steps))[: len(steps)]
    rows = list(rows) + [s["state"] for s in steps if s["state"] not in rows]
    row_h = 32 if len(rows) <= 7 else 25
    top = 22
    graph_h = top + len(rows) * row_h
    width = LABEL_W + START_W + len(steps) * COL_W + RIGHT
    height = graph_h + 86
    y_of = {r: top + (i + 0.5) * row_h for i, r in enumerate(rows)}
    x_of = [LABEL_W + START_W + (i + 0.5) * COL_W for i in range(len(steps))]
    sx, sy = LABEL_W + 16, top + len(rows) * row_h / 2
    visited = {s["state"] for s in steps}

    out = [f'<svg class="sg {"reveal" if animate else ""}" viewBox="0 0 {width} {height}" '
           f'style="max-width:{width * 1.12:.0f}px;min-width:{min(width, 640) * 0.86:.0f}px" role="img" '
           f'aria-label="Hidden state path: {esc(" ".join(s["state"] for s in steps))}">'
           f'<defs><filter id="glow-{uid}" x="-20%" y="-50%" width="140%" height="200%">'
           f'<feGaussianBlur stdDeviation="3.2" result="b"/><feMerge><feMergeNode in="b"/>'
           f'<feMergeNode in="SourceGraphic"/></feMerge></filter></defs>']

    # grid: row guides, row labels, column guides
    for r in rows:
        y = y_of[r]
        out.append(f'<line x1="{LABEL_W}" y1="{y}" x2="{width - RIGHT / 2}" y2="{y}" class="sg-row"/>')
        out.append(f'<text x="{LABEL_W - 20}" y="{y + 4}" class="sg-lbl {"on" if r in visited else ""}" '
                   f'text-anchor="end">{esc(r)}</text>'
                   f'<circle cx="{LABEL_W - 10}" cy="{y}" r="3" fill="{tag_color(r)}" '
                   f'opacity="{1 if r in visited else .35}"/>')
    for x in x_of:
        out.append(f'<line x1="{x}" y1="{top - 6}" x2="{x}" y2="{graph_h + 4}" class="sg-col"/>')
        for r in rows:
            out.append(f'<circle cx="{x}" cy="{y_of[r]}" r="1.6" class="sg-dot"/>')

    # start node
    out.append(f'<circle cx="{sx}" cy="{sy}" r="5" class="sg-start"/>'
               f'<text x="{sx}" y="{sy + 20}" text-anchor="middle" class="sg-small">start</text>')

    # branches (the roads not taken), drawn under the path
    for i, st in enumerate(steps):
        px, py = (sx, sy) if i == 0 else (x_of[i - 1], y_of[steps[i - 1]["state"]])
        for r, p in sorted((st.get("branches") or {}).items(), key=lambda kv: kv[1]):
            if r == st["state"] or r not in y_of or p < 0.005:
                continue
            x, y = x_of[i], y_of[r]
            op = 0.10 + 0.55 * p
            out.append(f'<path d="M {px:.1f} {py:.1f} {_curve(px, py, x, y)}" class="sg-branch" '
                       f'style="--d:{i};stroke-opacity:{op:.2f};stroke-width:{1 + 2.2 * p:.2f}">'
                       f'<title>{esc(steps[i - 1]["state"] if i else "start")} to {esc(r)}: {p:.2f}</title></path>')
            out.append(f'<circle cx="{x}" cy="{y}" r="3" class="sg-alt" style="--d:{i};opacity:{0.3 + 0.6 * p:.2f}"/>')
            if p >= 0.03:
                out.append(f'<text x="{x + 8}" y="{y + 3.5}" class="sg-alt-p" style="--d:{i}">{_fmt(p)}</text>')

    # the path itself
    d, px, py = f"M {sx} {sy}", sx, sy
    for i, st in enumerate(steps):
        x, y = x_of[i], y_of[st["state"]]
        d += " " + _curve(px, py, x, y)
        px, py = x, y
    out.append(f'<path d="{d}" class="sg-path-glow" pathLength="1" filter="url(#glow-{uid})"/>')
    out.append(f'<path d="{d}" class="sg-path" pathLength="1"/>')

    # nodes, emissions and words
    for i, st in enumerate(steps):
        x, y = x_of[i], y_of[st["state"]]
        prev = steps[i - 1]["state"] if i else "start"
        p_tr = (st.get("branches") or {}).get(st["state"])
        tip = f"{prev} to {st['state']}" + (f": {p_tr:.2f}" if p_tr is not None else "")
        if st.get("p_emit") is not None:
            tip += f". P({st['word']} | {st['state']}) = {st['p_emit']:.3f}"
        if st.get("confidence") is not None:
            tip += f". Posterior {st['confidence']:.0%}"
        g = [f'<g class="sg-step" style="--d:{i}"><title>{esc(tip)}</title>']
        g.append(f'<line x1="{x}" y1="{y + 8}" x2="{x}" y2="{graph_h + 18}" class="sg-emit"/>')
        g.append(f'<circle cx="{x}" cy="{y}" r="6.5" class="sg-node"/><circle cx="{x}" cy="{y}" r="2.4" class="sg-core"/>')
        if p_tr is not None:
            g.append(f'<text x="{x}" y="{y - 11}" text-anchor="middle" class="sg-p">{_fmt(p_tr)}</text>')
        word = esc(_short(st["word"]))
        q = '<tspan class="sg-q"> ?</tspan>' if oov[i] else ""
        g.append(f'<text x="{x}" y="{graph_h + 44}" text-anchor="middle" class="sg-word {"oov" if oov[i] else ""}">'
                 f'{word}{q}</text>')
        sub = (f'{st["confidence"]:.0%} sure' if st.get("confidence") is not None
               else _fmt(st["p_emit"]) if st.get("p_emit") is not None else st["state"])
        g.append(f'<text x="{x}" y="{graph_h + 66}" text-anchor="middle" class="sg-small">{esc(sub)}</text></g>')
        out.append("".join(g))
    out.append("</svg>")
    return '<div class="sg-wrap">' + "".join(out) + "</div>"


GRAPH_CSS = """
.sg-wrap { width: 100%; overflow-x: auto; display: block; padding: .4rem 0 .2rem; }
.sg-wrap::-webkit-scrollbar { height: 6px; } .sg-wrap::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }
svg.sg { display: block; margin: 0 auto; width: 100%; height: auto; overflow: visible; font-family: 'DM Sans', system-ui, sans-serif; }
.sg-row { stroke: #232323; stroke-width: 1; stroke-dasharray: 1 5; }
.sg-col { stroke: #1c1c1c; stroke-width: 1; }
.sg-dot { fill: #333; }
.sg-lbl { font-size: 11.5px; fill: #565656; letter-spacing: .02em; }
.sg-lbl.on { fill: #c4c4c4; }
.sg-start { fill: #151515; stroke: #9a9a9a; stroke-width: 1.5; }
.sg-small { font-size: 11px; fill: #6c6c6c; }
.sg-branch { fill: none; stroke: #d6d6d6; }
.sg-alt { fill: #9a9a9a; }
.sg-alt-p { font-size: 10px; fill: #7c7c7c; }
.sg-path { fill: none; stroke: #f7f7f7; stroke-width: 2.3; stroke-linecap: round; }
.sg-path-glow { fill: none; stroke: #ffffff; stroke-width: 5; stroke-opacity: .32; }
.sg-node { fill: #141414; stroke: #f5f5f5; stroke-width: 2; }
.sg-core { fill: #f5f5f5; }
.sg-p { font-size: 10.5px; fill: #e6e6e6; font-weight: 500; }
.sg-emit { stroke: #3c3c3c; stroke-width: 1; stroke-dasharray: 2 4; }
.sg-word { font-size: 19px; fill: #f1f1f1; letter-spacing: -.01em; }
.sg-word.oov { fill: #bdbdbd; text-decoration: underline dashed #777; }
.sg-q { fill: #8a8a8a; }
.sg-step:hover .sg-node { stroke-width: 3; }
.sg-step:hover .sg-word { fill: #ffffff; }

.sg.reveal .sg-path, .sg.reveal .sg-path-glow { stroke-dasharray: 1; stroke-dashoffset: 1;
  animation: sg-draw 1.6s cubic-bezier(.45,.05,.25,1) .15s forwards; }
.sg.reveal .sg-step, .sg.reveal .sg-branch, .sg.reveal .sg-alt, .sg.reveal .sg-alt-p {
  opacity: 0; animation: sg-in .5s ease forwards; animation-delay: calc(.2s + var(--d) * 110ms); }
.sg.reveal .sg-alt { animation-name: sg-in-soft; }
@keyframes sg-draw { to { stroke-dashoffset: 0; } }
@keyframes sg-in { to { opacity: 1; } }
@keyframes sg-in-soft { to { opacity: .7; } }
@media (prefers-reduced-motion: reduce) {
  .sg.reveal .sg-path, .sg.reveal .sg-path-glow { animation: none; stroke-dashoffset: 0; }
  .sg.reveal .sg-step, .sg.reveal .sg-branch, .sg.reveal .sg-alt, .sg.reveal .sg-alt-p { animation: none; opacity: 1; }
}
"""
