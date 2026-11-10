"""Visual system: graphite panels, grain and a glowing state-path graph (inspired by a dark voice-assistant UI)."""
from ui.graph import GRAPH_CSS

GRAIN = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' "
         "stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .55 0'/>"
         "</filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

:root {
  --page: #0c0c0c;
  --side: #101010;
  --stage: #151515;
  --raise: #1d1d1d;
  --line: #262626;
  --line-2: #333333;
  --text: #ececec;
  --muted: #8e8e8e;
  --faint: #5c5c5c;
  --cta: #f3f3f3;
  --cta-ink: #111111;
  --grain: url("__GRAIN__");
}

html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li, .stApp td, .stApp th, .stApp div {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-feature-settings: "tnum" 1;
}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded' !important;
}
.stApp { background: var(--page); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
[data-testid="stExpandSidebarButton"] { color: var(--text) !important; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid #1a1a1a; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: .5rem; }
.brand { display: flex; align-items: center; gap: .6rem; margin: -.25rem 0 1.4rem; }
.brand-mark { width: 22px; height: 22px; border-radius: 50%;
  background: radial-gradient(circle, #111 52%, #6f6f6f 78%, #f0f0f0 100%);
  box-shadow: 0 0 10px rgba(255,255,255,.25); }
.brand-name { font-size: 1.12rem; font-weight: 600; letter-spacing: -.01em; color: var(--text); }
.side-label { color: var(--muted); font-size: .86rem; margin: 1.1rem 0 .35rem .1rem; }

/* model list styled like navigation items */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; width: 100%; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div, [data-testid="stSidebar"] [data-testid="stRadio"] { width: 100%; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  width: 100%; padding: .55rem .75rem; border-radius: 10px; margin: 0;
  transition: background .15s ease; cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #171717; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: var(--raise); }
[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]) { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: .95rem; color: var(--text); }

/* inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, textarea {
  background: #171717 !important; border-color: var(--line) !important; border-radius: 10px !important;
}
[data-testid="stNumberInput"] button { background: #171717 !important; border-color: var(--line) !important; }
[data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-size: .86rem !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--cta) !important; box-shadow: 0 0 0 4px rgba(255,255,255,.08) !important; }
[data-testid="stSliderThumbValue"] { color: var(--text) !important; font-size: .8rem !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--faint) !important; }

/* buttons */
button[data-testid="stBaseButton-primary"] {
  background: var(--cta) !important; color: var(--cta-ink) !important; border: 0 !important;
  border-radius: 10px !important; font-weight: 500 !important; min-height: 2.6rem;
}
button[data-testid="stBaseButton-primary"]:hover { background: #ffffff !important; }
button[data-testid="stBaseButton-primary"] p { color: var(--cta-ink) !important; }
button[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {
  background: transparent !important; color: var(--text) !important; border: 1px solid var(--line-2) !important;
  border-radius: 10px !important;
}
button[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover {
  border-color: #6a6a6a !important; background: #1a1a1a !important;
}
button:focus-visible, input:focus-visible, [data-testid="stRadioOption"]:has(input:focus-visible) {
  outline: 2px solid #d8d8d8 !important; outline-offset: 2px;
}

.side-card { margin-top: 1.4rem; border: 1px solid var(--line); border-radius: 14px; padding: .95rem 1rem .4rem;
  background: #131313; }
.side-card h4 { margin: 0 0 .3rem; font-size: .95rem; font-weight: 500; color: var(--text); padding: 0; }
.side-card p { margin: 0 0 .6rem; font-size: .86rem; line-height: 1.45; color: var(--muted); }

/* ---------- main panel (the grained stage) ---------- */
[data-testid="stMain"] { padding: 14px; }
[data-testid="stMainBlockContainer"] {
  position: relative; max-width: 1180px; padding: 1.6rem 2.4rem 3rem;
  background: radial-gradient(1100px 620px at 50% 0%, #1d1d1d 0%, rgba(21,21,21,0) 70%) no-repeat, var(--stage);
  border: 1px solid #222; border-radius: 22px; overflow: clip; flex-shrink: 0; min-height: auto;
}
[data-testid="stMainBlockContainer"]::before {
  content: ""; position: absolute; inset: 0; background-image: var(--grain); opacity: .05;
  pointer-events: none; border-radius: inherit;
}
[data-testid="stMainBlockContainer"]::after {
  content: ""; position: absolute; left: 0; right: 0; top: 0; height: 1000px; pointer-events: none;
  background:
    linear-gradient(115deg, transparent 18%, rgba(255,255,255,.05) 26%, transparent 34%),
    linear-gradient(100deg, transparent 52%, rgba(255,255,255,.04) 60%, transparent 70%);
  -webkit-mask-image: linear-gradient(to bottom, #000 45%, transparent 100%);
          mask-image: linear-gradient(to bottom, #000 45%, transparent 100%);
}
[data-testid="stMainBlockContainer"] > div { position: relative; }

.topbar { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; margin-bottom: .4rem; }
.topbar .model { font-size: 1.02rem; color: var(--text); font-weight: 500; text-align: center; }
.topbar .right { justify-self: end; color: var(--muted); font-size: .86rem; }
.info-dot { width: 22px; height: 22px; border: 1.5px solid #bdbdbd; border-radius: 50%; display: grid;
  place-items: center; font-size: .78rem; font-weight: 600; color: #d6d6d6; }

.hero { display: flex; flex-direction: column; align-items: center; text-align: center; padding: .6rem 0 .4rem; }
.graph-cap { color: var(--faint); font-size: .8rem; max-width: 44rem; margin: .5rem auto 0; line-height: 1.5; }
.sentence-line { margin-top: 1.4rem; font-size: clamp(1.3rem, 2.3vw, 1.8rem); color: #f1f1f1; letter-spacing: -.012em;
  max-width: 46rem; line-height: 1.3; }
.verdict { margin-top: 1.1rem; color: var(--muted); font-size: .92rem; display: flex; gap: .6rem; align-items: center;
  flex-wrap: wrap; justify-content: center; }
.fail { display: inline-block; font-size: .76rem; padding: .1rem .55rem; border-radius: 999px; white-space: nowrap;
  border: 1px solid #555; color: #e0e0e0; }
.fail.f-structure { background: #e9e9e9; color: #111; border-color: #e9e9e9; }
.fail.f-meaning { border-style: dashed; }
.fail.f-memorisation { background: #3a3a3a; border-color: #3a3a3a; }
.fail.f-none { color: var(--muted); border-color: #333; }
.hero .sg-wrap { margin-top: .4rem; }
.lead { font-size: .95rem; color: #d2d2d2; line-height: 1.6; max-width: 46rem; margin: 0 0 .8rem; }
.lead b { color: #fff; font-weight: 500; }
.qa { border-top: 1px solid var(--line); padding: 1.1rem 0 .4rem; }
.qa h4 { font-size: 1rem; font-weight: 500; margin: 0 0 .5rem; color: var(--text); padding: 0; }
.qa h4 small { color: var(--faint); font-weight: 400; font-size: .82rem; margin-left: .4rem; }
.qa ul { margin: .2rem 0 .6rem 1.1rem; padding: 0; color: #cfcfcf; font-size: .92rem; line-height: 1.6; max-width: 48rem; }
.qa p { color: #cfcfcf; font-size: .92rem; line-height: 1.6; max-width: 48rem; margin: 0 0 .6rem; }
.muted-s { color: var(--faint); font-size: .84rem; }
/* tag chips */
.tg { display: inline-flex; align-items: baseline; gap: 1px; font-size: .68rem; font-weight: 600; letter-spacing: .03em;
  color: var(--c); border: 1px solid color-mix(in srgb, var(--c) 55%, transparent);
  background: color-mix(in srgb, var(--c) 13%, transparent); border-radius: 6px; padding: .02rem .32rem; line-height: 1.35;
  white-space: nowrap; }
.tg small { font-size: .58rem; opacity: .75; font-weight: 500; margin-left: 1px; }
.il { display: inline-flex; flex-wrap: nowrap; gap: .35rem .32rem; }
.tbl td.s { white-space: nowrap; }
table.tbl.fit { width: max-content; min-width: 100%; }
.tbl tr.copied td:first-child { box-shadow: inset 3px 0 0 #e8e8e8; }
.tbl.fit td, .tbl.fit th { padding-left: .7rem; padding-right: .7rem; }
.ilw { display: inline-flex; flex-direction: column; align-items: flex-start; gap: .18rem; }
.ilw b { font-weight: 400; color: var(--text); font-size: .92rem; }
.tbl tr.bad .ilw b { color: #a9a9a9; }
.legend { display: flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 .7rem; }
/* visual tables and bars */
.vt td { vertical-align: middle; }
.ib { display: flex; align-items: center; gap: .6rem; min-width: 9rem; }
.ib-track { position: relative; flex: 1; height: 7px; background: #222; border-radius: 4px; }
.ib-track > div { height: 100%; border-radius: 4px; }
.ib-track > i { position: absolute; top: -3px; width: 2px; height: 13px; background: #8a8a8a; border-radius: 1px; }
.ib span { color: var(--text); font-size: .84rem; min-width: 3.4rem; text-align: right; }
.wchips { display: flex; flex-wrap: wrap; gap: .3rem; }
.wchips span { font-size: .8rem; color: #d8d8d8; background: #1f1f1f; border: 1px solid #2c2c2c; border-radius: 6px;
  padding: .05rem .4rem; }
.wchips span em { font-style: normal; color: var(--faint); margin-left: .25rem; }
.sb { display: flex; height: 14px; border-radius: 7px; overflow: hidden; background: #222; margin: .3rem 0 .5rem; }
.sb > div { height: 100%; }
.sb-legend { display: flex; flex-wrap: wrap; gap: .4rem 1.1rem; font-size: .84rem; color: var(--muted); }
.sb-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: .4rem; vertical-align: -1px; }
.sb-structure { background: #ececec; } .sb-meaning { background: #8a8a8a; }
.sb-memorisation { background: #4a4a4a; } .sb-none { background: #2b2b2b; outline: 1px solid #444; outline-offset: -1px; }
.dv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 2rem; }
@media (max-width: 900px) { .dv-grid { grid-template-columns: 1fr; } }
.qa .tbl-wrap { margin: .4rem 0 .8rem; }

.pill { margin: 1.5rem auto 0; display: inline-grid; grid-template-columns: repeat(4, auto); background: #1b1b1b;
  border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
.pill > div { padding: .8rem 1.35rem; text-align: center; min-width: 8.5rem; }
.pill > div + div { border-left: 1px solid var(--line); }
.pill strong { display: block; font-size: 1.32rem; font-weight: 500; color: var(--text); }
.pill span { font-size: .8rem; color: var(--muted); }
.notice { display: flex; gap: .65rem; align-items: flex-start; max-width: 46rem; margin: 1.6rem auto .2rem;
  color: var(--muted); font-size: .86rem; line-height: 1.5; text-align: left; }
.notice .info-dot { flex: none; width: 18px; height: 18px; font-size: .68rem; border-color: #777; color: #aaa; margin-top: 1px; }
.stale { text-align: center; color: #cfcfcf; font-size: .88rem; margin-top: .9rem; }

/* ---------- tabs ---------- */
[data-testid="stTabs"] { margin-top: 1.6rem; }
[data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid var(--line); }
[data-baseweb="tab"] { padding: .55rem .9rem !important; border-radius: 10px 10px 0 0; }
[data-baseweb="tab"] p { color: var(--muted); font-size: .95rem; }
[data-baseweb="tab"][aria-selected="true"] p { color: var(--text); }
[data-baseweb="tab-highlight"] { background: var(--text) !important; height: 2px !important; }
[data-baseweb="tab-border"] { display: none; }

h3.sec { font-size: 1.05rem; font-weight: 500; margin: 1.2rem 0 .2rem; color: var(--text); padding: 0; }
p.sub { color: var(--muted); font-size: .9rem; margin: 0 0 .9rem; max-width: 46rem; line-height: 1.5; }

/* tables */
.tbl-wrap { border: 1px solid var(--line); border-radius: 14px; overflow-x: auto; overflow-y: hidden; background: #121212; }
table.tbl { width: 100%; border-collapse: collapse; font-size: .9rem; }
.tbl th { position: sticky; top: 0; background: #181818; color: var(--muted); font-weight: 400; text-align: left;
  padding: .65rem .9rem; border-bottom: 1px solid var(--line); white-space: nowrap; z-index: 1; }
.tbl td { padding: .62rem .9rem; border-bottom: 1px solid #1d1d1d; vertical-align: top; color: var(--text); }
.tbl tr:last-child td { border-bottom: 0; }
.tbl td.n { color: var(--faint); width: 3rem; }
.tbl td.states { color: var(--muted); font-size: .8rem; line-height: 1.55; }
.tbl tr.bad td.s { color: #a9a9a9; }
.tbl td.num { text-align: right; }
.tbl th.num { text-align: right; }
.tbl .best { color: #ffffff; font-weight: 600; }
.mark { display: inline-flex; align-items: center; gap: .35rem; white-space: nowrap; font-size: .84rem; }
.mark.ok { color: #f0f0f0; }
.mark.no { color: #8a8a8a; }
.mark.ok::before { content: "✓"; }
.mark.no::before { content: "✕"; }
.tag-copy { display: inline-block; font-size: .76rem; padding: .08rem .5rem; border-radius: 999px;
  background: #e8e8e8; color: #111; white-space: nowrap; }
.tag-new { color: var(--faint); font-size: .84rem; }

/* tagger */
.tokens { display: flex; flex-wrap: wrap; gap: .55rem; margin: .4rem 0 .6rem; }
.tok { min-width: 5.2rem; padding: .7rem .8rem .65rem; border: 1px solid var(--line); border-radius: 12px; background: #181818; }
.tok .word { font-size: 1.15rem; color: var(--text); }
.tok .tag { margin-top: .4rem; }
.tok .state { font-size: .74rem; color: var(--faint); }
.tok .bar { height: 3px; background: #2a2a2a; border-radius: 3px; margin-top: .5rem; overflow: hidden; }
.tok .bar > div { height: 100%; background: #e6e6e6; }
.tok .conf { font-size: .72rem; color: var(--faint); margin-top: .25rem; }
.tok.oov { border-style: dashed; border-color: #8a8a8a; background: #141414; }
.tok.oov .word::after { content: " ?"; color: var(--muted); }
.warn { display: flex; gap: .7rem; align-items: flex-start; border: 1px solid #5a5a5a; background: #1c1c1c;
  border-radius: 12px; padding: .75rem .95rem; margin: .4rem 0 1rem; color: #e2e2e2; font-size: .9rem; line-height: 1.5; }
.warn .info-dot { flex: none; border-color: #e2e2e2; color: #e2e2e2; width: 18px; height: 18px; font-size: .68rem; margin-top: 2px; }
.empty { color: var(--muted); font-size: .92rem; padding: 1.2rem 0; }

/* bars for confusions */
.bars { display: grid; grid-template-columns: auto 1fr auto; gap: .45rem .8rem; align-items: center; font-size: .86rem; max-width: 38rem; }
.bars .lbl { color: var(--muted); white-space: nowrap; }
.bars .track { height: 8px; background: #1f1f1f; border-radius: 4px; overflow: hidden; }
.bars .track > div { height: 100%; background: #cfcfcf; border-radius: 4px; }
.bars .val { color: var(--text); text-align: right; }

[data-testid="stVegaLiteChart"] { background: transparent; }
[data-testid="stCaptionContainer"] { color: var(--faint) !important; }

@media (max-width: 760px) {
  [data-testid="stMain"] { padding: 6px; }
  [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 2rem; border-radius: 16px; }
  .pill { grid-template-columns: repeat(2, auto); }
  .pill > div:nth-child(3) { border-left: 0; }
  .pill > div:nth-child(n+3) { border-top: 1px solid var(--line); }
  .pill > div { min-width: 7.5rem; padding: .7rem .8rem; }
  .topbar .right { display: none; }
}
__GRAPH_CSS__
</style>
""".replace("\n\n", "\n").replace("__GRAPH_CSS__", GRAPH_CSS).replace("__GRAIN__", GRAIN.replace('"', "'"))
