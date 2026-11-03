"""PaperPilot UI. Talks to the FastAPI backend over HTTP and never opens the index itself.

Run (with the API already running): uv run streamlit run app/streamlit_app.py
"""

import html
from collections import defaultdict

import httpx
import streamlit as st

from paperpilot.config import get_settings

API = get_settings().api_url

EXAMPLES = [
    "Does reranking improve passage retrieval?",
    "How does PPO clip the policy update?",
    "What does HyDE generate before retrieving documents?",
    "Which weights does LoRA freeze?",
    "What is linear regression?",
    "How does Mixtral route tokens to its experts?",
]
BADGES = {
    "direct": ("✓", "Answered directly", "badge-direct"),
    "corrected": ("↻", "Answered after a corrected search", "badge-corrected"),
    "general": ("◇", "General knowledge · not from the library", "badge-general"),
    "refused": ("!", "Not enough evidence in the library", "badge-refused"),
}
AREAS = {
    "rag-retrieval": "RAG & retrieval",
    "transformers": "Transformers & LLMs",
    "fine-tuning": "Fine-tuning & alignment",
    "vision": "Vision",
    "rl": "Reinforcement learning",
}

FONTS = (
    "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700"
    "&family=Source+Serif+4:ital,wght@0,400;0,600;1,400"
    "&family=JetBrains+Mono:wght@400;500&display=swap"
)

CSS = """
:root {
  --ink: #05070D; --navy: #0B1224; --panel: #101A33; --line: #1E2A4A;
  --text: #E6E9F2; --muted: #8A94AD; --yellow: #F5C518;
}
html, body, .stApp, button, input, textarea { font-family: 'Space Grotesk', sans-serif; }
.stApp {
    background: radial-gradient(1100px 560px at 88% -12%,
    #15224D 0%, var(--navy) 46%, var(--ink) 100%);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 860px; padding-top: 2.6rem; }

.pp-brand { display: flex; align-items: baseline; gap: .9rem; flex-wrap: wrap; }
.pp-logo { font-weight: 700; font-size: 2.2rem; letter-spacing: .01em; color: var(--text); }
.pp-logo span { color: var(--yellow); }
.pp-tag, .pp-label, .pp-time, .card-top, .lib-id, .lib-year, .pp-foot {
  font-family: 'JetBrains Mono', monospace;
}
.pp-tag { font-size: .7rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); }
.pp-rule {
  height: 2px; margin: .55rem 0 1.4rem;
  background: linear-gradient(90deg, var(--yellow) 0 72px, var(--line) 72px 100%);
}
.pp-sub { color: var(--muted); font-size: .98rem; line-height: 1.55; margin-bottom: 1.2rem; }
.pp-label {
  font-size: .68rem; letter-spacing: .18em; text-transform: uppercase;
  color: var(--muted); margin: 1.7rem 0 .55rem;
}

div[data-testid="stForm"] {
  border: 1px solid var(--line); border-radius: 12px; background: rgba(16, 26, 51, .55);
}
div[data-testid="stFormSubmitButton"] button {
  background: var(--yellow); color: var(--ink); border: none; font-weight: 700;
  letter-spacing: .04em;
}
div[data-testid="stFormSubmitButton"] button:hover { background: #FFD84A; color: var(--ink); }
.stButton button {
  border-radius: 999px; border: 1px solid var(--line); background: transparent;
  color: var(--text); font-size: .82rem; text-align: left;
}
.stButton button:hover { border-color: var(--yellow); color: var(--yellow); }

.badge {
  display: inline-flex; align-items: center; gap: .55rem; border-radius: 999px;
  padding: .36rem .8rem; font-family: 'JetBrains Mono', monospace; font-size: .72rem;
  letter-spacing: .09em; text-transform: uppercase;
}
.badge b { font-size: .95rem; }
.badge-direct { border: 1px solid var(--yellow); color: var(--yellow); }
.badge-corrected { background: var(--yellow); color: var(--ink); }
.badge-general { border: 1px dashed var(--muted); color: var(--text); }
.badge-refused { border: 1px solid #5C6683; color: #C3C9DA; }
.pp-time { font-size: .7rem; color: var(--muted); margin-left: .8rem; }

.card {
  background: var(--panel); border: 1px solid var(--line); 
  border-left: 3px solid var(--yellow);
  border-radius: 10px; padding: 1rem 1.15rem; margin: .75rem 0;
}
.card-top {
  display: flex; justify-content: space-between; gap: 1rem;
  font-size: .7rem; letter-spacing: .06em; color: var(--muted);
}
.card-num { color: var(--yellow); }
.card-title { font-weight: 600; font-size: 1.02rem; margin: .4rem 0 .15rem; color: var(--text); }
.card-authors { font-size: .82rem; color: var(--muted); }
.card-quote {
  font-family: 'Source Serif 4', serif; font-style: italic; font-size: 1.04rem;
  line-height: 1.6; color: #D9DEEA; border-left: 1px solid var(--line);
  padding-left: .9rem; margin: .8rem 0 .7rem;
}
.card a {
  font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .08em;
  color: var(--yellow) !important; text-decoration: none;
}
.card a:hover { text-decoration: underline; }

.lib-area {
  font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .16em;
  text-transform: uppercase; color: var(--yellow); margin: 1.1rem 0 .3rem;
}
.lib-row {
  display: grid; grid-template-columns: 92px 1fr 44px; gap: .75rem;
  padding: .25rem 0; border-bottom: 1px solid var(--line); font-size: .86rem;
}
.lib-id, .lib-year { font-size: .72rem; color: var(--muted); }
.pp-foot { font-size: .66rem; letter-spacing: .06em; color: var(--muted); margin-top: 2.5rem; }
"""


def ask_api(question: str, allow_general: bool) -> dict:
    response = httpx.post(
        f"{API}/ask",
        json={"question": question, "allow_general": allow_general},
        timeout=180,
    )
    if response.status_code == 422:
        raise ValueError("Please ask a question between 3 and 500 characters.")
    if response.status_code == 503:
        raise ValueError(response.json().get("detail", "The language model is busy, try again."))
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=600, show_spinner=False)
def get_papers() -> list[dict]:
    return httpx.get(f"{API}/papers", timeout=10).json()


def use_example(text: str) -> None:
    st.session_state.q = text
    st.session_state.pending = text


def card_html(i: int, c: dict) -> str:
    e = html.escape
    authors = ", ".join(c["authors"][:3]) + (" et al." if len(c["authors"]) > 3 else "")
    return "".join(
        [
            '<div class="card">',
            f'<div class="card-top"><span class="card-num">[{i}]</span>',
            f"<span>arXiv {e(c['arxiv_id'])} · {c['year']} · p.{c['page']}</span></div>",
            f'<div class="card-title">{e(c["title"])}</div>',
            f'<div class="card-authors">{e(authors)}</div>',
            f'<div class="card-quote">“{e(c["quote"])}”</div>',
            f'<a href="{e(c["url"])}" target="_blank">OPEN AT PAGE {c["page"]} →</a>',
            "</div>",
        ]
    )


def render_answer(result: dict) -> None:
    icon, label, cls = BADGES[result["path"]]
    seconds = result["duration_ms"] / 1000
    st.markdown(
        f'<div class="pp-label">Answer</div><span class="badge {cls}"><b>{icon}</b>{label}</span>'
        f'<span class="pp-time">{seconds:.1f}s</span>',
        unsafe_allow_html=True,
    )
    st.markdown(result["answer"])  # plain markdown, so $math$ renders
    if result["path"] == "general":
        st.warning(
            "Nothing in the library covers this. This answer comes from the model's general "
            "knowledge: it has no sources and was not verified."
        )
    if result["path"] == "refused":
        st.caption(
            "PaperPilot only answers from its 50 research papers. Try a question about "
            "transformers, RAG and retrieval, fine-tuning, vision or reinforcement learning, "
            "or switch on general-knowledge answers above."
        )
    if result["citations"]:
        st.markdown('<div class="pp-label">Verified sources</div>', unsafe_allow_html=True)
        for i, citation in enumerate(result["citations"], start=1):
            st.markdown(card_html(i, citation), unsafe_allow_html=True)


def render_library() -> None:
    with st.expander("The library · 50 papers"):
        try:
            papers = get_papers()
        except httpx.HTTPError:
            st.caption("Start the API to see the library.")
            return
        by_area: dict[str, list[dict]] = defaultdict(list)
        for p in papers:
            by_area[p["area"]].append(p)
        rows = []
        for area, name in AREAS.items():
            rows.append(f'<div class="lib-area">{name}</div>')
            for p in sorted(by_area[area], key=lambda p: p["year"]):
                rows.append(
                    f'<div class="lib-row"><span class="lib-id">{p["arxiv_id"]}</span>'
                    f"<span>{html.escape(p['title'])}</span>"
                    f'<span class="lib-year">{p["year"]}</span></div>'
                )
        st.markdown("".join(rows), unsafe_allow_html=True)


st.set_page_config(page_title="PaperPilot", page_icon="✈", layout="centered")
st.markdown(f"<style>@import url('{FONTS}');{CSS}</style>", unsafe_allow_html=True)
st.markdown(
    '<div class="pp-brand"><div class="pp-logo">Paper<span>Pilot</span></div>'
    '<div class="pp-tag">corrective rag · verified citations</div></div>'
    '<div class="pp-rule"></div>'
    '<p class="pp-sub">Ask a research question about 50 influential AI/ML papers. '
    "Every claim comes with a quote checked against its source page, "
    "or an honest “not enough evidence”.</p>",
    unsafe_allow_html=True,
)

with st.form("ask"):
    st.text_input(
        "Question",
        key="q",
        placeholder="e.g. Does reranking improve RAG?",
        label_visibility="collapsed",
    )
    st.toggle(
        "Allow general-knowledge answers when the library has nothing",
        key="allow_general",
        value=True,
    )
    submitted = st.form_submit_button("Ask PaperPilot")

st.markdown('<div class="pp-label">Try one</div>', unsafe_allow_html=True)
columns = st.columns(2)
for i, example in enumerate(EXAMPLES):
    columns[i % 2].button(example, key=f"ex{i}", on_click=use_example, args=(example,))

pending = st.session_state.pop("pending", None)
question = (st.session_state.get("q", "") if submitted else pending or "").strip()

if question:
    with st.status("Searching the library…", expanded=False) as status:
        status.write("retrieve → grade → answer → verify quotes")
        try:
            result = ask_api(question, st.session_state.get("allow_general", True))
        except httpx.ConnectError:
            status.update(label="API not reachable", state="error")
            st.error("Start the API first: `uv run uvicorn paperpilot.api.main:app`")
            st.stop()
        except (ValueError, httpx.HTTPError) as exc:
            status.update(label="Something went wrong", state="error")
            st.error(str(exc))
            st.stop()
        status.update(label="Done", state="complete")
    render_answer(result)

render_library()
st.markdown(
    '<div class="pp-foot">Library answers come only from the papers, and every quote is checked '
    "against its source text. General-knowledge answers are labelled and never cited.</div>",
    unsafe_allow_html=True,
)
