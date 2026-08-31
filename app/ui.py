"""Reusable light-only presentation components. No model or simulation work here."""
from dataclasses import dataclass
from html import escape

import streamlit as st

PALETTE = {
    "primary": "#2458C6", "info": "#007D91", "accent": "#976000",
    "danger": "#BB3444", "safe": "#16734D", "muted": "#596579",
    "grid": "#E4E9F1",
}


def page_header(title: str, eyebrow: str):
    """One heading rhythm for every page, with native heading semantics."""
    with st.container():
        st.markdown(f'<div class="kicker">{escape(eyebrow)}</div>', unsafe_allow_html=True)
        st.title(title)


def columns(spec, *, gap="medium", vertical_alignment="top"):
    """Keep weighted native columns wide; wrap relative to each row's width."""
    index = st.session_state.get("_ui_row", 0)
    st.session_state["_ui_row"] = index + 1
    with st.container(key=f"lab-columns-{index}"):
        return st.columns(spec, gap=gap, vertical_alignment=vertical_alignment)


@dataclass(frozen=True)
class Metric:
    label: str
    value: object
    note: str = ""
    tone: str = "default"


def metric_grid(metrics: list[Metric]):
    """Equal cards; notes contribute to height instead of overlapping siblings."""
    cards = []
    for metric in metrics:
        tone = metric.tone if metric.tone in ("safe", "danger", "accent") else "default"
        cards.append(
            f'<div class="metric-card"><dt>{escape(metric.label)}</dt>'
            f'<dd class="metric-value">{escape(str(metric.value))}</dd>'
            f'<dd class="metric-note tone-{tone}">{escape(metric.note)}</dd></div>')
    st.markdown('<dl class="metric-grid">' + "".join(cards) + '</dl>', unsafe_allow_html=True)


def card(title: str, body: str, *, eyebrow: str = "", caption: str = ""):
    """Escaped text card; use native bordered containers for rich content."""
    st.markdown(
        '<article class="card">'
        + (f'<div class="kicker">{escape(eyebrow)}</div>' if eyebrow else '')
        + f'<h3>{escape(title)}</h3><p>{escape(body)}</p>'
        + (f'<p class="card-caption">{escape(caption)}</p>' if caption else '')
        + '</article>', unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return f'<span class="pill" style="--pill-color:{escape(color, quote=True)}">{escape(text)}</span>'


def progress_steps(labels: list[str], active: int):
    items = []
    for i, label in enumerate(labels):
        state = "current" if i == active else "complete" if i < active else "upcoming"
        current = ' aria-current="step"' if i == active else ''
        marker = "✓" if i < active else str(i + 1)
        items.append(f'<li class="step-{state}"{current}><span aria-hidden="true">'
                     f'{marker}</span>{escape(label)}</li>')
    st.markdown('<ol class="progress-steps" aria-label="Demo progress">'
                + ''.join(items) + '</ol>', unsafe_allow_html=True)


def plot(fig, **kwargs):
    """Pin charts to light colors and reserve space for axes and legends."""
    fig.update_layout(
        template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#25324A", size=12),
        colorway=[PALETTE[k] for k in ("primary", "info", "safe", "accent", "danger", "muted")],
        margin=dict(l=24, r=24, t=56, b=72, autoexpand=True),
        legend=dict(orientation="h", y=-0.22, x=0, yanchor="top", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#25324A"),
    )
    fig.update_xaxes(automargin=True, gridcolor=PALETTE["grid"])
    fig.update_yaxes(automargin=True, gridcolor=PALETTE["grid"])
    kwargs.pop("width", None)
    st.plotly_chart(fig, width="stretch", theme=None, **kwargs)
