"""
dashboard/layout.py

The single shared layout system every page uses. Three things live
here:

  section(title, subtitle)  - the ONE way to render a bordered card
                               section (title + subtitle + content as
                               one visual block, not two stacked cards)
  card_grid(cards, columns)  - the ONE way to render a responsive,
                               equal-height grid of small cards
  spacer(height)             - the ONE way to add vertical space
                               between sections

Why this file exists: before it, every page had grown its own way of
making a "card" - some split a <div class="chart-card"> open/close
across two separate st.markdown calls (which does NOT actually
enclose native widgets placed in between - Streamlit does not treat
separate markdown calls as one continuous HTML stream, so this
silently failed to visually wrap things like checkboxes or radio
buttons), others used st.container(border=True, key=...) with a
different key scheme per page, and About built its own one-off
flexbox grid. Every page should import from here instead of
reinventing any of this.
"""

from contextlib import contextmanager

import streamlit as st

# ---------------------------------------------------------------
# SPACING SYSTEM
# Use these constants instead of picking a page-specific pixel value.
# ---------------------------------------------------------------

SECTION_GAP = 18     # vertical gap between sections on a page
TITLE_GAP = 8        # gap between a section's own title and its content
ROW_GAP = 14          # gap between cards within a grid row


def spacer(height: int = SECTION_GAP):
    """Vertical space between sections. Same constant everywhere -
    don't hand-roll `st.markdown("<div style='height:Npx'>...")`
    with a different N per page."""
    st.markdown(f"<div style='height:{height}px'></div>", unsafe_allow_html=True)


def _slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


@contextmanager
def section(title: str, subtitle: str = None, key: str = None):
    """
    The one way to render a bordered section on any page:

        with section("Global Feature Importance", "Ranked by importance"):
            st.plotly_chart(fig, use_container_width=True)

    Title, subtitle, and whatever you put inside the `with` block all
    land inside ONE real st.container(border=True) - so the chart (or
    table, or widgets) visually belongs to its title with a small
    gap, instead of the title looking like its own separate card with
    another card starting right below it.

    `key` is auto-derived from the title (stable across reruns) unless
    you pass one explicitly - only needed if two sections on the same
    page would otherwise end up with the same title.
    """
    if key is None:
        key = f"section_{_slugify(title)}"

    with st.container(border=True, key=key):
        sub_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
        st.markdown(
            f'<div class="section-title">{title}</div>{sub_html}',
            unsafe_allow_html=True,
        )
        yield


def card_grid(cards_html: list, columns: int = 4):
    """
    Renders a list of pre-built inner-HTML strings (each one card's
    contents, e.g. from grid_card()) as a responsive, equal-height
    CSS grid: `columns` per row on desktop, 2 on tablet, 1 on mobile
    (breakpoints are defined once in theme.py's .card-grid CSS - not
    duplicated here or per page).
    """
    inner = "".join(cards_html)
    st.markdown(
        f'<div class="card-grid" style="--grid-cols:{columns};">{inner}</div>',
        unsafe_allow_html=True,
    )


def grid_card(icon: str, title: str, description: str) -> str:
    """Builds one card's inner HTML for use with card_grid(). Returns
    a string rather than rendering directly, since card_grid() needs
    every card's HTML at once to build the surrounding grid."""
    return (
        '<div class="grid-card">'
        f'<div class="grid-card-icon">{icon}</div>'
        f'<div class="grid-card-title">{title}</div>'
        f'<div class="grid-card-desc">{description}</div>'
        '</div>'
    )


def workflow_strip(steps: list):
    """
    Renders a horizontal, wrapping strip of workflow steps connected
    by arrows: steps is a list of (icon, label) tuples.
    """
    parts = []
    for i, (icon, label) in enumerate(steps):
        parts.append(
            '<div class="workflow-step">'
            f'<div class="workflow-icon">{icon}</div>'
            f'<div class="workflow-label">{label}</div></div>'
        )
        if i < len(steps) - 1:
            parts.append('<div class="workflow-arrow">→</div>')
    st.markdown(f'<div class="workflow-strip">{"".join(parts)}</div>', unsafe_allow_html=True)