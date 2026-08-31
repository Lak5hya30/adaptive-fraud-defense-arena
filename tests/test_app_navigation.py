"""Offline UI checks: shared navigation and the Atlas's search/reset workflow."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(monkeypatch):
    # Match `streamlit run app/Home.py`, including its multipage discovery root.
    monkeypatch.syspath_prepend(str(ROOT / "app"))
    at = AppTest.from_file(str(ROOT / "app" / "Home.py")).run(timeout=30)
    assert not at.exception
    return at


def test_every_navigation_destination_renders_offline(app):
    from common import NAVIGATION

    for _, pages in NAVIGATION:
        for path, _, _ in pages:
            app.switch_page(path).run(timeout=30)
            assert not app.exception, (path, [e.message for e in app.exception])
            assert len(app.title) == 1


def test_atlas_search_combines_with_filters_and_recovers_from_empty_results(app):
    app.switch_page("pages/1_Threat_Atlas.py").run()
    app.text_input(key="atlas_search").set_value("  OTP relay  ").run()
    assert not app.exception
    assert any("Hyper-Personalized Cardholder" in e.label for e in app.expander)

    app.multiselect(key="atlas_category").select("Social Engineering").run()
    app.multiselect(key="atlas_status").select("IMPLEMENTED").run()
    assert not app.exception
    assert any("Hyper-Personalized Cardholder" in e.label for e in app.expander)

    app.text_input(key="atlas_search").set_value("no-such-attack-xyz").run()
    assert any("No attacks match" in info.value for info in app.info)
    app.radio(key="atlas_sort").set_value("Name").run()
    next(b for b in app.button if b.label == "Clear filters").click().run()
    assert not app.exception
    assert app.text_input(key="atlas_search").value == ""
    assert all(not control.value for control in app.multiselect)
    assert app.radio(key="atlas_sort").value == "Novelty"
    assert not app.info
    from common import get_taxonomy
    total = len(get_taxonomy().attacks)
    assert any(c.value == f"{total} of {total} attacks shown." for c in app.caption)
