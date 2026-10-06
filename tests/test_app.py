"""Smoke tests for the Streamlit app. Run with: pytest"""
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    # An empty value also stops load_dotenv() from picking up a local .env key
    monkeypatch.setenv("GROQ_API_KEY", "")
    st.cache_resource.clear()  # resets the shared daily AI counter (models refit, which is fast)


def run_app(**secrets) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    for k, v in secrets.items():
        at.secrets[k] = v
    return at.run()


@pytest.mark.parametrize("outcome", ["Hospitalizations", "ER Visits", "Deaths"])
def test_outcomes_render(outcome):
    at = run_app()
    at.segmented_control(key="outcome").set_value(outcome).run()
    assert not at.exception
    assert len(at.tabs) == 5


def test_filters_render():
    at = run_app()
    at.multiselect(key="states").set_value(["Arizona", "New York"])
    at.slider(key="years").set_value((2010, 2015))
    at.slider(key="pct").set_value(120)
    at.run()
    assert not at.exception


def test_empty_selection_warns():
    at = run_app()
    at.segmented_control(key="outcome").set_value("ER Visits")
    at.multiselect(key="states").set_value(["North Carolina"]).run()  # reports heat, not ER visits
    assert not at.exception
    assert any("No ER visits" in w.value for w in at.warning)


def test_ai_without_key_shows_notice():
    at = run_app()
    assert any("no Groq API key" in i.value for i in at.info)
    assert at.chat_input[0].disabled


def fake_groq(answer="Arizona has the highest rate."):
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
    client = mock.MagicMock()
    client.chat.completions.create.return_value = reply
    return mock.patch("groq.Groq", return_value=client), client


def test_ai_answers_and_counts_down():
    patcher, client = fake_groq()
    with patcher:
        at = run_app(GROQ_API_KEY="test-key", AI_SESSION_LIMIT="2")
        assert any("2 of 2 questions left" in c.value for c in at.caption)
        at.chat_input[0].set_value("Which state is highest?").run()
        assert not at.exception
        assert client.chat.completions.create.call_count == 1
        assert at.session_state.messages[-1]["content"] == "Arizona has the highest rate."
        assert any("1 of 2 questions left" in c.value for c in at.caption)

        at.chat_input[0].set_value("And the lowest?").run()
        assert client.chat.completions.create.call_count == 2
        assert at.chat_input[0].disabled
        assert any("used all the questions" in w.value for w in at.warning)


def test_ai_daily_limit():
    patcher, client = fake_groq()
    with patcher:
        at = run_app(GROQ_API_KEY="test-key", AI_DAILY_LIMIT="1")
        at.chat_input[0].set_value("First").run()
        # A new visitor (fresh session) shares the same daily counter
        other = run_app(GROQ_API_KEY="test-key", AI_DAILY_LIMIT="1")
        assert other.chat_input[0].disabled
        assert any("daily limit" in w.value for w in other.warning)
        assert client.chat.completions.create.call_count == 1
