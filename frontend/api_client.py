"""Shared authenticated API transport for local and container deployments."""
import os

import requests as transport
import streamlit as st

API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000/api/v1")


class Client:
    RequestException = transport.RequestException
    exceptions = transport.exceptions

    def request(self, method, url, **kwargs):
        headers = dict(kwargs.pop("headers", {}))
        token = st.session_state.get("api_token", "").strip()
        if token:
            headers["Authorization"] = "Bearer " + token
        return transport.request(method, url, headers=headers, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


client = Client()


def render_auth():
    def clear_actor_state():
        for key in ("messages", "sessions", "current_session", "pending_previews", "selected_eval_run", "confirm_clear"):
            st.session_state.pop(key, None)
        for key in list(st.session_state):
            if key.startswith("upload_"):
                st.session_state.pop(key, None)
    with st.sidebar:
        st.text_input("访问令牌", type="password", key="api_token", on_change=clear_actor_state)
