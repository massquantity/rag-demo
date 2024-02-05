import streamlit as st


def sidebar_params():
    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Enter your OpenAI API key",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",
        )
        temperature = st.slider(
            "LLM Temperature", min_value=0.0, max_value=2.0, step=0.1, value=1.0
        )

    if not openai_api_key:
        st.warning(
            "Please enter your OpenAI API key in the sidebar. "
            "You can get a key at https://platform.openai.com/account/api-keys."
        )
    return openai_api_key, temperature
