import pathlib
import tempfile
from io import BytesIO

import streamlit as st
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.llms import OpenAI

st.set_page_config(page_title="Chat with Documents", page_icon="🔥")
st.title("Chat with Documents")


@st.cache_resource(show_spinner=False)
def build_chat_engine(file: BytesIO, api_key: str) -> CondenseQuestionChatEngine:
    with st.spinner("Loading and indexing the document..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = pathlib.Path(temp_dir) / file.name
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())
            reader = SimpleDirectoryReader(input_files=[temp_file_path])
            documents = reader.load_data()

        llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        return index.as_chat_engine(chat_mode="condense_question", verbose=True)


openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="Enter your OpenAI API key",
    help="You can get your API key from https://platform.openai.com/account/api-keys.",
)

if not openai_api_key:
    st.warning(
        "Please enter your OpenAI API key in the sidebar. "
        "You can get a key at https://platform.openai.com/account/api-keys."
    )

uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
)

if not openai_api_key or not uploaded_file:
    st.stop()

chat_engine = build_chat_engine(uploaded_file, openai_api_key)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me questions about the uploaded document!",
        },
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_query := st.chat_input("Ask questions about the document..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"), st.spinner("Generating response..."):
        response = chat_engine.chat(user_query).response
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
