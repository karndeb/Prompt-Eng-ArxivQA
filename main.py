"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle

# Load the LangChain.
openapi_key = ''

with open("Faiss_Index/faiss_index_aiml.pickle", "rb") as f:
    store = pickle.load(f)

# store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0, openai_api_key=openapi_key), vectorstore=store)


# From here down is all the StreamLit UI.
st.set_page_config(page_title="Arxiv QA Bot", page_icon=":robot:")
st.header("Arxiv QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "What does the duration predictor do?", key="input")
    return input_text


user_input = get_text()

if user_input:
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
