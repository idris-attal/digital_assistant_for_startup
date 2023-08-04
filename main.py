# python 3.8 (3.8.16) or it doesn't work
# pip install streamlit streamlit-chat langchain python-dotenv
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI

from langchain import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # setup streamlit page
    st.set_page_config(
        page_title="Your Startup Assistant",
        page_icon="🤖 👨‍💻"
    )


def main():
    init()

    chat = ChatOpenAI(temperature=0)

   # addition of the logic prompt template for startup assistant chatbot    
    template = """ Pleae act a startup mentor. 
        {prompt_pattern}
        {history}
        {input}
    """
    p_pattern = """Please ask me two short meanigful questions to answer my following question. When you have enough information to
    answer my question, create an answer to my question with consideration of all information provided to you. Please do not
    generate answer until i did not provide you the answer to the asked questions."""

    template = template.format(prompt_pattern=p_pattern, history="{history}", input="{input}")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=chat)
    messages = conversation.predict(input="")
    # end of code from jupyter




    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=template)
            # SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("Your own startup assistan 🤖👨‍💻")

    # sidebar with user input
    with st.sidebar:
        user_input = st.text_input("Your message: ", key="input")

        # handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(
                AIMessage(content=response.content))

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')


if __name__ == '__main__':
    main()