import streamlit as st
from openai import OpenAI

st.title("Simple ChatGPT Clone")

# Set OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate response from OpenAI
def generate_response(prompt):
    response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Accept user input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from OpenAI
    with st.chat_message("assistant"):
        try:
            response = generate_response(prompt)
            st.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
