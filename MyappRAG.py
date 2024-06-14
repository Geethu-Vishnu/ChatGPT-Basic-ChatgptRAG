import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

st.title("ChatGPT with RAG")

# Set OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

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

# Initialize FAISS vector store and documents
documents = [
    "Artificial Intelligence (AI) has a long history dating back to ancient civilizations. However, modern AI began in the mid-20th century. In 1956, John McCarthy coined the term 'artificial intelligence' at the Dartmouth Conference. Early AI research focused on symbolic methods and problem-solving. The 1980s saw the rise of expert systems, which were designed to mimic human decision-making processes. With the advent of machine learning in the 1990s, AI began to shift towards data-driven approaches. Today, AI encompasses a wide range of technologies, including natural language processing, computer vision, and robotics.",
    "Machine learning is a subset of AI that involves the use of algorithms and statistical models to enable computers to learn from data. Key concepts in machine learning include supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training a model on labeled data, while unsupervised learning deals with finding hidden patterns in unlabeled data. Reinforcement learning, on the other hand, involves training agents to make sequences of decisions by rewarding them for good actions. Important techniques in machine learning include regression, classification, clustering, and neural networks."
]
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_texts(documents, embeddings)

# Define the Conversational Retrieval Chain
retriever = vectorstore.as_retriever(search_type="similarity")

# Manually create a QA chain using prompt templates
qa_prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know. Don't try to make up an answer.
{context}
Question: {question}
Answer:
"""
qa_prompt = PromptTemplate(input_variables=["context", "question"], template=qa_prompt_template)

def qa_chain(context, question):
    prompt = qa_prompt.format(context=context, question=question)
    response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

# Custom Conversational Retrieval Chain
def conversational_retrieval_chain(question):
    retrieved_docs = retriever.invoke(question)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    return qa_chain(context, question)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from OpenAI
    with st.chat_message("assistant"):
        try:
            response = conversational_retrieval_chain(prompt)
            st.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
