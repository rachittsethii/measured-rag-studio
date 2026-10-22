from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_conversational_rag_chain(retriever, model_name: str = "llama3.2:latest"):
    """
    Builds a RAG chain with chat history awareness and hybrid retrieval.
    Routes requests through active ngrok tunnel to local Mac Ollama instance.
    """
    llm = Ollama(
        base_url="https://embody-monotype-attribute.ngrok-free.dev",
        model=model_name,
        temperature=0.0,
        headers={"ngrok-skip-browser-warning": "true"}  # Bypasses ngrok 403 warning page
    )

    # Prompt to rephrase follow-up questions using chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = (
        contextualize_q_prompt 
        | llm 
        | StrOutputParser() 
        | retriever
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you do not know the answer, say that you "
        "do not know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=history_aware_retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain