from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_conversational_rag_chain(retriever, model_name: str = "llama3.2:latest"):
    """
    Builds a RAG chain with chat history awareness and hybrid retrieval.
    Routes requests through the active ngrok tunnel to local Ollama instance.
    """
    # Active ngrok tunnel base URL targeting local Ollama port 11434
    llm = Ollama(
        base_url="https://embody-monotype-attribute.ngrok-free.dev",
        model=model_name,
        temperature=0.0
    )

    # Prompt to rephrase follow-up questions into standalone queries using history
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
    
    # Chain to produce rephrased standalone questions
    history_aware_retriever = (
        contextualize_q_prompt 
        | llm 
        | StrOutputParser() 
        | retriever
    )

    # Question-answering prompt template
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

    # Helper function to format retrieved documents into a single text block
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Full conversational RAG pipeline execution sequence
    rag_chain = (
        RunnablePassthrough.assign(
            context=history_aware_retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain