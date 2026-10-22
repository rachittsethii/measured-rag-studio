from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_conversational_rag_chain(retriever, model_name: str = "llama3.2:latest"):
    """Builds a RAG chain with chat history awareness."""
    llm = Ollama(model=model_name, temperature=0.0)

    # Prompt to rephrase follow-up questions using chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # Chain to rephrase questions
    history_aware_retriever = (
        contextualize_q_prompt 
        | llm 
        | StrOutputParser() 
        | retriever
    )

    # RAG prompt with citations and history placeholder
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you do not know the answer, say that you do not know. "
        "Always cite source chunk IDs in square brackets like [chunk_id: ...] when stating facts.\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Final RAG chain
    def format_docs(docs):
        formatted = []
        for doc in docs:
            cid = doc.metadata.get("chunk_id", "unknown")
            formatted.append(f"[chunk_id: {cid}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(history_aware_retriever.invoke(x))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain