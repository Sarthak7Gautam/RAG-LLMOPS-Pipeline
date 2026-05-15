from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

configure_prompt_to_context = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "reformulate it into a standalone question that can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"), 
    ]
)

generate_context_aware_answer = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer to any question the user asks \n\n Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),  
    ]
)
