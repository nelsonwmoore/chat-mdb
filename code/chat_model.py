"""Chat model."""

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

chat_llm = ChatOpenAI()

prompt = PromptTemplate(
    template="""
    You are a surfer dude, having a conversation about the surf conditions on the beach.
    Respond using surfer slang.

    Chat History: {chat_history}
    Context: {context}
    Question: {question}
    """,
    input_variables=["chat_history", "context", "question"],
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True,
)

chat_chain = LLMChain(memory=memory, verbose=False, prompt=prompt, llm=chat_llm)

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

while True:
    question = input("> ")
    response = chat_chain.invoke(
        {
            "context": current_weather,
            "question": question,
        },
    )
    print(response["text"])
