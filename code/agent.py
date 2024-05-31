"""Agent."""

import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.tools import Tool
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_url = os.getenv("NEO4J_URI")

embedding_provider = OpenAIEmbeddings()
llm = ChatOpenAI()

prompt = PromptTemplate(
    template="""
    You are an expert on cancer data and data modeling. You answer
    questions about terms related to oncology and cancer data.

    Chat History: {chat_history}
    Question: {question}
    """,
    input_variables=["chat_history", "question"],
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

term_desc_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password,
    index_name="termDescs",
    embedding_node_property="embedding",
    text_node_property="origin_definition",
)

term_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=term_desc_vector.as_retriever(),
    verbose=True,
    return_source_documents=True,
)

chat_chain = LLMChain(memory=memory, prompt=prompt, llm=llm)


def run_retriever(query):
    results = term_retriever.invoke({"query": query})
    # format the results
    return "\n".join(
        [
            doc.metadata["value"] + " - " + doc.page_content
            for doc in results["source_documents"]
        ],
    )


tools = [
    Tool.from_function(
        name="Cancer Chat",
        description=(
            "For when you need to chat about cancer metadata."
            "The question will be a string. "
            "Return a string."
        ),
        func=chat_chain.run,
        return_direct=True,
    ),
    Tool.from_function(
        name="Cancer Term Search",
        description=(
            "For when you need to search for a term used in cancer research."
            "The question will be a string. Return a string."
        ),
        func=run_retriever,
        return_direct=True,
    ),
]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm=llm, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    max_iterations=3,
    verbose=True,
    handle_parsing_errors=True,
)

while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])
