"""Retrieve stuff."""

import os  # Import the os module

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_url = os.getenv("NEO4J_URI")

embedding_provider = OpenAIEmbeddings()
chat_llm = ChatOpenAI()

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
    llm=chat_llm,
    retriever=term_desc_vector.as_retriever(),
    verbose=True,
    return_source_documents=True,
)

result = term_retriever.invoke(
    {"query": "What terms are used by the GDC to describe Brain Cancer?"},
)
print(result)

# result = term_desc_vector.similarity_search(
#     "An anatomic site where several different types of cancer can occur.",
# )
# for doc in result:
#     print(doc.metadata["value"], "-", doc.page_content)
