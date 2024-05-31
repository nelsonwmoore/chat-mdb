import os

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_url = os.getenv("NEO4J_URI")

# A list of Documents
documents = [
    Document(
        page_content="Text to be indexed",
        metadata={"source": "local"},
    ),
]

# Service used to create the embeddings
embedding_provider = OpenAIEmbeddings()

new_vector = Neo4jVector.from_documents(
    documents,
    embedding_provider,
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password,
    index_name="myVectorIndex",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
    create_id_index=True,
)
