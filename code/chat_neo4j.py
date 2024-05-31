"""Neo4j Connect."""

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

load_dotenv(override=True)

graph = Neo4jGraph()

query = """MATCH (t:term) where t.embedding is not null return t limit 5"""
create_vector_index = """
    CREATE VECTOR INDEX termDescs IF NOT EXISTS
    FOR (t:term)
    ON (t.embedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }}"""
show_indexes = """SHOW INDEXES"""

result = graph.query(
    query=query,
)

print(result)
