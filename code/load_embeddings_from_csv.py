"""Load embeddings from CSV into Neo4j."""

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

load_dotenv(override=True)

graph = Neo4jGraph()

query = """
    LOAD CSV WITH HEADERS
    FROM 'file:///term-desc-embeddings.csv'
    AS row
    MATCH (t:term {nanoid: row.termid})
    CALL db.create.setNodeVectorProperty(t, 'embedding', apoc.convert.fromJsonList(row.embedding))
    RETURN count(*)
    """


result = graph.query(
    query=query,
)

print(result)
