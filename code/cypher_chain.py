"""
Use Cypher chain to generate Cypher queries for a Neo4j graph based on user questions.

Currently failing becauase it infers tags that don't exist from the schema?

To find anatomic sites shared by multiple models' properties, you can use the following Cypher query:

failure:
MATCH (n:node)-[:has_property]->(p:property)-[:has_concept]->(c:concept)-[:has_tag]->(:tag {key: "anatomic_site"})
WITH c, count(DISTINCT n) as numModels
WHERE numModels > 1
RETURN c.value as AnatomicSite, numModels as NumModels
ORDER BY numModels DESC
"""

from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI()

graph = Neo4jGraph()

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer 
questions about cancer metadata and terms.

Property entities generally are connected to terms via value_sets, for example:
    (:property)-[:has_value_set]->(:value_set)-[:has_term]->(:term)

Insructions:
Do not use (:tag) or (:concept) entities in your query at all.

Examples:
    Find terms that are used by the GDC to describe Brain Cancer.
    MATCH (p:property)-[:has_value_set]->(v:value_set)-[:has_term]->(t:term)
    WHERE p.model = "GDC" and t.value = "Brain Cancer"
    RETURN t.value, t.origin_definition, p.model

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
)

cypher_chain.invoke(
    {"query": "What anatomic sites are shared by multiple models' properties?"},
)
