import csv
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_term_descs(limit=None):
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    driver.verify_connectivity()

    query = """MATCH (t:term) WHERE t.origin_definition IS NOT NULL
    RETURN t.nanoid AS termid, t.value AS value, t.origin_definition as desc"""

    if limit is not None:
        query += f" LIMIT {limit}"

    terms, summary, keys = driver.execute_query(
        query,
    )

    driver.close()

    return terms


def generate_embeddings(file_name, limit=None):
    csvfile_out = open(file_name, "w", encoding="utf8", newline="")
    fieldnames = ["termid", "embedding"]
    output_plot = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
    output_plot.writeheader()

    terms = get_term_descs(limit=limit)

    print(len(terms))

    llm = OpenAI()

    for term in terms:
        print(term["value"])

        plot = f"{term['value']}: {term['desc']}"
        response = llm.embeddings.create(
            input=plot,
            model="text-embedding-ada-002",
        )

        output_plot.writerow(
            {
                "termid": term["termid"],
                "embedding": response.data[0].embedding,
            },
        )

    csvfile_out.close()


if __name__ == "__main__":
    generate_embeddings("./data/term-desc-embeddings.csv, limit=10")
