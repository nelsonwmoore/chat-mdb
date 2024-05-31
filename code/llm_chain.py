"""main"""
import os

from dotenv import load_dotenv
from icecream import ic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI(
    # model="gpt-3.5-turbo-instruct",
    # temperature=0,
)

template = PromptTemplate.from_template(template="""
    You are an expert on refactoring legacy code in Python. You are very familiar
    with Feathers' "Working Effectively with Legacy Code" as well as modern best
    practices in Python and software engineering in general.

    Provide advice regarding the user's question in the form of a numbered task
    list, e.g.
    1. Add Type Hints for all functions and methods that are missing them.
    2. Add unit tests for methods that don't have them to increase test coverage.
    3. ...

    Please help with this question: {question}
    """)

llm_chain = LLMChain(
    llm=llm,
    prompt=template,
    output_parser=StrOutputParser(),
)

response = llm_chain.invoke({
    "question": (
        "What are some techniques for reducting tight coupling?"
    )
})

ic(response.get("text"))
