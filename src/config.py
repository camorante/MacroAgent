from __future__ import annotations
import sys
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
import os

from langchain_postgres import PGVectorStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import Connection, AsyncConnection
from psycopg.rows import tuple_row, dict_row

sys.path.append('./')
load_dotenv(find_dotenv())

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_KEY = os.getenv('GOOGLE_API_KEY')
POSTGRESQL_VECTOR = os.getenv('POSTGRESQL_VECTOR')
POSTGRESQL_CHECKPOINT = os.getenv('POSTGRESQL_CHECKPOINT')
POSTGRESQL_DEFAULT = os.getenv('POSTGRESQL_DEFAULT')

def get_openai_llm(model="gpt-5-nano", max_tokens = None) -> ChatOpenAI:
    return ChatOpenAI(model=model, max_tokens=max_tokens)

def get_gemini_llm(
    model: str = "gemini-2.0-flash-lite",
    max_output_tokens: int | None = None,
    temperature: float = 0.7,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

def ensure_database():
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    with Connection.connect(POSTGRESQL_CHECKPOINT, **connection_kwargs) as conn:
        with conn.cursor(row_factory=tuple_row) as cur:
            database_name = POSTGRESQL_CHECKPOINT.split('/')[-1]
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(f'CREATE DATABASE "{database_name}"')

def get_checkpoint_store() -> PostgresSaver:
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    ensure_database()
    conn = Connection.connect(POSTGRESQL_CHECKPOINT, **connection_kwargs)
    saver = PostgresSaver(conn)
    saver.setup()
    return PostgresSaver(conn)

async def get_async_checkpoint_store() -> AsyncPostgresSaver:
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    }

    ensure_database()
    conn = await AsyncConnection.connect(POSTGRESQL_CHECKPOINT, **connection_kwargs)
    saver = AsyncPostgresSaver(conn)
    await saver.setup()
    return saver

"""
def get_vector_store():
    return PGVectorStore(embeddings=embeddings, collection_name='memory_colection', connection=postgres_url_vector, use_jsonb=True)
"""
