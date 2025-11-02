from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import re
import json

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

try:
    import boto3
except Exception:
    boto3 = None

try:
    from tabulate import tabulate
except Exception:
    tabulate = None


load_dotenv()


def get_env(name, default=None):
    return os.getenv(name, default)


PG_USER = get_env("PG_USER", "user")
PG_PASS = get_env("PG_PASS", "pass123")
PG_HOST = get_env("PG_HOST", "localhost")
PG_PORT = get_env("PG_PORT", "5432")
PG_DB = get_env("PG_DB", "fraud_detection_db")


def build_engine(user, password, host, port, db):
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def extract_sql(text_response: str) -> str:
    m = re.search(r"```sql\s*(.*?)```", text_response, flags=re.S | re.I)
    if not m:
        m = re.search(r"```\s*(.*?)```", text_response, flags=re.S)
    if m:
        sql = m.group(1).strip()
    else:
        sql = text_response.strip()
    sql = sql.strip('"')
    return sql


def is_safe_select(sql: str) -> bool:
    sql_clean = sql.strip().lower()
    if sql_clean.count(';') > 1:
        return False
    sql_no_sc = sql_clean.rstrip(';').strip()
    return sql_no_sc.startswith('select')


def ensure_limit(sql: str, default_limit=100):
    if re.search(r"\blimit\b", sql, flags=re.I):
        return sql
    return sql.rstrip().rstrip(';') + f" LIMIT {default_limit};"


def make_bedrock_client():
    token = get_env('AWS_BEARER_TOKEN_BEDROCK')
    if not boto3:
        raise RuntimeError('boto3 is not installed; install with pip install boto3')
    if not token:
        raise RuntimeError('AWS_BEARER_TOKEN_BEDROCK not set in env/.env')
    client = boto3.client(
        service_name='bedrock-runtime',
        region_name=get_env('BEDROCK_REGION', 'us-west-2'),
        aws_session_token=token,
    )
    return client


def query_llm_for_sql(client, model_id: str, user_question: str, schema_description: str) -> str:
    system_msg = (
        "You are an assistant that returns a single SQL SELECT statement for PostgreSQL (no explanations). "
        "Only use the schema provided and produce a valid PostgreSQL SQL SELECT query (single statement). "
        "Return the SQL only, in a single code block or plain text. Do not return any non-SQL text. "
        "Do not use destructive statements (no INSERT/UPDATE/DELETE/ALTER/DROP). "
        "Use proper PostgreSQL syntax. Always quote identifiers if they contain special characters or are reserved words."
    )

    user_prompt = (
        f"PostgreSQL Database Schema:\n{schema_description}\n\nQuestion: {user_question}\n\n"
        "Produce a single PostgreSQL SQL SELECT query that answers the question. Use proper PostgreSQL syntax. Limit results if necessary."
    )

    conversation = [
        {"role": "user", "content": [{"text": system_msg + "\n\n" + user_prompt}]},
    ]

    resp = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    try:
        text_out = resp['output']['message']['content'][0]['text']
    except Exception:
        raise RuntimeError(f'Unexpected response from LLM: {resp}')
    return text_out


def describe_schema(engine):
    q = text("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(q)
            rows = result.fetchall()
    except Exception as e:
        return ""

    schema = {}
    for row in rows:
        # Handle both SQLAlchemy 1.x (tuples) and 2.x (Row objects)
        # Try attribute access first (SQLAlchemy 2.0+ Row objects support this)
        try:
            table = row.table_name
            column = row.column_name
            dtype = row.data_type
        except (AttributeError, TypeError):
            # Fallback to indexing/tuple unpacking (SQLAlchemy 1.x or positional access)
            try:
                table, column, dtype = row
            except (TypeError, ValueError):
                # Last resort: use indices
                table = row[0]
                column = row[1]
                dtype = row[2]
        schema.setdefault(table, []).append((column, dtype))
    parts = []
    for t, cols in schema.items():
        col_desc = ", ".join([f"{c} ({dt})" for c, dt in cols])
        parts.append(f"{t}: {col_desc}")
    return "\n".join(parts)


app = FastAPI(title="DB Query via LLM")

# Enable CORS so browsers can make cross-origin requests (preflight OPTIONS).
# This is permissive for local development; narrow `allow_origins` in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    summarize: Optional[bool] = True
    limit: Optional[int] = 100


@app.on_event("startup")
def startup():
    global engine, schema_description, bedrock_client, model_id
    engine = build_engine(PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_DB)
    # warm check
    try:
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
    except Exception as e:
        # don't crash on startup; endpoints will report connection errors
        engine = None
    schema_description = describe_schema(engine) if engine else ""
    try:
        bedrock_client = make_bedrock_client()
    except Exception as e:
        print(e)
        bedrock_client = None
    model_id = get_env('CLAUDE_MODEL_ID', 'us.anthropic.claude-sonnet-4-20250514-v1:0')


@app.get('/health')
def health():
    return {"ok": True, "db": bool(engine), "bedrock": bool(bedrock_client)}


@app.post('/query')
def query(req: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not available")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    if not bedrock_client:
        raise HTTPException(status_code=503, detail="LLM/BEDROCK not configured on server")

    # First, check if the question is data-related or just a casual conversation
    try:
        is_data_question_prompt = (
            "You are an assistant that determines if a user's question is about analyzing data from a database, "
            "or if it's just casual conversation/greeting/off-topic.\n\n"
            f"User question: {question}\n\n"
            "Respond with ONLY one word: 'yes' if the question is about analyzing data (e.g., asking about transactions, "
            "users, fraud, statistics, counts, averages, etc.), or 'no' if it's casual conversation, greetings, "
            "or unrelated to data analysis."
        )
        conv = [{"role": "user", "content": [{"text": is_data_question_prompt}]}]
        resp_check = bedrock_client.converse(
            modelId=model_id,
            messages=conv,
            inferenceConfig={"maxTokens": 10, "temperature": 0.0},
        )
        is_data_related = resp_check['output']['message']['content'][0]['text'].strip().lower().startswith('yes')
    except Exception:
        # If check fails, assume it's data-related to maintain backward compatibility
        is_data_related = True

    # If not data-related, respond friendly without SQL
    if not is_data_related:
        try:
            friendly_prompt = (
                "You are a friendly and helpful assistant. The user sent you a message. "
                f"IMPORTANT: Respond in the EXACT same language as the user's message. "
                f"If the user wrote in Romanian, respond in Romanian. If in English, respond in English, etc. "
                f"User message: {question}\n\n"
                "Respond naturally and friendly in the same language. Keep it brief (1-2 sentences). "
                "If they're greeting you, greet them back in the same language. "
                "If they're asking something off-topic, be helpful and friendly, but always match their language exactly."
            )
            conv_friendly = [{"role": "user", "content": [{"text": friendly_prompt}]}]
            resp_friendly = bedrock_client.converse(
                modelId=model_id,
                messages=conv_friendly,
                inferenceConfig={"maxTokens": 150, "temperature": 0.7},
            )
            friendly_response = resp_friendly['output']['message']['content'][0]['text'].strip()
            return {"sql": None, "rows": [], "summary": friendly_response}
        except Exception as e:
            # Fallback to friendly response if LLM fails
            return {
                "sql": None,
                "rows": [],
                "summary": "Hello! I'm here to help you analyze your transaction data. Feel free to ask me questions about the database!"
            }

    # 1) get SQL from LLM (for data-related questions)
    try:
        llm_text = query_llm_for_sql(bedrock_client, model_id, question, schema_description)
        sql = extract_sql(llm_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # 2) safety checks
    if not is_safe_select(sql):
        raise HTTPException(status_code=400, detail="Unsafe SQL generated; only single SELECT statements allowed")

    sql_limited = ensure_limit(sql, default_limit=req.limit or 100)

    # 3) execute
    try:
        # Use text() wrapper and execute via connection for SQLAlchemy 2.0+ compatibility
        sql_query = text(sql_limited)
        with engine.connect() as conn:
            result = conn.execute(sql_query)
            # Convert result to DataFrame - this handles SQLAlchemy 2.0+ Row objects correctly
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error executing SQL: {e}")

    rows = df.to_dict(orient='records')

    summary_text = None
    if req.summarize and bedrock_client:
        try:
            preview_df = df.head(10)
            if tabulate:
                preview_text = tabulate(preview_df, headers='keys', tablefmt='psql', showindex=False)
            else:
                preview_text = preview_df.to_csv(index=False)
            try:
                num_stats_df = df.describe(include='number')
                numeric_stats_text = num_stats_df.to_json() if not num_stats_df.empty else '{}'
            except Exception:
                numeric_stats_text = '{}'

            summary_prompt = (
                'You are a helpful data analyst assistant. A user asked you a question about the data. '
                f'IMPORTANT: The user asked their question in this language: "{question}". '
                'You MUST respond in the EXACT same language as the user\'s question. '
                'If the user asked in Romanian, respond in Romanian. If in English, respond in English, etc.\n\n'
                'Here are the results from your analysis:\n\n'
                f'Data preview:\n{preview_text}\n\n'
                f'Statistics: {numeric_stats_text}\n\n'
                'Provide a clear, concise answer (2-4 sentences) that directly answers the user\'s question in simple language. '
                'Focus on the key findings and numbers. Keep it friendly and conversational. '
                'IMPORTANT: Do NOT mention SQL, queries, databases, or any technical implementation details. '
                'Do NOT say things like "SQL executed", "the query", "SELECT", or any database terminology. '
                'Just explain what the data shows in plain, natural language. '
                'Return ONLY the natural-language answer in the same language as the user\'s question, with no technical terms.'
            )
            conv = [{"role": "user", "content": [{"text": summary_prompt}]}]
            resp = bedrock_client.converse(modelId=model_id, messages=conv, inferenceConfig={"maxTokens":256, "temperature":0.3})
            try:
                summary_text = resp['output']['message']['content'][0]['text']
            except Exception:
                summary_text = str(resp)
        except Exception:
            summary_text = None

    return {"sql": sql, "rows": rows, "summary": summary_text}


if __name__ == '__main__':
    # Run with: uvicorn db.query_server:app --reload --port 8000
    print('This file is a FastAPI app. Run with: uvicorn db.query_server:app --reload --port 8000')