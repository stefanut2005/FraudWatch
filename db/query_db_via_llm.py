import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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


def describe_schema(engine):
    """Return a compact string description of tables and columns in public schema."""
    q = text("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(q).fetchall()
    except Exception as e:
        print(f"Could not read schema: {e}")
        return ""

    schema = {}
    for table, column, dtype in rows:
        schema.setdefault(table, []).append((column, dtype))

    parts = []
    for t, cols in schema.items():
        col_desc = ", ".join([f"{c} ({dt})" for c, dt in cols])
        parts.append(f"{t}: {col_desc}")
    return "\n".join(parts)


def extract_sql(text_response: str) -> str:
    # try to find a SQL code block ```sql ... ``` or ``` ... ``` or fallback to the whole response
    m = re.search(r"```sql\s*(.*?)```", text_response, flags=re.S | re.I)
    if not m:
        m = re.search(r"```\s*(.*?)```", text_response, flags=re.S)
    if m:
        sql = m.group(1).strip()
    else:
        sql = text_response.strip()
    # remove leading/trailing quotes
    sql = sql.strip('"')
    return sql


def is_safe_select(sql: str) -> bool:
    # Very simple safety checks: only allow a single statement starting with SELECT
    sql_clean = sql.strip().lower()
    # disallow semicolons (more than one statement) — allow a trailing semicolon but remove it
    if sql_clean.count(';') > 1:
        return False
    # remove trailing semicolon for checking
    sql_no_sc = sql_clean.rstrip(';').strip()
    return sql_no_sc.startswith('select')


def ensure_limit(sql: str, default_limit=100):
    # If query already contains a LIMIT, return as-is; otherwise add a LIMIT
    if re.search(r"\blimit\b", sql, flags=re.I):
        return sql
    # strip trailing semicolon and append limit
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
        "You are an assistant that returns a single SQL SELECT statement (no explanations). "
        "Only use the schema provided and produce a valid SQL SELECT query (single statement). "
        "Return the SQL only, in a single code block or plain text. Do not return any non-SQL text. "
        "Do not use destructive statements (no INSERT/UPDATE/DELETE/ALTER/DROP)."
    )

    user_prompt = (
        f"Schema:\n{schema_description}\n\nQuestion: {user_question}\n\n"
        "Produce a single SQL SELECT query that answers the question. Limit results if necessary."
    )

    # Bedrock Converse currently accepts only 'user' and 'assistant' roles.
    # Put the system instructions together with the user prompt as a single 'user' message.
    conversation = [
        {"role": "user", "content": [{"text": system_msg + "\n\n" + user_prompt}]},
    ]

    resp = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    # Extract text
    try:
        text_out = resp['output']['message']['content'][0]['text']
    except Exception:
        raise RuntimeError(f'Unexpected response from LLM: {resp}')
    return text_out


def main():
    engine = build_engine(PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_DB)

    # quick check
    try:
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
    except Exception as e:
        print('Could not connect to Postgres:', e)
        sys.exit(1)

    schema = describe_schema(engine)
    if not schema:
        print('Warning: could not retrieve schema description or schema is empty.')

    try:
        client = make_bedrock_client()
    except Exception as e:
        print('Unable to create Bedrock client:', e)
        print('You can still run raw SQL locally; set AWS_BEARER_TOKEN_BEDROCK to enable LLM-driven queries.')
        client = None

    model_id = get_env('CLAUDE_MODEL_ID', 'us.anthropic.claude-sonnet-4-20250514-v1:0')

    print('Interactive DB query via LLM. Type your question (or "exit")')
    while True:
        try:
            question = input('\nQuestion> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye')
            break
        if not question:
            continue
        if question.lower() in ('exit', 'quit'):
            print('Goodbye')
            break

        if client:
            try:
                llm_text = query_llm_for_sql(client, model_id, question, schema)
            except Exception as e:
                print('LLM error:', e)
                continue

            sql = extract_sql(llm_text)
            print('\n-- SQL generated by LLM:')
            print(sql)
        else:
            # prompt the user to input SQL directly
            sql = input('Enter SQL to run (only SELECT allowed):\n')

        if not is_safe_select(sql):
            print('Rejected: only a single SELECT statement is allowed.')
            continue

        sql = ensure_limit(sql, default_limit=100)

        # Execute and show results
        try:
            df = pd.read_sql_query(sql, engine)
            pd.set_option('display.max_rows', 200)
            pd.set_option('display.max_columns', 50)
            print('\n== Results ==')
            if df.empty:
                print('(no rows)')
            else:
                # Pretty-print using tabulate if available, otherwise fallback to pandas
                try:
                    if tabulate:
                        print(tabulate(df.head(50), headers='keys', tablefmt='psql', showindex=False))
                    else:
                        print(df.head(50).to_string(index=False))
                except Exception:
                    print(df.head(50).to_string(index=False))

                # concise summary
                print('\n-- Summary --')
                print(f'Rows returned: {len(df)}')
                print(f'Columns: {len(df.columns)} -> {", ".join(df.columns.tolist())}')

                # Automatically generate an LLM natural-language summary if client is available
                if client:
                    try:
                        print('\nGenerating natural-language summary...')
                        # prepare a small human-readable preview of results
                        preview_df = df.head(10)
                        if tabulate:
                            preview_text = tabulate(preview_df, headers='keys', tablefmt='psql', showindex=False)
                        else:
                            # fallback to CSV-like text
                            preview_text = preview_df.to_csv(index=False)

                        # prepare numeric stats if available
                        try:
                            num_stats_df = df.describe(include='number')
                            if num_stats_df.empty:
                                numeric_stats_text = '{}'
                            else:
                                numeric_stats_text = num_stats_df.to_json()
                        except Exception:
                            numeric_stats_text = '{}'

                        # Ask the LLM to produce a human-like answer in the same language as the question.
                        summary_prompt = (
                            'You are a helpful assistant. The user asked a question and ran the following SQL query:\n'
                            f'{sql}\n\nHere is a short preview of the query results:\n{preview_text}\n\n'
                            f'Basic numeric statistics (JSON):\n{numeric_stats_text}\n\n'
                            'Produce a concise, human-friendly answer (3-6 sentences) that explains the result in plain language, '
                            'mentions any obvious notable values, and suggests a sensible follow-up if appropriate. '
                            'Return only the natural-language summary (no SQL, no code blocks). Match the language of the user question.'
                        )

                        conv = [{"role": "user", "content": [{"text": summary_prompt}]}]
                        resp = client.converse(modelId=model_id, messages=conv, inferenceConfig={"maxTokens":256, "temperature":0.3})
                        try:
                            summary_text = resp['output']['message']['content'][0]['text']
                        except Exception:
                            summary_text = str(resp)
                        print('\n== LLM Summary ==')
                        print(summary_text)
                    except Exception as e:
                        print('Error asking LLM for summary:', e)
        except Exception as e:
            print('Error executing SQL:', e)


if __name__ == '__main__':
    main()
