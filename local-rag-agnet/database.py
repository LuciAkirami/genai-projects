import psycopg
from psycopg.rows import dict_row

# DB Parameters
DB_PARAMS = {
    "dbname": "memory_agent",
    "user": "example_user",
    "password": "12345",
    "host": "localhost",
    "port": "5432",
}


# fetch the conversation from the database
def fetch_conversations():
    conn = psycopg.connect(**DB_PARAMS)

    with conn.cursor(row_factory=dict_row) as cursor:
        fetched_data = cursor.execute("SELECT * FROM conversations;")
        conversation = fetched_data.fetchall()

    conn.close()

    return conversation


# save a conversation in the database
def save_conversation(prompt, response):
    conn = psycopg.connect(**DB_PARAMS)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            INSERT INTO conversations (timestamp, prompt, response)
            VALUES (CURRENT_TIMESTAMP, %s, %s);
            """,
            (prompt, response),
        )
        conn.commit()
    conn.close()
