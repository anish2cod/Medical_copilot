# mcp_servers/sql_server.py
from mcp.server.fastmcp import FastMCP
import sqlite3
import os

# Initialize the Server
mcp = FastMCP("PatientDataSQL")

# Define the Path to DB (Go up one level from 'mcp_servers')
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mimic_demo.db"))

@mcp.tool()
def query_mimic_db(query: str) -> str:
    """
    Executes a read-only SQL query on the MIMIC-IV demo database.
    Useful for finding labs, diagnoses, and medications.
    """
    if not os.path.exists(DB_PATH):
        return f"Error: Database not found at {DB_PATH}"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            # Fetch results and headers
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            results = [dict(zip(cols, row)) for row in rows]
            return str(results)
    except Exception as e:
        return f"SQL Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()
