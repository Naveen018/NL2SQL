import mysql.connector
from flask import Flask, render_template, request, jsonify, abort
from flask_cors import CORS
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
import pandas as pd
from typing import List, Dict, Any
import re
from dotenv import load_dotenv
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../frontend")
CORS(app)

# MySQL connection configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "test123",
    "database": "llm_db",
}

# SQLDatabase setup for LangChain
db_uri = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
db = SQLDatabase.from_uri(db_uri)

# LangChain setup with OpenAI (using GPT-4o-mini)
llm = ChatOpenAI(model="gpt-4o-mini")

# Database schema description
table_info = """
Table: student_courses
Columns: student_id (INT), student_name (VARCHAR), student_email (VARCHAR), course_id (INT), course_name (VARCHAR), course_fees (FLOAT), course_instructor (VARCHAR)

Table: question_answers
Columns: answer_id (INT), course_id (INT), student_id (INT), question_id (INT), question_text (TEXT), options (JSON), correct_answer (VARCHAR), student_answer (VARCHAR), is_correct (BOOLEAN)

Table: exam_results
Columns: result_id (INT), student_name (VARCHAR), course_id (INT), course_name (VARCHAR), result_percentage (FLOAT)
"""

# General prompt template
general_prompt_template = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template="You are a MySQL expert. Given the following database schema:\n{table_info}\n\nGenerate a valid MySQL query for the following natural language request:\n{input}\n\nUse a LIMIT clause with {top_k} if appropriate. Escape reserved keywords (e.g., `rank`) with backticks. Return only the SQL query without any explanation or additional text.",
)

# Specialized prompt for "top X per group" queries
group_prompt_template = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template="You are a MySQL expert. Given the following database schema:\n{table_info}\n\nGenerate a valid MySQL query for the following natural language request:\n{input}\n\nFor queries requesting 'top X' items per group (e.g., per course or subject), use window functions like ROW_NUMBER() to rank items within each group. If the query involves frequency-based metrics (e.g., most wrong answers), aggregate data (e.g., COUNT) by the grouping column (e.g., course_name) and item (e.g., question_id), then rank by the aggregated value in descending order. If the query involves direct value-based metrics (e.g., highest scores or percentages), rank directly by the value (e.g., result_percentage) in descending order without aggregation. Do not join tables unless additional columns are needed (e.g., course_instructor from student_courses; note that exam_results already has course_name). Escape reserved keywords (e.g., `rank`) with backticks. Select the top {top_k} per group, returning the group column (e.g., course_name), item column (e.g., student_name), and the ranked value (e.g., result_percentage or count). Order results by group and ranked value descending. Return only the SQL query without any explanation or additional text.",
)

# Create SQL query chains
general_sql_chain = create_sql_query_chain(
    llm=llm, db=db, prompt=general_prompt_template
)
group_sql_chain = create_sql_query_chain(llm=llm, db=db, prompt=group_prompt_template)


class QueryRequest(BaseModel):
    query: str


def clean_sql_query(sql_query: str) -> str:
    """
    Clean the SQL query by removing Markdown code block markers, extra whitespace, and invalid characters.
    """
    # Remove Markdown markers and other code block indicators
    cleaned_query = re.sub(
        r"```sql\s*|```mysql\s*|```", "", sql_query, flags=re.IGNORECASE
    )
    # Remove extra whitespace and newlines
    cleaned_query = " ".join(cleaned_query.split())
    # Remove any leading/trailing non-SQL characters
    cleaned_query = cleaned_query.strip("` \t\n")
    # Ensure query ends with semicolon
    if not cleaned_query.endswith(";"):
        cleaned_query += ";"
    return cleaned_query


def validate_sql_query(sql_query: str) -> bool:
    """
    Validate the SQL query by attempting a dry run with EXPLAIN.
    """
    try:
        cleaned_query = clean_sql_query(sql_query)
        # Escape reserved keyword 'rank' if not already escaped
        cleaned_query = re.sub(
            r"\bAS rank\b", "AS `rank`", cleaned_query, flags=re.IGNORECASE
        )
        logger.debug(f"Validating SQL: {cleaned_query}")
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN {cleaned_query}")
        cursor.fetchall()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        logger.error(f"SQL validation failed: {e}")
        return False


def execute_query(sql_query: str) -> List[Dict[str, Any]]:
    """
    Execute the SQL query and return results.
    """
    try:
        cleaned_query = clean_sql_query(sql_query)
        # Escape reserved keyword 'rank'
        cleaned_query = re.sub(
            r"\bAS rank\b", "AS `rank`", cleaned_query, flags=re.IGNORECASE
        )
        logger.debug(f"Executing query: {cleaned_query}")
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(cleaned_query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except mysql.connector.Error as e:
        logger.error(f"Database error: {e}")
        raise Exception(f"Database error: {str(e)}")


def suggest_chart_type(df: pd.DataFrame) -> str:
    """
    Suggest a chart type based on the DataFrame structure.
    """
    if df.empty:
        return "none"
    num_columns = len(df.columns)
    num_rows = len(df)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    
    # Faceted bar for grouped ranking queries (e.g., top X per group)
    if num_columns >= 3 and len(numeric_cols) >= 1 and 'course_name' in df.columns and 'student_name' in df.columns:
        return "faceted_bar"

    if num_columns == 2 and len(numeric_cols) == 1 and df.iloc[:, 0].nunique() <= 10:
        return "pie"
    elif num_columns >= 2 and len(numeric_cols) >= 1:
        return "bar"
    elif num_columns == 1 and len(numeric_cols) == 1 and num_rows > 10:
        return "histogram"
    else:
        return "bar"


def extract_top_k(query: str) -> int:
    """
    Extract the number of results requested from the query (e.g., 'Top 100' -> 100).
    Default to 10 if no number is found.
    """
    match = re.search(r"\btop\s+(\d+)\b", query, re.IGNORECASE)
    return int(match.group(1)) if match else 10


@app.route("/", methods=["GET"])
def serve_home():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def process_query():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            abort(400, description="Missing 'query' in request body")

        query = data["query"]
        top_k = extract_top_k(query)
        logger.debug(f"Processing query: {query}, top_k: {top_k}")

        # Use group prompt for "top X per group" queries
        group_keywords = [
            "each course",
            "per course",
            "each subject",
            "by course",
            "by subject",
        ]
        if any(keyword in query.lower() for keyword in group_keywords):
            sql_chain = group_sql_chain
        else:
            sql_chain = general_sql_chain

        sql_query = sql_chain.invoke(
            {"question": query, "top_k": top_k, "table_info": table_info}
        ).strip()
        logger.debug(f"Generated SQL: {sql_query}")

        # Validate SQL query
        if not validate_sql_query(sql_query):
            logger.warning("Invalid SQL query detected, attempting to regenerate")
            sql_query = sql_chain.invoke(
                {
                    "question": query
                    + " (ensure valid MySQL syntax, avoid unnecessary joins, escape reserved keywords like `rank`)",
                    "top_k": top_k,
                    "table_info": table_info,
                }
            ).strip()
            logger.debug(f"Regenerated SQL: {sql_query}")
            if not validate_sql_query(sql_query):
                raise Exception("Generated SQL query is invalid after retry")

        results = execute_query(sql_query)
        df = pd.DataFrame(results)
        chart_type = suggest_chart_type(df)

        response = {
            "sql_query": sql_query,
            "results": results,
            "chart_type": chart_type,
            "columns": list(df.columns) if not df.empty else [],
        }
        logger.debug(f"Response: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
