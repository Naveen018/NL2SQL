from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llama_index.core import SQLDatabase as LlamaSQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai_like import OpenAILike
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import logging
import re
import json
from typing import Dict, List, Any
import pandas as pd
from ratelimit import limits, sleep_and_retry
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app with static and template folders
app = Flask(__name__, 
    static_folder="../frontend",
    static_url_path="",
    template_folder="../frontend")
CORS(app, resources={r"/query": {"origins": ["http://localhost:5080", "http://localhost:3000"]}})

# MySQL connection configuration
db_config = {
    "host": os.getenv('DB_HOST', 'localhost'),
    "user": os.getenv('DB_USER', 'root'),
    "password": os.getenv('DB_PASSWORD', 'test1234'),
    "database": os.getenv('DB_NAME', 'llm_db'),
}

# Initialize LangChain SQLDatabase
db_uri = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
langchain_db = SQLDatabase.from_uri(db_uri)
logger.info(f"Connected to MySQL database: {db_config['database']}")

# Initialize SQLAlchemy engine for execute_query and validate_sql_query
sqlalchemy_engine = create_engine(db_uri)

# Initialize LlamaIndex
llama_db = LlamaSQLDatabase.from_uri(db_uri)
Settings.llm = OpenAILike(
    model="deepseek-r1-distill-llama-70b",
    api_base="https://api.groq.com/openai/v1",
    api_key=os.getenv('GROQ_API_KEY'),
    is_chat_model=True,
    temperature=0.5
)
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=llama_db,
    tables=["exam_results", "student_courses", "question_answers"],
    verbose=True
)

# Initialize LangChain LLM
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.5
)

# Rate limiting: 30 requests per minute (60 seconds)
CALLS = 30
PERIOD = 60

# Cache setup
cache_dir = "./cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir, verbose=0)

# Database schema description (for prompts)
table_info = """
Table: student_courses
Columns: student_id (INT NOT NULL), student_name (VARCHAR(100) NOT NULL), student_email (VARCHAR(255) NOT NULL UNIQUE), course_id (INT NOT NULL), course_name (VARCHAR(100) NOT NULL), course_fees (DECIMAL(10,2) NOT NULL), course_instructor (VARCHAR(100) NOT NULL)
Constraints: PRIMARY KEY (student_id, course_id), UNIQUE (student_email), INDEX idx_course_id (course_id)

Table: question_answers
Columns: answer_id (INT PRIMARY KEY), course_id (INT NOT NULL), student_id (INT NOT NULL), question_id (INT NOT NULL), question_text (TEXT NOT NULL), options (JSON NOT NULL), correct_answer (VARCHAR(255) NOT NULL), student_answer (VARCHAR(255) NOT NULL), is_correct (BOOLEAN NOT NULL)
Constraints: FOREIGN KEY (student_id, course_id) REFERENCES student_courses(student_id, course_id), INDEX idx_question_id (question_id)

Table: exam_results
Columns: result_id (INT PRIMARY KEY), student_id (INT NOT NULL), student_name (VARCHAR(100) NOT NULL), course_id (INT NOT NULL), course_name (VARCHAR(100) NOT NULL), result_percentage (DECIMAL(5,2) NOT NULL)
Constraints: FOREIGN KEY (student_id, course_id) REFERENCES student_courses(student_id, course_id), INDEX idx_student_name (student_name)
"""

# Rate-limited and cached SQL query generation
@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
@memory.cache
def generate_sql_query(query: str) -> str:
    """Generate SQL query using LlamaIndex."""
    try:
        # For "top N in each group" queries, provide explicit context
        if "top" in query.lower() and "each" in query.lower():
            query = f"{query}. Use window functions like ROW_NUMBER() to rank within groups, as shown in schema examples."
        # For "top N in a specific course", include course_name
        elif "top" in query.lower() and " in " in query.lower():
            query = f"{query}. Include course_name in the output for clarity."
        response = sql_query_engine.query(query)
        sql = str(response.metadata["sql_query"]).strip()
        logger.debug(f"Generated SQL: {sql}")
        return sql
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.error(f"Rate limit exceeded in generate_sql_query: {str(e)}")
            raise
        logger.error(f"Failed to generate SQL: {str(e)}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Failed to generate SQL: {str(e)}")
        return f"Error: {str(e)}"

# Rate-limited SQL query validation
@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def validate_sql_query(sql_query: str, user_query: str = "") -> Dict[str, Any]:
    """Validate SQL query syntax and semantics. Returns JSON with 'is_valid' (boolean) and 'error' (string)."""
    try:
        # Handle dictionary input from agent
        if isinstance(sql_query, dict):
            logger.debug(f"Received dict input: {sql_query}")
            user_query = sql_query.get('user_query', user_query)
            sql_query = sql_query.get('sql_query', '')
            # Handle nested dictionary
            if isinstance(sql_query, dict):
                user_query = sql_query.get('user_query', user_query)
                sql_query = sql_query.get('sql_query', '')
        
        logger.debug(f"validate_sql_query input: sql_query={sql_query}, user_query={user_query}")
        cleaned_query = clean_sql_query(sql_query)
        logger.debug(f"Cleaned query for validation: {cleaned_query}")
        
        # Check if query starts with SELECT or WITH
        if not cleaned_query.lower().lstrip().startswith(('select', 'with')):
            return {"is_valid": False, "error": "Query must start with SELECT or WITH"}
        
        # Semantic check for "top N in each group" queries
        if "top" in user_query.lower() and "each" in user_query.lower():
            if not all(keyword in cleaned_query.upper() for keyword in ["ROW_NUMBER()", "PARTITION BY"]):
                return {"is_valid": False, "error": "Query for 'top N in each group' must use ROW_NUMBER() with PARTITION BY"}
        
        # Check for course_name in "top N in a specific course" queries
        if "top" in user_query.lower() and " in " in user_query.lower() and "each" not in user_query.lower():
            if "course_name" not in cleaned_query.lower():
                return {"is_valid": False, "error": "Query for 'top N in a specific course' must include course_name in the output"}
        
        with sqlalchemy_engine.connect() as conn:
            try:
                conn.execute(f"EXPLAIN {cleaned_query}")
                logger.debug(f"Query validated: {cleaned_query}")
                return {"is_valid": True, "error": ""}
            except Exception as e:
                logger.error(f"Validation failed: {str(e)}")
                try:
                    conn.execute(cleaned_query)
                    logger.warning(f"Query passed execution but failed EXPLAIN: {cleaned_query}")
                    return {"is_valid": True, "error": ""}
                except Exception as exec_e:
                    return {"is_valid": False, "error": str(exec_e)}
    except Exception as e:
        logger.error(f"Unexpected validation error: {str(e)}")
        return {"is_valid": False, "error": str(e)}

# Rate-limited and cached SQL query fixing
@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
@memory.cache
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=46),
    retry=retry_if_exception_type(httpx.HTTPStatusError)
)
def fix_sql_query(sql_query: str, error: str = None) -> str:
    """Fix an invalid SQL query based on error message."""
    try:
        # Handle dictionary input from agent
        if isinstance(sql_query, dict):
            logger.debug(f"Received dict input for fixer: {sql_query}")
            error = sql_query.get('error', error)
            sql_query = sql_query.get('sql_query', '')
            # Handle nested dictionary
            if isinstance(sql_query, dict):
                error = sql_query.get('error', error)
                sql_query = sql_query.get('sql_query', '')
        
        if not error:
            error = "Unknown error"
        
        # Clean the input query first
        sql_query = clean_sql_query(sql_query)
        
        # In fix_sql_query
        prompt = ChatPromptTemplate.from_template(
            """
            The following SQL query failed with error: {error}
            Query: {sql_query}
            Schema: {table_info}
            Fix the query to resolve the error and ensure it matches the schema.
            If the error involves ONLY_FULL_GROUP_BY, remove non-unique columns from GROUP BY and use aggregation (e.g., MAX) for non-grouped SELECT columns.
            For 'top N in each group', use ROW_NUMBER() with PARTITION BY and alias as row_num.
            For 'top N in a specific course', include course_name in the output.
            If the error involves a reserved keyword like 'rank', replace it with 'row_num'.
            Return only the corrected SQL query, starting with SELECT or WITH.
            """
        )
        chain = prompt | llm
        fixed_query = chain.invoke({
            "error": error,
            "sql_query": sql_query,
            "table_info": table_info
        }).content.strip()
        
        # Clean and deduplicate the fixed query
        fixed_query = clean_sql_query(fixed_query)
        if fixed_query.count(';') > 1:
            fixed_query = fixed_query.split(';')[0] + ';'
        
        logger.debug(f"Fixed SQL: {fixed_query}")
        return fixed_query
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.error(f"Rate limit exceeded in fix_sql_query: {str(e)}")
            raise
        logger.error(f"Failed to fix SQL: {str(e)}")
        return sql_query
    except Exception as e:
        logger.error(f"Failed to fix SQL: {str(e)}")
        return sql_query

# LangChain tools
from langchain_core.tools import tool

# In sql_generator
@tool
def sql_generator(query: str) -> str:
    """Generate an SQL query from a natural language query."""
    enhanced_query = f"""
    {query}. When using GROUP BY, include only columns that uniquely identify the group (e.g., question_id for questions).
    For non-grouped columns in SELECT, use aggregation functions (e.g., MAX) or ensure they are functionally dependent.
    Comply with MySQL's ONLY_FULL_GROUP_BY mode.
    """
    return generate_sql_query(enhanced_query)

@tool
def sql_validator(sql_query: str, user_query: str = "") -> Dict[str, Any]:
    """Validate an SQL query. Returns JSON with 'is_valid' (boolean) and 'error' (string)."""
    return validate_sql_query(sql_query, user_query)

@tool
def sql_fixer(sql_query: str, error: str = None) -> str:
    """Fix an invalid SQL query based on the error message."""
    return fix_sql_query(sql_query, error)

tools = [sql_generator, sql_validator, sql_fixer]

# LangChain ReAct agent
prompt = ChatPromptTemplate.from_template(
    """
    You are an SQL query assistant. Use these tools:
    {tools}
    Tool names: {tool_names}

    Format:
    Thought: <reasoning>
    Action: <tool_name>
    Action Input: <input>
    Observation: <output>

    Steps:
    1. Use sql_generator to create SQL from user query
    2. Use sql_validator to check query validity
    3. If invalid, use sql_fixer with error message
    4. Return valid query in format:
    Final Answer: ```sql
    <query>
    ```

    Rules:
    - For "top N in each group": Use ROW_NUMBER() with PARTITION BY, alias as row_num
    - For "top N in specific course": Include course_name, use simple SELECT with LIMIT
    - Fix queries missing required fields or window functions
    - Minimize tool calls

    Schema: {table_info}
    Query: {query}
    {agent_scratchpad}
    """
)

# Create the agent with the correct input variables
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Configure the agent executor with proper input handling
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,  # Reduced to minimize API calls
    return_intermediate_steps=True
)

def clean_sql_query(sql_query: str) -> str:
    """Clean SQL query by removing Markdown, comments, and extra whitespace."""
    if not isinstance(sql_query, str):
        logger.error(f"clean_sql_query received non-string input: {type(sql_query)}")
        sql_query = str(sql_query)
    sql_query = re.sub(r'--.*?\n|/\*.*?\*/', '', sql_query, flags=re.DOTALL)
    sql_query = re.sub(r"```sql\s*|```mysql\s*|```", "", sql_query, flags=re.IGNORECASE)
    sql_query = sql_query.strip()
    sql_query = " ".join(sql_query.split())
    sql_query = re.sub(r';+', ';', sql_query.strip("` \t\n;"))
    if not sql_query.endswith(";"):
        sql_query += ";"
    logger.debug(f"Raw query: {sql_query}")
    return sql_query

def execute_query(sql_query: str) -> Dict[str, Any]:
    """Execute SQL query and return results as DataFrame-compatible dict."""
    try:
        cleaned_query = clean_sql_query(sql_query)
        logger.debug(f"Executing query: {cleaned_query}")
        with sqlalchemy_engine.connect() as conn:
            df = pd.read_sql(cleaned_query, conn)
        logger.info(f"Query executed, returned {len(df)} rows")
        return {"results": df.to_dict(orient="records"), "columns": df.columns.tolist()}
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise Exception(f"Database error: {str(e)}")

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
@app.route("/query", methods=["POST"])
def process_query():
    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        if not data or "query" not in data:
            abort(400, description="Missing 'query' in request body")

        query = data["query"]
        logger.debug(f"Processing query: {query}")
        
        # Initialize steps tracking
        processing_steps = []

        # Run LangChain ReAct agent with the correct input variables
        try:
            result = agent_executor.invoke({
                "query": query,
                "table_info": table_info,
                "agent_scratchpad": ""
            })
            
            # Extract steps from intermediate steps
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if isinstance(step, tuple) and len(step) == 2:
                        action, observation = step
                        if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                            step_info = {
                                "tool": action.tool,
                                "input": action.tool_input,
                                "output": observation
                            }
                            processing_steps.append(step_info)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded in agent_executor: {str(e)}")
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
            logger.error(f"Agent execution failed: {str(e)}")
            return jsonify({"error": "Failed to process query. Please try again with a simpler query."}), 500
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            return jsonify({"error": "Failed to process query. Please try again with a simpler query."}), 500

        # Extract SQL query from agent's response
        response_text = result["output"].strip()
        logger.debug(f"Agent response: {response_text}")

        # Look for SQL query in Final Answer section
        final_answer_match = re.search(r'Final Answer:\s*```sql\n?(.*?)\n?```', response_text, re.DOTALL)
        if final_answer_match:
            sql_query = final_answer_match.group(1).strip()
        else:
            # Check for error message
            if response_text.startswith("Error:") or "Agent stopped" in response_text:
                logger.error(f"Agent failed to produce a valid query: {response_text}")
                return jsonify({"error": "Failed to generate a valid SQL query. Please try rephrasing your question."}), 500
            # Try fallback regex for SELECT/WITH query
            sql_match = re.search(r'```sql\n?(.*?)\n?```|SELECT.*?;|WITH.*?;', response_text, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1) or sql_match.group(0).strip()
            else:
                logger.error(f"Could not extract SQL query from response: {response_text}")
                return jsonify({"error": "Failed to generate a valid SQL query. Please try rephrasing your question."}), 500

        # Clean and deduplicate the final query
        sql_query = clean_sql_query(sql_query)
        if sql_query.count(';') > 1:
            sql_query = sql_query.split(';')[0] + ';'
        
        logger.debug(f"Final SQL query: {sql_query}")

        # Execute query
        try:
            query_result = execute_query(sql_query)
            # Add execution step
            processing_steps.append({
                "tool": "execute_query",
                "input": sql_query,
                "output": f"Successfully executed query, returned {len(query_result['results'])} rows"
            })
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return jsonify({"error": f"Failed to execute query: {str(e)}"}), 500

        # Prepare response
        response = {
            "sql_query": sql_query,
            "results": query_result["results"],
            "columns": query_result["columns"],
            "processing_steps": processing_steps
        }
        logger.debug(f"Response: {response}")
        return jsonify(response)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.error(f"Rate limit exceeded in process_query: {str(e)}")
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        logger.exception(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def serve_home():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5080, debug=True)