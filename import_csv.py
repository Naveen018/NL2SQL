import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os
import time
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Database connection configuration
db_config = {
    'host': 'localhost',
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'llm_db',
    'raise_on_warnings': True
}

# Get project root and data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
data_dir = os.path.join(project_root, 'data')
logging.info(f"Script directory: {script_dir}")
logging.info(f"Project root: {project_root}")
logging.info(f"Data directory: {data_dir}")

# CSV file paths
student_courses_csv = os.path.join(data_dir, 'students_courses.csv')
question_answers_csv = os.path.join(data_dir, 'questions_answers.csv')
exam_results_csv = os.path.join(data_dir, 'exam_results.csv')

# Verify CSV files exist
for csv_file in [student_courses_csv, question_answers_csv, exam_results_csv]:
    logging.info(f"Checking file: {csv_file}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

def validate_csv(df, table_name, columns):
    """Validate CSV data against schema constraints."""
    # Check columns
    if list(df.columns) != columns:
        raise ValueError(f"CSV columns {list(df.columns)} do not match expected columns {columns}")
    
    # Check for duplicates
    if table_name == 'student_courses':
        if df['student_email'].duplicated().any():
            raise ValueError("Duplicate student_email found in student_courses")
        if df[['student_id', 'course_id']].duplicated().any():
            raise ValueError("Duplicate (student_id, course_id) found in student_courses")
    
    # Check for NaN in NOT NULL columns
    not_null_columns = {
        'student_courses': ['student_id', 'student_name', 'student_email', 'course_id', 'course_name', 'course_fees', 'course_instructor'],
        'question_answers': ['answer_id', 'course_id', 'student_id', 'question_id', 'question_text', 'options', 'correct_answer', 'student_answer', 'is_correct'],
        'exam_results': ['result_id', 'student_id', 'student_name', 'course_id', 'course_name', 'result_percentage']
    }
    for col in not_null_columns[table_name]:
        if df[col].isna().any():
            raise ValueError(f"Null values found in NOT NULL column {col} for {table_name}")

def transform_student_courses(df):
    """Transform student_courses DataFrame."""
    df['student_id'] = df['student_id'].astype(int)
    df['course_id'] = df['course_id'].astype(int)
    df['course_fees'] = df['course_fees'].astype(float).round(2)
    df['student_name'] = df['student_name'].astype(str)
    df['student_email'] = df['student_email'].astype(str)
    df['course_name'] = df['course_name'].astype(str)
    df['course_instructor'] = df['course_instructor'].astype(str)
    return df

def transform_question_answers(df):
    """Transform question_answers DataFrame."""
    df['answer_id'] = df['answer_id'].astype(int)
    df['course_id'] = df['course_id'].astype(int)
    df['student_id'] = df['student_id'].astype(int)
    df['question_id'] = df['question_id'].astype(int)
    df['question_text'] = df['question_text'].astype(str).fillna('')
    df['options'] = df['options'].astype(str).apply(lambda x: x if x and x != 'nan' else '{}')
    df['correct_answer'] = df['correct_answer'].astype(str).fillna('')
    df['student_answer'] = df['student_answer'].astype(str).fillna('')
    df['is_correct'] = df['is_correct'].map({'True': 1, 'False': 0, True: 1, False: 0, 'nan': 0, '': 0}).astype(int)
    # Validate JSON
    for opt in df['options']:
        try:
            json.loads(opt)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in options: {opt}")
    return df

def transform_exam_results(df):
    """Transform exam_results DataFrame."""
    df['result_id'] = df['result_id'].astype(int)
    df['student_id'] = df['student_id'].astype(int)
    df['course_id'] = df['course_id'].astype(int)
    df['student_name'] = df['student_name'].astype(str)
    df['course_name'] = df['course_name'].astype(str)
    df['result_percentage'] = df['result_percentage'].astype(float).round(2)
    return df

def validate_foreign_keys(df, table_name, conn):
    """Validate (student_id, course_id) against student_courses."""
    if table_name in ['question_answers', 'exam_results']:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, course_id FROM student_courses")
        valid_pairs = set(tuple(row) for row in cursor.fetchall())
        cursor.close()
        
        invalid_rows = df[~df[['student_id', 'course_id']].apply(tuple, axis=1).isin(valid_pairs)]
        if not invalid_rows.empty:
            logging.error(f"Invalid (student_id, course_id) pairs in {table_name}:\n{invalid_rows[['student_id', 'course_id']].head()}")
            raise ValueError(f"Found {len(invalid_rows)} invalid (student_id, course_id) pairs in {table_name}")

def import_csv_to_table(csv_file, table_name, columns, transform=None, batch_size=5000):
    """Import CSV data into a MySQL table using batch inserts."""
    logging.info(f"Importing {csv_file} into {table_name}...")
    start_time = time.time()
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Validate CSV
    validate_csv(df, table_name, columns)
    
    # Apply transformations
    if transform:
        df = transform(df)
    
    try:
        # Connect to database
        conn = mysql.connector.connect(**db_config)
        
        # Validate foreign keys for question_answers and exam_results
        if table_name != 'student_courses':
            validate_foreign_keys(df, table_name, conn)
        
        cursor = conn.cursor()
        
        # Prepare SQL insert statement with INSERT IGNORE
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT IGNORE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Batch insert
        total_rows = len(df)
        inserted_rows = 0
        for start in range(0, total_rows, batch_size):
            batch = df.iloc[start:start + batch_size].values.tolist()
            try:
                cursor.executemany(sql, batch)
                inserted_rows += cursor.rowcount
                conn.commit()
                logging.info(f"Imported {min(start + batch_size, total_rows)}/{total_rows} rows")
            except Error as e:
                logging.error(f"Error in batch {start}-{start+batch_size}: {e}")
                logging.error(f"Problematic batch: {batch[:5]}")  # Log first 5 rows
                raise
        
        logging.info(f"Successfully imported {inserted_rows} rows into {table_name} in {time.time() - start_time:.2f} seconds")
        if inserted_rows < total_rows:
            logging.warning(f"Skipped {total_rows - inserted_rows} duplicate rows due to INSERT IGNORE")
        
    except Error as e:
        logging.error(f"Error importing {table_name}: {e}")
        raise
    
    finally:
        cursor.close()
        conn.close()
    
    return inserted_rows

def truncate_tables():
    """Truncate all tables before importing."""
    tables = ['exam_results', 'question_answers', 'student_courses']
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        for table in tables:
            cursor.execute(f"TRUNCATE TABLE {table};")
            logging.info(f"Truncated table {table}")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        conn.commit()
    except Error as e:
        logging.error(f"Error truncating tables: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def main(truncate=False):
    if truncate:
        truncate_tables()
    
    # Import student_courses
    student_courses_columns = [
        'student_id', 'student_name', 'student_email', 'course_id',
        'course_name', 'course_fees', 'course_instructor'
    ]
    import_csv_to_table(student_courses_csv, 'student_courses', student_courses_columns, transform_student_courses)
    
    # Import question_answers
    question_answers_columns = [
        'answer_id', 'course_id', 'student_id', 'question_id', 'question_text',
        'options', 'correct_answer', 'student_answer', 'is_correct'
    ]
    import_csv_to_table(question_answers_csv, 'question_answers', question_answers_columns, transform_question_answers)
    
    # Import exam_results
    exam_results_columns = [
        'result_id', 'student_id', 'student_name', 'course_id', 'course_name', 'result_percentage'
    ]
    import_csv_to_table(exam_results_csv, 'exam_results', exam_results_columns, transform_exam_results)
    
    logging.info("All imports completed successfully.")

if __name__ == "__main__":
    main(truncate=True)  # Set to False to skip truncation