import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Database connection configuration
db_config = {
    'host': 'localhost',
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'llm_db',  # Using llm_db as per your snippet
    'raise_on_warnings': True
}

# Get project root and data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # Script is in project root
data_dir = os.path.join(project_root, 'data')
print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"Data directory: {data_dir}")

# CSV file paths
student_courses_csv = os.path.join(data_dir, 'students_courses.csv')
question_answers_csv = os.path.join(data_dir, 'questions_answers.csv')
exam_results_csv = os.path.join(data_dir, 'exam_results.csv')

# Verify CSV files exist
for csv_file in [student_courses_csv, question_answers_csv, exam_results_csv]:
    print(f"Checking file: {csv_file}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

def import_csv_to_table(csv_file, table_name, columns, transform=None, batch_size=1000):
    """Import CSV data into a MySQL table using batch inserts."""
    print(f"Importing {csv_file} into {table_name}...")
    start_time = time.time()
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Apply transformations
    if transform:
        df = transform(df)
    
    # Handle NaN values
    if table_name == 'question_answers':
        df['question_text'] = df['question_text'].fillna('')
        df['options'] = df['options'].fillna('{}')
        df['correct_answer'] = df['correct_answer'].fillna('')
        df['student_answer'] = df['student_answer'].fillna('')
        df['is_correct'] = df['is_correct'].fillna(0).map({'True': 1, 'False': 0, True: 1, False: 0, 'nan': 0})
    
    # Ensure columns match
    if list(df.columns) != columns:
        raise ValueError(f"CSV columns {list(df.columns)} do not match expected columns {columns}")
    
    try:
        # Connect to database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Prepare SQL insert statement
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Batch insert
        total_rows = len(df)
        for start in range(0, total_rows, batch_size):
            batch = df.iloc[start:start + batch_size].values.tolist()
            cursor.executemany(sql, batch)
            conn.commit()
            print(f"Imported {min(start + batch_size, total_rows)}/{total_rows} rows")
        
        print(f"Successfully imported {total_rows} rows into {table_name} in {time.time() - start_time:.2f} seconds")
        
    except Error as e:
        print(f"Error importing {table_name}: {e}")
        raise
    
    finally:
        cursor.close()
        conn.close()

def transform_question_answers(df):
    """Transform question_answers DataFrame."""
    df['is_correct'] = df['is_correct'].map({'True': 1, 'False': 0, True: 1, False: 0, 'nan': 0})
    return df

def main():
    # Disable foreign key checks
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        conn.commit()
    except Error as e:
        print(f"Error disabling foreign key checks: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
    
    # Import student_courses
    student_courses_columns = [
        'student_id', 'student_name', 'student_email', 'course_id',
        'course_name', 'course_fees', 'course_instructor'
    ]
    import_csv_to_table(student_courses_csv, 'student_courses', student_courses_columns)
    
    # Import question_answers
    question_answers_columns = [
        'answer_id', 'course_id', 'student_id', 'question_id', 'question_text',
        'options', 'correct_answer', 'student_answer', 'is_correct'
    ]
    import_csv_to_table(question_answers_csv, 'question_answers', question_answers_columns, transform_question_answers)
    
    # Import exam_results
    exam_results_columns = [
        'result_id', 'student_name', 'course_id', 'course_name', 'result_percentage'
    ]
    import_csv_to_table(exam_results_csv, 'exam_results', exam_results_columns)
    
    # Re-enable foreign key checks
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        conn.commit()
    except Error as e:
        print(f"Error re-enabling foreign key checks: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
    
    print("All imports completed successfully.")

if __name__ == "__main__":
    main()