import pandas as pd
from faker import Faker
import random
import json
import os

# Initialize Faker for realistic data
fake = Faker()

# Set random seed for reproducibility
random.seed(42)

# Constants
NUM_ENROLLMENTS = 50000  # Students_Courses records
NUM_ANSWERS = 60000      # Questions_Answers records
NUM_RESULTS = 50000      # Exam_Results records

# Course details with fixed course_id
COURSES = [
    {'course_id': 1, 'course_name': 'Python'},
    {'course_id': 2, 'course_name': 'Java'},
    {'course_id': 3, 'course_name': 'C++'},
    {'course_id': 4, 'course_name': 'JavaScript'},
    {'course_id': 5, 'course_name': 'SQL'},
    {'course_id': 6, 'course_name': 'Data Science'},
    {'course_id': 7, 'course_name': 'Machine Learning'}
]
INSTRUCTORS = ['Dr. Smith', 'Prof. Johnson', 'Dr. Lee', 'Prof. Brown', 'Dr. Davis']
FEES = [500.00, 600.00, 750.00, 800.00, 1000.00]

# Question templates for variety
QUESTION_TEMPLATES = {
    'Python': [
        'What is a list in Python?', 'What does len() do?', 'How to define a function in Python?',
        'What is a dictionary in Python?', 'What is the purpose of __init__?'
    ],
    'Java': [
        'What is a Java interface?', 'What does public static void main do?', 'What is a class in Java?',
        'What is polymorphism?', 'What is an abstract class?'
    ],
    'C++': [
        'What is a pointer in C++?', 'What is a reference?', 'What is operator overloading?',
        'What is a virtual function?', 'What is a destructor?'
    ],
    'JavaScript': [
        'What is a closure in JavaScript?', 'What does async/await do?', 'What is a Promise?',
        'What is the DOM?', 'What is event delegation?'
    ],
    'SQL': [
        'What is a primary key?', 'What does JOIN do?', 'What is a subquery?',
        'What is normalization?', 'What is an index?'
    ],
    'Data Science': [
        'What is a pandas DataFrame?', 'What is overfitting?', 'What is a confusion matrix?',
        'What is cross-validation?', 'What is a decision tree?'
    ],
    'Machine Learning': [
        'What is supervised learning?', 'What is a neural network?', 'What is gradient descent?',
        'What is a support vector machine?', 'What is clustering?'
    ]
}

# Options for multiple-choice questions with corresponding correct answers
OPTIONS_WITH_CORRECT = [
    {
        'options': ['A mutable sequence', 'A class', 'A function', 'A loop'],
        'correct': 'A'
    },
    {
        'options': ['Returns length', 'Prints text', 'Loops over items', 'Defines a function'],
        'correct': 'A'
    },
    {
        'options': ['A blueprint for classes', 'A loop', 'A variable', 'A method'],
        'correct': 'A'
    },
    {
        'options': ['True', 'False', 'Maybe', 'None'],
        'correct': 'A'
    },
    {
        'options': ['Increases speed', 'Reduces memory', 'Improves accuracy', 'None of the above'],
        'correct': 'C'
    }
]

# 1. Generate Students_Courses
def generate_students_courses():
    data = []
    student_emails = set()
    used_combinations = set()  # Track (student_id, course_id)
    student_id = 1
    
    while len(data) < NUM_ENROLLMENTS:
        # Generate unique email
        while True:
            email = fake.email()
            if email not in student_emails:
                student_emails.add(email)
                break
        student_name = fake.name()
        
        # Random course
        course = random.choice(COURSES)
        course_id = course['course_id']
        course_name = course['course_name']
        combination = (student_id, course_id)
        
        # Avoid duplicate (student_id, course_id)
        if combination not in used_combinations:
            used_combinations.add(combination)
            data.append({
                'student_id': student_id,
                'student_name': student_name,
                'student_email': email,
                'course_id': course_id,
                'course_name': course_name,
                'course_fees': random.choice(FEES),
                'course_instructor': random.choice(INSTRUCTORS)
            })
            student_id += 1
    
    df = pd.DataFrame(data)
    # Validate unique emails
    if df['student_email'].duplicated().any():
        raise ValueError("Duplicate student emails detected")
    # Validate unique (student_id, course_id)
    if df[['student_id', 'course_id']].duplicated().any():
        raise ValueError("Duplicate (student_id, course_id) detected")
    return df

# 2. Generate Questions_Answers
def generate_questions_answers(students_courses):
    data = []
    answer_id = 1
    # Course-specific question ID counter
    question_id_counter = {course['course_id']: 100 * course['course_id'] for course in COURSES}
    
    # Get valid enrollments
    valid_enrollments = students_courses[['student_id', 'course_id', 'course_name']].to_dict('records')
    
    for _ in range(NUM_ANSWERS):
        # Sample a valid enrollment
        enrollment = random.choice(valid_enrollments)
        student_id = enrollment['student_id']
        course_id = enrollment['course_id']
        course_name = enrollment['course_name']
        
        # Generate question
        question_text = random.choice(QUESTION_TEMPLATES[course_name])
        question_id = question_id_counter[course_id]
        question_id_counter[course_id] += 1  # Increment for next question
        
        # Generate options
        option_set = random.choice(OPTIONS_WITH_CORRECT)
        options = option_set['options']
        correct_answer = option_set['correct']
        options_dict = {chr(65+i): opt for i, opt in enumerate(options)}  # A, B, C, D
        
        # Simulate student answer
        student_answer = random.choice([chr(65+i) for i in range(len(options))])
        is_correct = student_answer == correct_answer
        
        data.append({
            'answer_id': answer_id,
            'course_id': course_id,
            'student_id': student_id,
            'question_id': question_id,
            'question_text': question_text,
            'options': json.dumps(options_dict),
            'correct_answer': correct_answer,
            'student_answer': student_answer,
            'is_correct': is_correct
        })
        answer_id += 1
    
    df = pd.DataFrame(data)
    # Validate foreign key
    merged = df.merge(
        students_courses[['student_id', 'course_id']],
        on=['student_id', 'course_id'],
        how='left',
        indicator=True
    )
    if (merged['_merge'] != 'both').any():
        raise ValueError("Invalid (student_id, course_id) pairs in question_answers")
    return df

# 3. Generate Exam_Results
def generate_exam_results(students_courses):
    data = []
    result_id = 1
    
    # Get valid enrollments
    valid_enrollments = students_courses[['student_id', 'student_name', 'course_id', 'course_name']].to_dict('records')
    
    for _ in range(NUM_RESULTS):
        # Sample a valid enrollment
        enrollment = random.choice(valid_enrollments)
        
        data.append({
            'result_id': result_id,
            'student_id': enrollment['student_id'],
            'student_name': enrollment['student_name'],
            'course_id': enrollment['course_id'],
            'course_name': enrollment['course_name'],
            'result_percentage': round(random.uniform(50.0, 100.0), 2)
        })
        result_id += 1
    
    df = pd.DataFrame(data)
    # Validate foreign key
    merged = df.merge(
        students_courses[['student_id', 'course_id']],
        on=['student_id', 'course_id'],
        how='left',
        indicator=True
    )
    if (merged['_merge'] != 'both').any():
        raise ValueError("Invalid (student_id, course_id) pairs in exam_results")
    return df

# Generate and save CSVs
def main():
    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    students_courses = generate_students_courses()
    questions_answers = generate_questions_answers(students_courses)
    exam_results = generate_exam_results(students_courses)
    
    # Save to CSV
    students_courses.to_csv('data/students_courses.csv', index=False)
    questions_answers.to_csv('data/questions_answers.csv', index=False)
    exam_results.to_csv('data/exam_results.csv', index=False)
    
    print(f"Generated {len(students_courses)} records for Students_Courses")
    print(f"Generated {len(questions_answers)} records for Questions_Answers")
    print(f"Generated {len(exam_results)} records for Exam_Results")

if __name__ == "__main__":
    main()