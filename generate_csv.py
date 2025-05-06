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

# Options for multiple-choice questions
GENERIC_OPTIONS = [
    ['A mutable sequence', 'A class', 'A function', 'A loop'],
    ['Returns length', 'Prints text', 'Loops over items', 'Defines a function'],
    ['A blueprint for classes', 'A loop', 'A variable', 'A method'],
    ['True', 'False', 'Maybe', 'None'],
    ['Increases speed', 'Reduces memory', 'Improves accuracy', 'None of the above']
]

# 1. Generate Students_Courses
def generate_students_courses():
    data = []
    student_emails = set()
    student_id = 1
    used_combinations = set()
    
    while len(data) < NUM_ENROLLMENTS:
        # Generate unique student
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
        combination = (student_name, course_name)
        
        # Avoid duplicate enrollments
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
    
    return pd.DataFrame(data)

# 2. Generate Questions_Answers
def generate_questions_answers(students_courses):
    data = []
    answer_id = 1
    
    # Get valid student_id and course_id pairs
    valid_enrollments = students_courses[['student_id', 'course_id', 'course_name']].to_dict('records')
    
    for _ in range(NUM_ANSWERS):
        # Sample a valid enrollment
        enrollment = random.choice(valid_enrollments)
        student_id = enrollment['student_id']
        course_id = enrollment['course_id']
        course_name = enrollment['course_name']
        
        # Select a question
        question_text = random.choice(QUESTION_TEMPLATES[course_name])
        question_id = random.randint(1, 1000)  # Simplified question ID
        
        # Generate options
        options = random.choice(GENERIC_OPTIONS)
        options_dict = {chr(65+i): opt for i, opt in enumerate(options)}  # A, B, C, D
        correct_answer = options[0]  # First option is correct for simplicity
        
        # Simulate student answer
        student_answer = random.choice(options)
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
    
    return pd.DataFrame(data)

# 3. Generate Exam_Results
def generate_exam_results(students_courses):
    data = []
    result_id = 1
    
    # Get valid student_id, course_id, and names
    valid_enrollments = students_courses[['student_id', 'course_id', 'student_name', 'course_name']].to_dict('records')
    
    for _ in range(NUM_RESULTS):
        # Sample a valid enrollment
        enrollment = random.choice(valid_enrollments)
        
        data.append({
            'result_id': result_id,
            'student_name': enrollment['student_name'],
            'course_id': enrollment['course_id'],
            'course_name': enrollment['course_name'],
            'result_percentage': round(random.uniform(50.0, 100.0), 2)
        })
        result_id += 1
    
    return pd.DataFrame(data)

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