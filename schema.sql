create database llm_db;

use llm_db;

CREATE TABLE student_courses (
    student_id INT NOT NULL,
    student_name VARCHAR(100) NOT NULL,
    student_email VARCHAR(255) NOT NULL UNIQUE,
    course_id INT NOT NULL,
    course_name VARCHAR(100) NOT NULL,
    course_fees DECIMAL(10, 2) NOT NULL,
    course_instructor VARCHAR(100) NOT NULL,
    PRIMARY KEY (student_id, course_id),
    INDEX idx_course_id (course_id)
);

CREATE TABLE question_answers (
    answer_id INT PRIMARY KEY,
    course_id INT NOT NULL,
    student_id INT NOT NULL,
    question_id INT NOT NULL,
    question_text TEXT NOT NULL,
    options JSON NOT NULL,
    correct_answer VARCHAR(255) NOT NULL,
    student_answer VARCHAR(255) NOT NULL,
    is_correct BOOLEAN NOT NULL,
    FOREIGN KEY (student_id, course_id) REFERENCES student_courses(student_id, course_id),
    INDEX idx_question_id (question_id)
);


CREATE TABLE exam_results (
    result_id INT PRIMARY KEY,
    student_id INT NOT NULL,
    student_name VARCHAR(100) NOT NULL,
    course_id INT NOT NULL,
    course_name VARCHAR(100) NOT NULL,
    result_percentage DECIMAL(5, 2) NOT NULL,
    FOREIGN KEY (student_id, course_id) REFERENCES student_courses(student_id, course_id),
    INDEX idx_student_name (student_name)
);