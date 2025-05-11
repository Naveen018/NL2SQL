# NL2SQL - Natural Language to SQL Query Generator

This project converts natural language queries into SQL queries using advanced language models. It provides a web interface where users can input questions in plain English and get corresponding SQL queries and results.

## Features

- Natural language to SQL query conversion
- Real-time query validation and error fixing
- Interactive web interface using Streamlit
- Support for complex queries including "top N" and grouping operations
- Rate limiting and caching for optimal performance

## Prerequisites

- Python 3.8 or higher
- MySQL Server

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd NL2SQL
```

### 2. Backend Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory with the following variables:
```env
DB_HOST=localhost
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_NAME=llm_db
GROQ_API_KEY=your_groq_api_key
```

### 3. Database Setup

1. Create a MySQL database:
```sql
CREATE DATABASE llm_db;
```

2. Import the database schema (schema.sql will be provided separately)

## Running the Application

### 1. Start the Backend Server

```bash
cd backend
python app.py
```
The backend server will run on http://localhost:5080

### 2. Start the Frontend (Streamlit)

```bash
cd frontend
streamlit run frontend.py
```
The frontend will be available at http://localhost:8501

## Project Structure

```
NL2SQL/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   └── frontend.py
└── README.md
```

## API Endpoints

- `POST /query`: Process natural language queries
  - Request body: `{"query": "your natural language query"}`
  - Response: SQL query, results, and processing steps

## Dependencies

### Backend Dependencies
- Flask
- Flask-CORS
- LangChain
- SQLAlchemy
- pandas
- python-dotenv
- ratelimit
- joblib
- tenacity
- httpx

### Frontend Dependencies
- Streamlit

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For support, please contact naveenv3112000@gmail.com
