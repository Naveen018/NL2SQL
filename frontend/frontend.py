import streamlit as st
import requests
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="NL2SQL Query Interface", page_icon="🔍", layout="wide")

# Title
st.title("Natural Language to SQL Query Interface")

# Input form
query = st.text_area("Enter your query (e.g., Top 5 students from each course)", height=100)

# Submit button
if st.button("Submit", disabled=not query.strip()):
    with st.spinner("Processing your query..."):
        try:
            # Send query to Flask backend
            response = requests.post(
                "http://localhost:5080/query",
                json={"query": query},
                headers={"Content-Type": "application/json"}
            )

            # Check response status
            if response.status_code != 200:
                st.error(f"Error: HTTP {response.status_code} - {response.json().get('error', 'Unknown error')}")
            else:
                data = response.json()

                # Display processing steps
                if "processing_steps" in data and data["processing_steps"]:
                    st.subheader("Processing Steps")
                    for step in data["processing_steps"]:
                        # Get icon and title for the step
                        icon = {
                            "sql_generator": "🔍",
                            "sql_validator": "✓",
                            "sql_fixer": "🔧",
                            "execute_query": "⚡"
                        }.get(step["tool"], "•")
                        title = {
                            "sql_generator": "Generating SQL Query",
                            "sql_validator": "Validating SQL Query",
                            "sql_fixer": "Fixing SQL Query",
                            "execute_query": "Executing Query"
                        }.get(step["tool"], step["tool"])

                        # Display step in an expander
                        with st.expander(f"{icon} {title}"):
                            st.write("**Input:**")
                            st.code(step["input"] if isinstance(step["input"], str) else str(step["input"]), language="json")
                            st.write("**Output:**")
                            st.code(step["output"] if isinstance(step["output"], str) else str(step["output"]), language="json")

                # Display SQL query
                if "sql_query" in data:
                    st.subheader("Generated SQL Query")
                    st.code(data["sql_query"], language="sql")

                # Display results
                if "results" in data and data["results"] and "columns" in data:
                    st.subheader("Results")
                    if len(data["results"]) > 0:
                        # Convert results to DataFrame for table display
                        df = pd.DataFrame(data["results"], columns=data["columns"])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No results returned.")
                else:
                    st.error("No results or columns returned.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {str(e)}")
        except ValueError as e:
            st.error(f"Error parsing response: {str(e)}")

# Footer
st.markdown("---")
st.write("Built with Streamlit and Flask for NL2SQL processing.")