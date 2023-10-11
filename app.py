import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from analyze import AnalyzeGPT, SQL_Query, ChatGPT_Handler
import openai
from pathlib import Path
from dotenv import load_dotenv
import os
import datetime
import sqlite3


########### Ask api key
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
openai.api_key = openai_api_key


max_response_tokens = 1500
token_limit = 6000
temperature = 0.2

Enter = "False"
def runapp():
    Enter = "True"

st.markdown(
    """# **Database OpenAI Assistant**
This is an experimental assistant that requires OpenAI access. The app demonstrates the use of OpenAI to support getting insights from Database by just asking questions. The assistant can also generate SQL and Python code for the Questions.
"""
)

col1, col2 = st.columns((3, 1))

with st.sidebar:
    options = ("SQL Assistant", "Data Analysis Assistant")
    index = st.radio(
        "Choose the app", range(len(options)), format_func=lambda x: options[x]
    )
    if index == 0:
        system_message = """
        You are an agent designed to interact with a Sqlite3 with schema detail in Sqlite3.
        Given an input question, create a syntactically correct Sqlite3 query to run, then look at the results of the query and return the answer.
        You can order the results by a relevant column to return the most interesting data in the database.
        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        Remember to format SQL query as in ```sql\n SQL QUERY HERE ``` in your response.

        """
        few_shot_examples = ""
        extract_patterns = [("sql", r"```sql\n(.*?)```")]
        extractor = ChatGPT_Handler(extract_patterns=extract_patterns)
        
        show_code = st.checkbox("Show code", value=True)
        show_prompt = st.checkbox("Show prompt", value=True)
        question = st.sidebar.text_area("Ask me a question")
        

    elif index == 1:
        system_message = """
        You are a smart AI assistant to help answer business questions based on analyzing data. 
        You can plan solving the question with one or multiple thought step. At each thought step, you can write python code to analyze data to assist you. Observe what you get at each step to plan for the next step.
        You are given following utilities to help you retrieve data and communicate your result to end user.
        1. execute_sql(sql_query: str): A Python function can query data from the Sqlite3 given a query which you need to create. The query has to be syntactically correct for Sqlite3 and only use tables and columns under <<data_sources>>. The execute_sql function returns a Python pandas dataframe contain the results of the query.
        2. Use plotly library for data visualization. 
        3. Use observe(label: str, data: any) utility function to observe data under the label for your evaluation. Use observe() function instead of print() as this is executed in streamlit environment. Due to system limitation, you will only see the first 10 rows of the dataset.
        4. To communicate with user, use show() function on data, text and plotly figure. show() is a utility function that can render different types of data to end user. Remember, you don't see data with show(), only user does. You see data with observe()
            - If you want to show  user a plotly visualization, then use ```show(fig)`` 
            - If you want to show user data which is a text or a pandas dataframe or a list, use ```show(data)```
            - Never use print(). User don't see anything with print()
        5. Lastly, don't forget to deal with data quality problem. You should apply data imputation technique to deal with missing data or NAN data.
        6. Always follow the flow of Thought: , Observation:, Action: and Answer: as in template below strictly. 

        """

        few_shot_examples = """
        <<Template>>
        Question: User Question
        Thought 1: Your thought here.
        Action: 
        ```python
        #Import neccessary libraries here
        import numpy as np
        #Query some data 
        sql_query = "SOME SQL QUERY"
        step1_df = execute_sql(sql_query)
        # Replace 0 with NaN. Always have this step
        step1_df['Some_Column'] = step1_df['Some_Column'].replace(0, np.nan)
        #observe query result
        observe("some_label", step1_df) #Always use observe() instead of print
        ```
        Observation: 
        step1_df is displayed here
        Thought 2: Your thought here
        Action:  
        ```python
        import plotly.express as px 
        #from step1_df, perform some data analysis action to produce step2_df
        #To see the data for yourself the only way is to use observe()
        observe("some_label", step2_df) #Always use observe() 
        #Decide to show it to user.
        fig=px.line(step2_df)
        #visualize fig object to user.  
        show(fig)
        #you can also directly display tabular or text data to end user.
        show(step2_df)
        ```
        Observation: 
        step2_df is displayed here
        Answer: Your final answer and comment for the question
        <</Template>>

        """

        extract_patterns = [
            ("Thought:", r"(Thought \d+):\s*(.*?)(?:\n|$)"),
            ("Action:", r"```python\n(.*?)```"),
            ("Answer:", r"([Aa]nswer:) (.*)"),
        ]
        extractor = ChatGPT_Handler(extract_patterns=extract_patterns)
        
        show_code = st.checkbox("Show code", value=True)
        show_prompt = st.checkbox("Show prompt", value=True)
        question = st.sidebar.text_area("Ask me a question")
    
         
    if question:
        # conn = sqlite3.connect('chinook.db')
        analyzer = AnalyzeGPT(
            content_extractor=extractor,
            # sql_query_tool=conn,
            system_message=system_message,
            few_shot_examples=few_shot_examples,
            st=st,
            max_response_tokens=max_response_tokens,
            token_limit=token_limit,
            temperature=temperature,
        )
        if index == 0:
            analyzer.query_run(question, show_code, show_prompt, col1)
        elif index == 1:
            analyzer.run(question, show_code, show_prompt, col1)
        else:
            st.error("Not implemented yet!")


