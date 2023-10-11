import openai
import string
import ast
from datetime import timedelta
import os
import pandas as pd
import numpy as np
import random
from urllib import parse
import re
import json
from sqlalchemy import create_engine
import sqlalchemy as sql
from plotly.graph_objects import Figure
import time
import sqlite3

def get_table_schema():
	conn = sqlite3.connect('TechItOut2DB.db')
	c = conn.cursor()

	def sq(str,con=conn):
		return pd.read_sql('''{}'''.format(str), con)
	
	tables_List = sq(
		'''select distinct name
		from sqlite_master
		where type='table';'''
		,conn)

	#Ouptput Variable
	output=[]
	# Initialize variables to store table and column information
	current_table = ""
	columns = []

	for index,row in tables_List.iterrows():
		#print(row["name"])
		#table_schema="DWH"
		table_name_single=row["name"]
		# table_name = f"{table_schema}.{table_name_single}"
		df = sq(f'''PRAGMA table_info({table_name_single});''',conn)
		for index,row in df.iterrows():
			# table_name = f"{table_schema}.{table_name_single}"
			table_name = f"{table_name_single}"
			column_name = row["name"]
			data_type = row["type"]
			if " " in table_name:
				table_name = f"[{table_name}]"
				column_name = row["name"]
			if " " in column_name:
				column_name = f"[{name}]"

			# If the table name has changed, output the previous table's information
			if current_table != table_name and current_table != "":
				output.append(f"table: {current_table}, columns: {', '.join(columns)}")
				columns = []

			# Add the current column information to the list of columns for the current table
			columns.append(f"{column_name} {data_type}")

			# Update the current table name
			current_table = table_name

	# Output the last table's information
	output.append(f"table: {current_table}, columns: {', '.join(columns)}")
	output = "\n".join(output)
	return output


class ChatGPT_Handler:  # designed for chat completion API
    def __init__(
        self,
        gpt_deployment=None,
        max_response_tokens=None,
        token_limit=None,
        temperature=None,
        extract_patterns=None,
    ) -> None:
        self.max_response_tokens = max_response_tokens
        self.token_limit = token_limit
        self.gpt_deployment = gpt_deployment
        self.temperature = temperature
        self.extract_patterns = extract_patterns

    def _call_llm(self, prompt , stop):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_response_tokens,
            stop=stop
        )

        llm_output = response["choices"][0]["message"]["content"]
        return llm_output

    def extract_output(self, text_input):
        output = {}
        if len(text_input) == 0:
            return output
        for pattern in self.extract_patterns:
            if "sql" in pattern[1]:
                sql_query = ""
                sql_result = re.findall(pattern[1], text_input, re.DOTALL)

                if len(sql_result) > 0:
                    sql_query = sql_result[0]
                    output[pattern[0]] = sql_query
                else:
                    return output
                text_before = (
                    text_input.split(sql_query)[0]
                    .strip("\n")
                    .strip("```sql")
                    .strip("\n")
                )

                if text_before is not None and len(text_before) > 0:
                    output["text_before"] = text_before
                text_after = text_input.split(sql_query)[1].strip("\n").strip("```")
                if text_after is not None and len(text_after) > 0:
                    output["text_after"] = text_after
                return output

            if "python" in pattern[1]:
                result = re.findall(pattern[1], text_input, re.DOTALL)
                if len(result) > 0:
                    output[pattern[0]] = result[0]
            else:
                result = re.search(pattern[1], text_input, re.DOTALL)
                if result:
                    output[result.group(1)] = result.group(2)

        return output


class SQL_Query():
    def __init__(
        self,
        system_message="",
        data_sources="",
    ):
        super().__init__(**kwargs)
        if len(system_message) > 0:
            self.system_message = f"""
            {data_sources}
            {system_message}
            """
    conn = sqlite3.connect('TechItOut2DB.db')
    
    def execute_sql_query(str,con=conn):
        return pd.read_sql('''{}'''.format(str), con)

class AnalyzeGPT(ChatGPT_Handler,SQL_Query):
    def __init__(
        self,
        content_extractor,
        # sql_query_tool,
        system_message,
        few_shot_examples,
        st,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        table_schema = get_table_schema()
        system_message = f"""
        <<data_sources>>
        {table_schema}
        {system_message.format()}
        {few_shot_examples}
        """
        self.conversation_history = [{"role": "system", "content": system_message}]
        self.st = st
        self.content_extractor = content_extractor
        # self.sql_query_tool = sql_query_tool

    def get_next_steps(self, updated_user_content, stop):
        old_user_content = ""
        if len(self.conversation_history) > 1:
            old_user_content = self.conversation_history.pop()  # removing old history
            old_user_content = old_user_content["content"] + "\n"
        self.conversation_history.append(
            {"role": "user", "content": old_user_content + updated_user_content}
        )
        n = 0
        try:
            llm_output = self._call_llm(self.conversation_history, stop)

        except Exception as e:
            time.sleep(8)  # sleep for 8 seconds
            while n < 5:
                try:
                    llm_output = self._call_llm(self.conversation_history, stop)
                except Exception as e:
                    n += 1
                    print(
                        "error calling open AI, I am retrying 5 attempts , attempt ", n
                    )
                    time.sleep(8)  # sleep for 8 seconds
                    print(e)

            llm_output = "OPENAI_ERROR"

        output = self.content_extractor.extract_output(llm_output)
        if len(output) == 0 and llm_output != "OPENAI_ERROR":  # wrong output format
            llm_output = "WRONG_OUTPUT_FORMAT"

        return llm_output, output

    def run(self, question: str, show_code, show_prompt, st) -> any:
        import numpy as np
        import plotly.express as px
        import plotly.graph_objs as go
        import pandas as pd

        st.write(f"Question: {question}")

        conn = sqlite3.connect('TechItOut2DB.db')
    
        def execute_sql(str,con=conn):
            return pd.read_sql('''{}'''.format(str), con)

        observation = None

        def show(data):
            if type(data) is Figure:
                st.plotly_chart(data)
            else:
                st.write(data)
            i = 0
            for key in self.st.session_state.keys():
                if "show" in key:
                    i += 1
                self.st.session_state[f"show{i}"] = data
                if type(data) is not Figure:
                    self.st.session_state[f"observation: show_to_user{i}"] = data

        def observe(name, data):
            try:
                data = data[:5]  # limit the print out observation to 15 rows
            except:
                pass
            self.st.session_state[f"observation:{name}"] = data

        max_steps = 3
        count = 1

        finish = False
        new_input = f"Question: {question}"

        while not finish:
            llm_output, next_steps = self.get_next_steps(
                new_input, stop=["Observation:", f"Thought {count+1}"]
            )
            if llm_output == "OPENAI_ERROR":
                st.write(
                    "Error Calling Open AI, probably due to max service limit, please try again"
                )
                break
            elif (
                llm_output == "WRONG_OUTPUT_FORMAT"
            ):  # just have open AI try again till the right output comes
                count += 1
                continue

            new_input += f"\n{llm_output}"
            for key, value in next_steps.items():
                new_input += f"\n{value}"

                if "ACTION" in key.upper():
                    if show_code:
                        st.write(key)
                        st.code(value)
                    observations = []
                    serialized_obs = []
                    try:
                        exec(value, locals())
                        for key in self.st.session_state.keys():
                            if "observation:" in key:
                                observation = self.st.session_state[key]
                                observations.append((key.split(":")[1], observation))
                                if type(observation) is pd:
                                    serialized_obs.append(
                                        (key.split(":")[1], observation.to_string())
                                    )

                                elif type(observation) is not Figure:
                                    serialized_obs.append(
                                        {key.split(":")[1]: str(observation)}
                                    )
                                del self.st.session_state[key]
                    except Exception as e:
                        observations.append(("Error:", str(e)))
                        serialized_obs.append(
                            {"Encounter following error, can you try again?\n:": str(e)}
                        )

                    for observation in observations:
                        st.write(observation[0])
                        st.write(observation[1])

                    obs = (
                        f"\nObservation on the first 10 rows of data: {serialized_obs}"
                    )
                    new_input += obs
                else:
                    st.write(key)
                    st.write(value)
                if "Answer" in key:
                    print("Answer is given, finish")
                    finish = True
            if show_prompt:
                self.st.write("Prompt")
                self.st.write(self.conversation_history)

            count += 1
            if count >= max_steps:
                print("Exceeding threshold, finish")
                break

    def query_run(self, question: str, show_code, show_prompt, st) -> any:
        st.write(f"Question: {question}")

        conn = sqlite3.connect('TechItOut2DB.db')
    
        def execute_sql(str,con=conn):
            return pd.read_sql('''{}'''.format(str), con)

        max_steps = 3
        count = 1

        new_input = f"Question: {question}"
        while count <= max_steps:
            llm_output, next_steps = self.get_next_steps(
                new_input, stop=["Observation:", f"Thought {count+1}"]
            )
            if llm_output == "OPENAI_ERROR":
                st.write(
                    "Error Calling Open AI, probably due to max service limit, please try again"
                )
                break
            elif (
                llm_output == "WRONG_OUTPUT_FORMAT"
            ):  # just have open AI try again till the right output comes
                count += 1
                continue
            output = None
            error = False

            new_input += f"\n{llm_output}"
            for key, value in next_steps.items():
                new_input += f"\n{value}"

                if "SQL" in key.upper():
                    if show_code:
                        st.write("SQL Code")
                        st.code(value)
                    try:
                        output = execute_sql(value)
                    except Exception as e:
                        new_input += (
                            "Encounter following error, can you try again?\n" + str(e)
                        )
                        error = str(e)
                else:
                    if show_code:
                        st.write(value)
            if show_prompt:
                self.st.write("Prompt")
                self.st.write(self.conversation_history)

            if output is not None:
                st.write(output)
                break

            if error:
                st.write(error)

            count += 1
            if count >= max_steps:
                st.write(
                    "Cannot handle the question, please change the question and try again"
                )
