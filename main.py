#rag+llm
import asyncio  # Required for async sleep
import io
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, TypedDict

# Load your CSV or Excel
import chainlit as cl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate,
                                    PromptTemplate)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (ConfigurableField, RunnableBranch,
                                      RunnableLambda, RunnableParallel,
                                      RunnablePassthrough)
from langchain_core.runnables.config import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from yfiles_jupyter_graphs import GraphWidget

embeddings = FastEmbedEmbeddings()
llm = ChatGroq(model_name="llama3-8b-8192", api_key='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')
class GraphState(TypedDict):
    question: Optional[str] = None
    classification: Optional[str] = None
    response: Optional[str] = None
    length: Optional[int] = None
    greeting: Optional[str] = None

workflow = StateGraph(GraphState)



def classify(question):
    # Use invoke() and directly access the content of the response
    response = llm.invoke([HumanMessage(content="classify intent of given input as normal query or data retrieval, data related stuffs. there are only two option either 'normal_query' or 'data_retrieval' .Output just the class. Input: {}".format(question))])
    print('clasify',response.content.strip())
    return response.content.strip()
  # Directly access the content of the AIMessage

def classify_input_node(state):
    question = state.get('question', '').strip()
    classification = classify(question)
    print("classification", classification)
    return {"classification": classification}


def handle_greeting_node(state):
    question = state.get('question', '').strip()
    response = llm.invoke([HumanMessage(content='''Answer the user in formal way.
                                        Dont add an extra information or stuff
                                        Try to answer in lesser words
                                        its can be greeting, talkitive response or FAQs
                                        be friendly but not talktive
                                        guide user as per his query
                                        Input: {}'''.format(question))])
    return {'response':response.content.strip()}  # Directly access the content of the AIMessage


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''1. You are an assistant that converts user queries into executable Python code.
            2. Provide code only, without comments or additional information.
            3. Start your response with the code directly.
            4. Ensure the code is accurate and executable.
            5. Automatically select columns based on user input.
            6. For visualizations, use Plotly code only.
            7. Do not include the word 'python' in the output.
            8. Use print() for non-visualization tasks, since this isn't a Jupyter or Colab notebook.
            9. The dataset is named 'df', and you already know the column names.
            10. If clarification is needed, return 'None'.
            11. Always provide code, regardless of the query.
            12. Avoid using black in plots.
            13. use proper plots what user is asking to
            14. Most Important: What not to do: when user asking for a specific plot and you are giving another irrelevant plot based on user query.
                eg: give me a scatter plot of longitude and latitude
                    instead you must give a scatter plot of longitude and latitude not a any other plot
            15. Don't do like :
                import plotly.express as px
                print(px.scatter(df.head(50), x='Captured Strip ', y='Captured Strip '))
                
                instead do:
                import plotly.express as px
                fig = px.scatter(df, x='longitude', y='latitude')
                fig.show()
                ''',
        ),
        ("human", "{input}"),
    ]
)

            # 16. just for your understanding about column names 
                
            #     do not directly give this column name if user asking for column names, instead give code like print(df.columns)

llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-8b-8192",
    api_key='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'
)

# Invoke the LLM chain
chain = prompt | llm

# Function to extract and run the code, and return the output or figure
def get_fig_from_code(code, df):
    local_variables = {'df': df}
    output_buffer = io.StringIO()  # Buffer to capture printed output
    sys.stdout = output_buffer  # Redirect stdout to capture print statements

    try:
        exec(code, {}, local_variables)
    except Exception as e:
        sys.stdout = sys.__stdout__  # Reset stdout
        return str(e)
    finally:
        sys.stdout = sys.__stdout__  # Reset stdout after execution
    
    # Check if there's a figure created in local variables
    for var in local_variables.values():
        if isinstance(var, go.Figure):  # Check for a Plotly figure
            return var
    
    # If no plot is created, return the captured output
    output = output_buffer.getvalue()
    return output if output.strip() else "Code executed but no output produced."



def handle_RAG(state):
    question = state.get('question', '').strip()
    print('handle Rag ',question)
    return {"response": question,"length":len(question)}
        
    
def bye(state):
    return{"greeting":"The graph has finished"}

workflow.add_node("classify_input", classify_input_node)
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_RAG", handle_RAG)
workflow.add_node("bye", bye)
workflow.set_entry_point("classify_input")
workflow.add_edge('handle_greeting', END)
workflow.add_edge('bye', END)
def decide_next_node(state):
    classification = state.get('classification')
    if classification == "normal_query":
        return "handle_greeting"
    elif classification in ["data_related",'RAG', "data_retrieval"]:  # Ensure varied classifications lead to different nodes
        return "handle_RAG"
    else:
        print("Unknown classification, terminating flow.")
        return "bye"  # Ensure a termination condition


def check_RAG_length(state):
    return "handle_RAG" if state.get("length")>500 else "bye"

workflow.add_conditional_edges(
    "classify_input",
    decide_next_node,
    {
        "handle_greeting": "handle_greeting",
        "handle_RAG": "handle_RAG"
    }
)

workflow.add_conditional_edges(
    "handle_RAG",
    check_RAG_length,
    {
        "bye": "bye",
        "handle_RAG": "handle_RAG"
    }
)
app = workflow.compile()
config = RunnableConfig(recursion_limit=10000)
print(config)

import re

import numpy as np

#you can skip this
def dms_to_decimal(degree_str):
    if isinstance(degree_str, str):  # Check if it's a string
        try:
            lat_str, lon_str = degree_str.split('/')
            # Convert latitude
            lat_match = re.match(r"(\d+)°(\d+)'(\d+)''([NSEW])", lat_str)
            if lat_match:
                lat_degrees = float(lat_match.group(1)) + float(lat_match.group(2)) / 60 + float(lat_match.group(3)) / 3600
                if lat_match.group(4) in ['S']:
                    lat_degrees = -lat_degrees
            else:
                lat_degrees = np.nan  # Assign NaN for invalid formats
            
            # Convert longitude
            lon_match = re.match(r"(\d+)°(\d+)'(\d+)''([NSEW])", lon_str)
            if lon_match:
                lon_degrees = float(lon_match.group(1)) + float(lon_match.group(2)) / 60 + float(lon_match.group(3)) / 3600
                if lon_match.group(4) in ['W']:
                    lon_degrees = -lon_degrees
            else:
                lon_degrees = np.nan  # Assign NaN for invalid formats
            
            return lat_degrees, lon_degrees
        except Exception as e:
            print(f"Error processing {degree_str}: {e}")
            return np.nan, np.nan  # Return NaN if there's an error
    else:
        return np.nan, np.nan  # Handle non-string types



@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    global last_image_mod_time
    last_image_mod_time = time.time()

    # File upload process
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a CSV or Excel file to begin!", 
            accept=["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
            max_size_mb=100,
            timeout=180,
        ).send()

    # Load the uploaded file
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    csv_file = (file.path)
    
    # Determine the file type and read accordingly
    if file.name.endswith('.csv'):
        df = pd.read_csv(csv_file, encoding="utf-8")
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(csv_file)
#    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S:%f', errors='coerce')  #based on your requirements
 #   df[['latitude', 'longitude']] = df['(Lat/Lon)'].apply(dms_to_decimal).apply(pd.Series)
    df.columns = df.columns.str.strip()

    cl.user_session.set('data', df)
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    question = message.content
    output1=app.invoke({'question':question,'length':3},config)
    print(len(output1['response']),len(question))
    print("Generated:\n", output1['response'])
    if str(question.replace(' ',''))==str(output1['response'].replace(' ','')):
        p = chain.invoke({
                "input": output1['response']
            })

        generated_code = p.content
        print('generated code',generated_code)
        df = cl.user_session.get('data')
        output = get_fig_from_code(generated_code, df)

        # Send the output or plot based on what was generated
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    '''1. You are an assistant that converts pandas code output into a readable, formal format.
                    2. Provide the output in a clear, well-presented way.
                    3. Respond with only the output, but you may add a brief description.
                    4. If the output is a DataFrame, display it in an aesthetically improved format.
                    5. If you receive code as output, say "I didn't get it, could you please clarify?".
                    6. Be friendly in your responses.
                    7. If an error occurs, ask for user clarification based on the original question in one line.
                    This is the user's question: {user}''',
                ),
                ("human", "{input}"),
            ]
        )



        # Invoke the LLM chain
        chain2 = prompt | llm
        p = chain2.invoke({
            'user':question,
            "input": output,
        })


        search_result= p.content
        print('last result',search_result)
        if isinstance(output, go.Figure):  # If output is a Plotly figure
            elements = [cl.Plotly(name="chart", figure=output, display="inline",size='large')]
            await cl.Message(content="Here is your plot", elements=elements).send()

        else:
            await cl.Message(content=str(search_result)).send()
    else:
        await cl.Message(content=str(output1['response'])).send()  # Show the printed output
