# Databricks notebook source
# MAGIC %pip install dbtunnel[gradio] aiohttp

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
os.environ['DATABRICKS_TOKEN'] = ""

os.environ['API_ENDPOINT'] = "https://dbc-8d1d0ef1-d7b2.cloud.databricks.com/serving-endpoints/pdf-chatbot-endpoint/invocations"


# COMMAND ----------

import itertools
import gradio as gr
import requests
import json

from gradio.themes.utils import sizes


def respond(message, history):

    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"
    
    print ("#### Message #####")
    print (message)

    local_token = os.getenv('DATABRICKS_TOKEN')
    local_endpoint = os.getenv('API_ENDPOINT')

    if local_token is None or local_endpoint is None:
        return "ERROR missing env variables"

    # Add your API token to the headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {local_token}'
    }

    payload = {
        "inputs" : [message]

    }

    try:
        response = requests.post(
            local_endpoint, json=payload, headers=headers, timeout=100)

        response_json = response.json()
        print ("#### RESPONSE JSON")
        print (response_json)

        response_data = response_json["predictions"][0]  # Updated line to extract the result field
        print(response_data)

    except Exception as error:
        response_data = f"ERROR status_code: {type(error).__name__}"
        # + str(response.status_code) + " response:" + response.text

    # print(response.json())
    return response_data


theme = gr.themes.Soft(
    text_size=sizes.text_sm,radius_size=sizes.radius_sm, spacing_size=sizes.spacing_sm,
)


demo = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
    textbox=gr.Textbox(placeholder="Ask me a question",
                       container=False, scale=7),
    title="pdf Chatbot Demo",
    description="This chatbot is a demo example to interact with an pdf chatbot",
    examples=[["What is DAG in spark?"],],
    cache_examples=False,
    theme=theme,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

# COMMAND ----------

from dbtunnel import dbtunnel
dbtunnel.gradio(demo).run()
