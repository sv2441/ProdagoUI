## import the CSV file and output the CSV file 

import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import csv
import os
import base64


load_dotenv()

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
    
chat_llm = ChatOpenAI(temperature=0.0)


def dict_to_csv(data, filename, append=False):
    mode = 'a' if append else 'w'
    with open(filename, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not append:
            writer.writeheader()
        writer.writerow(data)

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="final.csv">Download CSV File</a>'
    return href

def convert_dict_to_csv(data_dict):
    with open('data21.csv', 'a', newline='') as csvfile:
        fieldnames = ['OP type', 'Activity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if the file is empty and write the header only if it's empty
        is_file_empty = csvfile.tell() == 0
        if is_file_empty:
            writer.writeheader()

        for key, value in data_dict.items():
            if isinstance(value, list):
                for item in value:
                    writer.writerow({'OP type': key, 'Activity': item})
            else:
                writer.writerow({'OP type': key, 'Activity': value})



def result(df):
    
    #####output parser #############################################

    Action_schema = ResponseSchema(name="Action",
                             description="List broad actionables relating to requirements. Classify each actionable into Requirements or Support practices.")

    response_schemas = [ Action_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    ###########################################################################

    title_template = """ \ You are an AI Governance bot. Just Execute the set of steps.
                1)provide List broad actionables relating to "{topic}".Classify each actionable into Requirements or Support practices. 
                If the "{topic}" demands some actions, that is referred as 'Requirement'. 
                If the "{topic}" expects an outcome, the actionables that will help achieving that is referred as 'Support Practices'.
                Ensure that the actionables are activites and Ensure each point has an action word, the subject and the activity.
               {format_instructions}
                """


    prompt = ChatPromptTemplate.from_template(template=title_template)

    
    
    df2 = df
    for index, row in df.iterrows():
        messages = prompt.format_messages(topic=row, format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        convert_dict_to_csv(data)
    data21 = pd.read_csv('data21.csv')
    results = pd.concat([df, data21], axis=1).fillna(0)
    results.to_csv('final21.csv')
    data21.to_csv('data21.csv')
    
    # final = pd.read_csv('final.csv')
    st.subheader("Final Result")
    st.dataframe(data21)
    st.markdown(get_download_link(data21), unsafe_allow_html=True)


def results2(df):
    summary_schema = ResponseSchema(name="Summary",
                             description="Summary of Action Associated in Text in 15 to 20 words.")

    response_schemas = [summary_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    title_template = """ \ You are an AI Governance bot.
                Summarize actions associated with the following statement in 15-20 words paragraphin "{topic}"
                {format_instructions}
                """

    prompt = ChatPromptTemplate.from_template(template=title_template)
    # df=pd.read_csv("final1.csv")
    df2=df['Activity']
    for i in range(len(df2)):
        messages = prompt.format_messages(topic=df2[i], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        dict_to_csv(data, 'data22.csv', append=True)
    data22 = pd.read_csv('data22.csv', names=['Description'])
    result = pd.concat([df, data22], axis=1)
    output= pd.concat([df2, data22], axis=1)
    result.to_csv('final22.csv')
    data22.to_csv('data22.csv')
    
    st.subheader("Summary Result")
    st.dataframe(output)
    st.markdown(get_download_link(data22), unsafe_allow_html=True)
    

def results4(df):
    Artefact_basis_schema = ResponseSchema(name="Artefact Name",
                                description="Provide a name for the artefact basis")

    response_schemas = [Artefact_basis_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    title_template = """ \ You are an AI Governance bot.
                Provide a name for the artefact basis the following text “{topic}”.
                {format_instructions}
                """

    prompt = ChatPromptTemplate.from_template(template=title_template)
    # df=pd.read_csv("final1.csv")
    df2=df['Description']
    for i in range(len(df2)):
        messages = prompt.format_messages(topic=df2[i], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        dict_to_csv(data, 'data24.csv', append=True)
    data24 = pd.read_csv('data24.csv', names=['Artefact Name'])
    result = pd.concat([df, data24], axis=1)
    output= pd.concat([df2, data24], axis=1)
    result.to_csv('final24.csv')
    data24.to_csv('data24.csv')
    
    st.subheader("Artefact Result")
    st.dataframe(output)
    st.markdown(get_download_link(data24), unsafe_allow_html=True)
    
    
def results5(df):
    Artefact_description_schema = ResponseSchema(name="Artefact Description",
                             description="Provide an artefact description.")

    response_schemas = [Artefact_description_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    title_template = """ \ You are an AI Governance bot.
                Provide an artefact description based on the following “{topic}”.
                {format_instructions}
                """

    prompt = ChatPromptTemplate.from_template(template=title_template)
    # df=pd.read_csv("final1.csv")
    df2=df['Artefact Name']
    
    for i in range(len(df2)):
        messages = prompt.format_messages(topic=df2[i], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        dict_to_csv(data, 'data25.csv', append=True)
    data25 = pd.read_csv('data25.csv', names=['Artefact Description'])
    result = pd.concat([df, data25], axis=1)
    output= pd.concat([df2, data25], axis=1)
    result.to_csv('final25.csv')
    data25.to_csv('data25.csv')
    
    st.subheader("Artefact Description")
    st.dataframe(output)
    st.markdown(get_download_link(data25), unsafe_allow_html=True)
    
    
def results3(df):
    intended_results_schema = ResponseSchema(name="Intended Results",
                             description="Summary of intended results.")

    response_schemas = [intended_results_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    
    title_template = """ \ You are an AI Governance bot.
                Summarize intended results of doing the activity from a third person perspective for “{topic}” in 15 to 20 words. 
                {format_instructions}
                """

    prompt = ChatPromptTemplate.from_template(template=title_template)
    # df=pd.read_csv("final1.csv")
    df2=df['Description']
    for i in range(len(df2)):
        messages = prompt.format_messages(topic=df2[i], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        dict_to_csv(data, 'data23.csv', append=True)
    data23 = pd.read_csv('data23.csv', names=['Intended Results'])
    result = pd.concat([df, data23], axis=1)
    output= pd.concat([df2, data23], axis=1)
    result.to_csv('final23.csv')
    data23.to_csv('data23.csv')
    
    st.subheader("Intended Result")
    st.dataframe(output)
    st.markdown(get_download_link(data23), unsafe_allow_html=True)
    
    


def main():
    st.image('logo.png')
    st.title("Upload Requirements")

    # File upload
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        # Read CSV file
        # df = pd.read_csv(file)
        df = pd.read_csv(file, usecols=['Intended Results', 'OP Key', 'OP Description', 'OP Title'])

        # Display preview
        st.subheader("CSV File Preview")
        st.dataframe(df)

        # Button to process the file
        if st.button("Generate Activity"):
            result(df)
            
        if st.button("Generate Description"):
            results2(pd.read_csv('data21.csv'))
            
        if st.button("Generate Intended Results"):
            results3(pd.read_csv('data22.csv'))
            
        if st.button("Generate Artefact"):
            results4(pd.read_csv('data22.csv'))
            
        if st.button("Generate Artefact Description"):
            results5(pd.read_csv('data24.csv'))
            
        


if __name__ == "__main__":
    main()
