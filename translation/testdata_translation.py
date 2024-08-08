from typing import Literal
import pandas as pd
from io import StringIO
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

df = pd.read_csv("PubMedRS-200/PubMedRS-200.csv")

print(df.columns)
# print(df.iterrows())

model = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    api_key="hahaahah",
    verbose=True,
    streaming=True,
    max_tokens=10000,
)

SYSTEM_PROMPT = """
Your job is to translate an input to Vietnamese.
Return as a single string.
Don't add any extra information, just return the translated result.
"""
# ['specialty', 'SubTopic', 'Title', 'Abstract', 'Introduction', 'Methods',
#        'Results', 'Conclusion', 'PMID', 'Ref_PMIDs', 'Ref_DOIs', 'questions',
#        'PublishedDate', 'HumanQuestions', 'HumanAnswer'
allowed_col = [0, 1, 2, 3, 4, 5, 6, 7, 11, 13, 14]

with open("ViPubMedRS-200.csv", "a") as f:
    for index, row in df.iterrows():

        print(f"row: {index}")
        row_res = []
        for col_index, col_value in enumerate(row):
            print(f"col: {col_index}, value: {col_value}")
            if col_index in allowed_col:
                # Translate the column value to Vietnamese
                USER_PROMPT = f"{col_value}"

                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=USER_PROMPT),
                ]

                chain = model | StrOutputParser()

                translated_value = chain.invoke(messages)
                row_res.append(translated_value)
            else:
                row_res.append(col_value)

        # Write list to csv
        temp = pd.DataFrame([row_res], columns=df.columns)
        temp.to_csv(f, mode="a", header=False, index=False)
