from lit_review_machine import core, utils


from dotenv import load_dotenv
import os
from openai import OpenAI
import os
import pandas as pd


#---------
import importlib
from lit_review_machine import core, utils

importlib.reload(utils)
importlib.reload(core)


#---------

# Access env variables
load_dotenv()
# Securely load our open ai API key  
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
# Create the LLM client 
llm_client = OpenAI(api_key=OPEN_API_KEY)


# Input our questions that will drive the review
questions = [
    "What drivers account for the resurgence of industrial policy in both highly industrial and industrializing countries?", 
    "How have the definitions and approaches to industrial policy shifted from the post-World War II period to present day? And, more specifically, how have these shifts considered (if at all) inclusive and sustainable growth, and respect of human rights and gender equality?",
    "What challenges and constraints do less-industrialized countries face in realizing effective industrial policy?",
    "What key recommendations can Oxfam make to more industrialized countries so that their industrial policies do less harm to industrialized countries?",
    "What key recommendations can Oxfam make to rich countries to advance reforms among different transnational institutions, so as to increase the policy space available to less-industrialized countries, to a point that it is, at least, comparable with the policy space afforded to industrialized nations?"
]

############
#Restart the pipeline if you want
############
utils.restart_pipeline()


# Create the ingesstion dataframe
start_state = pd.DataFrame.from_dict({
    "question_id":[f"question_{i}" for i in range(len(questions))], 
    "question_text": questions
    })

# Instantiate in Ingestor class
ingestor = core.Ingestor(llm_client=llm_client,
                         ai_model="gpt-4o",
                         papers = start_state,
                         file_path=os.path.join(os.getcwd(), "data", "docs"))

# Ingest the papers
ingestor.ingest_papers()
# update the metadata 
# metadata is a first class object here as the result is citaiton achored synthesis
# For this reason everythig gets passed through a metadata check to ensure  metatdata matches the actual text
ingestor.update_metadata()
# Chunk the papers so that they can be used to acquire insights
ingestor.chunk_papers()

