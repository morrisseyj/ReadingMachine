from lit_review_machine import core, utils, state


from dotenv import load_dotenv
import os
from openai import OpenAI
import os
import pandas as pd


#---------
import importlib
from lit_review_machine import core, utils, state

importlib.reload(utils)
importlib.reload(core)
importlib.reload(state)


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

example_grey_literature_sources = (
    "The Center for Global Development, The Brookings Institution, The Overseas Development Institute, UNCTAD, UNIDO, The World Bank, or regional development banks"
)


paper_context = (
    'The purpose of this literature review, conducted by Oxfam America, is to articulate advocacy priorities to ensure that the industrial policy '
    'strategies of industrialized nations—particularly the United States—do not unduly constrain the policy options available to less industrialized countries. '
    'The underlying hypothesis is that the recent resurgence of industrial policy risks impoverishing low-income countries, as wealthy countries are better positioned '
    'to dominate global markets due to their superior financial and political resources.'
    )

############
#Restart the pipeline if you want
############
utils.restart_pipeline()

questions_dict = {f"question_{idx}": question_text for idx, question_text in enumerate(questions)}
questions_df = pd.DataFrame(list(questions_dict.items()), columns=["question_id", "question_text"])
insights_df = questions_df.copy() # initialize insights df with question ids and text, the insights will be added in later steps of the pipeline. This ensures that the question ids and text are always present in the insights df, which is important for the metadata anchored synthesis that we are doing in this pipeline.

# Instantiate in Ingestor class
ingestor = core.Ingestor(
    questions = questions_df, 
    papers=insights_df,
    llm_client=llm_client,
    ai_model="gpt-4o",
    file_path=os.path.join(os.getcwd(), "data", "docs")
    )

# Ingest the papers
ingestor.ingest_papers()
# update the metadata 
# metadata is a first class object here as the result is citaiton achored synthesis
# For this reason everythig gets passed through a metadata check to ensure  metatdata matches the actual text
ingestor.update_metadata()
# Chunk the papers so that they can be used to acquire insights
ingestor.chunk_papers()


latest_state = state.QuestionState.load(filepath = r'C:\Users\jmorrissey\Documents\python_projects\ReadingMachine\lit_review_machine\data\runs\06_full_text_and_chunks')

insights_generator = core.Insights(state = latest_state,
                                   llm_client=llm_client,
                                   ai_model="gpt-4o", 
                                   paper_context=paper_context)

insights_generator.get_chunk_insights()


with open(os.path.join(insights_generator.pickle_path, insights_generator.chunk_insights_pickle_file), "rb") as f:
    recover_chunk_insights = pickle.load(f)
