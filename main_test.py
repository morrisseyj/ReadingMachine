from typing import List

from lit_review_machine import core, utils, state, render


from dotenv import load_dotenv
import os
from openai import OpenAI
import os
import pandas as pd


#---------
import importlib
from lit_review_machine import core, utils, state, render

importlib.reload(utils)
importlib.reload(core)
importlib.reload(state)
importlib.reload(render)


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

utils.restart_pipeline()

############

# We need to pass a df of both the questions and the papers to the ingestor. 
# If our papers are just in a single folder and not associated with specific questions, then we can just pass the same df for both the questions and the papers, as the ingestor will be able to associate the papers with the questions based on the question ids and text that are present in both dfs. This is a bit of a hack, but it allows us to keep the pipeline flexible and adaptable to different use cases. In this case, we are just using the questions df as a placeholder for the papers df, as we don't have specific papers associated with each question at this point in the pipeline. The insights will be added in later steps of the pipeline, and they will be associated with the questions based on the question ids and text that are present in both dfs.
# First create a questions dict from the ist
questions_dict = {f"question_{idx}": question_text for idx, question_text in enumerate(questions)}
# Then create a df from the questions dict, with columns for question id and question text
questions_df = pd.DataFrame(list(questions_dict.items()), columns=["question_id", "question_text"])
# Then initialize insights df with question ids and text, the insights will be added in later steps of the pipeline. This ensures that the question ids and text are always present in the insights df, which is important for the metadata anchored synthesis that we are doing in this pipeline.
insights_df = questions_df.copy() 

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

# To recover the state of the pipeline at any point, we can use the QuestionState class to load the state from a specific filepath. This allows us to pick up where we left off in the pipeline without having to re-run previous steps. In this case, we are loading the state from a specific filepath that corresponds to a previous run of the pipeline. This is useful for testing and debugging purposes, as it allows us to see how the pipeline is progressing and to identify any issues that may arise.
latest_state = state.QuestionState.load(filepath = r'C:\Users\jmorrissey\Documents\python_projects\ReadingMachine\lit_review_machine\data\runs\06_full_text_and_chunks')

# Now we create the insights generator, which is responsible for taking the chunked papers and generating insights from the chunks as well as meta insights from the whole paper 
insights_generator = core.Insights(state = latest_state,
                                   llm_client=llm_client,
                                   ai_model="gpt-4o", 
                                   paper_context=paper_context)

# Get the chunk insights
insights_generator.get_chunk_insights()
# Get the meta insights from the whole paper
insights_generator.get_meta_insights()

# Get the latest state again
latest_state = state.QuestionState.load(filepath = r'C:\Users\jmorrissey\Documents\python_projects\ReadingMachine\lit_review_machine\data\runs\07_insights')
# Initialize the cluster class
cluster = core.Clustering(state = latest_state, llm_client=llm_client, embedding_model='text-embedding-3-small')
# Embed the insights
cluster.embed_insights()

# Now we have to do dimensionality reduction so that we can cluster without the "curse of dimensionality"
# The logic here is as follows: we want to make sure that our dim reduction does not remove the meaningful structure in the data
# To do this we sweep UMAP params and calculate the silhoete score against the question_ids as our labels. So the test here is how well does the clustering separate out reduced dim insights by research question
# This is a proxy testing whether enough structure remains in the data after dimensionality reduction
# Notbaly, we expect natural language to not separate perfectly. Also if we have similar research questions we expect the separation of insights to be weak
# What we do then is select the params we want to sweep and optionally exclude any research questions that we think are very similar that might be polluting our silhoette score proxy
# The params below are the defaults to sweep over, but you can adjust as you like - adding more params will increase the run time
# You want high numbers here (close to 1). But note that the standards for silhoette need to be relaxed for this case, where natural language overlaps and labels are not mutually exclusive. Anything above or close to 0.1 is likely acceptable.
cluster.tune_umap_params(n_neighbors_list = [5, 15, 30, 50, 75, 100],
                         min_dist_list = [0.0, 0.1, 0.2, 0.5],
                         n_components_list = [5, 10, 20],
                         metric_list = ["cosine", "euclidean"],
                         rq_exclude=["question_3"]) # Exclude question with id 4 as it is quite similar to question 3 and it will cause the silhouette score to overlap 
# Now we reduce dimensions using the best params that we found in the above sweep 
cluster.reduce_dimensions(n_neighbors = 75,
                          min_dist = 0,
                          n_components = 5,
                          metric = "cosine",
                          random_state = 42)
# With dimensions reduced we now look to tune HDBSCAN params so that we can cluster. 
cluster.tune_hdbscan_params(min_cluster_sizes=[5, 10, 15, 20],
                            metrics=["euclidean", "manhattan"],
                            cluster_selection_methods=["eom", "leaf"])

# Useful to save the results to html to inspect - you want low db score and few outliers
cluster.hdbscan_tuning_results.to_html("hdbscan_tuning_results.html")

# Now we read the results (hdbscan_tuning_results.html) to identify params. Not we select these for each question to try and maximize coherence within the question
# We want to get lowest DBI scores, maximum cluster size and minimize outliers. 
cluster.generate_clusters(clustering_param_dict={
    # Best DBI (0.44) before hitting the <NA> wall at size 15
    "question_0": {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "leaf"},
    # Excellent DBI (0.38) with decent data retention (212 outliers)
    "question_1": {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "leaf"},
    # Messiest question; size 15 leaf selection yields the best balance (0.61)
    "question_2": {"min_cluster_size": 15, "metric": "euclidean", "cluster_selection_method": "leaf"},
    # Size 5 is the clear winner here (0.36) - larger sizes caused <NA> failure
    "question_3": {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_method": "eom"},
    # Absolute best score in the entire sweep (0.34); highly stable at size 20
    "question_4": {"min_cluster_size": 20, "metric": "manhattan", "cluster_selection_method": "eom"}
})

# As final sense check on your clustering you can look at the cumulative proportion of clusters per research question. 
# Essentially if you have 90% of the insights in three clusters and another 8% in a large number of small clusters (with 2% in the outliers) you can just shift the small clusters into outliers. 
# Note all outliers are handled in summarization so you are not losing insights by shifting small clusters into outliers, you are just shrinking the number of cliusters that get passed for synthesis and then for theme mapping
cluster.cum_prop_cluster.to_html("cum_prop_cluster.html")
# Now you can clean your clusters by optionally shifting your small clusters into outliers - passed as a dict with question ids as keys and top_n as the number of clusters to keep before shifting the rest into outliers
# For this small corpus i am happy with the way the clusters look so i don't pass any dict.
cluster.clean_clusters()

# Now we move to summarizing the clusters and generating the themes. 
# Initialze the Summarize class using eithe the Clustering class or by loading the state from the latest step of the pipeline.
latest_state = state.QuestionState.load(filepath = r'C:\Users\jmorrissey\Documents\python_projects\ReadingMachine\lit_review_machine\data\runs\08_clusters')
summarize = core.Summarize(state=latest_state, 
                           llm_client=llm_client,
                           ai_model="gpt-4o",
                           paper_output_length=8000)

# First we summarize the clusters. This only happens once
# Specifically we calculate the shortest path between the cluster centroids and feed them to the LLM in that order with already summarized clusters passed as frozen context for the next summariation
# This maximizes semantic coherence when asking the LLM to summarize clusters and will better enable conceptual coheremce when we undertake theme mapping
summarize.summarize_clusters()
# Next we build a schema of the themes that will be used to allocate insight to themes
summarize.gen_theme_schema()
# Then we apply the schema to actually map the insights to themes
summarize.map_insights_to_themes()
# Then we populate the themes based on the insights that have been allocated to them bu the mapping process
summarize.populate_themes()
# Then we check for any orphans that might have been dropped in the process - this captures insights that might not have been exposed via the cluster summaries that drove the firt mapping process
summarize.address_orphans()

# Now we iterate the above process to improve the schema - so that orphans are likely accounted for in a more complete manner than was possible with the cluster summaries
summarize.gen_theme_schema()
summarize.map_insights_to_themes()
summarize.populate_themes()

# Finally we handle redundancy
summarize.address_redundancy()

#THen we pass the output of the redudancy check to the render class

render.Render()













if os.path.exists(os.path.join(os.path.join(summarize.summary_save_location, "summaries.parquet"))):
    recover = None
    while recover not in ['r', 'n']:
        recover = input("Summaries already exist on disk. Do you want to recover (r) or generate new ones (n)? (r/n): ").lower()
    if recover == 'r':
        summarize.summaries = pd.read_parquet(os.path.join(summarize.summary_save_location, "summaries.parquet"))
        return summarize.summaries
    else:
        print("Re-running cleaning of summaries...")

# We are going to send the insights to the LLM in the order of the shortest path, so that the most similar clusters are summarized close together
# This will add coherence to the final paper when the summaries are stitched together
# It will also aid in the applicaion of the sliding window for summary clean up
shortest_paths = summarize._estimate_shortest_path()

# Add calculated lengths to insights
summarize.state.insights = summarize._calculate_summary_length()

summaries_dict_lst: List[dict] = []

total_clusters = len(summarize.state.insights.groupby("question_id")["cluster"].nunique(dropna=False))
count = 1

# Loop over unique research questions
for _, row in summarize.state.questions.iterrows():
    rq_id = row["question_id"]
    rq_text = row["question_text"]
    rq_df: pd.DataFrame = summarize.state.insights[summarize.state.insights["question_id"] == rq_id].copy()

    # Loop over clusters for this research question - in shortest path order
    for cluster in shortest_paths[rq_id]["order"]:
        print(f"Summarizing cluster {cluster} for research question {rq_id} ({count} of {total_clusters})...")
        count += 1
        # Skip any cases where chunks might have had no insights (and therefore no cluster)
        if pd.isna(cluster) or cluster == "NA":
            continue

        cluster_df: pd.DataFrame = rq_df[rq_df["cluster"] == cluster]
        length_str: str = cluster_df["length_str"].iloc[0]
        # get the insights, they are list of single strings. So make sure they are valid string to send to the LLM
        insights: List[str] = cluster_df["insight"].apply(
            lambda x: x if isinstance(x, str) else (
                x[0] if isinstance(x, list) and len(x) == 1 and isinstance(x[0], str) else None
            )).tolist()
        
        if any(i is None for i in insights):
            raise ValueError("Insight format error: each insight must be a string or a single-item list containing a string.")

        # Build system prompt from predefined method
        sys_prompt: str = Prompts().summarize(summary_length=length_str)

        # Get the summaries frozen so far if there are any
        frozen_summaries = pd.DataFrame(summaries_dict_lst)["summary"].tolist() if summaries_dict_lst else []

        # Build user prompt for LLM
        user_prompt: str = (
            f"Research question id: {rq_id}\n"
            f"Research question text: {rq_text}\n"
            "PRECEDING CLUSTER SUMMARIES (for context only; may be empty):\n"
            f"{'\n'.join(frozen_summaries) if frozen_summaries else ''}\n"
            f"Cluster: {cluster}\n"
            "INSIGHTS:\n" +
            "\n".join(insights)
        )

        json_schema = {
            "name": "cluster_summary",
            "schema": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "question_text": {"type": "string"},
                    "cluster": {"type": "number"},
                    "summary": {"type": "string"}
                },
                "required": ["question_id", "question_text", "cluster", "summary"],
                "additionalProperties": False
            }
        }
        fall_back = {"question_id": rq_id, "question_text": rq_text, "cluster": cluster, "summary": ""}

        response = utils.call_chat_completion(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            llm_client=summarize.llm_client,
            ai_model=summarize.ai_model,
            return_json=True,
            json_schema=json_schema,
            fall_back=fall_back
        )

        summaries_dict_lst.append(response)

# Now convert the list of dict responses to a dataframe for easier handling and saving
summaries_df: pd.DataFrame = pd.DataFrame(summaries_dict_lst)

print(
    f"Summaries saved here: {summarize.summary_save_location}\n"
    "Returned object is a Summaries instance. Access via `variable.summaries`.\n"
    f"Or load later with: Summaries.from_parquet('{summarize.summary_save_location}')"
)

summarize.summaries = summaries_df

# Save summaries as this is LLM output we may want to reuse later - save as parquet
os.makedirs(summarize.summary_save_location, exist_ok=True)
summarize.summaries.to_parquet(os.path.join(summarize.summary_save_location, "summaries.parquet"))
return summaries_df