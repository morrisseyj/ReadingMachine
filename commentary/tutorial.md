**Tutorial for ReadingMachine**

**A Computational Methodology for Structured Corpus Reading and Large- Scale Synthesis**

**Introduction**

Most fields today produce far more text than any individual—or even a team—can realistically read. Whether you’re working in research, policy, or organizational analysis, the bottleneck is no longer access to information. It’s making sense of it.

The default response is still human-led review: reading papers, taking notes, synthesizing arguments. This works, but it doesn’t scale well. It’s slow, expensive, and shaped by the path the reader takes through the material. Existing AI approaches don’t fully solve this either. Retrieval-based systems only look at small parts of a corpus. Summarization pipelines compress information early and often lose nuance. Agentic workflows explore selectively, depending heavily on intermediate decisions. In all cases, it’s easy to miss things—and hard to know what was missed.

ReadingMachine takes a different approach. It treats large language models not as systems that “reason over” a corpus, but as tools for performing bounded reading tasks at scale. Instead of trying to produce a synthesis in one step, it breaks reading into smaller operations—extracting insights, organizing them, and then building up a structured view of the corpus. The goal is not just to summarize, but to do so in a way that preserves structure and reduces omission.

ReadingMachine is an experimental, open-source method (https://github.com/morrisseyj/ReadingMachine/). This tutorial walks through how to run the pipeline step by step, explaining what each part of the code is doing along the way. It complements the GitHub README, the examples documentation, and the accompanying white paper.

**Overview**

Conceptually, the methodology formalizes a familiar qualitative research workflow into a computational pipeline, with large language models performing bounded reading and synthesis tasks. In abstract form, this workflow can be understood as:

reading a corpus → extracting notes → identifying themes → synthesizing those themes → checking for completeness → refining for clarity

ReadingMachine decomposes this process into a more granular and structured sequence of steps. The following pipeline overviews the computational implementation of the above workflow:

    [Generate Research Questions]  
                    ↓  
            [Ingest Papers]  
                    ↓  
            [Chunk Papers]  
                    ↓  
            [Generate Insights]  
                    ↓  
            [Cluster Insights]  
                    ↓  
            [Summarize Clusters]  
                    ↓  
            [Generate Theme Schema]  
                    ↓  
            [Map Insights to Themes]  
                    ↓  
            [Summarize Themes]  
                    ↓  
            [Identify Orphans]  
                    ↓  
            [Reinsert Orphans]  
                    ↓  
        ┌───────────────────────────────┐  
        │      Iteration Loop           │  
        │  (Re-theme if needed)         │  
        │                               │  
        │  Generate Theme Schema        │  
        │            ↓                  │  
        │  Map Insights to Themes       │  
        │            ↓                  │  
        │  Summarize Themes             │  
        │            ↓                  │  
        │  Identify + Reinsert Orphans  │  
        └────────────↑───────────────────┘  
                     │  
                     ↓  
        [Address Redundancy (Optional)]  
                     ↓  
        [Generate Title, Executive Summary, Question Summaries]  
                     ↓  
        [Render Output]  

The specific steps to run the pipeline are as follows (the below draws heavily on the example pipeline run contained [here](https://github.com/morrisseyj/ReadingMachine/blob/main/examples/run_core_pipeline.py)). 

**What you should expect**

Running the full pipeline will:

- take time (minutes to hours depending on corpus size)
- incur API costs
- generate intermediate files at each stage
- produce a structured synthesis in /outputs


**1. Setup**

Assuming you have python on your system, with git installed.

First we need to set up our environment and prepare our data

*1.1. Clone the github repo:*

This will copy the module's code from the github repo to your local machine.

```
git clone https://github.com/morrisseyj/ReadingMachine
```

*1.2. Install uv*

Dependency management is done via uv to handle conflicts. It is strongly recommended you use the uv.lock file to set up the environment. So install uv with instructions from [here](https://docs.astral.sh/uv/getting-started/installation/)

*1.3. Set up the environment*

This makes sure all the dependencies are installed without conflicts in a virtual environment.

In a terminal in /ReadingMachine/
```
uv sync
```

You will need to access this virtual environment and inialize a python instance there in order to execute python code. The steps for doing this depend on where you running the code and how your IDE operates.

*1.4. Upload your corpus*

ReadingMachine can read any corpus (though its capacities across different types of corpuses are still being determined). You should therefore put the documents that you want to read in `/ReadingMachine/data/corpus/`. If you have research questions and want to retrieve literature to answer them there is a [getlit.py](https://github.com/morrisseyj/ReadingMachine/blob/main/examples/run_getlit_pipeline.py) tool that can support that process. For this tutorial a list of open access publications relevant to the example research questions used below is available [here](https://github.com/morrisseyj/ReadingMachine/blob/main/examples/toy_corpus.md).

*1.5. Create your secrets*

You need an OpenAI API key to execute the pipeline (currently only OpenAI access is enabled, future development involves adding other services). This step safely stores your key outside of your codebase.

Create a file `.env`. Populate this with:

```
OPENAI_API_KEY=<your_api_key_here>
```

*1.6. Load the modules you will use for the core pipeline*

Now we move to a python script, with most code now executing in a single python instance (should be loaded from your virtual environment).

This step imports the libraries you need to access the modules and execute the code.

```
from readingmachine import core, render, config
from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd

```
*1.7. Load your secrets from .env and set up your LLM client object*

Next we load the environment variables which gives us access to an OpenAI llm_client instance that we can use to call OpenAIs language and embedding models.

```
# Load environment variables from .env
load_dotenv()

# Read OpenAI API key*
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create LLM client
llm_client = OpenAI(api_key=OPENAI_API_KEY)

```

*1.8. Articulate your research questions*

Your research questions drive the entire process. They should be clear, focused and answerable, and well distinguished from one another. The tool cannot compensate well for poorly framed research questions. The research questions will be used to extract the insights on top of which the entire synthesis is built.

```
# Example research questions:
questions = [
	"What effects does remote work have on worker productivity?",
	"How does remote work affect worker satisfaction and retention?",
	"What organizational practices enable successful remote work?",\
	"What constraints limit the adoption of remote work?",
	"How has the role of remote work changed over time?"
]
```
*1.9. Briefly describe the context for the paper*

To try and contextualize the insight generation step further we also provide the paper context – essentially why are you posing the particular research questions. This is effectively the topline rationale for the reading you want to do. This rationale is passed to the LLM when asking it to identify relevant insights. 

```
# Example context
paper_context = (
	"This literature review examines the growth of remote and hybrid work arrangements "
	"and their implications for productivity, worker well-being, and organizational structure. "
	"The goal of the review is to synthesize empirical findings about when remote work "
	"improves or reduces productivity, how it affects job satisfaction and retention, "
	"and what organizational practices enable successful implementation."
)

```

*1.10. Prepare the data for ingestion into the pipeline*

With our environment set up and our data ready, we are almost ready to begin to use the ReadingMachine modules. 

ReadingMachine is implemented as a sequence of classes, each corresponding to a distinct stage in the reading process. These classes operate over a shared state object, with each stage transforming that state in a controlled and sequential manner.

The first task therefore is to instantiate the first class in the pipeline. The instantiation requires us to pass the state some minimum content. We prepare that as follows:

```
# Create canonical question IDs
questions_dict = {
    f"question_{idx}": q
    for idx, q in enumerate(questions)
}

questions_df = pd.DataFrame(
    list(questions_dict.items()),
    columns=["question_id", "question_text"]
)

# Initialize an insights dataframe with the question information - this will be updated with the paper metadata and insights as we go through the pipeline
insights_df = questions_df.copy()

```

**Sidenote: How to resume a run**

This pipeline can involve long running times especially if the corpus is large. It is quite possible that you may have to close your python instance during a run. The tool includes multiple supports for resume - from continuously saving expensive language model calls, to saving the overall pipeline state after the last method of each class is called. As such if you re-run elements of the pipeline you will see prompts asking if you would like to reload or re-run analysis steps. 

If you close your python instance and want to reload from save you can call the following convenience function:

```
from readingmachine import utils
utils.restart_pipeline()
```

This will print the most recent complete class, and give you instructions on how to 1) load the last complete state and 2) instantiate the next class in the pipeline. You simply need asign this instantiaion call to a variable to proceed. The corpus_state saves after all the required methods in the class have been run. The summary_state (see below) saves at the completion of each summarize step.

**2 Start the pipeline: Ingestor class**

Ingestor class is responsible for ingesting papers, pulling metadata, dedpulicating papers and chunking the papers. First we initialize the class.

ingestor = core.Ingestor(
    questions=questions_df,
    papers=insights_df,
    llm_client=llm_client,
    ai_model="gpt-4o"
)

*2.1 Ingest papers*

Then we pull all the papers into the system. Currently .html and .pdf are supported filetypes
```
ingestor.ingest_papers()

```

*2.2 Update metadata for the papers*
For traceability metadata is a first class concern for ReadingMachine. This process uses an LLM to look at the raw content of the documents and retrieve metadata.
```
ingestor.update_metadata()
```

*2.3 Drop duplicates*
We drop duplicate papers as maintaining them is expensive in terms of tokens and can cause insights to show up more than once compromising the synthesis weighting.

```
ingestor.drop_duplicates()
```

*2.4 Manual duplicate check* 
The above step involves dropping exact duplicates and identifying possible duplicates. The user has to manually inspect a file and delete duplicates. The file is here: `~/ReadingMachine/data/fuzzy_check/ingest/duplicate_check.csv`

Manually amend the file so that one unique versions of each document remains. This is also a good time to manually complete any meta data that the model failed to generate. 

*2.5 Update the state*
Now we update the state to reflect the deduplicated and metadata complete file.

```
ingestor.update_state("duplicate_check.csv")
```

*2.6 Chunk documents*
The final step of this class is chunking the documents so that they can be passed to the LLM for insight retrieval.

```
ingestor.chunk_papers()
```

At the end of this class the corpus_state has four components:
- questions – rows are research questions linked with a question_id
- insights – rows are chunk_ids linked to a paper_id, along with paper metadata.
- full_text – rows are paper full text linked with a paper_id 
- chunks – rows are chunk text linked with a chunk_id

We move now to generating the insights

**3. Generate insights**

This class handles insight extraction. First we instantiate the class - passing the final corpus_state attribte from the Ingestor class to the Insights class.

```
insights_generator = core.Insights(
    corpus_state=ingestor.corpus_state,
    llm_client=llm_client,
    ai_model="gpt-4o",
    paper_context=paper_context
)
```

*3.1 Generate chunk insights*

This step extracts atomic insights from each chunk, forming the core unit of analysis used throughout the pipeline.

```
insights_generator.get_chunk_insights()
```

*3.2 Generate meta insights*

We also need to pass the questions and the whole document (or as large a chunks as will fit in the context window) to the LLM to extract insights that might span long sections and been missed in our chunk pass.

```
insights_generator.get_meta_insights()
```

**4. Cluster insights**

Next we move to clustering the insights. This class handles embeddnig generation, dimensionality reduction and clustering. Instantiate the class as follows:

```
cluster = core.Clustering(
    corpus_state=insights_generator.corpus_state,
    llm_client=llm_client,
    embedding_model="text-embedding-3-small"
)
```

*4.1. Generate the embeddings*

First we generate vector representations of the insights ("embeddings") so that we can cluster them.

```
cluster.embed_insights()
```

*4.2 Reduce dimensions*

Next we have to reduce dimensions to compensate for sparsity effects in high dimensional data. We use UMAP for this, but first we have to select our UMAP parameters. 

<u>4.2.1. UMAP parameter sweep</u>

ReadingMachine includes a parameter sweep function to identify the best configuration. 

```
cluster.tune_umap_params(
    n_neighbors_list=[5, 15, 30, 50, 75, 100],
    min_dist_list=[0.0, 0.1, 0.2, 0.5],
    n_components_list=[5, 10, 20],
    metric_list=["cosine", "euclidean"]
)
```

You want to select parameters that maximize the silhoette score. 1 is the maximum but because natural language data is messy we don't expect very high scores here (likely closer to ~0.15-0.3). 

Note this parameter sweep uses research questions as a proxy for estimating the silhoette score. If you have two very multiple research questions that are very similar, this will worsen silhoette scores. You can exclude any rresearch questions from the sweep by passing the parameter: `rq_exclude = [<question_id>]` to the .tune_umap_params() method.

<u>4.2.2. Apply UMAP</u>

With the parameters selected we now reduce the dimensions.

```
cluster.reduce_dimensions(
    n_neighbors=5,
    min_dist=0.5,
    n_components=5,
    metric="cosine",
    random_state=config.seed
)
```

*4.3. Clustering*

Now we can cluster on the reduced dimensional embeddings. We use HDBSCAN for this. Again we have to select parameters and the tool includes another pramater sweep capability.

<u>4.3.1. HDBSCAN parameter sweeep</u>

```
cluster.tune_hdbscan_params(
    min_cluster_sizes=[5, 10, 15, 20],
    metrics=["euclidean", "manhattan"],
    cluster_selection_methods=["eom", "leaf"]
)
```

The clustering parameters are calculated per question. The easiest way to inspect the results is via html:

```
cluster.hdbscan_tuning_results.to_html("hdbscan_tuning_results.html")
```

You want to select parameter combinations that both minimize the db score and minimize outliers. Note however that unlike other ML approaches, outliers here are not discarded (they will be worked into themes) so it minimizing them should likely recieve less emphasis than is normal in other approaches. 

<u>4.3.2. Cluster with HDBSCAN</u>

With the parameters selected we now cluster with HDBSCAN

```
cluster.generate_clusters({
    "question_0": {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "eom"},
    "question_1": {"min_cluster_size": 5, "metric": "manhattan", "cluster_selection_method": "eom"},
    "question_2": {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_method": "eom"},
    "question_3": {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_method": "eom"},
    "question_4": {"min_cluster_size": 5, "metric": "manhattan", "cluster_selection_method": "eom"}
})
```

*4.4. Optimizing clusters*

You can now optionally collapse very small clusters into outliers by selecting a cutoff. 
Note even if you don't want to reduce the clusters you should run `cluster.clean_clusters()` to trigger the state of the current class to save. 

```
# Inspect the cumulative distribution of cluster sizes to choose a cutoff
cluster.cum_prop_cluster
# Clean to your new cutoff if you want
cluster.clean_clusters(final_cluster_count = <int of the number of clusters you want>)

```

**5. Summarize the data**

At this point in the pipeline you have all the papers, all the chunks and all the insights, with the insights organized into clusters. We now go about organizing and summarizing these insights into the final syntheis. 

The Summarize class contains a lot of the methods for generating the summary. You initialize it as follows:

```
summarize = core.Summarize(
    corpus_state=cluster.corpus_state,
    llm_client=llm_client,
    ai_model="gpt-4o",
    paper_output_length=10000
)
```

Note that while the state that was managing the first two classes was the corpus_state attribute, the Summarize class modifies the summary_state attribute. Saves happen here after every summary mehod gets called.

*5.1. Summarize the clusters*

The first step is to have an LLM summarize all the clusters of insights.

```
summarize.summarize_clusters()
```

*5.2. Generate theme schema*

Then we have the LLM look at these summaries and build a better conceptual mapping of the dominant themes, expressed as a theme schema - a set of theme labels, theme descriptions and rules for whether to include or exclude an insight. 

```
summarize.gen_theme_schema()
```

*5.3. Map insights to themes*

Now we pass the theme schema and batches of the insights to the LLM and ask it to allocate each insight to a theme id. 

```
summarize.map_insights_to_themes()
```

*5.4. Populate the themes*

With the insights organized into themes we can ask the LLM to summarize each theme using only the insights allocated to it. 

```
summarize.populate_themes()
```

*5.5. Address orphans*

Now we check the theme summaries to make sure all the insights that were allocated to a theme actually got incorporated. For any insights that are identified as missing we force them into the themes. This two step process happens in a single call. 

```
summarize.address_orphans()
```

**6. Iterate themes and finalize**

Now we have a set of themes containing summaries of the insights allocated to them, along with reinserted orphans. However since the themes were generated on top of compressed cluster summaries the schema may not have accounted for minority insights or edge cases. In those cases the forced orphan reinsertion may destabilize theme boundaries, or break their internal conceptual cohernece. Iteration on themes is a means to restore this coherence. To achieve this practically we pass the current set of summarized themes back to the `gen_theme_schema()` method that will try to generate an improved schema accounting for all the insights that are now reflected in the summaries.

*6.1. Iterate themes*

So we regenerate the schema, re-map the insights, re-populate the themes, and address orphans again:

```
# Regenerate themes incorporating previously orphaned insights
summarize.gen_theme_schema() # This will prompt you whether you want to use the latest theme summaries or cluster summaries to generate the new theme schema.
summarize.map_insights_to_themes()
summarize.populate_themes()
summarize.address_orphans()
```

You can run this iteration as many times as you like but for now running it twice is suggested. 

*6.2. Address redundancy*

The final step in this class is optional. You can create an updated output that tries to reduce redundacies across themes within each research question. This risks losing insights, but makes the output more readable. 

```
summarize.address_redundancy()
```

**7. Render**

The final stage of the pipeline involves the Render class that will optionally generate cosmetic features: title, exec summary, question summaries and a stylistic re-write. All are optional. It also renders the final synthesis. Initialize the class using the corpus_state and summary_state from the complete Summarize class. 

```
renderer = render.Render(
    summary_state=summarize.summary_state,
    corpus_state=summarize.corpus_state,
    llm_client=llm_client,
    ai_model="gpt-4o"
)
```

*7.1. Generate optional cosmetic elements*

Here we just pass content from the summarized themes and ask the LLM to generate different cosmetic elements: stylistic rewrite, title and exec summary and question summaries.  

```
renderer.stylistic_rewrite()
renderer.gen_exec_summary()
renderer.gen_question_summaries()
```

*7.2. Knit cosmetic elements into the themes*

Penultimately we incorporate whatever cosmetic elements you have chosen to create, into the data holding the theme summaries.

```
renderer.integrate_cosmetic_changes()
```

*7.3. Export the results*

Finally we render the output. You can render to any of "docx", "md" or "pdf". Set `use_stylized = True` to use the stylized re-write or `False` to use whatever was finally generated by the Summarize class (i.e. whether you ran the redundancy pass or not) 

```
renderer.render_output("md", use_stylized=True)
```

You can now inspect your results at ~/ReadingMachine/outputs

**8. Trace claims**

Because the tool summarizes themes directly from insights on every pass, we can trace all claims in the paper back to the original insight and chunk. A convenience methodd in the Render class enables this:

```
renderer.trace_claim(
    question_text="How does remote work affect worker satisfaction and retention?",
    theme_label="Work-Life Balance",
    citation_lastname= "Lingfeng",
    citation_year=2021
    )
```

That is the complete pipeline. For more details on the methodology see (arxiv whitepaper). 

As mentioned this project is in an experimental phase and it still seeking collaboration, testing and validation. Should you have questions or an interest in collaborating please reach out to: james.morrissey@oxfam.org. 

