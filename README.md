# ReadingMachine

**ReadingMachine implements a structured methodology for large-scale thematic synthesis across natural-language corpora.**

The system coordinates large language models in a **highly constrained reading workflow** that extracts structured insights from documents and recombines them into an inspectable synthesis.

Rather than retrieving fragments of text or repeatedly summarizing documents, ReadingMachine performs a **structured analytical pass over the entire defined corpus**, rather than sampling subsets through retrieval.

The result is a structured map of the arguments and insights contained in a document collection.

ReadingMachine should therefore be understood primarily as a **semantic research methodology implemented through machine reading**, rather than as a conventional AI application.

---

## Motivation

Many domains now generate far more text than individuals can realistically read. In many of these contexts, the primary risk is not insufficient access to information, but silent omission, compression drift, and the loss of minority or dissenting claims during synthesis.

Academic literature is an obvious example, but similar challenges appear in areas such as:

- policy research  
- government reporting  
- program evaluations  
- organizational documentation  
- legal materials  
- qualitative survey responses  
- institutional records  

In these settings the problem is rarely the availability of information. Instead, the bottleneck lies in **reading, organizing, and synthesizing large volumes of natural language**.

ReadingMachine coordinates machine reading across a corpus to address this problem. By structuring the reading process and preserving intermediate artifacts, the system allows large document collections to be analyzed in a systematic and inspectable way.

Literature review is therefore only one application. The same architecture can be used wherever large natural-language corpora need to be understood.

---

## Conceptual Approach

ReadingMachine separates **reading** from **interpretation**.

The system performs the large-scale reading process:

documents
→ chunking
→ insight extraction
→ clustering
→ theme formation
→ synthesis

The model is constrained to perform narrow tasks such as:

- reading bounded text segments  
- extracting atomic insights  
- organizing insights semantically  
- synthesizing themes from structured inputs  

Human researchers remain responsible for:

- defining the corpus  
- framing research questions  
- selecting specific parameters
- determining the stopping conditions
- interpreting results  
- evaluating conclusions  

This division preserves human analytical judgment while removing the practical bottleneck imposed by large-scale reading.

---

## Analytical Model

The pipeline treats **insights (atomic claims)** as the core unit of analysis.

document
→ chunk
→ insight 

Insights represent arguments, mechanisms, or findings identified in the text that are pertinent to the queries posed by the user. Each insight retains metadata linking it to its source chunk and document.

Using insights rather than document summaries allows the system to:

- preserve minority claims and disagreements  
- cluster semantically similar arguments across documents  
- trace synthesized claims back to their sources  
- delay compression until the final stages of synthesis  

---

## Pipeline Overview

### Document ingestion

Documents are imported into the corpus along with metadata such as:

- author  
- year  
- title   

---

### Chunking

Documents are divided into meaningful semantic units such as paragraphs or sentences.

Chunking ensures that reading tasks operate on bounded segments of text.

---

### Insight extraction

The system extracts **atomic insights** from the corpus. An insight is the smallest unit of meaning that expresses a discrete claim relevant to the research question.

Insight extraction does not aim to recover a neutral or singular “true” meaning of a text. Interpretation is inherent to reading. The pipeline instead exposes interpretive transformations as structured artifacts that can be inspected, compared, and interrogated.

Two complementary passes are used.

**Chunk pass**

Insights are extracted from individual chunks of text.

**Whole-document pass**

The model reads the full document (or the largest portion that fits within the context window) to capture insights that span multiple sections.

This dual approach helps capture both local arguments and cross-document reasoning.

---

### Embedding

Insights are embedded into vector representations.

Citation metadata is removed during embedding so that clustering reflects semantic similarity rather than authorship.

Embedding models and versions should be fixed and recorded as part of the analytical configuration. Reproducibility assumes consistent embedding and clustering settings.

---

### Dimensionality reduction

Insight embeddings initially exist in a high-dimensional space. Because of the "curse of dimensionality" - as dimensions increase volume expands exponentially resulting in data being too sparse to meaningfully cluster - dimensionality reduction is necessary for clustering.

UMAP is used to reduce dimensionality prior to clustering. Parameter sweeps can be used to evaluate different configurations. 

---

### Clustering

Insights are clustered using HDBSCAN.

Clusters serve as **organizational scaffolding**, grouping semantically similar insights before synthesis. They are not treated as analytical conclusions.

Parameter sweeps allow clustering behavior to be tuned and evaluated.

---

### Cluster summarization

Each cluster is summarized sequentially.

Clusters are processed in **shortest-path order within the embedding space** (using the full dimensions of the embedding), with previously summarized clusters provided as frozen context. This ordering improves semantic continuity across summaries.

Outliers are preserved and included.

---

### Theme generation

Using cluster summaries and research questions, the system generates a **theme schema**. This moves the eventual synthesis structure from semantic density (clusters) to conceptual density (themes).

Themes include:

- substantive thematic categories  
- an **Other** category to capture minority insights and prevent theme bloat
- a **Conflicts** category to explicitly represent disagreements in the corpus and account for LLMs' inclination to smooth disagreement when summarizing
- Sets of rules for assigning insights to each theme

The system only classifies conflict when substantively incompatible claims are present; trade-offs or cumulative critiques are not treated as conflict.

Clusters themselves are not used as themes; they only organize insights before thematic synthesis.

---

### Insight-to-theme mapping

Each insight is mapped to one or more themes.

Themes therefore represent structured collections of insights rather than summaries of documents.

---

### Theme population and synthesis

Theme summaries are generated using only the insights mapped to each theme.

This stage produces the primary narrative structure of the synthesis.

---

### Orphan detection

After synthesis, the system checks whether all mapped insights appear in the theme summary.

Insights that are omitted are flagged as **orphans**.

---

### Orphan reinsertion loop

Orphans are forcibly reinserted into their assigned theme and theme summaries are regenerated.

This process reduces silent omission, a common problem in automated summarization.

---

### Iteration

The steps from theme schema generation to orphan handling are then repeated until the user chooses to stop. Thus the themes with orphans inserted are handed back to the theme schema generator which updates the themes. Mapping, population, summarization and orphan handling are reapplied. In future tools for assessing stopping conditions will be added (focussed on assessing theme, orphan and "other" stabilization), but for the current architecture it is suggested the user run this pass twice. 

### Redundancy pass

Because insights may appear in multiple themes, a final pass reduces repetition while preserving information.

Repeated insights are replaced with cross-references rather than deleted.

---

### Rendering

The final synthesis can be rendered to formats such as:

- Markdown  
- DOCX  
- PDF  

An optional stylistic rewrite can improve readability while preserving informational fidelity.

---

## Reproducibility, Inspectability, and Interpretation

Corpus analysis inevitably involves analytical design choices. Decisions such as corpus selection, research questions, clustering parameters, and theme structures shape the resulting synthesis.

ReadingMachine is designed to make these choices **reproducible and inspectable**.

### Reproducibility

Analytical configurations—such as corpus definition, prompts, clustering parameters, and models—can be fixed and recorded. This allows the same workflow to be rerun under comparable conditions, making it possible to reproduce the analytical process and evaluate how changes in configuration affect the results.

### Inspectability

The pipeline preserves intermediate artifacts at every stage of the analysis:

chunk
→ insight
→ cluster
→ theme
→ synthesis

This makes it possible to trace how particular claims in the final synthesis emerged from the underlying corpus. Researchers can inspect the transformations that occur between stages and examine how different analytical decisions influence the resulting thematic structure.

### Interpretation

Interpretation is inherent in the act of reading. Both human readers and language models interpret text through prior assumptions—whether those arise from human cognitive frameworks or from patterns present in a model’s training data. ReadingMachine therefore does not claim to produce neutral or objective representations of a corpus.

Instead, the pipeline is designed to make interpretive processes **visible and examinable**. By varying models, prompts, or analytical parameters, researchers can explore how different interpretive lenses shape the resulting synthesis. In this sense, the inevitable semantic judgements embedded in a model’s reading become a source of signal to be interrogated (and potentially studied), rather than noise that must simply be corrected for.

While language models introduce some stochastic variation in wording, the structured workflow—semantic indexing through insight extraction combined with computational clustering—produces a stable intermediate representation of the corpus. As a result, **similar analytical configurations and research questions should produce similar analytical outcomes**, even if the exact wording of the final synthesis varies.

### Model Error and Hallucination Risk

Like all language model workflows, ReadingMachine cannot eliminate the risk of extraction errors or hallucinated claims. The pipeline mitigates this risk by constraining tasks to bounded text segments, preserving source-linked metadata for every extracted insight, and restricting synthesis to mapped insights rather than open-ended reasoning. Because each insight remains traceable to its originating chunk, researchers can audit and verify claims at every stage of the workflow. 

---

## What This System Is Not

ReadingMachine differs from several common approaches to automated research.

**Retrieval-based systems (RAG)** generate answers from subsets of documents returned by search queries. In these systems most of the corpus is never read, and conclusions depend heavily on retrieval performance. Relevant material that is not retrieved simply does not enter the analysis, meaning omission is often invisible.

**Hierarchical summarization pipelines** repeatedly compress documents through successive layers of summaries. This can compound information loss and often flattens disagreements or minority claims across sources.

**Agentic research systems** attempt to explore a topic through iterative search, retrieval, and reasoning. Because their trajectories depend on intermediate queries and decisions, small variations in prompts can lead to different research paths and different subsets of the corpus being examined. Here too, omissions are typically implicit rather than observable.

ReadingMachine instead performs **corpus mapping**. It reads the entire corpus, extracts claims, organizes them semantically, and synthesizes the resulting structure.

For this reason, ReadingMachine is not a replacement for retrieval or agentic workflows. Those approaches remain useful for exploration and question answering. ReadingMachine instead provides a **high-fidelity structural mapping of a defined corpus**, which can then inform further exploration and analysis - see below.

---
### Complementarity with Other Approaches

ReadingMachine is best understood as a complement to retrieval and agentic research workflows rather than a replacement for them. Each approach therefore operates at a different stage of the research process: discovery, exploration, structural mapping, and reasoning.

Different tools are well suited to different stages of working with large document collections:

- **Agentic research systems** are useful for early conceptual exploration and can help identify relevant literature or assemble an initial corpus.
- **RAG systems** allow flexible exploration of that literature through ad hoc queries and targeted retrieval.
- **ReadingMachine** performs a structured reading pass across the corpus, extracting insights and producing a thematic map of the arguments contained within it.
- **Agentic workflows or targeted retrieval** can then be used again to reason over the resulting thematic structure, investigate edge cases, or explore specific claims in more detail.

A typical workflow might therefore look like:

agentic search → corpus identification  
↓  
RAG exploration → initial familiarity with the literature  
↓  
ReadingMachine → structural mapping of arguments and themes  
↓  
agentic or retrieval workflows → deeper analysis, edge-case exploration, or follow-up questions 

In this sense, ReadingMachine fills a specific role within a broader ecosystem of tools: it provides **high-fidelity thematic mapping of a corpus**, which can then inform further exploration and reasoning.

---  

## Scope and Limitations  

ReadingMachine is designed for a specific analytical task: **high-fidelity thematic synthesis of large natural-language corpora**. It is not intended to replace other approaches to working with language models, and in many contexts those approaches remain more appropriate.

### Relationship to Retrieval Systems

ReadingMachine is not a replacement for retrieval-augmented generation (RAG).

RAG systems are well suited to situations where users want to:

- ask individual questions about a corpus
- perform flexible or exploratory querying
- retrieve supporting passages quickly

ReadingMachine addresses a different problem. It is designed for situations where a researcher already has a defined corpus and wants to understand **the structure of arguments within that corpus**.

In particular, the pipeline is useful when:

- omission risks are costly
- minority or dissenting claims must be preserved
- the goal is thematic mapping rather than question answering

RAG systems retrieve fragments of text relevant to a query. ReadingMachine instead performs an **structured reading pass over the entire corpus** before synthesis occurs.

---

### Relationship to Agentic Research Workflows

ReadingMachine is also not a replacement for agentic research systems.

Agentic workflows typically involve models formulating queries, retrieving documents, and reasoning about them through iterative exploration. These systems are designed to answer questions *about* a corpus.

ReadingMachine deliberately avoids that role. The pipeline does **not perform autonomous reasoning about the corpus** beyond the interpretive tasks required for reading, comprehension, and synthesis.

As a result, ReadingMachine cannot answer analytical questions such as:

- *How many papers use a particular method?*  
- *Which approach appears to perform best?*  
- *What proportion of the literature supports a given claim?*

These are questions *about* the corpus rather than claims *within* it.

Instead, ReadingMachine produces a **faithful thematic representation of the claims present in the text itself**. The system maps what the corpus says, leaving interpretation and evaluation to the researcher (or potentially downstream agent).

---

### Scale Considerations

The pipeline is designed to operate on corpora substantially larger than those typically handled by retrieval systems or hierarchical summarization workflows. However, it is not infinitely scalable.

The key architectural difference is where scale constraints appear.

In many retrieval or agentic workflows, the context bottleneck occurs during the final synthesis step, when the model must integrate all relevant information gathered across the corpus. As the amount of material grows, this can stress the context window and lead to known failure modes such as attention loss or “missing middle” effects - further raising omission risk and challenges with citation anchoring.

ReadingMachine addresses this by converting documents into a structured intermediate representation before synthesis:

documents
→ chunks
→ insights
→ clusters
→ themes
→ synthesis

Because synthesis occurs at the **theme level rather than the corpus level**, the model never needs to reason over the entire corpus simultaneously (the only stage where broader integration occurs is theme schema generation, which is scaffolded by cluster summaries or prior theme structures and repeated iteratively). Instead, it integrates the insights associated with one thematic area at a time.

This shifts the scaling constraint from:  

entire corpus exceeds context window

to:  

a single theme exceeds context window  

In practice this allows the system to scale to larger corpora before encountering the same limitations.

Two scale constraints are nevertheless anticipated.

**Insight density**

As corpora grow, the number of extracted insights may become large enough to introduce noise in embedding space and clustering behavior. At sufficient scale, semantically similar insights may need to be consolidated or deduplicated before thematic analysis.

**Theme context limits**

Thematic synthesis still requires integrating many insights within the context window of a model. Extremely dense themes may eventually stress this stage of the pipeline.

**Redundancy trade-off**

Because themes are synthesized independently, the system prioritizes **local completeness** within each theme. Even when previous theme summaries are provided as frozen context, models must balance preserving all relevant information locally with avoiding repetition globally. In practice, this often results in some redundancy across themes in the final synthesis. This is considered acceptable as a tradeoff for lowering omission risk and ensuring marginal claims are persisted in the final output.

The architecture has been designed to accommodate future extensions that address these issues—for example through insight consolidation, staged synthesis, or hierarchical thematic structures—but these remain areas for further development.

In practice, these constraints are expected to appear **later than the scale limits typically encountered by retrieval-based or hierarchical summarization workflows**, because ReadingMachine delays compression and operates on structured insight representations rather than raw document context.

**Corpus heterogeneity**

The effects of extreme corpus heterogeneity on clustering stability and theme convergence remain an open empirical question. The current architecture is designed to expose instability (for example, through persistent orphan churn or shifting theme schemas across iterations) rather than conceal it. Future benchmarking will examine how heterogeneity interacts with clustering and synthesis behavior.
---

### Interpretive Limits

ReadingMachine does not eliminate interpretation.

Decisions about corpus selection, research questions, clustering parameters, and theme structures all shape the resulting synthesis. The goal of the pipeline is not to produce neutral or definitive interpretations, but to make the analytical process **transparent and inspectable**.

By preserving intermediate artifacts—from chunks to insights to themes—the system allows researchers to examine how particular conclusions emerge from the corpus and how different analytical choices affect the results.
---

## Potential Applications

ReadingMachine can support many types of corpus analysis, including:

- literature synthesis  
- policy analysis  
- organizational knowledge mapping  
- legal corpus analysis  
- qualitative research coding  
- analysis of evaluation reports or institutional documentation  

The pipeline may also serve as an input layer for more complex analytical workflows, where structured insight sets are used as inputs to downstream reasoning systems - see above.

---

## Design Philosophy

ReadingMachine treats large language models primarily as **industrial-scale readers**.

The pipeline coordinates that capability within a constrained and inspectable analytical workflow that structures extraction and recombination rather than using free-form reasoning. Notably, rather than attempting to replace human reasoning, the system focuses on organizing large-scale reading so that researchers can more effectively interpret the resulting structures.

The goal is not automated knowledge production, but **structured access to the arguments and insights contained within large document collections**.

In this sense, ReadingMachine treats large language models not as autonomous reasoners, but as components in a reproducible method for coordinating large-scale reading.

## Collaboration and Method Development

ReadingMachine is presented as an evolving methodological framework rather than a finished product. Its robustness, limitations, and boundary conditions can only be meaningfully assessed through broader use and critical engagement.

We are actively seeking collaboration in the following areas:

- expert evaluation of thematic quality across domains

- adversarial testing designed to stress or break the pipeline

- benchmarking both against human-led synthesis and formal relevant benchmark development

- parameter sensitivity and clustering diagnostics

- large-scale heterogeneity testing

- code hardening and performance optimization

Researchers are encouraged not only to use the system, but to challenge it. Identifying failure modes, instability conditions, or systematic distortions is considered a productive contribution to the development of the method.

This project should be understood as an open methodological experiment in large-scale structured reading.

## Environment

ReadingMachine has been tested on:

- Python 3.12
- scikit-learn 1.3.2
- umap-learn 0.5.7

These versions are pinned to avoid compatibility issues between UMAP, HDBSCAN, and sklearn.