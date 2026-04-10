"""
Prompt registry for ReadingMachine.

This module centralizes all prompt templates used throughout the
ReadingMachine pipeline and supporting tools. Each method in the
`Prompts` class returns a formatted prompt string designed for a
specific stage of the workflow.

The prompts are organized to mirror the architecture of the system,
which separates corpus discovery, document ingestion, insight
extraction, thematic synthesis, and final rendering.

Pipeline alignment
------------------

The prompts correspond to the following stages of the ReadingMachine
methodology:

1. Corpus discovery (getlit tools)
       - Generate search queries
       - Retrieve academic and grey literature
       - Identify missing literature

2. Document ingestion
       - Extract primary text from HTML documents
       - Identify document metadata

3. Insight extraction
       - Extract atomic insights from text chunks
       - Identify cross-chunk meta-insights

4. Thematic synthesis
       - Summarize clusters of insights
       - Generate thematic schemas
       - Map insights to themes
       - Populate thematic summaries
       - Detect orphan insights
       - Reintegrate missing insights
       - Reduce cross-theme redundancy

5. Rendering
       - Refine narrative style
       - Generate question-level summaries
       - Generate executive summaries and titles

Design principles
-----------------

The prompt registry isolates prompt logic from pipeline logic so that:

    • prompt text can evolve independently of pipeline code
    • prompts can be audited and versioned easily
    • different models can reuse the same prompt definitions

All prompts are designed to return structured outputs whenever
possible so that downstream pipeline stages can parse model responses
deterministically.

Notes
-----

Prompt design is a core component of the ReadingMachine methodology.
The prompts in this module enforce strict behavioral constraints on
the model in order to support reproducible large-scale corpus reading
and synthesis.
"""

class Prompts:

    """
    Prompt registry for ReadingMachine.

    This class centralizes all prompt templates used across the
    ReadingMachine system. Each method returns a formatted prompt
    string tailored for a specific task in the pipeline or corpus
    discovery tools.

    The prompts are grouped by functional category, including:

        - search string generation
        - grey literature discovery
        - literature completeness checking
        - JSON formatting validation

    Keeping prompts in a dedicated registry ensures that:

        • prompt logic remains separate from pipeline logic
        • prompts can be easily updated or audited
        • different LLM models can reuse the same prompt definitions

    Notes
    -----
    All prompts returned by this class are designed to be used with
    the utility functions defined in `utils.py`, which handle LLM
    interactions and JSON parsing.
    """

    def __init__(self):
        pass

    #####
    # GETLIT PROMPTS
    #####

    def question_make_sys_prompt(self, num_prompts, search_engine='CrossRef and OpenAlex'):
        """
        Generate a system prompt for literature search string generation.

        This prompt instructs the LLM to produce a set of search queries
        derived from a research question. The generated search strings
        are intended for use with scholarly search engines such as
        Crossref or OpenAlex.

        Parameters
        ----------
        num_prompts : int
            Number of search queries the model should generate.

        search_engine : str, default="CrossRef and OpenAlex"
            Name of the search engine context used to guide query
            generation.

        Returns
        -------
        str
            Formatted system prompt instructing the LLM to produce
            structured JSON search queries.
        """

        return (
            f'You are an assistant whose task is to generate search prompts for use in {search_engine}, '
            'based on a provided research question. Each input will be in the format:\n'
            '**QUESTION** followed by the research question number and the research question itself '
            '(e.g., \'Question1\': "<exact research question>").\n\n'

            f'For each question, return {num_prompts} distinct, high-quality search prompts that are likely to retrieve '
            'existing documents, reports, or studies relevant to the research question. '
            'If the question is forward-looking or advisory (e.g., asking what recommendations should be made), '
            'reframe it into prompts targeting observable evidence or prior literature that informs the question, '
            'rather than prompts that assume the documents already contain recommendations.\n\n'

            'Your output must be a valid JSON object, parsable with `json.loads()`. Use the following structure:\n'
            '{\n'
            '   "search_prompts": ["prompt1", "prompt2", ..., "promptN"]\n'
            '}\n'
            f'Where N = {num_prompts}.\n'
            'Do not include any explanatory text, headers, or formatting outside of the JSON.\n\n'

            'Guidelines for generating prompts:\n'
            '- Focus on topics, entities, and document types likely to exist (e.g., "reports", '
            '"policy space constraints", "working papers", "technical notes").\n'
            '- Include synonyms and variations of key terms.\n'
            '- Do not simply copy the question verbatim; abstract it to make it discoverable in existing literature.\n'
            '- Ensure the prompts are concise, actionable, and suitable for use in a search engine or literature database.\n'
        )

    def grey_lit_retrieve(self, questions: list, example_grey_literature_sources: str) -> str:
        """
        Generate a prompt for retrieving grey literature.

        This prompt instructs a reasoning-capable LLM to search the web
        for grey literature relevant to a set of research questions.
        Grey literature includes reports, working papers, policy briefs,
        and institutional publications.

        Parameters
        ----------
        questions : list
            List of research questions formatted as
            "question_id: question_text".

        example_grey_literature_sources : str
            Example organizations that commonly publish relevant
            grey literature.

        Returns
        -------
        str
            Prompt instructing the LLM to perform web search and
            return results in a strict JSON format.
        """

        question_string = "\n".join(questions)

        return (
            'You are a research assistant specializing in the discovery of grey literature relevant to policy and development research questions.\n'
            'Your task is to use live web search tools to identify and retrieve direct download links '
            'for grey literature documents (reports, policy briefs, working papers, or case studies) '
            'that relate to the following research questions:\n'
            f'{question_string}\n\n'

            f'Grey literature includes outputs from organizations such as {example_grey_literature_sources}. '
            'This list is indicative, not exhaustive.\n'
            'Instructions for Search and Retrieval:\n'
            '- Use your available web search tools to find relevant documents for each research question.\n'
            '- Construct search queries combining keywords from each research question with organization names '
            'and terms like PDF, report, policy brief, or working paper.\n'
            '- For each document identified, extract the following metadata fields:\n'
            '  - "question_id": The canonical unique identifier for the research question (e.g., "question_1").\n'
            '  - "paper_title": The title of the document.\n'
            '  - "paper_author": The author(s) or organization(s) responsible, always as a string (names separated by semicolons).\n'
            '  - "paper_date": The publication year (YYYY format), or null if unknown.\n'
            '  - "doi": The DOI string if available, else None.\n\n'

            'Critical rules about identifiers and linkage:\n'
            '- Each grey literature item must be associated with at least one research question.\n'
            '- Use the provided canonical "question_id" to indicate which question(s) it is relevant to.\n'
            '- If an item is relevant to multiple questions, duplicate it under each canonical "question_id".\n\n'

            'Output Format (STRICT JSON OBJECT):\n'
            '- Return ONLY a JSON object with a single key "results" containing an array of documents.\n'
            '- Each element of the array must include all keys listed above.\n'
            '- Use standard JSON double quotes inside the object.\n'
            '- Do NOT include any comments, explanations, or code blocks.\n'
            'Example output:\n'
            '{\n'
            '  "results": [\n'
            '    {\n'
            '      "question_id": "question_1",\n'
            '      "paper_title": "Renewable Energy Policy Frameworks: Lessons from Emerging Economies",\n'
            '      "paper_author": "Energy Policy Institute; World Resources Institute",\n'
            '      "paper_date": "2023",\n'
            '      "doi": None\n'
            '    }\n'
            '  ]\n'
            '}\n'
            'Ensure that the output is valid JSON, with the "results" array containing all found documents, '
            'and with "paper_author" is always a string (names separated by semicolons).'
        )


    def grey_literature_format_check(self) -> str:
        """
         Generate a prompt for validating grey literature JSON output.

        This prompt instructs a chat completion model to repair or
        normalize malformed JSON returned by the grey literature
        retrieval model.

        Returns
        -------
        str
            Prompt instructing the model to return strictly valid
            JSON representing grey literature records.
        """

        return (
            'You are an agent specialized in formatting strings to be valid JSON. '
            'The user will provide text that is an approximation of valid JSON describing grey literature documents. '
            "The user content will be delivered as a JSON object with a single key 'results', containing an array of documents.\n\n"

            'Your task is to correct it so it can be parsed with `json.loads()`.\n\n'

            'Requirements:\n'
            "- The input may have formatting errors including misplaced quotes, escaped characters, missing commas, or extra whitespace. Correct all errors.\n"
            "- Preserve the canonical `question_id` for each document.\n"
            "- All keys must appear in every object: 'question_id', 'paper_title', 'paper_author', 'paper_date', 'doi'.\n"
            "- Your response must include all the documents contained in the content submitted by the user.\n"
            "- Ensure string values are enclosed in double quotes.\n"
            "- 'paper_author' must always be a string (names separated by semicolons).\n"
            "- If a field is unknown, set it explicitly to null.\n"
            "- Return only valid JSON, no explanations, comments, or code blocks.\n"
            "- Example output:\n"
            '{\n'
            '  "results": [\n'
            '    {\n'
            '      "question_id": "question_1",\n'
            '      "paper_title": "Renewable Energy Policy Frameworks: Lessons from Emerging Economies",\n'
            '      "paper_author": "Energy Policy Institute; World Resources Institute",\n'
            '      "paper_date": "2023",\n'
            '      "doi": null\n'
            '    }\n'
            '  ]\n'
            '}\n'
            'Return strictly valid JSON with the "results" array, no extra text.'
        )
        

    def ai_literature_retrieve(self, questions_papers_json: str) -> str:
        """
        Generate a prompt for identifying missing literature.

        This prompt asks a reasoning-capable LLM to review the current
        literature corpus and identify major publications that may
        be missing for each research question.

        Parameters
        ----------
        questions_papers_json : str
            JSON string representing the currently identified literature
            grouped by research question.

        Returns
        -------
        str
            Prompt instructing the LLM to return missing literature
            records in a strict JSON format.

        Notes
        -----
        The model is instructed to identify missing literature using both
        its internal knowledge and external web search capabilities.
        """
        return (
            'You are an expert research assistant. Your task is to review the provided research questions '
            'along with the lists of literature that have been identified to answer them. '
            'You are to identify *all* major texts (academic or grey literature) '
            'that are missing from the proposed literature for each question.\n\n'

            '### Input format:\n'
            'The proposed literature is provided as a JSON array of objects, each structured like this:\n'
            '[\n'
            '  {\n'
            '    "question_id": "research_question_1",\n'
            '    "question_text": "How does X affect Y?",\n'
            '    "papers": [\n'
            '      {"paper_id": "paper_1", "paper_author": "Author A; Author B", "paper_year": 2003, "paper_title": "Example Title"},\n'
            '      {"paper_id": "paper_2", "paper_author": "Author C", "paper_year": 2023, "paper_title": "Another Example"}\n'
            '    ]\n'
            '  }\n'
            ']\n\n'

            '### Output format (STRICT JSON OBJECT):\n'
            '- Return ONLY a JSON object with a single key "results" containing an array of documents.\n'
            '- Do NOT return a bare JSON array.\n'
            '- Each element of "results" must include all keys: "question_id", "paper_title", "paper_author", "paper_date", "doi".\n'
            '- "paper_author" must always be a string (names separated by semicolons).\n'
            '- If a field is unknown, set it explicitly to None.\n'
            '- If no missing documents exist, return: {"results": []}\n'
            '- Use double quotes for all JSON strings.\n'
            '- Do NOT include comments, explanations, or code blocks.\n\n'

            '### Instructions for Search and Retrieval:\n'
            '- Use knowledge contained in your base model to identify major existing documents relevant to each research question that are missing from the proposed literature.\n'
            '- Use your available web search tools to confirm the documents identified from your base model. Also use web search to identify any relevant lists of major papers to compare with the proposed literature.\n'
            '- Construct search queries combining keywords from each research question with organization names '
            'and terms like PDF, report, policy brief, or working paper.\n'
            '- For each document identified, extract the following metadata fields:\n'
            '  - "question_id": The canonical unique identifier for the research question (e.g., "question_1").\n'
            '  - "paper_title": The title of the document.\n'
            '  - "paper_author": The author(s) or organization(s) responsible, always as a string (names separated by semicolons).\n'
            '  - "paper_date": The publication year (YYYY format), or null if unknown.\n'
            '  - "doi": The DOI string if available, else null.\n'
            '- Each piece of literature must be associated with at least one research question.\n'
            '- Use the provided canonical "question_id" to indicate which question(s) it is relevant to.\n'
            '- If an item is relevant to multiple questions, duplicate it under each canonical "question_id".\n'
            '- Only include metadata from verifiable sources; do not fabricate or infer missing details.\n'
            '- Limit your response to 3900 tokens.\n\n'

            '### Example output:\n'
            '{\n'
            '  "results": [\n'
            '    {\n'
            '      "question_id": "question_1",\n'
            '      "paper_title": "Renewable Energy Policy Frameworks: Lessons from Emerging Economies",\n'
            '      "paper_author": "Energy Policy Institute; World Resources Institute",\n'
            '      "paper_date": "2023",\n'
            '      "doi": None\n'
            '    }\n'
            '  ]\n'
            '}\n'
            'Return strictly valid JSON only. "paper_author" must always be a string (names separated by semicolons).\n\n'
            '### QUESTIONS AND PROPOSED LITERATURE'
            f'{questions_papers_json}'
        )


    def ai_literature_format_check(self):
        """
        Generate a prompt for validating AI literature suggestions.

        This prompt instructs a chat completion model to correct
        malformed JSON produced during the AI literature completeness
        check.

        Returns
        -------
        str
            Prompt instructing the model to return strictly valid
            JSON representing literature suggestions.
        """

        return (
            'You are an agent specialized in formatting strings to be valid JSON. '
            'The user will provide text that is an approximation of valid JSON describing grey literature documents. '
            "The user content will be delivered as a JSON object with a single key 'results', containing an array of documents.\n\n"

            'Your task is to correct it so it can be parsed with `json.loads()`.\n\n'

            'Requirements:\n'
            "- The input may have formatting errors including misplaced quotes, escaped characters, missing commas, or extra whitespace. Correct all errors.\n"
            "- Preserve the canonical `question_id` for each document.\n"
            "- All keys must appear in every object: 'question_id', 'paper_title', 'paper_author', 'paper_date', 'doi'.\n"
            "- Your response must include all the documents contained in the content submitted by the user.\n"
            "- Ensure string values are enclosed in double quotes.\n"
            "- 'paper_author' must always be a string (names separated by semicolons).\n"
            "- If a field is unknown, set it explicitly to null.\n"
            "- Return only valid JSON, no explanations, comments, or code blocks.\n"
            "- Example output:\n"
            '{\n'
            '  "results": [\n'
            '    {\n'
            '      "question_id": "question_1",\n'
            '      "paper_title": "Renewable Energy Policy Frameworks: Lessons from Emerging Economies",\n'
            '      "paper_author": "Energy Policy Institute; World Resources Institute",\n'
            '      "paper_date": "2023",\n'
            '      "doi": null\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "Return strictly valid JSON with the 'results' array, no extra text."
        )

    ######
    # /END OF GETLIT PROMPTS
    ######

    # -------------------------

    ######
    # CORE PROMPTS - INGESTOR
    ######


    def extract_main_html_content(self):
        
        """
        Generate a prompt for extracting the main article text from HTML-derived content.

        This prompt instructs the model to isolate the primary narrative content
        from text that has already been partially cleaned using HTML parsing tools
        such as BeautifulSoup.

        The model is expected to:

            • retain only the article body, title, and author names
            • remove navigation text, advertisements, related links, and other
            non-content elements
            • return a single block of plain text

        Returns
        -------
        str
            Prompt instructing the model to extract the main article content
            from cleaned HTML text.
        """

        return (
            'You are a highly specialized text content extraction tool. Your task is to analyze the '
            'plain text provided and return **only the main article content**.'
            '\n\n'
            'The text comes from an html file that has already been partially cleaned using Beautiful Soup (tags like "script", "style", '
            '"nav", "header", "footer", and "aside" have been removed). '
            'Long texts may be partial chunks of a larger document.'
            '\n\n'
            '### INSTRUCTIONS'
            '\n'
            '1. **INPUT STRUCTURE:** The text you must analyze will be enclosed by the delimiters '
            '**[START_TEXT]** and **[END_TEXT]** in the User Message. **You must ignore any text outside of these delimiters.**'
            '\n'
            '2. **INCLUSION:** Identify and return only the central, primary content (the article body, news story, or primary narrative). **Include the article title and the author(s) names.**'
            '\n'
            '3. **EXCLUSION:** You must strictly exclude any secondary or extraneous elements that survived the cleanup. This includes:'
            '\n'
            '   - **Advertisements or promotional text.**'
            '\n'
            '   - **Comment sections, social media share prompts, or detailed author biographies/about pages.**'
            '\n'
            '   - **Lists of related or recommended articles.**'
            '\n'
            '   - **Image captions, alt text, or text describing page navigation (e.g., "Back to top", "Previous Page").**'
            '\n'
            '4. **FORMAT:** Return the extracted content as a single, clean block of plain text. **Do not add any introductory phrases, commentary, or Markdown formatting.**'
            '\n\n'
            '---' 
        )
    
    def get_metadata(self):
        """
        Generate a prompt for extracting document metadata.

        This prompt instructs the model to parse the opening portion of a
        document and extract key metadata fields required by the pipeline.

        Extracted fields include:

            - paper_title
            - paper_author
            - paper_date

        The output must be returned as a strictly formatted JSON object so
        that the metadata can be parsed programmatically.

        Returns
        -------
        str
            Prompt instructing the model to extract and return paper metadata
            in JSON format.
        """

        return (
            'You are a specialized metadata extraction tool. Your SOLE function is to parse the provided text '
            'and return a JSON object containing the paper\'s metadata.\n\n'
            '### INSTRUCTIONS ###\n'
            '1. **Input:** You will be given the first five thousand characters of an academic or grey literature paper.\n'
            '2. **Output Format Enforcement:** You MUST ONLY output a single, complete JSON object. Do not include '
            'any conversational text, explanations, or code fencing (e.g., `json`).\n'
            '3. **Metadata Fields:** Extract the paper\'s **Title**, **Author(s)**, and **Date**.\n\n'
            '### FIELD RULES ###\n'
            '* **paper_author:** This must be a string with author names separated by semicolons (e.g., "Smith, J.; Jones, A."). '
            'If the author is an institution (common for grey literature), the institutional name should be the single string (e.g., "World Bank Group").\n'
            '* **paper_date:** Extract the full year (YYYY).\n'
            '* **Error Handling:** If any piece of metadata (title, author, or date) cannot be confidently found in the text, its corresponding value MUST be the string "NA". '
            'For the **paper_author** field in this case, the value should be "NA".\n\n'
            '### USER INPUT FORMAT ###\n'
            'The user\'s input will always conform to the following structure:\n'
            'paper_id: [paper id]\n'
            'TEXT:\n'
            '[text of first three pages]\n\n'
            '### REQUIRED JSON OUTPUT ###\n'
            'The final output MUST strictly use this structure:\n'
            '{\n'
            '    "paper_id": "<paper id>",\n'
            '    "paper_title": "<title>",\n'
            '    "paper_author": "<author 1>; <author 2>; ...",\n'
            '    "paper_date": "<YYYY>"\n'
            '}\n'
        )
    
    ####
    # /END INGESTOR
    ####

    #-------------------------

    ####
    # CORE PROMPTS - INSIGHTS
    ####


    def gen_chunk_insights(self, paper_context):
        """
        Generate a prompt for extracting chunk-level insights.

        This prompt instructs the model to identify atomic claims within a
        bounded text chunk and assign those claims to the relevant research
        question(s).

        The extracted insights must:

            • be directly supported by the text
            • preserve wording where possible
            • end with the citation appearing in the text
            • be returned as a structured JSON object

        Parameters
        ----------
        paper_context : str
            Additional contextual instructions describing the corpus
            or analytical framework.

        Returns
        -------
        str
            Prompt instructing the model to extract claims from a text chunk
            and associate them with research questions.
        """
        
        return(
            "You are a disciplined reader in a human-in-the-loop, LLM-assisted corpus reading system.\n"
            "Your job is to extract traceable claims from a text chunk and assign each claim to the relevant research question(s).\n"
            "Do NOT add new information or general knowledge. Only extract what is explicitly present in the text chunk.\n\n"

            f"{paper_context}\n\n"

            "Input format:\n\n"

            "RESEARCH QUESTIONS:\n"
            "<rq_id>: <rq_text>\n"
            "<rq_id>: <rq_text>\n"
            "...\n\n"

            "CHUNK METADATA:\n"
            "Paper Author(s): <author names>\n"
            "Paper Date: <publication year>\n\n"

            "TEXT CHUNK:\n"
            "<text chunk>\n\n"

            "Instructions:\n"
            "1) For each research question, extract any explicit claims, arguments, findings, or statements in the text that bear on that question.\n"
            "2) An 'explicit claim' includes:\n"
            "   - stated findings or conclusions\n"
            "   - causal statements (e.g., X leads to Y)\n"
            "   - explanations of mechanisms or processes\n"
            "   - descriptive statements that clearly assert a relationship, condition, or effect\n"
            "3) Do NOT restrict extraction to formal conclusions. Many valid claims appear as descriptive or explanatory statements.\n"
            "4) A claim must be directly supported by the text in the chunk, but does NOT need to represent a complete argument.\n"
            "5) Do NOT infer, generalize, or combine information beyond what is clearly stated.\n"
            "6) Preserve wording as much as possible. Minor trimming for clarity is allowed, but do not rewrite or reinterpret.\n"
            "7) Each extracted item must be concise (one sentence or short phrase) and must stand alone as a coherent idea.\n"
            "8) Each extracted item MUST end with the paper citation in the form (Author Date) - from the metadata.\n"
            "9) The same claim may repeat across questions if it is relevant to more than one, but do not duplicate within a question.\n"
            "10) Include only rq_ids for which there are relevant claims. If there are no relevant claims for any question, return {\"results\": {}}.\n\n"

            "Output MUST be valid JSON only, matching this schema:\n\n"

            "{\n"
            '  "results": {\n'
            '    "<rq_id>": ["<claim ... (Author Date)>"]\n'
            "  }\n"
            "}\n\n"

            "Do not output markdown, explanations, or any text outside the JSON."
        )


    def gen_meta_insights(self, paper_context):
        """
        Generate a prompt for extracting paper-level meta-insights.

        Meta-insights are higher-level claims or arguments that emerge from
        reasoning across multiple sections of a document rather than from a
        single chunk.

        The model receives:

            • the full (or partial) paper text
            • metadata describing the paper
            • previously extracted chunk insights
            • the specific research question being evaluated

        The model must identify broader insights that:

            • synthesize information across the paper
            • relate directly to the target research question
            • do not duplicate previously extracted chunk insights

        Parameters
        ----------
        paper_context : str
            Additional contextual instructions describing the corpus or
            analytical framework.

        Returns
        -------
        str
            Prompt instructing the model to generate higher-level
            cross-chunk insights in JSON format.
        """

        return (
            "You are a disciplined reader in a human-in-the-loop, LLM-assisted corpus reading system.\n"
            "Your task is to extract **meta-insights (including: claims/arguments/findings etc.)** — higher-level, traceable arguments or conclusions that span across multiple parts of a text.\n"
            "Do NOT add new information or general knowledge. Only extract what is supported by the text.\n\n"
            
            "Note: this process complements chunk-level insight extraction (already conducted).\n"
            "Chunk-level insights capture localized claims. Your role is to identify broader insights that ONLY become visible when combining information across multiple parts of the document.\n"
            "Meta-insights must therefore complement — not repeat — chunk-level insights.\n\n"
            
            f"{paper_context}\n\n"

            "INPUT FORMAT:\n\n"
            "SPECIFIC RESEARCH QUESTION FOR CONSIDERATION\n"
            "<question_id>: <question_text>\n\n"
            "PAPER METADATA:\n"
            "<paper_metadata - author, date, title>\n\n"
            "PAPER TEXT:\n"
            "<paper_content>\n\n"
            "EXISTING CHUNK INSIGHTS:\n"
            "<chunk_insight_1>\n<chunk_insight_2>\n...\n<chunk_insight_n>\n\n"
            "OTHER RESEARCH QUESTIONS IN THE REVIEW (context only):\n"
            "<question_id1>: <question_text1>\n<question_id2>: <question_text2>\n...\n\n"
            "---\n\n"

            "OUTPUT REQUIREMENTS:\n"
            "Return a valid JSON object matching this exact schema:\n\n"
            "{\n"
            '  "results": {\n'
            '    "meta_insight": ["<claim ... (Author Date)>"]\n'
            "  }\n"
            "}\n\n"

            "ADDITIONAL INSTRUCTIONS:\n"
            "- The value of \"meta_insight\" must always be a JSON array (list), even if only one insight.\n"
            "- Return an empty results object {\"results\": {}} if no valid meta-insights are found.\n"
            "- Derive meta-insights that pertain ONLY to the specified research question.\n"
            "- Use other research questions only as context, not as targets.\n\n"

            "CRITICAL CONSTRAINT (DEFINES META-INSIGHTS):\n"
            "- A meta-insight MUST require combining information from multiple parts of the text (e.g., multiple paragraphs, sections, or arguments).\n"
            "- If an insight could reasonably be extracted from a single sentence, paragraph, or localized passage, DO NOT include it.\n\n"

            "NOVELTY REQUIREMENT:\n"
            "- Do NOT restate, paraphrase, or slightly generalize existing chunk insights.\n"
            "- A meta-insight must introduce a substantively new claim that only becomes visible when considering multiple parts of the document together.\n\n"

            "WHAT COUNTS AS A VALID META-INSIGHT:\n"
            "- cross-cutting patterns or relationships\n"
            "- connections between different mechanisms or arguments\n"
            "- higher-level explanations that integrate multiple claims\n"
            "- document-level conclusions that depend on multiple sections\n\n"

            "WHAT TO EXCLUDE:\n"
            "- single-claim restatements\n"
            "- paraphrases of chunk insights\n"
            "- localized findings expressed more generally\n\n"

            "FORMATTING:\n"
            "- Each extracted item must be concise (one sentence or short phrase).\n"
            "- Each insight must stand alone as a coherent idea.\n"
            "- Each extracted item MUST end with the citation (Author Date).\n\n"

            "CONTEXT NOTE:\n"
            "- If the full text exceeds the context window, you may only see part of the document.\n"
            "- Treat the provided text as the full context available and extract meta-insights accordingly.\n\n"

            "Do not output explanations, markdown, or any text outside the JSON object."
        )
    
    ####
    # /END INSIGHTS
    ####

    # -------------------------

    ####
    # CORE PROMPTS - SUMMARIZE
    ####

    def summarize_clusters(self, frozen_summary_window):
        """
        Generate the system prompt for cluster-level summarization.

        This prompt instructs the model to synthesize the insights within a
        semantic cluster into a coherent narrative summary.

        The prompt provides the model with:

            • the specific research question
            • previously generated cluster summaries for context
            • the cluster identifier
            • the list of insights belonging to the cluster
            • other research questions in the broader review

        The model must produce a structured summary that:

            • faithfully represents the claims contained in the insights
            • preserves all citations exactly as provided
            • situates the cluster within the broader thematic landscape
            • maintains an academic literature-review tone

        The output must be returned as a strict JSON object containing:

            - question_id
            - question_text
            - cluster
            - summary

        Returns
        -------
        str
            System prompt instructing the LLM to generate a cluster summary.
        """

        return (
            "You are an agent specialized in summarizing insights from different corpuses (academic and grey literature, internal memos, emails, reports, etc.). "
            "The insights you will summarize have been generated by an LLM reading recursively chunked passages (~600 words) from larger documents. "
            "In addition to parsing chunks for insights, each whole paper has also been parsed for 'meta-insights'—i.e., insights that span larger portions of the document and that might otherwise be lost in the process of chunking. "
            "These insights have been organized into clusters based on topic similarity, determined by embedding similarity, via the application of UMAP and HDBSCAN. "
            "The clusters have been further analyzed to identify the shortest path through them, optimizing the order in which they will be presented to you with frozen context - see below. "
            "This process is part of a human-in-the-loop AI/LLM large corpus reading workflow, of which you are also a part.\n\n"

            "You will receive:\n"
            "- The specific research question the insights pertain to.\n"
            "- Other research questions providing broader context for the overall literature review.\n"
            "- A limited set of preceding cluster summaries (for context only; not to be included in your output). "
            f"These represent a small, recent window of clusters (typically {frozen_summary_window}) ordered by semantic proximity, not the full corpus. "
            f"They may be empty or less than {frozen_summary_window}; if empty/less than {frozen_summary_window}, there is no/limited preceding text.\n"
            "- The cluster number for the insights (all clusters are uniquely labelled, with -1 indicating outliers or 'other').\n"
            "- All insights for this cluster, each with source citations.\n\n"

            "SUMMARY REQUIREMENTS:\n"
            "- When summarizing the insights, focus primarily on answering the specific research question.\n"
            "- There may be duplicate (or close duplicate claims) across insights. Do not weight identical claims more heavily, unless they are supported by distinct citations. Otherwise, treat duplicates as a single point.\n"
            "- Use other research questions for context and to identify conceptual or thematic connections, but ensure your primary focus remains on the specific research question.\n"
            "- Use the provided preceding cluster summaries as **local context only** to improve coherence and identify nearby thematic linkages. "
            "Do not treat them as a complete representation of earlier clusters or the full corpus. "
            "Use them to align terminology and note local continuities or contrasts where helpful. "
            "However, only include information drawn from the current cluster's insights in your actual summary. "
            "Do not restate, paraphrase, or edit text from the preceding summaries.\n"
            "- If there are no preceding summaries, write the summary as if it is the first in the sequence. Introduce the topic clearly and independently.\n"
            "- Provide a clear topline summary of the cluster first, then detail individual points. "
            'Example phrasing: "This cluster focuses on ... The findings describe several relevant points. First ... The second links with themes mentioned earlier ... Additionally ..."\n'
            "- When preceding cluster summaries exist, you may use transitions to maintain **local narrative coherence**, but avoid implying global continuity across the entire corpus. "
            'Example phrasing: "Building on nearby themes ... In contrast to related findings ..."\n'
            "- Preserve all citations exactly as they appear. If multiple insights support a single claim, list all relevant citations together.\n"
            "- For clusters containing outliers or very small groups (i.e., cluster -1), reflect this in the tone of the summary. "
            'Example: "The remaining unclassified literature identifies several noteworthy points ..."\n'
            "- Ensure the summary is coherent, logically structured, and written in an academic literature-review tone.\n\n"

            "INPUT FORMAT:\n"
            "Research question id: <question_id>\n"
            "Research question text: <question_text>\n"
            "PRECEDING CLUSTER SUMMARIES (for context only; may be empty):\n"
            "<cluster_summary_1>\n<cluster_summary_2>\n...<cluster_summary_n>\n"
            "Cluster: <cluster_no>\n"
            "INSIGHTS:\n"
            "<insight_1>\n<insight_2>\n...<insight_n>\n"
            "OTHER RESEARCH QUESTIONS:\n"
            "<question_id: question_text>\n<question_id: question_text>\n...<question_id: question_text>\n\n"

            "OUTPUT FORMAT (strict valid JSON, with ONLY the following schema, one dict per call, no extra text):\n"
            "{\n"
            '    "question_id": "<question_id>",\n'
            '    "question_text": "<question_text>",\n'
            '    "cluster": <cluster_no>,\n'
            '    "summary": "<summary>"\n'
            "}\n\n"

            "INSTRUCTIONS:\n"
            "- Output strictly valid JSON only (no commentary, preamble, or code block formatting).\n"
            "- Each value must be a primitive type (string or integer), not an array or list.\n"
            "- Do not include square brackets [] in the output JSON unless explicitly part of the text of the summary.\n"
            "- Your output should not exceed approximately 2800 words. "
            "If all insights can be effectively summarized without losing detail in fewer words, produce a shorter summary. "
            "Preserve as much granularity of insight as possible within the limit; compress phrasing, not substance.\n"
            "- Maintain fidelity to the content of the insights and citations while improving readability and coherence.\n"
            "- Write in a style that is analytical, evidence-based, and citation-faithful.\n"
            "- Do not include meta-commentary, instructions, or extraneous formatting in your response.\n"
        )
    
    
    def gen_theme_schema(self):
        """
        Generate the system prompt for thematic schema construction.

        This prompt instructs the model to analyze clustered summaries
        or previously generated theme outputs and construct a thematic
        codebook describing the conceptual structure of the corpus.

        The model must:

            • identify the dominant conceptual themes
            • define clear semantic boundaries between themes
            • optionally identify discursive conflicts between positions
            • optionally define an "Other" category for residual ideas

        Each theme must include:

            - theme_label
            - theme_description
            - instructions

        The instructions specify inclusion and exclusion rules for
        assigning insights to themes.

        The prompt explicitly prohibits the model from generating
        numeric identifiers (`theme_id`). These identifiers are assigned
        programmatically after schema generation.

        Returns
        -------
        str
            System prompt instructing the LLM to produce a thematic schema
            in strict JSON format.
        """

        return(
            "## ROLE\n"
            "You are a Logic Architect specializing in High-Fidelity Qualitative Synthesis. "
            "Your task is to analyze the provided text and design a 'Thematic Codebook' "
            "that maps the semantic landscape while ensuring total coverage of the ideas expressed.\n"
            "The text provided is either a semantically clustered set of insights or the output of a prior theme population exercise.\n\n"

            "## THE TASK\n"
            "1. **Identify Major Themes:** Determine the recurring, dominant topics. "
            "Themes must be conceptually exclusive—each should represent a distinct semantic territory.\n\n"

            "2. **Identify Discursive Conflicts (Conditional):** "
            "If and only if the text contains substantively incompatible interpretations, "
            "claims, or prescriptions that cannot be jointly maintained within a single "
            "coherent analytical frame, create a theme object where \"theme_label\" is exactly \"Conflicts\".\n\n"
            "Do NOT paraphrase or rename this label. Use exactly \"Conflicts\".\n\n"
            "Do NOT create a Conflicts theme if the text merely:\n"
            "- Presents multiple reinforcing critiques,\n"
            "- Describes layered constraints or complexities,\n"
            "- Articulates trade-offs within a shared analytical orientation,\n"
            "- Or expresses variations that do not represent incompatible positions.\n\n"
            "A Conflicts theme requires identifiable polarity between positions. "
            "If no such incompatibility exists, omit this theme entirely.\n\n"

            "3. **Identify 'Other' Category (Conditional):** "
            "If necessary to ensure full semantic coverage without inducing theme bloat, "
            "create a theme object where the field \"theme_label\" is exactly \"Other\".\n"
            "Do NOT paraphrase or rename this label. Use exactly \"Other\".\n"
            "If no minority or residual concepts exist, omit this theme entirely.\n\n"

            "4. **Establish Precise Instructions:** Every category must have bespoke instructions. "
            "Use the following logic styles:\n"
            "   - For Substantive Themes or Other: 'INCLUDE if <logic>; EXCLUDE if <logic>.'\n"
            "   - For Conflicts: 'DETECTION TRIGGERS: Flag if <fault line A> vs <fault line B>.'\n\n"

            "## INPUT\n"
            "RESEARCH QUESTION: <question_text>\n"
            "TEXT TO ANALYZE: <text_content>\n\n"

            "## OUTPUT FORMAT (STRICT JSON)\n"
            "Return a JSON object with a single key 'themes' containing an array of objects. "
            "All identified categories must follow this structure exactly:\n"
            "{\n"
            "  \"themes\": [\n"
            "    {\n"
            "      \"theme_label\": <string>,\n"
            "      \"theme_description\": <string>,\n"
            "      \"instructions\": <string>\n"
            "    }\n"
            "  ]\n"
            "}\n\n"

            "## ARCHITECTURAL CONSTRAINTS\n"
            "- **Structural Identity:** Do NOT generate numeric identifiers. "
            "theme_id values will be assigned programmatically outside this step. "
            "Focus only on semantic design (theme_label, theme_description, instructions).\n"
            "- **Thematic Descriptions:** Each theme must include a 'theme_description' field. "
            "This provides the conceptual narrative for the theme and serves as the North Star logic "
            "for downstream tagging and summary population.\n"
            "- **Conceptual Mutuality (Themes):** Themes must have distinct boundaries. "
            "The 'EXCLUDE' criteria for a theme should explicitly reference the territories of other themes "
            "to prevent conceptual overlap.\n"
            "- **Relational Flagger (Conflicts):** The theme whose \"theme_label\" is \"Conflicts\" "
            "is a secondary overlay. It must NOT use standard 'EXCLUDE' logic. "
            "Use DETECTION TRIGGERS to define the precise dimension along which "
            "positions are incompatible (e.g., interpretation, causal explanation, "
            "normative claim, or proposed course of action). "
            "The object of disagreement must be explicitly stated in abstract terms. "
            "When generating a Conflicts category, preserve polarity rather than harmonizing positions.\n"
            "- **The 'Other' Bucket:** The theme whose \"theme_label\" is \"Other\" prevents theme bloat. "
            "It should house valid but lower-frequency ideas that do not warrant a standalone theme. "
            "It requires standard INCLUDE/EXCLUDE logic.\n"
            "- **Multi-Labeling Awareness:** Categories are not mutually exclusive. "
            "A single point of data may satisfy criteria for multiple themes or the Conflicts flag.\n"
            "- **Anti-Smoothing:** If the input text indicates multiple viewpoints, explicitly preserve those distinctions. "
            "Do not collapse opposing positions into neutralized consensus language.\n"
            "- **Total Coverage:** Every concept in the input must be addressable by the substantive themes or the 'Other' category.\n"
            "- **Theme Optimization:** You have full authority to merge, split, or revise themes to best serve the research question. "
            "You are not tethered to the original structure of the provided text."
        )
       
    
    def theme_map_to_schema(self, allowed_ids: list, other_theme_id: int, conflicts_theme_id: int = None):
        """
        Generate the system prompt for mapping insights to themes.

        This prompt instructs the model to classify a batch of insights
        according to the previously generated thematic schema.

        The schema includes:

            • theme identifiers
            • theme descriptions
            • inclusion and exclusion rules

        The model must evaluate each insight against all themes and
        assign one or more theme identifiers when appropriate.

        Key constraints enforced in the prompt include:

            • multi-label classification is permitted
            • exclusion rules must be strictly respected
            • only provided theme_ids may be returned
            • identifiers must be returned as arrays of strings

        Optional overrides may include:

            - an "Other" theme for residual insights
            - a "Conflicts" theme for incompatible positions

        Parameters
        ----------
        allowed_ids : list
            List of valid theme identifiers that may be assigned.

        other_theme_id : int
            Identifier for the "Other" category used when no substantive
            theme applies.

        conflicts_theme_id : int, optional
            Identifier used to flag discursive conflicts when present.

        Returns
        -------
        str
            System prompt instructing the model to map insights to themes
            using strict JSON output.
        """

        allowed_ids_str = ", ".join(str(id) for id in allowed_ids)

        if other_theme_id is not None:
            other_theme = (
                f"- **Mandatory Residual Assignment (HARD RULE):** Every insight MUST be assigned at least one theme_id.\n"
                f"  If an insight does not clearly satisfy any theme, you MUST assign it to theme_id {other_theme_id}.\n"
                f"  This rule OVERRIDES all other constraints, including exclusion rules and semantic strictness.\n"
                f"  Returning an empty array, null, or no assignment is strictly forbidden.\n"
            )
        else:
            other_theme = (
                "- **Mandatory Assignment (HARD RULE):** Every insight MUST be assigned at least one theme_id.\n"
                "  If no theme clearly applies, assign the closest matching theme_id.\n"
                "  Returning an empty array, null, or no assignment is strictly forbidden.\n"
            )

        if conflicts_theme_id is not None:
            conflicts_theme = (
                f"- **Conflict Flagging:** If the insight reflects substantive discursive conflict and matches detection triggers,\n"
                f"  you should assign it to theme_id {conflicts_theme_id}. This can be applied alongside other themes.\n"
            )
        else:
            conflicts_theme = ""

        return (
            "## ROLE\n"
            "You are a Logic Architect specializing in High-Fidelity Qualitative Synthesis. "
            "Your task is to map batches of insights to a Thematic Codebook Schema with full coverage.\n\n"

            "## THEMATIC SCHEMA STRUCTURE\n"
            "You will be provided with a JSON codebook representing the thematic pillars. Each theme follows this structure:\n"
            "{\n"
            '  "theme_id": "<numeric identifier>",\n'
            '  "theme_label": "<short label>",\n'
            '  "theme_description": "<summary of semantic scope>",\n'
            '  "instructions": "<INCLUDE/EXCLUDE logic or DETECTION TRIGGERS>"\n'
            "}\n\n"

            "## INPUT\n"
            "RESEARCH QUESTION: <question_text>\n"
            "THEMATIC CODEBOOK:\n"
            "<JSON array of themes>\n\n"
            "INSIGHTS TO MAP:\n"
            "<insight_id>: <insight_text>\n"
            "...\n\n"

            "## MAPPING LAWS\n"
            "- **Active Best-Match:** Evaluate every insight against ALL themes using theme_description and instructions.\n"
            "- **Strict Exclusions:** If an insight meets an EXCLUDE criterion for a theme, do not assign that theme.\n"
            "- **Multi-Labeling:** Assign ALL themes that validly apply.\n"
            f"{other_theme}"
            f"{conflicts_theme}"
            "- **Semantic Integrity:** Use conceptual meaning, not keyword matching.\n"
            "- **Bias Toward Assignment:** When uncertain, prefer assigning the closest valid theme_id rather than leaving an insight unmapped.\n\n"

            "## OUTPUT CONTRACT (STRICT JSON ONLY)\n"
            "Return ONLY a JSON object. No commentary.\n"
            "{\n"
            '  "mapped_data": [\n'
            '    { "insight_id": "string", "theme_id": ["string"] }\n'
            "  ]\n"
            "}\n\n"

            "## RULES FOR theme_ids\n"
            "- Always return an array (e.g., ['1']).\n"
            "- Use ONLY provided theme_id values.\n"
            "- Do NOT return theme_label.\n"
            "- Do NOT return text such as 'other' or 'conflicts'.\n"
            "- Do NOT invent new IDs.\n"
            "- Never return null, a single string, or an empty array.\n"
            f"- Valid theme_id values: [{allowed_ids_str}].\n\n"
        )


    def populate_themes(self, theme_len: int, theme_type: str):

        """
         Generate the system prompt for theme population.

        This prompt instructs the model to synthesize the insights assigned
        to a single theme into a cohesive thematic narrative.

        The synthesis must:

            • adhere strictly to the theme description ("North Star logic")
            • preserve all substantive insights and citations
            • maintain the relative salience of claims
            • remain within the allocated word limit
            • produce the most concise synthesis possible without losing meaning

        The prompt also adapts its structural instructions depending on
        the type of theme being synthesized.

        Supported theme types
        ---------------------

        general
            Standard thematic synthesis of conceptually related insights.

        other
            Residual category capturing lower-frequency insights that do
            not justify a standalone theme.

        conflicts
            Theme describing structured disagreement or discursive fault
            lines present in the literature.

        Parameters
        ----------
        theme_len : int
            Maximum word count allocated to the theme synthesis.

        theme_type : str
            Type of theme being synthesized. Must be one of:

                "general"
                "other"
                "conflicts"

        Returns
        -------
        str
            System prompt instructing the model to synthesize insights into
            a thematic section and return the result as strict JSON.
        """
               
        # -----------------------------------------------------
        # Specific instructions for different theme types
        # -----------------------------------------------------

        general_theme_instructions = (
            "## STRUCTURAL INSTRUCTION — GENERAL THEME\n\n"
            "Organize the section as a coherent thematic narrative.\n"
            "Integrate related insights into a logically structured synthesis.\n"
            "Avoid list-like presentation.\n"
            "Preserve conceptual distinctions while maintaining flow.\n\n"
        )

        other_theme_instructions = (
            "## STRUCTURAL INSTRUCTION — 'OTHER' THEME\n\n"
            "This theme captures valid but lower-frequency or residual claims.\n\n"

            "REQUIREMENTS:\n"
            "- Organize minority claims coherently.\n"
            "- Treat them as substantively meaningful but less dominant.\n"
            "- Preserve conceptual distinctions among minority claims.\n"
            "- Do NOT inflate their weight relative to dominant themes.\n\n"
        )

        conflict_theme_instructions = (
            "## STRUCTURAL INSTRUCTION — DISCURSIVE CONFLICT THEME\n\n"
            "This theme captures structured disagreement, definitional divergence, or "
            "interpretive fault lines present in the literature.\n\n"

            "STRUCTURAL REQUIREMENTS:\n"

            "- Organize the section explicitly around identifiable positions or camps.\n"
            "- Clearly demarcate positions using analytic contrastive language "
            "  (e.g., 'One strand argues...', 'A contrasting view holds...', "
            "  'A minority perspective contends...').\n"
            "- Explicitly state what is in dispute (e.g., definition, mechanism, policy design, "
            "  normative goal, institutional constraint).\n"
            "- Present positions sequentially and in contrast.\n"
            "- Do NOT blend positions into a single harmonized narrative.\n"
            "- Do NOT resolve or adjudicate the disagreement.\n"
            "- Do NOT imply convergence unless it is explicitly present in the material you are summarizing.\n"
            "- Preserve visible tension where present in the source material.\n\n"
        )

        instructions_dict = {
            "general": general_theme_instructions,
            "other": other_theme_instructions,
            "conflicts": conflict_theme_instructions
            }
        
        if theme_type not in instructions_dict:
            raise ValueError("Invalid theme_type. Must be one of: 'general', 'other', 'conflicts'.")
        
        specific_instructions = instructions_dict[theme_type]


        return (
            "## ROLE\n"
            "You are a Qualitative Research Lead specializing in High-Fidelity Synthesis. "
            "Your task is to analyze a collection of Research Insights mapped to a specific theme "
            "and transform them into a cohesive, evidence-based thematic section.\n\n"

            "## INPUT STRUCTURE\n"
            "You will receive a user message in the following format:\n"
            "RESEARCH QUESTION: <question_text>\n"
            "THEME LABEL: <theme_label>\n"
            "THEME DESCRIPTION: <theme_description (the North Star logic)>\n"
            "INSIGHTS TO SYNTHESIZE:\n"
            "<list of specific insights identified as relevant to this theme>\n\n"

            "## SYNTHESIS LAWS\n\n"

            "1. Adhere Strictly to the North Star:\n"
            "   The 'theme_description' defines the conceptual territory of this section. "
            "The synthesis must remain tightly bounded by this logic and read as a "
            "self-contained thematic section aligned to the overarching research question.\n\n"

            "2. Full Insight Coverage (Non-Negotiable):\n"
            "   Every distinct insight must be substantively represented. "
            "Do not omit meaningfully different claims. "
            "Do not collapse distinct ideas into overly generic umbrella statements.\n\n"

            "3. Salience Preservation:\n"
            "   Reflect the relative salience of insights within the theme. "
            "Widely supported, central, or conceptually generative claims should receive "
            "proportionally more emphasis than marginal or minority claims. "
            "Preserve visible weighting through structure and phrasing.\n\n"

            "4. Compression Objective:\n"
            "   Produce the shortest possible synthesis that satisfies the above constraints. "
            "Compress phrasing, not substance. "
            "Prioritize preservation of conceptual distinctions and salience structure "
            "over stylistic elaboration.\n\n"

            f"5. Length Constraint:\n"
            f"   Do not exceed {theme_len} words.\n\n"

            "6. Fidelity:\n"
            "   Preserve all factual details and citations exactly as they appear in the source text. "
            "Do not introduce new information or external knowledge.\n\n"

            "7. Tone:\n"
            "   Maintain a formal, academic, analytic tone.\n\n"

            "NOTE\n"
            "Some insights may be duplicates. If the exact same claims appears in multiple insights with the same citation, treat it as a single point. " 
            "However, if the same claim is supported by distinct citations in different insights, this should increase its salience and be reflected in the synthesis accordingly.\n\n"

            f"{specific_instructions}"

            "## OUTPUT CONTRACT (STRICT JSON ONLY)\n\n"
            "Return ONLY a JSON object. No commentary.\n\n"
            "{\n"
            "  \"thematic_summary\": \"The synthesized thematic narrative including citations.\"\n"
            "}\n"
        )

    

     def identify_orphans(self):
        """
        Generate the system prompt for identifying orphan insights.

        This prompt instructs the model to audit a thematic summary against
        the insights originally mapped to that theme.

        The model must determine which insights are substantively reflected
        in the summary and return the identifiers of those insights.

        An insight is considered reflected if:

            • its core claim appears in the summary
            • its contribution is preserved within a synthesized claim
            • it is represented at an appropriate level of abstraction

        Insights that are not reflected are considered "orphans" and must
        be reintroduced into the thematic synthesis.

        Returns
        -------
        str
            System prompt instructing the model to identify which insights
            are reflected in the thematic summary and return the result
            as strict JSON.
        """
        return(
            '# ROLE\n'
            'You are a Research Auditor. Your task is to verify the groundedness of a thematic summary by mapping source insights to the text.\n\n'

            '# TASK\n'
            'I will provide you with:\n'
            '1. A THEMATIC SUMMARY (a synthesis of the text).\n'
            '2. A LIST OF SOURCE INSIGHTS (numbered with unique IDs).\n\n'

            '# INPUT FORMAT\n'
            'THEMATIC SUMMARY:\n'
            '<thematic_summary_text>\n\n'
            'SOURCE INSIGHTS:\n'
            '<insight_id_1>: <insight_text_1>\n'
            '<insight_id_2>: <insight_text_2>\n'
            '...\n\n'

            'You must determine which specific insights are substantively reflected in the summary.\n\n'

            '# DEFINITION OF "REFLECTED"\n'
            'An insight is considered reflected if:\n'
            '- Its core claim, finding, or argument is clearly represented in the summary, even if expressed at a higher level of abstraction.\n'
            '- It meaningfully contributes to a synthesized claim in the summary, even if not individually distinguishable.\n'
            '- It is incorporated as part of a broader grouping of similar insights, where the shared mechanism, relationship, or implication is clearly represented.\n\n'

            'An insight is NOT reflected if:\n'
            '- The specific claim, finding, or argument is absent from the summary.\n'
            '- The summary contradicts the insight without explicitly acknowledging that tension.\n'
            '- The insight is reduced to a vague generalization that erases its substantive contribution.\n\n'

            '# IMPORTANT\n'
            '- Reflection requires substantive representation, not mere topic overlap.\n'
            '- You may infer inclusion when a generalized or synthesized claim clearly captures the core mechanism or implication of the insight, even if not all details are explicitly stated.\n'
            '- However, do not mark an insight as reflected if its specific contribution is plausibly omitted under the abstraction.\n\n'   

            '# OUTPUT PROTOCOL\n'
            '- Return ONLY a JSON object.\n'
            '- The object must contain a single key "mentioned_insight_ids" containing an array of strings.\n'
            '- Only include IDs from the provided list that are substantively reflected in the summary.\n'
            '- Do not provide explanations or commentary.\n\n'

            '# JSON SCHEMA\n'
            '{\n'
            '  "mentioned_insight_ids": ["ID_1", "ID_2", ...]\n'
            '}\n\n'
        )
    
    def integrate_orphans(self):
        """
        Generate the system prompt for reintegrating orphan insights.

        This prompt instructs the model to revise a thematic summary so
        that insights identified as missing ("orphans") are substantively
        incorporated into the narrative.

        The revised summary must:

            • preserve the original findings
            • integrate orphan insights coherently
            • maintain the original analytical tone
            • preserve all citations exactly as written
            • avoid mechanical insertion of orphan sentences

        The updated summary must remain faithful to:

            • the research question
            • the theme label
            • the theme description

        Returns
        -------
        str
            System prompt instructing the model to integrate orphan insights
            into an updated thematic summary returned as strict JSON.
        """
        return (
            '# ROLE\n'
            'You are a Research Synthesizer. Your task is to update an existing thematic summary so that all listed orphan insights are substantively reflected.\n\n'

            '# TASK\n'
            'I will provide you with:\n'
            '1. THEMATIC CONTEXT: The Theme Label, Theme Description, and Research Question.\n'
            '2. ORIGINAL SUMMARY: The current version of the summary.\n'
            '3. ORPHAN INSIGHTS: A list of insights that must be integrated.\n\n'

            '# INPUT FORMAT\n'
            'RESEARCH QUESTION: <question_text>\n'
            'THEME LABEL: <theme_label>\n'
            'THEME DESCRIPTION:\n'
            '<theme_description>\n'
            'ORIGINAL SUMMARY:\n'
            '<original_summary_text>\n'
            'ORPHAN INSIGHTS:\n'
            '<insight_text_1>\n'
            '<insight_text_2>\n'
            '...\n\n'

            '# OBJECTIVE\n'
            'Produce a revised summary in which every orphan insight is substantively reflected, '
            'using the same definition of reflection as defined below.\n\n'

            '# DEFINITION OF "REFLECTED"\n'
            'An orphan insight is reflected if:\n'
            '- Its core claim, finding, or argument is clearly represented in the revised summary.\n'
            '- It contributes meaningfully to a synthesized claim.\n'
            '- Its substantive contribution is preserved even if phrased at a higher level of abstraction.\n\n'

            'An orphan insight is NOT reflected if:\n'
            '- It is only loosely implied without preserving its conceptual contribution.\n'
            '- It is reduced to a vague generalization that erases its distinct meaning.\n\n'

            '# INTEGRATION GUIDELINES\n'
            '- DO NOT remove or contradict existing findings in the original summary.\n'
            '- EXPAND or refine the narrative where necessary to integrate orphan insights coherently.\n'
            '- MAINTAIN the existing abstraction level; do not append isolated sentences mechanically.\n'
            '- If an orphan introduces contradiction or nuance, explicitly articulate that tension.\n'
            '- Preserve all citations exactly as they appear in the source insights.\n'
            '- Multiple insights may be synthesized together, but each must remain substantively represented.\n'
            '- Use the Research Question and Theme Description as guiding constraints.\n\n'

            '# CONVERGENCE REQUIREMENT\n'
            'After revision, the updated summary should allow an auditor applying the reflection definition '
            'to identify all orphan insights as reflected.\n\n'

            '# OUTPUT PROTOCOL\n'
            '- Return ONLY a JSON object.\n'
            '- The object must contain a single key "updated_summary".\n'
            '- Do not provide explanations, preamble, or commentary.\n\n'

            '# JSON SCHEMA\n'
            '{\n'
            '  "updated_summary": "The full revised thematic summary..."\n'
            '}'
        )
    
    def address_redundancy(self):
        """
        Generate the system prompt for redundancy reduction across themes.

        This prompt instructs the model to reduce unnecessary repetition
        across previously synthesized themes while preserving the full
        informational content of each theme.

        Redundancy arises when insights assigned to multiple themes produce
        overlapping statements across different thematic sections.

        The model must:

            • eliminate surface-level repetition
            • preserve all substantive claims
            • preserve all citations exactly as written
            • maintain the logical integrity of each theme

        If a claim already appears in earlier themes, the model should
        reference it concisely rather than restating it in full.

        Returns
        -------
        str
            System prompt instructing the model to perform redundancy
            reduction and return the refined theme text as strict JSON.
        """
        return (
            "# ROLE\n"
            "You are a Research Editor performing a structural redundancy reduction pass on a set of themes.\n\n"

            "# PURPOSE\n"
            "The themes have already been generated independently from a set of assigned insights. "
            "Because some insights are assigned to multiple themes, portions of this theme may "
            "substantively repeat material already expressed in earlier themes.\n\n"

            "Your task is to reduce unnecessary surface-level redundancy while preserving:\n"
            "- Full structural coverage of assigned insights.\n"
            "- All citations exactly as written.\n"
            "- All substantive claims.\n\n"

            "# IMPORTANT CONSTRAINTS\n"
            "1. Zero Information Loss: Do NOT remove or alter any substantive claim.\n"
            "2. Do NOT introduce new claims or interpretations.\n"
            "3. Do NOT alter theme scope or structure.\n"
            "4. Preserve all citation markers exactly as written.\n\n"

            "# REDUNDANCY DEFINITION\n"
            "Redundancy refers only to repeated full articulation of substantively identical claims "
            "that already appear in previously written themes.\n\n"

            "# HOW TO HANDLE REDUNDANCY\n"
            "- If a claim has already been fully articulated in earlier themes, do NOT restate it verbatim.\n"
            "- Instead, provide a concise reference-based expression (e.g., "
            "\"As discussed/noted/described in earlier sections/above, ...\") while preserving its presence.\n"
            "- Maintain the logical integrity of this theme.\n"
            "- Ensure that all assigned insights remain substantively represented.\n\n"

            "# WHAT YOU WILL RECEIVE\n"
            "- You will receive the research qyestion the themes are answering, the previously cleaned themes in the order they are written, the current theme label, and the current theme text to refine.\n" \
            "- The previously cleaned themes are frozen context. Do NOT edit or alter them. They are read-only and serve only to inform your redundancy reduction in the current theme.\n\n"

            "# INPUT FORMAT\n"
            "RESEARCH QUESTION: <question_text>\n"
            "PREVIOUSLY CLEANED THEMES:\n"
            "<previous_theme_text>\n\n"
            "CURRENT THEME LABEL: <theme_label>\n"
            "CURRENT THEME TEXT TO REFINE:\n"
            "<current_theme_text>\n\n"

            "# OUTPUT\n"
            "Return ONLY a JSON object with the following structure:\n"
            "{\n"   
            "  \"refined_theme\": \"<Your refined theme text here>\"\n"
            "}\n"
            
            "Return ONLY a JSON object with key 'refined_theme'. "
            "Do not include commentary."
        )
    
    ####
    # /END PROMPTS FOR SYNTHESIS
    ####

    #-----------------------------

    ####
    # PROMPTs FOR RENDER
    ####

    def stylistic_rewrite(self, style:str , label: str, index: int):
        """
        Generate the system prompt for stylistic refinement of thematic summaries.

        This prompt instructs the model to improve the narrative quality of a
        thematic summary while preserving all underlying information and citations.

        The refinement process focuses on:

            - improving narrative flow
            - reducing repetitive phrasing
            - Address rhetorical padding and mechanical language
            - strengthening transitions between themes
            - maintaining a specified stylistic tone

        The prompt dynamically selects stylistic guidance depending on the
        type of theme being edited.

        Theme categories
        ----------------
        dominant
            Themes representing the central analytical findings of the corpus.

        other
            Themes capturing residual or lower-frequency insights.

        conflict
            Themes describing structured disagreement or interpretive tension.

        Parameters
        ----------
        style : str
            Narrative style to be applied (e.g., academic, journalistic).

        label : str
            Theme label used to determine the stylistic category.

        index : int
            Position of the theme within the research question sequence.
            Used to rotate stylistic framing guidance.

        Returns
        -------
        str
            System prompt instructing the model to produce a refined thematic
            summary while preserving all substantive content.
            """
        
        style_guidelines = {
            "dominant": [
                "Focus on the weight of evidence (e.g., 'The most prominent narrative identified across the corpus...')",
                "Use a structural framing (e.g., 'A primary pillar of the analysis concerns...')",
                "Use an evidentiary lens (e.g., 'The material most densely clusters around...')",
                "Use a centralizing opener (e.g., 'At the core of the discussion lies...')",
                "Focus on frequency of observation (e.g., 'A recurring point of emphasis across the record is...')",
                "Use a foundational approach (e.g., 'Fundamental to this body of work is...')",
                "Highlight a consistent pattern (e.g., 'A highly consistent pattern emerges regarding...')",
                "Frame as a primary driver (e.g., 'The analysis suggests that a central driver of this theme is...')"
            ],
            "other": [
                "Frame as divergent material (e.g., 'While not forming part of the dominant narrative, several distinct strands also emerge...')",
                "Use a breadth framing (e.g., 'The analysis also captures a series of less central but substantively meaningful positions...')",
                "Frame as complementary nuance (e.g., 'Beyond the primary thematic clusters, the material also reflects...')",
                "Use a niche framing (e.g., 'Less frequent, but nonetheless analytically significant, are discussions of...')",
                "Frame as localized insight (e.g., 'A subset of the corpus provides more focused insight into...')",
                "Frame as an emerging or isolated perspective (e.g., 'Isolated but noteworthy strands also highlight...')",
                "Use a peripheral framing (e.g., 'On the periphery of the dominant themes, the record also indicates...')",
                "Frame as granular detail (e.g., 'Providing additional granularity, specific contributions note...')"
            ],
            "conflict": [
                "Frame as structured tension (e.g., 'The literature reveals a significant tension concerning...')",
                "Use an internal friction lens (e.g., 'A notable point of contradiction emerges where...')",
                "Use a divergence framing (e.g., 'The material reflects a clear divergence regarding...')",
                "Focus on the complexity of disagreement (e.g., 'The record presents a complex landscape of competing interpretations of...')",
                "Frame as lack of consensus (e.g., 'Consensus is noticeably absent on the question of...')",
                "Use a dual-position framing (e.g., 'The debate is characterized by a clear division between...')",
                "Frame as interpretative disagreement (e.g., 'The analysis highlights conflicting accounts concerning...')",
                "Highlight competing priorities (e.g., 'A friction emerges between competing priorities, specifically...')"
            ]
        }

        label_lower = label.lower()
        if label_lower == "other":
            category = "other"
        elif label_lower in ["conflicts", "conflict"]:
            category = "conflict"
        else:
            category = "dominant"
        
        guidelines = style_guidelines[category][index % len(style_guidelines[category])]  # Rotate through style guidelines for the category

        return (
            '# ROLE\n'
            f'You are a Research Editor. Your task is to refine a {category.upper()} thematic summary into a cohesive, dynamic narrative in an {style} style.\n\n'

            '# TASK\n'
            'You will receive a thematic summary that potentially contains monotonous or mechanical phrasing and prose. '
            'Your task is to refine the thematic summary into a cohesive, dynamic narrative.\n'
            'To achieve this task you will receive previously cleaned up summaries as frozen context as well as the summary to clean.\n'

            '# INPUT FORMAT\n'
            'CURRENT RESEARCH QUESTION: <question_text>\n'
            'FROZEN CONTEXT:\n'
            '<frozen_context>\n'
            'CURRENT THEME LABEL: <theme_label>\n'
            'THEMATIC SUMMARY TO STYLE:\n'
            '<thematic summary to clean>\n\n'

            "OUTPUT FORMAT:\n"
                '{\n'
                '  "refined_summary": "<Your refined thematic summary text here>"\n'
                '}\n\n'

            '# MANDATORY OPENING STYLE\n'
            f'Begin this theme using a refined framing inspired by the following stylistic guidance: {guidelines}. Ensure the phrasing feels natural and integrated into the narrative.\n\n'

            '# EDITORIAL GUIDELINES\n'
            '- **Zero Information Loss**: Every data point, finding, and citation from the ORIGINAL THEME must be retained. Do not prune.\n'
            '- Maintain approximate length parity with the original summary.\n'
            '- Do not compress, expand, or alter substantive meaning.\n'
            '- **Explicit Linkage**: Use the FROZEN CONTEXT (previously written themes) to create narrative bridges. If information is repeated, do not delete it; instead, make the connection explicit (e.g., "As noted in the previous section..." or "Consistent with earlier findings...").\n'
            '- **Dynamic Tone**: Use varied, professional prose. Avoid monotonic "Theme X is..." structures and repeated phrasing.\n'
            '- **Citation Integrity**: Preserve all citations (author date) exactly as they appear.\n\n'

            '# NOTES\n'
            '- This is a refinement process, not a rewrite. The core content must remain intact; your role is to enhance prose while maintaining coherence and keeping to the specified style.\n'
            '- If there is no frozen context (i.e., this is the first theme), simply follow the opening style guidance and ensure a strong, engaging introduction to the theme.\n\n'

            '# OUTPUT PROTOCOL\n'
            '- Return ONLY a JSON object with the key "refined_summary".\n'
            '- Do not provide preamble or commentary.'
        )

    def exec_summary(self, word_count: int):
        """
        Generate the system prompt for producing the executive summary.

        This prompt instructs the model to synthesize the full corpus review
        into a concise executive summary and generate a descriptive title.

        The executive summary must:

            • synthesize insights across all research questions
            • highlight cross-cutting patterns in the literature
            • identify key divergences or variations
            • summarize implications emerging from the corpus

        The output must be structured as a strict JSON object containing:

            - executive_summary
            - title

        Parameters
        ----------
        word_count : int
            Target length for the executive summary.

        Returns
        -------
        str
            System prompt instructing the model to generate an executive
            summary and title in strict JSON format.
        """

        return (
            "## ROLE\n"
            "You are a writing agent specialized in developing executive summaries for corpus reviews. "
            "Your task is to synthesize a concise, coherent executive summary that captures the key insights and themes across a body of literature organized by research question. "
            "You will also generate a concise, descriptive title for the review.\n\n"

            "## INPUT FORMAT\n"
            "You will receive a single string containing the full review content organized by research question. "
            "It includes theme labels and citations. Question and theme boundaries are explicit.\n\n"

            "## OUTPUT FORMAT\n"
            "{\n"
            '  "executive_summary": "<final text only>",\n'
            '  "title": "<concise descriptive title, max 12 words>"\n'
            "}\n\n"

            "## INSTRUCTIONS\n"
            "- Tone/style: formal, concise, neutral, suitable for an academic content review.\n"
            "- No headings, no bullet lists, no citations; write continuous prose only.\n"
            "- Structure: 4-6 short paragraphs with clear topic sentences.\n"
            "- Order: (1) cross-cutting themes; (2) notable divergences by geography, sector, or time; "
            "(3) implications emerging from the literature.\n"
            "- Coverage: synthesize across all research questions; do not restate the questions verbatim.\n"
            "- Fidelity: preserve qualifiers, uncertainty, and limitations from the source text. "
            "Do not introduce new claims, numbers, trends, causal inferences, or recommendations unless they are explicitly supported across multiple themes in the input.\n"
            f"- Length target: approximately {word_count} words (±10%). "
            "Compress phrasing rather than omitting key findings. End at a sentence boundary.\n"
            "- Acronyms: define on first use if not defined in the input.\n"
            "- De-duplication: consolidate overlapping statements and avoid repeating the same point.\n"
            "- Title: maximum 12 words; no subtitle; avoid colons.\n"
            "- Do not include any text outside the JSON object."
        )
    
    def question_summaries(self):
        """
        Generate the system prompt for research-question summaries.

        This prompt instructs the model to produce a short overview
        describing how the themes collectively answer a specific
        research question.

        The generated summary serves as a narrative bridge between the
        research question and the detailed theme sections that follow.

        The summary must:

            • synthesize the themes conceptually
            • remain faithful to the source text
            • avoid introducing new claims or interpretations
            • be written in formal academic prose

        Returns
        -------
        str
            System prompt instructing the model to produce a concise
            question-level summary in strict JSON format.
        """
        return (
            "## ROLE\n"
            "You are an agent specialized in text synthesis. Your task is to synthesize thematic descriptions for a specific research question. Do not invent facts. Use only the provided content.\n"
            "You will receive the research question and the full thematic summaries. The boundary between each theme will be clearly marked.\n"
            "The content you generate will appear between the question text and the first theme. It is intended to overview the thematic landscape that follows and to orient the reader to the way the question was answered.\n\n"

            "## INPUT FORMAT\n"
            "- Research question: <question_text>\n"
            "- CONTENT TO SUMMARIZE:\n"
            "<theme_1>\n"
            "<theme_2>\n"
            "...\n\n"

            "## OUTPUT FORMAT\n"
            '{\n'
            '  "summary": "<one concise paragraph>"\n'
            '}\n\n'

            "## INSTRUCTIONS\n"
            "- Write a single paragraph (3-5 sentences) that previews what follows.\n"
            "- Be specific but concise. Synthesize; do not list every detail.\n"
            "- Maintain fidelity to the content. No new claims, numbers, or sources.\n"
            "- Do not introduce causal interpretations or policy implications unless explicitly present in the content.\n"
            "- Keep formal academic tone. No headings, no bullets, no first person, no meta commentary.\n"
            "- Ensure the paragraph clearly answers the research question through synthesis, without restating it verbatim.\n"
            "- Avoid duplication: do not repeat long phrases from the content; abstract them.\n"
            "- If the content clearly enumerates themes, you may note the count briefly (e.g., 'three themes'). Only do this when unambiguous.\n"
            "- End at a natural sentence boundary. Output valid JSON only."
        )
    
    ####
    # /END PROMPTS FOR RENDER
    ####
    