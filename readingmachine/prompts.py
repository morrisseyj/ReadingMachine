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
            '* **paper_author:** This must be a string with author names separated by semicolons (e.g. "Smith, J.; Jones, A."). '
            'The author names should be in the form: last_name, first_initial.\n'
            '* **paper_institution:** If the author is an institution (common for grey literature), the institutional name should be the single string (e.g. "World Bank").\n'
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

    def gen_in_text_citation(self):
        """
        Generate a prompt for extracting in-text citations.

        This prompt instructs the model to identify and format in-text citations
        for a given set of papers.

        Returns
        -------
        str
            Prompt instructing the model to extract and return in-text citations
            in a structured format.
        """
        return (
            "## ROLE\n"
            "You are a specialized citation formatting tool.\n\n"

            "## TASK\n"
            "Convert author and year metadata into standardized in-text citations.\n"
            "You will receive a JSON object where:\n"
            "- keys are paper_id values\n"
            "- values contain author and year information\n\n"

            "## RULES\n"
            "1. Return exactly one output entry for every input paper_id.\n"
            "2. Do not omit, modify, reorder, or invent paper_id values.\n"
            "3. Return ONLY valid JSON.\n"
            "4. Do not include explanations, markdown, comments, or additional text.\n"
            "5. Use the following style requirements:\n"
            "   - Use author last names only.\n"
            "   - Use the format 'Author Year' e.g. 'Smith 2020'.\n"
            "   - For more than one author use: <first_author> et al. <date> e.g. 'Smith et al. 2020'.\n"
            "       - It is understood that this is not common in academic styles but is is necessary for brevity in this application.\n"
            "   - For institutions use the institutional name e.g. 'World Bank 2020'.\n"
            "   - For no date or nd/n.d./n.d use n.d. e.g. 'Smith n.d.'.\n"
            "   - if dates are formatted as floats (e.g. 2020.0), convert to integers (e.g. 2020).\n"
            "13. Do not use parentheses.\n"
            "14. If formatting is uncertain, return the closest reasonable citation rather than omitting the entry.\n\n"

            "## OUTPUT FORMAT\n"
            "{\n"
            '  "<paper_id>": "<formatted citation>"\n'
            "}\n"
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
            "8) The same claim may repeat across questions if it is relevant to more than one, but do not duplicate within a question.\n"
            "9) Include only rq_ids for which there are relevant claims. If there are no relevant claims for any question, return {\"results\": {}}.\n\n"

            "Output MUST be valid JSON only, matching this schema:\n\n"

            "{\n"
            '  "results": {\n'
            '    "<rq_id>": ["<claim>"]\n'
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
            '    "meta_insight": ["<claim)>"]\n'
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
            "- Each insight must stand alone as a coherent idea.\n\n"

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

    def summarize_clusters(self, frozen_summary_window, max_output_words=2500):
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
            "- The cluster number for the insights (all clusters are uniquely labelled; cluster labels > 0 indicate valid clusters generated via HDBSCAN, cluster labels < 0 indicate outliers, sometimes split via KMeans).\n"
            "- All insights for this cluster, each with source citations.\n\n"

            "SUMMARY REQUIREMENTS:\n"
            "- When summarizing the insights, focus primarily on answering the specific research question.\n"
            "- There may be duplicate (or close duplicate claims) across insights. Do not weight identical claims more heavily, unless they are supported by distinct citations. Otherwise, treat duplicates as a single point.\n"
            "- Use other research questions for context and to identify conceptual or thematic connections, but ensure your primary focus remains on the specific research question.\n"
            "- Use the provided preceding cluster summaries as local context only to improve coherence and identify nearby thematic linkages. "
            "Do not treat them as a complete representation of earlier clusters or the full corpus. "
            "Only include information drawn from the current cluster's insights in your summary.\n"
            "- If there are no preceding summaries, write the summary as if it is the first in the sequence.\n"
            "- Provide a clear topline summary of the cluster first, then detail individual points.\n"
            "- Maintain local narrative coherence where appropriate, but do not imply global continuity across the corpus.\n"
            "- Preserve all citations exactly as they appear. Do not alter or change the citations in any way. When amalgamating insights from various insights preserve up to four citations per point, using the most prominent. Represent them as semicolon-separated, e.g. (Jones 2023; Smith and Paul, 2022)\n"
            "- For clusters containing outliers or small groups (cluster -1), reflect this in the tone.\n"
            "- Ensure the summary is coherent, logically structured, and written in an academic literature-review tone.\n\n"

            "INPUT FORMAT:\n"
            "Research question id: <question_id>\n"
            "Research question text: <question_text>\n"
            "PRECEDING CLUSTER SUMMARIES:\n"
            "<cluster_summary_1>\n<cluster_summary_2>\n...<cluster_summary_n>\n"
            "Cluster: <cluster_no>\n"
            "INSIGHTS:\n"
            "<insight_1>\n<insight_2>\n...<insight_n>\n"
            "OTHER RESEARCH QUESTIONS:\n"
            "<question_id: question_text>\n...<question_id: question_text>\n\n"

            "OUTPUT FORMAT (STRICT JSON ONLY):\n"
            "{\n"
            '    "summary": "<summary>"\n'
            "}\n\n"

            "CONSTRAINTS (STRICT PRIORITY ORDER):\n"

            f"1. LENGTH (HARD LIMIT)\n"
            f"- The summary MUST NOT exceed {max_output_words} words.\n"
            "- This is a strict upper bound and must be respected.\n\n"

            "2. COVERAGE (SOFT)\n"
            "- All insights should be represented in the summary.\n"
            "- Each insight must contribute to the synthesized output either directly or through accurate abstraction.\n\n"

            "3. GRANULARITY (PRESERVE WHERE POSSIBLE)\n"
            "- Preserve distinct claims, mechanisms, and relationships wherever possible.\n"
            "- Do NOT collapse substantively different insights into vague generalizations.\n\n"

            "4. WHEN CONSTRAINTS CONFLICT\n"
            "- You MUST prioritize staying within the length limit.\n"
            "- Then ensure all insights are represented.\n"
            "- Then ensure granularity is preserved"
            "- If necessary to satisfy the length constraint:\n"
            "  • combine clearly similar insights\n"
            "  • abstract repeated mechanisms or findings\n"
            "  • compress phrasing while preserving meaning\n"
            "- Do NOT omit unique or substantively different claims, but if necessary compress aggressively to fit the length constraint.\n\n"

            "OUTPUT RULES:\n"
            "- Return strictly valid JSON.\n"
            "- Do not include commentary, explanations, or formatting outside the JSON.\n"
            "- Ensure the JSON is complete and properly closed.\n"
            "- Write in an analytical, evidence-based, citation-faithful style.\n"
        )
    
    
    def gen_theme_schema_cluster_source(self):
        """
        Generate the system prompt for initial thematic schema construction.

        This prompt instructs the model to transform semantically clustered
        summaries into a conceptual codebook that defines the structure of the
        corpus. The model identifies major themes, establishes conceptual
        boundaries, and defines assignment rules for mapping insights to themes.

        The resulting codebook represents a shift from semantic grouping
        (similarity in language or content) to conceptual organization
        (coherent, interpretable categories aligned with the research question).

        Each theme must include:
            - theme_label
            - theme_description (conceptual "North Star")
            - instructions (INCLUDE / EXCLUDE logic)

        Special themes:
            - "Conflicts": captures incompatible positions using detection triggers
            - "Other": captures residual concepts that do not warrant a standalone theme

        The prompt also guides the model toward selecting an appropriate level of
        abstraction:
            - themes may contain multiple related concepts
            - themes must preserve conceptual distinctions
            - themes should not be overly broad or diffuse

        Returns
        -------
        str
            System prompt instructing the LLM to generate a thematic codebook
            in strict JSON format.

        Notes
        -----
        - This is a generative step that defines the initial conceptual partition
        of the corpus.
        - The prompt does not consider downstream capacity constraints directly;
        those are evaluated later during synthesis and integration.
        - The resulting schema is expected to be iteratively refined based on
        integration failures.
        """

        return(
            "## ROLE\n"
            "You are a Logic Architect specializing in High-Fidelity Qualitative Synthesis. "
            "Your task is to analyze the provided text and design a 'Thematic Codebook' "
            "that maps the conceptual landscape while ensuring total coverage of the ideas expressed.\n"
            "The text provided is the result of a semantic clustering process, where the core data has been synthesized into cluster summaries. "
            "A central task for you is transforming this map of semantic density to one of conceptual density.\n\n"

            "## THE TASK\n"
            "1. **Identify Major Themes:** Determine the recurring, dominant topics. "
            "Themes must have clearly defined conceptual boundaries, even though a single data point may be assigned to multiple themes.\n\n"

            "2. **Identify Discursive Conflicts (Conditional):** "
            "If and only if the text contains substantively incompatible interpretations, "
            "claims, or prescriptions that cannot be jointly maintained within a single "
            "coherent analytical frame, create a theme object where \"theme_label\" is exactly \"Conflict\".\n\n"

            "Do NOT paraphrase or rename this label. Use exactly \"Conflict\".\n\n"

            "Do NOT create a Conflict theme if the text merely:\n"
            "- Presents multiple reinforcing critiques,\n"
            "- Describes layered constraints or complexities,\n"
            "- Articulates trade-offs within a shared analytical orientation,\n"
            "- Or expresses variations that do not represent incompatible positions.\n\n"
            "A Conflict theme requires identifiable polarity between positions. "
            "If no such incompatibility exists, omit this theme entirely.\n\n"

            "3. **Identify 'Other' Category (Conditional):** "
            "If necessary to ensure full conceptual coverage without inducing theme bloat, "
            "create a theme object where the field \"theme_label\" is exactly \"Other\".\n"
            "Do NOT paraphrase or rename this label. Use exactly \"Other\".\n"
            "If no minority or residual concepts exist, omit this theme entirely.\n\n"

            "4. **Establish Precise Instructions:** Every category must have bespoke operational assignment instructions. "
            "Use the following logic styles:\n"
            "   - For Substantive Themes or Other: 'INCLUDE if <conceptual territory>; EXCLUDE if <conceptual territories assigned to other themes>.'\n"
            "   - For Conflicts: 'DETECTION TRIGGERS: Flag if <fault line A> vs <fault line B>.'\n\n"

            "INCLUDE rules define the conceptual territory assigned to the theme.\n"
            "EXCLUDE rules must explicitly reference conceptual territories that belong to OTHER THEMES in the schema.\n"
            "Do NOT write EXCLUDE rules as the inverse of the INCLUDE rule.\n"
            "Do NOT write generic EXCLUDE rules such as 'exclude if the text does not address this theme.'\n"
            "A strong EXCLUDE rule identifies conceptually distinct material that should instead be routed to neighboring themes.\n\n"

            "## IDEAL CODEBOOK PROPERTIES\n"
            "An effective thematic codebook will:\n\n"
            "- Define themes that are internally conceptually coherent\n"
            "- Ensure clear conceptual boundaries between themes\n"
            "- Capture the full conceptual landscape without forcing conceptually distinct ideas into the same theme\n"
            "- Avoid unnecessary fragmentation into overly fine-grained themes\n"
            "- Minimize reliance on the 'Other' category\n\n"

            "Themes should represent stable conceptual categories that can accommodate their assigned content without requiring excessive compression or loss of conceptual granularity.\n\n"
            "Themes should be articulated at a suitable level of abstraction to provide clear insight into the research question.\n\n"
            "A theme may encompass multiple related concepts, provided they can be expressed coherently without collapsing meaningful distinctions.\n\n"
            "Themes should not be so broad that they require excessively long or diffuse descriptions to explain their scope.\n\n"
            
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
            "  ],\n"
            "   \"no_change\": false\n"
            "}\n\n"

            "## ARCHITECTURAL CONSTRAINTS\n"
            "The \"no_change\" field must always be present and must always be set to false.\n"
            "Do not omit this field and do not set it to true.\n"
            "- **Structural Identity:** Do NOT generate numeric identifiers. "
            "theme_id values will be assigned programmatically outside this step. "
            "Focus only on conceptual design (theme_label, theme_description, instructions).\n"
            "- **Thematic Descriptions:** Each theme must include a 'theme_description' field. "
            "This provides the conceptual narrative for the theme and serves as the North Star logic "
            "for downstream tagging and summary population.\n"
            "- **Conceptual Mutuality (Themes):** Themes must operate as a mutually constraining partition. "
            "Each theme should define both:\n"
            "   - what conceptual territory belongs inside the theme, and\n"
            "   - what conceptual territories belong to OTHER THEMES.\n"
            "EXCLUDE rules should therefore actively route conceptually distinct material toward neighboring themes rather than merely restating the inverse of the INCLUDE rule.\n"
            "- **Relational Flagger (Conflict):** The theme whose \"theme_label\" is \"Conflict\" "
            "is a secondary overlay. It must NOT use standard 'EXCLUDE' logic. "
            "Use DETECTION TRIGGERS to define the precise dimension along which "
            "positions are incompatible (e.g., interpretation, causal explanation, "
            "normative claim, or proposed course of action). "
            "The object of disagreement must be explicitly stated in abstract terms. "
            "When generating a Conflict category, preserve polarity rather than harmonizing positions.\n"
            "- **The 'Other' Bucket:** The theme whose \"theme_label\" is \"Other\" prevents theme bloat. "
            "It should house valid but lower-frequency ideas that do not warrant a standalone theme. "
            "It requires standard INCLUDE/EXCLUDE logic.\n"
            "- **Multi-Labeling Awareness:** Categories are not mutually exclusive. "
            "A single point of data may satisfy criteria for multiple themes or the Conflict flag.\n"
            "- **Anti-Smoothing:** If the input text indicates multiple viewpoints, explicitly preserve those distinctions. "
            "Do not collapse opposing positions into neutralized consensus language.\n"
            "- **Total Coverage:** Every concept in the input must be addressable by the substantive themes or the 'Other' category.\n"
            "- **Theme Optimization:** You have full authority to merge, split, or revise themes to best serve the research question. "
            "You are not tethered to the original structure of the provided text."
        )

    def gen_theme_schema_repair_instructions(self):
        return(
            "## ROLE\n"
            "You are a synthesis capacity architect specializing in the decomposition of thematic clusters into components small enough to operationalize bounded synthesis constraints.\n\n"

            "## THE TASK\n"
            "You are working as part of an iterative loop to refine a thematic codebook.\n"
            "That iterative loop works as follows: " \
            "1. Generate a thematic codebook.\n" 
            "2. Assign content to the codebook themes based on the instructions defined in the codebook.\n" 
            "3. Attempt to synthesize the content assigned to each theme under bounded output constraints.\n"
            "4. Check whether synthesis included all the assigned content\n"
            "5. Forcibly reinsert any content that was lost\n"
            "   - Insertion is done in batches to ensure full insertion considering stress on the LLM context window\n"
            "   - If insertion for a batch causes the theme to exceed bounded synthesis constraints, the batch content is summarized under 'FAILED BATCH SUMMARIES' and the theme is marked as failing.\n"
            "6. Update the codebook to partition failing themes into smaller independently synthesizable claim-families so that synthesis and reinsertion can pass.\n\n"

            "You are currently at step 6 of this process."
            "Your task is to generate a decomposition plan that separates failing themes into sufficiently small claim-families that they can pass subsequent synthesis and reinsertion without exceeding bounded synthesis constraints.\n\n"

            "## UNDERSTANDING THE CODEBOOK STRUCTURE\n"
            "### STRUCTURE\n"
            "Each theme defines an operational synthesis region using:\n"
            "- theme_label\n"
            "- theme_description (the North Star logic)\n"
            "- instructions (INCLUDE / EXCLUDE rules)\n\n"

            "### INCLUDE/EXCLUDE LOGIC\n"
            "All themes define precise operational assignment rules:\n"
            "- Substantive Themes or Other: 'INCLUDE if <conceptual territory>; EXCLUDE if <conceptual territories assigned to other themes>.'\n"
            "- Conflict: 'DETECTION TRIGGERS: Flag if <fault line A> vs <fault line B>.'\n\n"

            "INCLUDE rules define the conceptual material assigned to the synthesis region.\n"
            "EXCLUDE rules define conceptual material that must be routed to neighboring synthesis regions.\n"
            "EXCLUDE rules should not be the simple inverses of the INCLUDE rule.\n"
            "A strong EXCLUDE rule explicitly routes ambiguous or neighboring material toward other themes so the full schema behaves as a mutually constraining assignment partition.\n"
            "When generating the scope logic in the repair plan, provide sufficient specificity to allow for subsequent EXCLUDE rules to be written that can:\n"
            "- identify the neighboring themes most likely to overlap with the current theme\n"
            "- explicitly exclude those conceptual territories\n"
            "- route ambiguous material toward the appropriate neighboring themes\n\n"

            "### SPECIAL THEMES\n"
            "**Conflict Theme (Conditional)**\n"
            "Have \"theme_label\" as exactly \"Conflict\" ONLY if the data contains "
            "substantively incompatible interpretations, claims, or prescriptions that cannot be "
            "maintained within a single coherent conceptual frame.\n\n"

            "Do NOT paraphrase or rename this label. Use exactly \"Conflict\".\n\n"

            "Do NOT create a Conflict theme if the material merely:\n"
            "- Presents reinforcing critiques\n"
            "- Describes layered constraints or interacting factors\n"
            "- Articulates trade-offs within a shared conceptual frame\n"
            "- Expresses variation in emphasis without incompatible positions\n\n"

            "A Conflict theme requires identifiable polarity between positions.\n\n"

            "Instructions for conflict will invoke DETECTION TRIGGERS (not INCLUDE/EXCLUDE). If scope logic amends conflict it must be sufficiently specific to allow subsequent rules to:\n"
            "- Define the conceptual dimension of disagreement (e.g. mechanism, definition, policy logic, normative claim)\n"
            "- Preserve opposing positions as distinct\n"
            "- Avoid harmonizing or resolving disagreement\n\n"

            "**'Other' Theme (Conditional)**\n"
            "Have \"theme_label\" as exactly \"Other\" ONLY if needed to ensure full conceptual coverage "
            "without fragmenting the schema into excessively fine-grained themes.\n\n"

            "Do NOT paraphrase or rename this label. Use exactly \"Other\".\n\n"

            "The 'Other' theme should:\n"
            "- Capture valid but low-frequency or residual concepts\n"
            "- Not contain a coherent or dominant conceptual grouping\n"
            "- Not substitute for poorly defined or overly broad themes elsewhere\n\n"

            "If no residual concepts exist, omit this theme entirely.\n\n"

            "## INPUT\n"
            "You will receive:\n"
            "1. The research question\n"
            "2. Efforts at previous schema development to date, which will include:\n"
            "   - Prior schema\n"
            "   - The theme summaries those schema generated\n"
            "       - For failing themes, these summaries will include the 'FAILED BATCH SUMMARIES' that identify the content from any batch (numbered by batch) that exceeded the bounded input constraint when being forcibly reinserted.\n"
            "   - The pass/fail status of each theme reflecting the result of complete reinsertion.\n"
            "   - The word count of all currently passing themes.\n"
            "   - Themes with `word_count = null` failed during reinsertion. Their true representational load is therefore unknown and should be assumed to exceed operational capacity.\n"
            "Prior codebooks are arranged by iteration with higher iterations representing more recent versions. \n"
            "The most recent codebook is flagged as such. This should be the focus of your revision.\n"
            "Each codebook is flagged as to whether **all** the themes passed completion checks.\n\n"

            "## UNDERSTANDING FAILURES\n"
            "The synthesis system operates under bounded output constraints (4096 tokens/~2500 words).\n"
            "A theme fails when the assigned content cannot be synthesized by a subsequent LLM call without excessive compression or output failure (i.e. truncation).\n"
            "A coherent theme can compress many related insights into a smaller number of generalized statements.\n"
            "A heterogeneous theme cannot be compressed safely without loss of nuance, because preserving conceptual fidelity requires many distinct statements.\n"
            "As conceptual heterogeneity increases, the number of statements required for faithful synthesis also increases.\n"
            "Failures therefore indicate that the assigned conceptual territory requires more representational capacity than is available under bounded synthesis constraints - the failure mode is truncated output.\n"
            "All failed themes will include summaries of the content that could not be integrated ('FAILED BATCH SUMMARIES'), which should be used as evidence for how to revise the schema.\n"

            "## REQUIRED STRUCTURAL REPAIR\n"
            "If 'schema_has_failures' = True for the most recent iteration, every failing non-Conflict theme must be decomposed.\n"
            "For each failing theme, identify the largest independently synthesizable claim-family by representational load currently assigned to that theme.\n"
            "An independently synthesizable claim-family is a recurring group of claims with its own mechanism, actor system, policy instrument, causal structure, constraint type, or argumentative logic that can be synthesized as a bounded unit.\n\n"

            "By 'largest', prioritize the claim-family that:\n"
            "1. accounts for the most distinct claims in the current summary and failed batch summaries;\n"
            "2. recurs across multiple sources or batches;\n"
            "3. can plausibly function as an independent bounded synthesis region;\n"
            "4. can plausibly be synthesized within bounded output constraints (4096 tokens/~2500 words).\n\n"

            "Extract that claim-family from the failed source theme by assigning it to a new theme, unless an existing theme has both clear conceptual fit and spare representational capacity.\n"
            "If extracting only the largest claim-family is unlikely to make the residual source theme pass, extract additional independently synthesizable claim-families until the residual is expected to pass or the source theme should be dissolved entirely.\n"
            "If removing the largest claim-family leaves no coherent bounded residual, dissolve the source theme and reallocate all remaining content.\n\n"

            "Do not write final theme descriptions or final INCLUDE/EXCLUDE prose. Provide scope logic that a second-stage schema rewriter can use to generate labels, descriptions, and operational INCLUDE/EXCLUDE rules.\n"
            "The scope logic must be specific enough to support future INCLUDE rules for retained/new/receiving themes and future EXCLUDE rules that prevent extracted content from drifting back into the narrowed source theme.\n"
            "Scope logic should identify neighboring themes most likely to overlap, the conceptual territory to route away from the source theme, and the destination for that territory.\n\n"

            "## USING FAILED BATCH SUMMARIES and CURRENT SUMMARY LENGTHS in repair decisions.\n"
            "- Treat summary length as an approximate proxy for representational capacity. Passing themes approaching the system limit (4096 tokens/~2500 words) are near capacity and should not be expanded further.\n"
            "- Treat content in failed batch summaries as conceptual content correctly assigned to this theme under its existing instructions.\n"
            "- Use the synthesized theme content **and** FAILED BATCH SUMMARIES to identify the largest independently synthesizable claim-families currently assigned to a failing theme.\n\n"

            "## USING PRIOR ITERATIONS\n"
            "- Use prior iterations only to identify recurring instability patterns, repeated failures, and ineffective prior repairs. Do not optimize older iterations independently from the current schema state.\n"
            "- Do not repeat failed conceptual aggregations from prior iterations, defined by their inclusion/exclusion instructions.\n\n"
   
            "## UPDATE PRINCIPLES\n"
            "Do not generate repair plans that repeat failed conceptual aggregations from prior iterations, including semantically similar rearticulations.\n"
            "A repair plan may preserve a similar broad topic only if the resulting assignment behavior will materially change through extraction and narrowed scope logic.\n"
            "Do not overload passing themes merely to avoid creating additional themes.\n"
            "Do not return repair plans that merely polish boundaries, rename themes, or clarify wording without materially reducing representational load.\n"
            "A repair is insufficient if it removes only minor examples, edge cases, citation-specific details, or already-covered neighboring concepts.\n"
            "When narrowing a failed theme, the scope logic must reduce included territory and identify excluded territory so extracted material is not reassigned back in later iterations.\n\n"
           
           "## OUTPUT FORMAT (STRICT JSON)\n"
            "{\n"
            "  \"repair_plan\": {\n"
            "    \"theme_repairs\": [\n"
            "      {\n"
            "        \"source_theme_id\": <integer>,\n"
            "        \"source_theme_label\": <string>,\n"
            "        \"completeness_check\": \"fail\",\n"
            "        \"concepts_ranked_by_representational_load\": [\n"
            "          {\n"
            "            \"concept\": <string>,\n"
            "            \"estimated_load\": \"high\" | \"medium\" | \"low\",\n"
            "            \"evidence_from_summary_or_failed_batches\": <string>,\n"
            "            \"independently_synthesizable\": <boolean>\n"
            "          }\n"
            "        ],\n"
            "        \"extractions\": [\n"
            "          {\n"
            "            \"concept\": <string>,\n"
            "            \"action\": \"new_theme\" | \"move_to_existing_theme\",\n"
            "            \"target_theme_id\": <integer | null>,\n"
            "            \"new_theme_label\": <string | null>,\n"
            "            \"new_theme_core_scope\": <string | null>,\n"
            "            \"new_theme_inclusions\": [<string>],\n"
            "            \"new_theme_exclusions\": [<string>],\n"
            "            \"receiving_theme_scope_update\": <string | null>,\n"
            "            \"reason\": <string>\n"
            "          }\n"
            "        ],\n"
            "        \"source_theme_resolution\": {\n"
            "          \"outcome\": \"rename_and_narrow\" | \"dissolve_and_reallocate\",\n"
            "          \"residual_label\": <string | null>,\n"
            "          \"residual_core_scope\": <string | null>,\n"
            "          \"residual_inclusions\": [<string>],\n"
            "          \"residual_exclusions\": [<string>],\n"
            "          \"residual_expected_to_pass\": <boolean>,\n"
            "          \"dissolution_reason\": <string | null>\n"
            "        },\n"
            "        \"repair_narrative\": <string>\n"
            "      }\n"
            "    ],\n"
            "    \"schema_repairs\": [\n"
            "      {\n"
            "        \"affected_theme_ids\": [<integer>],\n"
            "        \"repair_narrative\": <string>\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n\n"

            "Field definitions:\n"
            "- repair_plan: contains decomposition operations needed for the second-stage schema rewrite.\n"
            "- theme_repairs: one decomposition plan for each failed non-Conflict theme.\n"
            "- source_theme_id: the input theme_id of the failed theme being decomposed.\n"
            "- source_theme_label: the input label of the failed theme being decomposed.\n"
            "- concepts_ranked_by_representational_load: major claim-families inside the failed theme, ordered from largest to smallest expected synthesis burden.\n"
            "- concept: a distinct claim-family, mechanism, actor system, policy instrument, causal structure, or constraint type.\n"
            "- estimated_load: rough estimate of how much synthesis capacity this concept consumes.\n"
            "- evidence_from_summary_or_failed_batches: concise evidence that this concept recurs in the current summary or failed batch summaries.\n"
            "- independently_synthesizable: true if the concept can plausibly function as a bounded synthesis region.\n"
            "- extractions: claim-families removed from the failed source theme.\n"
            "- action: use \"new_theme\" when extracted material should become a new theme; use \"move_to_existing_theme\" only when an existing theme has clear conceptual fit and spare representational capacity.\n"
            "- target_theme_id: existing receiving theme_id, or null if action is \"new_theme\".\n"
            "- new_theme_label: proposed label when action is \"new_theme\", otherwise null.\n"
            "- new_theme_core_scope: proposed conceptual scope when action is \"new_theme\", otherwise null.\n"
            "- new_theme_inclusions: concepts that should be included in the new theme.\n"
            "- new_theme_exclusions: neighboring concepts that should be excluded from the new theme.\n"
            "- receiving_theme_scope_update: scope update required for an existing receiving theme, or null if action is \"new_theme\".\n"
            "- reason: concise explanation for why this extraction reduces representational load.\n"
            "- source_theme_resolution: describes what happens to the original failed theme after extractions.\n"
            "- outcome: use \"rename_and_narrow\" when a bounded successor to the source theme remains; use \"dissolve_and_reallocate\" when no coherent bounded residual remains.\n"
            "- residual_label: revised label for the narrowed source theme, or null if dissolved.\n"
            "- residual_core_scope: remaining conceptual scope after extraction, or null if dissolved.\n"
            "- residual_inclusions: concepts that remain assigned to the narrowed residual theme.\n"
            "- residual_exclusions: extracted or neighboring concepts that must be excluded from the residual theme.\n"
            "- residual_expected_to_pass: whether the residual is expected to synthesize within bounded output constraints.\n"
            "- dissolution_reason: required if outcome is \"dissolve_and_reallocate\", otherwise null.\n"
            "- repair_narrative: one concise sentence explaining the decomposition.\n"
            "- schema_repairs: schema-level repairs affecting relationships across multiple themes.\n"
            "- affected_theme_ids: input theme_ids affected by the schema-level repair.\n\n"

            "Validation rules:\n"
            "- Every theme repair must include at least one extraction.\n"
            "- Every failed non-Conflict theme must appear exactly once in theme_repairs.\n"
            "- Every theme repair must identify concepts_ranked_by_representational_load.\n"
            "- Every theme repair must extract at least the largest independently synthesizable high-load claim-family.\n"
            "- If the residual is still likely to fail, extract additional claim-families until the residual is expected to pass or dissolve the source theme.\n"
            "- Do NOT create new numeric theme identifiers; only reference input theme_id values when identifying source or receiving themes.\n"
            "- Do NOT use move_to_existing_theme unless the receiving theme has both clear conceptual fit and spare representational capacity.\n"
            "- Do NOT output final INCLUDE/EXCLUDE prose; provide scope logic only.\n"
            "- Do NOT claim a new theme is needed unless at least one extraction uses action='new_theme' with that exact new_theme_label.\n"
            "- Do NOT output a rewritten schema.\n"
            "- Do NOT output polished final theme prose.\n"
            "- Output only decomposition operations and conceptual reallocations.\n"
            "- Cosmetic edits do NOT count as repairs.\n"
            "- The Conflict theme must preserve conceptual polarity.\n"
            "- The Other theme must remain a residual category.\n"
            "- The repaired codebook must support full assignment of the conceptual content.\n"
        )


    # def gen_theme_schema_repair_instructions(self):
    #     """
    #     Generate the system prompt for iterative refinement of a thematic schema.

    #     This prompt instructs the model to revise an existing thematic codebook
    #     based on the results of a synthesis and completeness-checking process.
    #     It uses both successful and failed theme summaries to diagnose structural
    #     issues and improve conceptual alignment.

    #     The model receives:
    #         - the current codebook
    #         - theme-level summaries
    #         - pass/fail completeness indicators
    #         - summaries of content that could not be integrated ("FAILED BATCH SUMMARIES")
    #         - summary lengths as a proxy for representational load

    #     Failures are interpreted as signals of conceptual overload:
    #         - the theme contains more distinct ideas than can be represented
    #         without loss of granularity under length constraints

    #     The model must refine the schema by:
    #         - splitting overloaded themes
    #         - tightening or expanding inclusion boundaries
    #         - reallocating content where appropriate
    #         - introducing new themes when necessary

    #     The goal is to produce a stable conceptual partition that:
    #         - maintains internal coherence
    #         - has clear boundaries between themes
    #         - supports full coverage of the data
    #         - avoids excessive reliance on "Other"
    #         - avoids over-expansion of already dense themes

    #     A convergence flag is required:
    #         - "no_change": true if no clear improvements are needed
    #         - "no_change": false if any structural modification is made

    #     Returns
    #     -------
    #     str
    #         System prompt instructing the LLM to return an updated thematic
    #         codebook and convergence flag in strict JSON format.

    #     Notes
    #     -----
    #     - This is a corrective step that operates on an existing conceptual schema.
    #     - It uses integration failures as a structural diagnostic, not as
    #     classification errors.
    #     - The prompt enforces conservative updates: changes should only be made
    #     when clearly justified by the input.
    #     - This step is part of an iterative loop that converges toward a stable,
    #     capacity-compatible schema.
    #     """
    #     return(
    #         "## ROLE\n"
    #         "You are a Logic Architect specializing in High-Fidelity Qualitative Synthesis. "
    #         "You are working as part of an iterative loop to refine a thematic codebook based on the results of a synthesis and completeness-checking process. "
    #         "Your task is to articulate refinements to an existing Thematic Codebook so that it:\n"
    #         "1. Can be successfully operationalized in subsequent calls to an LLM (i.e. complete without theme failures) - first priority\n"
    #         "2. Maintains strong conceptual coherence\n"
    #         "3. Accurately partitions the conceptual landscape of the data\n\n"

    #         "## CODEBOOK STRUCTURE\n"
    #         "Each theme defines a conceptual territory using:\n"
    #         "- theme_label\n"
    #         "- theme_description (the North Star logic)\n"
    #         "- instructions (INCLUDE / EXCLUDE rules)\n\n"

    #         "## INCLUDE/EXCLUDE LOGIC\n"
    #         "All themes must define precise operational assignment rules:\n"
    #         "- Substantive Themes or Other: 'INCLUDE if <conceptual territory>; EXCLUDE if <conceptual territories assigned to other themes>.'\n"
    #         "- Conflict: 'DETECTION TRIGGERS: Flag if <fault line A> vs <fault line B>.'\n\n"

    #         "INCLUDE rules define the bounded conceptual territory assigned to the theme.\n"
    #         "EXCLUDE rules must define conceptual territories that belong to OTHER THEMES in the current schema.\n"
    #         "Do NOT write EXCLUDE rules as simple inverses of the INCLUDE rule.\n"
    #         "Do NOT write generic EXCLUDE rules such as 'exclude if the text does not address this theme.'\n"
    #         "A strong EXCLUDE rule explicitly routes ambiguous or neighboring material toward other themes so the full schema behaves as a mutually constraining conceptual partition.\n"
    #         "When writing EXCLUDE rules:\n"
    #         "- identify the neighboring themes most likely to overlap with the current theme\n"
    #         "- explicitly exclude those conceptual territories\n"
    #         "- route ambiguous material toward the appropriate neighboring themes\n\n"

    #         "## SPECIAL THEMES\n"
    #         "**Conflict Theme (Conditional)**\n"
    #         "Create a theme where \"theme_label\" is exactly \"Conflict\" ONLY if the data contains "
    #         "substantively incompatible interpretations, claims, or prescriptions that cannot be "
    #         "maintained within a single coherent conceptual frame.\n\n"

    #         "Do NOT paraphrase or rename this label. Use exactly \"Conflict\".\n\n"

    #         "Do NOT create a Conflict theme if the material merely:\n"
    #         "- Presents reinforcing critiques\n"
    #         "- Describes layered constraints or interacting factors\n"
    #         "- Articulates trade-offs within a shared conceptual frame\n"
    #         "- Expresses variation in emphasis without incompatible positions\n\n"

    #         "A Conflict theme requires identifiable polarity between positions.\n\n"

    #         "Instructions must use DETECTION TRIGGERS (not INCLUDE/EXCLUDE), and must:\n"
    #         "- Define the conceptual dimension of disagreement (e.g. mechanism, definition, policy logic, normative claim)\n"
    #         "- Preserve opposing positions as distinct\n"
    #         "- Avoid harmonizing or resolving disagreement\n\n"

    #         "**'Other' Theme (Conditional)**\n"
    #         "Create a theme where \"theme_label\" is exactly \"Other\" ONLY if needed to ensure full conceptual coverage "
    #         "without fragmenting the schema into excessively fine-grained themes.\n\n"

    #         "Do NOT paraphrase or rename this label. Use exactly \"Other\".\n\n"

    #         "The 'Other' theme should:\n"
    #         "- Capture valid but low-frequency or residual concepts\n"
    #         "- Not contain a coherent or dominant conceptual grouping\n"
    #         "- Not substitute for poorly defined or overly broad themes elsewhere\n\n"

    #         "If no residual concepts exist, omit this theme entirely.\n\n"

    #         "## INTERPRETING THE INPUT\n"
    #         "You will receive:\n"
    #         "1. The research question\n"
    #         "2. Efforts at previous schema development to date, which will include:\n"
    #         "   - Prior schema\n"
    #         "   - The theme summaries those schema generated\n"
    #         "   - The pass/fail status of each theme reflecting the result of a completeness check on the theme.\n"
    #         "   - The word count of all currently passing themes.\n"
    #         "   - Themes with `word_count = null` failed before synthesis completed. Their true representational load is therefore unknown and should be assumed to exceed operational capacity.\n"
    #         "Prior codebooks are arranged by iteration with higher iterations representing more recent versions. The most recent codebook is flagged as such. This should be the focus of your revision.\n"
    #         "Each codebook is flagged as to whether all the themes passed completion checks.\n\n"

    #         "## UNDERSTANDING FAILURES\n"
    #         "The synthesis system operates under bounded output constraints (4096 tokens/~2500 words).\n"
    #         "A theme fails when the assigned content cannot be synthesized by a subsequent LLM call without excessive compression or output failure (i.e. truncation).\n"
    #         "A coherent theme can compress many related insights into a smaller number of generalized statements.\n"
    #         "A heterogeneous theme cannot be compressed safely without loss of nuance, because preserving conceptual fidelity requires many distinct statements.\n"
    #         "As conceptual heterogeneity increases, the number of statements required for faithful synthesis also increases.\n"
    #         "Failures therefore indicate that the assigned conceptual territory requires more representational capacity than is available under bounded synthesis constraints - the failure mode is truncated output.\n"
    #         "All failed themes will include summaries of the content that could not be integrated ('FAILED BATCH SUMMARIES'), which should be used as evidence for how to revise the schema.\n"

    #         ###############################################
    #         "## REQUIRED STRUCTURAL REPAIR\n"
    #         "If 'schema_has_failures' = True for the most recent iteration, the current schema contains failing themes. Therefore, you **MUST** return a repair plan that would produce a structurally changed schema.\n"
    #         "A structurally changed schema means that **for every failing theme** ('completeness_check' = 'fail') you should at least:\n"
    #         "   - Identify the largest major concept cluster that is separable from the theme's core conceptual territory.\n"
    #         "       - By 'largest', prioritize the cluster that:\n"
    #         "           1. accounts for the most distinct claims in the current summary and failed batch summaries;\n"
    #         "           2. recurs across multiple sources or batches;\n"
    #         "           3. could plausibly function as an independent theme;\n"
    #         "           4. could plausibly be synthesized as a coherent standalone theme within bounded output constraints;\n"
    #         "   - Move that cluster out of the failed theme by creating a new theme unless there is an existing receiving theme with both clear conceptual fit and spare representational capacity.\n"
    #         "   - Update the theme labels, descriptions and instructions for all affected themes (new and source) in order to maintain clear conceptual boundaries and ensure the full schema behaves as a mutually constraining partition.\n"
    #         "   - If removing the largest conceptual cluster fundamentally breaks the coherence of the source theme and leaves only a small residual, address the residual by either merging it with another theme or creating a new theme for it.\n"
    #         "   - If one extracted cluster is unlikely to reduce the failed theme below operational capacity, extract additional separable major clusters until the remaining source theme is coherent and likely to pass the completion requirement.\n"
    #         "**DO NOT** return the same schema when the current schema contains failures.\n"
    #         "A repair is insufficient if it removes only a minor example, edge case, citation-specific detail, or already-covered neighboring concept. The removed cluster must be large enough that its extraction would materially reduce the number of distinct claims assigned to the failed theme in the next iteration.\n"
    #         "When repairing a failed theme, revise both INCLUDE and EXCLUDE rules so future assignment behavior changes materially.\n"
    #         "When reducing the number of concepts in a failed theme, this process should include reducing the set of concepts that are included AND expanding the set of concepts that are excluded. This is necessary to avoid reintroducing conceptually distinct material into a narrowed theme during future reassignment.\n"

    #         #############################################################

    #         "## IDEAL CODEBOOK PROPERTIES\n"
    #         "An effective thematic codebook/schema will:\n\n"
    #         "- Allow for the successful expression of all assigned conceptual content, without loss of granularity, under the constraints on output length (i.e. no failing themes).\n"
    #         "- Define themes that are internally conceptually coherent\n"
    #         "- Ensure clear conceptual boundaries between themes\n"
    #         "- Capture the full conceptual landscape without forcing conceptually distinct ideas into the same theme\n"
    #         "- Avoid unnecessary fragmentation into overly fine-grained themes\n"
    #         "- Minimize reliance on the 'Other' category\n\n"

    #         "## PRIORITIZATION RULE\n"
    #         "When resolving conflicts between the imperatives to 1) address failed themes and 2) minimize theme counts, maximize conceptual coherence and minimize reliance on the 'Other' category; you must prioritize addressing failed themes and resolving completeness failures first.\n"
    #         "Only once failures are addressed, should you optimize toward the other ideal properties defined above.\n\n"

    #         "## UPDATE PRINCIPLES\n"
    #         "Revise the codebook to advance its ideal form as stated above.\n"
    #         "Always diagnose and revise the CURRENT ITERATION.\n"
    #         "Historical iterations should be used to identify recurring instability patterns, repeated failures, and ineffective prior repairs.\n"
    #         "Do not optimize older iterations independently from the current schema state.\n"
    #         "Do not repeat articulations of themes that failed in previous iterations - substantively similar (i.e. only semantically distinct) rearticulations are not acceptable.\n"
    #         "A revised theme may retain a similar conceptual label or broad topic if appropriate, but the assignment behavior of the rules must change substantively and materially (both inclusion and exclusion criteria).\n"
    #         "When updating a schema, INCLUDE/EXCLUDE rules should change for both the failed theme and any receiving themes to which content is reassigned, so that the full schema behaves as a mutually constraining conceptual partition.\n" \
    #         "If an 'Other' theme is failing you should either expand some themes to accommodate its content, or create a new theme based on the most coherent subset of the 'Other' content, rather than simply expanding the 'Other' theme to accommodate more content.\n\n"

    #        "#### Use FAILED BATCH SUMMARIES and current summary lengths to determine how failed themes should be revised.\n"
    #         "- Do not reuse previously failed INCLUDE/EXCLUDE rules or DETECTION TRIGGERS.\n"
    #         "- Do not rely on theme_id continuity when interpreting prior iterations; themes may be renamed, reordered, merged, or split.\n"
    #         "- Repeated failure indicates that the underlying conceptual aggregation is structurally too broad for bounded synthesis.\n"
    #         "- Treat summary length as an approximate proxy for representational capacity. Passing themes approaching the system limit (4096 tokens/~2500 words) are near capacity and should not be expanded further.\n"
    #         "- Use FAILED BATCH SUMMARIES together with current summary lengths to decide whether failed material should be reassigned to existing themes or split into narrower new themes.\n"
    #         "- Reallocate failed material to an existing theme only when there is BOTH clear conceptual alignment and clear representational capacity in the receiving theme.\n"
    #         "   - If either conceptual alignment or representational capacity is uncertain, create a narrower new theme instead of reallocating.\n"
    #         "- When content is reassigned, update the receiving theme's instructions to explicitly accommodate it and narrow the originating theme so boundaries remain clear and non-overlapping.\n"
    #         "- Do not overload passing themes merely to avoid creating additional themes.\n\n"

    #         "## REPEATED FAILURE PATTERNS\n"
    #         "Identify repeated failure patterns when substantially similar conceptual territories continue to appear inside themes that fail across iterations, even if the themes are renamed, reordered, merged, or split.\n"
    #         "Repeated failure indicates that the underlying conceptual aggregation is operationally invalid for bounded synthesis, regardless of how conceptually elegant, comprehensive, or theoretically coherent the theme may appear.\n"
    #         "When a conceptual territory repeatedly fails:\n"
    #         "- do NOT preserve the broad aggregation through minor reformulation\n"
    #         "- do NOT prioritize conceptual elegance over operational viability\n"
    #         "- strongly prefer narrower, less elegant, more operationally bounded themes instead\n"
    #         "- treat successful bounded synthesis as more important than preserving high-level conceptual integration\n\n"

    #         "### Improve overall schema quality and conceptual coherence.\n"
    #         "- Only prioritize conceptual coherence on boundary clarity after making efforts to resolve failures. An elegant and coherent schema is desirable but secondary to operational viability.\n"
    #         "- Use both passing and failing themes as evidence for improving overall conceptual structure and capacity balance.\n"
    #         "- If the current schema has no failing themes, treat the passing schema conservatively and do not optimize it speculatively.\n"
    #         "- If the current schema has any failing themes, do not let individually passing themes prevent structural repair; passing themes may be narrowed, split, or reorganized when needed to resolve failures and rebalance conceptual load.\n"            
    #         "- A correct and operationally viable conceptual partition is more important than continuity with the previous schema.\n"
    #         "- Avoid unnecessary restructuring ONLY when the current schema already resolves completeness failures.\n\n"

    #         "You may revise multiple parts of the schema simultaneously when attempting to resolve failed themes or improve conceptual coherence and theme distinction.\n\n"

    #         "## CONVERGENCE CONDITION\n"
    #         "You must always include a field \"no_change\" in your output.\n\n"

    #         "HARD RULE:\n"
    #         "- If the most recent iteration contains any theme with \"completeness_check\": \"fail\", you MUST set \"no_change\": false.\n\n"

    #         "- Set \"no_change\": true ONLY if:\n"
    #         "   - all themes in the most recent iteration pass the completeness check, AND\n"
    #         "   - there are no obvious unresolved opportunities to improve conceptual partitioning without risking new cases of theme overload.\n\n"

    #         "Do NOT set \"no_change\": true if any theme has failed the completeness check.\n"
    #         "Do NOT make speculative improvements. Only propose repairs when improvements would be obvious based on the input.\n\n"
            
    #        "## OUTPUT FORMAT (STRICT JSON)\n"
    #         "{\n"
    #         "  \"repair_plan\": {\n"
    #         "    \"theme_repairs\": [\n"
    #         "      {\n"
    #         "        \"source_theme_id\": <integer>,\n"
    #         "        \"source_theme_label\": <string>,\n"
    #         "        \"completeness_check\": \"fail\",\n"
    #         "        \"repair_action\": \"retain_and_shrink\" | \"remove_and_redistribute\",\n"
    #         "        \"receiving_theme_ids\": [<integer>],\n"
    #         "        \"new_theme_needed\": <boolean>,\n"
    #         "        \"new_theme_label\": <string | null>,\n"
    #         "        \"label_change\": <string | null>,\n"
    #         "        \"description_change\": <string>,\n"
    #         "        \"include_change\": <string>,\n"
    #         "        \"exclude_change\": <string>,\n"
    #         "        \"trigger_change\": <string | null>,\n"
    #         "        \"reallocated_concepts\": [\n"
    #         "          {\n"
    #         "            \"concept\": <string>,\n"
    #         "            \"from_theme_id\": <integer>,\n"
    #         "            \"to_theme_id\": <integer | null>,\n"
    #         "            \"to_new_theme_label\": <string | null>\n"
    #         "          }\n"
    #         "        ],\n"
    #         "        \"repair_narrative\": <string>\n"
    #         "      }\n"
    #         "    ],\n"
    #         "    \"schema_repairs\": [\n"
    #         "      {\n"
    #         "        \"affected_theme_ids\": [<integer>],\n"
    #         "        \"repair_narrative\": <string>\n"
    #         "      }\n"
    #         "    ]\n"
    #         "  },\n"
    #         "  \"no_change\": <boolean>\n"
    #         "}\n\n"

    #         "Field definitions:\n"
    #         "- repair_plan: contains all repair operations needed for the second-stage schema rewrite.\n"
    #         "- theme_repairs: theme-level repair operations for failed non-Conflict themes.\n"
    #         "- source_theme_id: the input theme_id of the failed theme being repaired.\n"
    #         "- source_theme_label: the input label of the failed theme being repaired.\n"
    #         "- repair_action: use \"retain_and_shrink\" if the failed theme remains in narrower form; use \"remove_and_redistribute\" if no substantially equivalent successor remains.\n"
    #         "- receiving_theme_ids: existing theme_ids that must receive reallocated content. Use [] if all reallocated content goes to new themes.\n"
    #         "- new_theme_needed: true if any removed content should become a new theme.\n"
    #         "- new_theme_label: proposed label for the new theme, or null if no new theme is needed.\n"
    #         "- label_change: the revised label for the source theme, or null if unchanged or removed.\n"
    #         "- description_change: how the source theme description should change.\n"
    #         "- include_change: how the source theme INCLUDE rule should change.\n"
    #         "- exclude_change: how the source theme EXCLUDE rule should change, including where excluded content should be routed.\n"
    #         "- trigger_change: only for Conflict themes; otherwise null.\n"
    #         "- reallocated_concepts: specific conceptual territories removed from failed themes and assigned elsewhere.\n"
    #         "- repair_narrative: one concise sentence explaining the substantive repair.\n"
    #         "- schema_repairs: schema-level repair operations that affect relationships or boundaries across multiple themes.\n"
    #         "- affected_theme_ids: theme_ids affected by the schema-level repair.\n"
    #         "- no_change: whether no repair is needed for the current schema.\n\n"

    #         "Important:\n"
    #         "- If no_change=true, repair_plan.theme_repairs and repair_plan.schema_repairs must both be empty arrays.\n"
    #         "- If no_change=false, every failed non-Conflict theme must appear exactly once in repair_plan.theme_repairs.\n"

    #         "- Every newly proposed conceptual territory must either:\n"
    #         "   - identify an existing receiving theme_id, or\n"
    #         "   - specify new_theme_needed=true with a concrete new_theme_label.\n"

    #         "- If repair_action='retain_and_shrink', at least one of:\n"
    #         "   - label_change\n"
    #         "   - include_change\n"
    #         "   - exclude_change\n"
    #         "must materially alter the conceptual partition of the theme.\n"

    #         "- Do NOT claim a new theme is needed unless at least one reallocated_concepts item uses to_new_theme_label with that exact label.\n"

    #         "- Do NOT output a rewritten schema.\n"
    #         "- Do NOT output polished final theme prose.\n"
    #         "- Output only repair operations and conceptual reallocations.\n"
    #         "- The repair plan must be operationally executable by a second-stage schema rewrite system.\n"
    #         "- Every failed non-Conflict theme must appear exactly once in repair_plan.\n"
    #         "- Cosmetic edits do NOT count as repairs.\n"
    #         "- If a theme is retained_and_shrunk, the repair plan must explicitly identify conceptual territory removed from the theme.\n"
    #         "- If content is reassigned to an existing theme, identify the receiving theme_id.\n"
    #         "- If content requires a new theme, set new_theme_needed=true and provide new_theme_label.\n"

    #         "## CONSTRAINTS\n"
    #         "- Do NOT create new numeric theme identifiers; only reference input theme_id values when identifying source or receiving themes.\n"            "- The Conflict theme must preserve conceptual polarity\n"
    #         "- The Other theme must remain a residual category\n"
    #         "- The final codebook after repair must support full assignment of the conceptual content\n"
    #     )


    def implement_schema_repairs(self):
        """
        """
        return(
            "## ROLE\n"
            "You are a Schema Rewrite Engine.\n"
            "Your task is to mechanically implement a previously generated decomposition repair plan onto an existing thematic codebook.\n"
            "You are NOT performing conceptual diagnosis, optimization, reinterpretation, or strategic reasoning.\n"
            "You must only implement the supplied repair operations faithfully and consistently.\n\n"

            "## INPUTS\n"
            "You will receive:\n"
            "1. The current research question\n"
            "2. The current schema\n"
            "3. A validated repair_plan generated by a prior planning stage\n"
            "4. The most recent thematic summaries, including any FAILED BATCH SUMMARIES\n"
            " - FAILED BATCH SUMMARIES describe content that could not be reinserted into a theme without exceeding bounded synthesis constraints. Treat them as evidence of representational load and as context for implementing the repair_plan, not as a reason to invent additional repairs.\n\n"

            "## REPAIR PLAN FIELD DEFINITIONS\n"
            "- repair_plan: contains decomposition operations that must be implemented in the rewritten schema.\n"
            "- theme_repairs: one decomposition plan for each failed non-Conflict theme.\n"
            "- source_theme_id: the theme_id in the input schema that the repair operation targets.\n"
            "- source_theme_label: the original label of the targeted theme.\n"
            "- concepts_ranked_by_representational_load: major claim-families inside the failed theme, ordered from largest to smallest expected synthesis burden. Use this as context only; implement the extractions.\n"
            "- extractions: claim-families that must be removed from the failed source theme.\n"
            "- concept: the claim-family being extracted.\n"
            "- action: if \"new_theme\", create a new theme for the extracted concept; if \"move_to_existing_theme\", update the existing target_theme_id to receive it.\n"
            "- target_theme_id: existing theme_id that receives the extracted concept, or null if action is \"new_theme\".\n"
            "- new_theme_label: label to use for a newly created theme, or null if the concept moves to an existing theme.\n"
            "- new_theme_core_scope: conceptual scope of the new theme.\n"
            "- new_theme_inclusions: concepts that must be included in the new theme's INCLUDE rule.\n"
            "- new_theme_exclusions: concepts that must be excluded from the new theme's EXCLUDE rule.\n"
            "- receiving_theme_scope_update: scope update that must be incorporated into an existing receiving theme.\n"
            "- source_theme_resolution: specifies whether the original failed theme becomes a narrowed successor or is dissolved.\n"
            "- outcome: if \"rename_and_narrow\", keep a narrowed successor theme using the residual fields; if \"dissolve_and_reallocate\", remove the source theme entirely.\n"
            "- residual_label: revised label for the narrowed source theme, or null if dissolved.\n"
            "- residual_core_scope: remaining conceptual scope after extraction, or null if dissolved.\n"
            "- residual_inclusions: concepts that must remain included in the narrowed residual theme.\n"
            "- residual_exclusions: concepts that must be excluded from the narrowed residual theme, especially extracted concepts that should not drift back in.\n"
            "- dissolution_reason: explanation for dissolution; use it only to understand intent, not as output text.\n"
            "- schema_repairs: schema-level repairs affecting relationships across multiple themes.\n"
            "- affected_theme_ids: theme_ids whose descriptions or instructions may need adjustment for a schema-level repair.\n\n"

            "## TASK\n"
            "Apply the repair_plan to rewrite the schema.\n\n"

            "You must:\n"
            "- create new themes for extractions with action='new_theme'\n"
            "- update receiving themes for extractions with action='move_to_existing_theme'\n"
            "- narrow source themes when source_theme_resolution.outcome='rename_and_narrow'\n"
            "- remove source themes when source_theme_resolution.outcome='dissolve_and_reallocate'\n"
            "- convert residual_core_scope, residual_inclusions, and residual_exclusions into clear theme descriptions and INCLUDE/EXCLUDE rules\n"
            "- convert new_theme_core_scope, new_theme_inclusions, and new_theme_exclusions into clear theme descriptions and INCLUDE/EXCLUDE rules\n"
            "- convert receiving_theme_scope_update into updated descriptions and INCLUDE/EXCLUDE rules for receiving themes\n\n"

            "## IMPLEMENTATION RULES\n"
            "- Implement the repair plan exactly.\n"
            "- The repair_plan remains authoritative. If summaries suggest additional possible improvements, ignore them unless required to faithfully implement the repair_plan.\n"
            "- Do NOT reinterpret the repair plan.\n"
            "- Do NOT preserve broad conceptual aggregations that the repair plan decomposes.\n"
            "- Do NOT reintroduce extracted claim-families into narrowed source themes.\n"
            "- Do NOT optimize for elegance, theoretical completeness, or conceptual compression beyond the repair plan.\n"
            "- Do NOT invent additional restructurings not implied by the repair plan.\n"
            "- Do NOT make speculative improvements.\n"
            "- Preserve unchanged themes unless the repair plan requires modification.\n\n"

            "## INCLUDE/EXCLUDE REQUIREMENTS\n"
            "All substantive themes must use:\n"
            "'INCLUDE if ...; EXCLUDE if ...'\n\n"

            "INCLUDE rules must reflect the scope logic in the repair plan, including residual_inclusions, new_theme_inclusions, and receiving theme additions where applicable.\n\n"

            "EXCLUDE rules must:\n"
            "- explicitly route neighboring conceptual material toward named destination themes whenever possible\n"
            "- reflect residual_exclusions and new_theme_exclusions from the repair plan\n"
            "- explicitly exclude extracted claim-families from narrowed source themes\n"
            "- prevent extracted content from drifting back into source themes in later assignment\n"
            "- preserve non-overlapping assignment behavior\n\n"

            "Preferred EXCLUDE form:\n"
            "'EXCLUDE if discussing <excluded territory>, which should be routed to <theme_label>.'\n\n"

            "Do NOT write generic, inverse, or vague exclusions.\n\n"

            "## CONFLICT THEME\n"
            "If a Conflict theme exists:\n"
            "- preserve the label exactly as \"Conflict\"\n"
            "- preserve conceptual polarity\n"
            "- use DETECTION TRIGGERS instead of INCLUDE/EXCLUDE\n\n"

            "## OTHER THEME\n"
            "If an Other theme exists:\n"
            "- preserve it as a residual category only\n"
            "- do NOT allow it to absorb coherent conceptual groupings that should become substantive themes\n\n"

            "## IMPORTANT\n"
            "- This is an implementation task, not a planning task.\n"
            "- The repair plan is the authoritative source of structural change.\n"
            "- Your job is to produce a clean rewritten schema that faithfully realizes the repair plan.\n"
            "- Every extraction in the repair plan must be visibly implemented in the resulting schema.\n\n"

            "## OUTPUT FORMAT (STRICT JSON)\n"
            "{\n"
            "  \"themes\": [\n"
            "    {\n"
            "      \"theme_label\": <string>,\n"
            "      \"theme_description\": <string>,\n"
            "      \"instructions\": <string>\n"
            "    }\n"
            "  ]\n"
            "}\n\n"

            "## CONSTRAINTS\n"
            "- Do NOT generate numeric identifiers\n"
            "- Do NOT output explanations\n"
            "- Do NOT output repair narratives\n"
            "- Do NOT output markdown\n"
            "- Output only the rewritten schema JSON\n"
        )
    
    def gen_theme_schema_optimize(self):
        """
        """
        return(
            "## ROLE\n"
            "You are a Schema Optimization Engine.\n"

            "## TASK\n"
            "You are part of an iterative loop of schema refinement. Your task is to propose improvements to the current schema that would enhance its overall quality and coherence without reintroducing completeness failures.\n\n"

            "## INPUTS\n"
            "You will receive:\n"
            "1. The current research question\n"
            "2. The history of schema iterations, including the thematic summaries they produced.\n"
            "   - Each iteration is marked with the highest iteration number representing the most recent version of the schema.\n"
            "   - Prior iterations include both efforts to partition themes to ensure all themes pass a completeness check and efforts to optimize the schema by improving conceptual coherence, boundary clarity, and overall quality without risking new failures.\n"
            "   - Iterations with different objectives are marked as such.\n"
            "   - The most recent iteration is flagged as such and should be the focus of your optimization efforts.\n"

            "## CODEBOOK STRUCTURE\n"
             "Each theme defines a conceptual territory using:\n"
            "- theme_label\n"
            "- theme_description (the North Star logic)\n"
            "- instructions (INCLUDE / EXCLUDE rules)\n\n"

            "## INCLUDE/EXCLUDE LOGIC\n"
            "All themes must define precise operational assignment rules:\n"
            "- Substantive Themes or Other: 'INCLUDE if <conceptual territory>; EXCLUDE if <conceptual territories assigned to other themes>.'\n"
            "- Conflict: 'DETECTION TRIGGERS: Flag if <fault line A> vs <fault line B>.'\n\n"

            "INCLUDE rules define the bounded conceptual territory assigned to the theme.\n"
            "EXCLUDE rules must define conceptual territories that belong to OTHER THEMES in the current schema.\n"
            "Do NOT write EXCLUDE rules as simple inverses of the INCLUDE rule.\n"
            "Do NOT write generic EXCLUDE rules such as 'exclude if the text does not address this theme.'\n"
            "A strong EXCLUDE rule explicitly routes ambiguous or neighboring material toward other themes so the full schema behaves as a mutually constraining conceptual partition.\n"
            "When writing EXCLUDE rules:\n"
            "- identify the neighboring themes most likely to overlap with the current theme\n"
            "- explicitly exclude those conceptual territories\n"
            "- route ambiguous material toward the appropriate neighboring themes\n\n"

            "## SPECIAL THEMES\n"
            "**Conflict Theme (Conditional)**\n"
            "Create a theme where \"theme_label\" is exactly \"Conflict\" ONLY if the data contains "
            "substantively incompatible interpretations, claims, or prescriptions that cannot be "
            "maintained within a single coherent conceptual frame.\n\n"

            "Do NOT paraphrase or rename this label. Use exactly \"Conflict\".\n\n"

            "Do NOT create a Conflict theme if the material merely:\n"
            "- Presents reinforcing critiques\n"
            "- Describes layered constraints or interacting factors\n"
            "- Articulates trade-offs within a shared conceptual frame\n"
            "- Expresses variation in emphasis without incompatible positions\n\n"

            "A Conflict theme requires identifiable polarity between positions.\n\n"

            "Instructions must use DETECTION TRIGGERS (not INCLUDE/EXCLUDE), and must:\n"
            "- Define the conceptual dimension of disagreement (e.g. mechanism, definition, policy logic, normative claim)\n"
            "- Preserve opposing positions as distinct\n"
            "- Avoid harmonizing or resolving disagreement\n\n"

            "**'Other' Theme (Conditional)**\n"
            "Create a theme where \"theme_label\" is exactly \"Other\" ONLY if needed to ensure full conceptual coverage "
            "without fragmenting the schema into excessively fine-grained themes.\n\n"

            "Do NOT paraphrase or rename this label. Use exactly \"Other\".\n\n"

            "The 'Other' theme should:\n"
            "- Capture valid but low-frequency or residual concepts\n"
            "- Not contain a coherent or dominant conceptual grouping\n"
            "- Not substitute for poorly defined or overly broad themes elsewhere\n\n"

            "If no residual concepts exist, omit this theme entirely.\n\n"

            "## UNDERSTANDING FAILURES\n"
            "The synthesis system operates under bounded output constraints (4096 tokens/~2500 words).\n"
            "A theme fails when the assigned content cannot be synthesized by a subsequent LLM call without excessive compression or output failure (i.e. truncation).\n"
            "A coherent theme can compress many related insights into a smaller number of generalized statements.\n"
            "A heterogeneous theme cannot be compressed safely without loss of nuance, because preserving conceptual fidelity requires many distinct statements.\n"
            "As conceptual heterogeneity increases, the number of statements required for faithful synthesis also increases.\n"
            "Failures therefore indicate that the assigned conceptual territory requires more representational capacity than is available under bounded synthesis constraints - the failure mode is truncated output.\n"
            "All previously failed themes will be marked as such.\n\n"

            "## USING CURRENT SUMMARY LENGTHS IN OPTIMIZATION DECISIONS\n"
            "- Treat summary length as an approximate proxy for representational capacity. Themes approaching the system limit (4096 tokens/~2500 words) are near capacity and should not be expanded further.\n"

            "## COMPRESSION DISCIPLINE\n"
            "Optimization must not rely on future synthesis compression to make overloaded or near-overloaded themes viable.\n"
            "When considering merges, reallocations, or broader theme scopes, assume each substantively distinct claim-family, mechanism, causal logic, actor system, policy instrument, implication, minority view, or contradiction must remain separately representable in later synthesis.\n"
            "Paraphrases, near-duplicates, repeated examples, and differently cited versions of the same claim may be consolidated.\n"
            "Distinct mechanisms, causal logics, actor systems, policy instruments, contradictions, minority positions, geographies, sectors, or implications must not be collapsed merely to create a more elegant schema.\n"
            "If an optimization would require collapsing distinct claim-families into generalized statements to remain within bounded synthesis constraints, do not make that optimization.\n\n"
              
            "## OPTIMIZATION OBJECTIVES\n"
            "The ideal codebook/schema will:\n\n"
            "- Allow for the successful expression of all assigned conceptual content, without loss of granularity, under the constraints on output length (i.e. no failing themes).\n"
            "- Define themes that are internally conceptually coherent\n"
            "- Ensure clear conceptual boundaries between themes\n"
            "- Capture the full conceptual landscape without forcing conceptually distinct ideas into the same theme\n"
            "- Avoid unnecessary fragmentation into overly fine-grained themes\n"
            "- Minimize reliance on the 'Other' category\n\n"

            "## OPTIMIZATION CONSTRAINTS\n"
            "Only the current schema iteration is relevant for your optimization task. Use prior iterations only as context for understanding the history of schema development, not as targets for optimization.\n"
            "When proposing improvements, you must ensure that the revised schema:\n"
            "- does NOT contain any themes that fail the completeness check\n"
            "- does NOT reintroduce previously resolved completeness failures\n"
            "- If proposing reallocation of content between themes and it is unclear whether bounded synthesis constraints can be maintained, you must avoid making such changes\n"
            "Only suggest changes to the schema if there are obvious and non-speculative improvements that can be made based on the input. Do NOT make speculative improvements.\n"
            "If making improvements, you should update theme descriptions, instructions and labels as needed to maintain clear conceptual boundaries. \n"
            "- Changes to INCLUSION/EXCLUSION (or TRIGGERS in the case of conflict) should be applied to both the theme being changed and any other affected themes so that the full schema behaves as a mutually constraining partition.\n"
            "Do not merge themes unless both conceptual coherence and bounded synthesis viability are clearly preserved without collapsing distinct claim-families into lossy generalizations.\n\n"

            "## CONVERGENCE CONDITION\n"
            "You must always include a field \"no_change\" in your output.\n\n"

            "Set \"no_change\": true if either:\n"
            "- there are no obvious unresolved opportunities to improve conceptual partitioning without risking new cases of theme overload; or\n"
            "- all obvious improvements would require expanding near-capacity themes.\n\n"

            "Set \"no_change\": false only if there are obvious, non-speculative improvements that preserve bounded synthesis viability.\n\n"

            "## OUTPUT FORMAT (STRICT JSON)\n"
            "{\n"
            "  \"no_change\": <boolean>,\n"
            "  \"themes\": [\n"
            "    {\n"
            "      \"theme_label\": <string>,\n"
            "      \"theme_description\": <string>,\n"
            "      \"instructions\": <string>\n"
            "    }\n"
            "  ]\n"
            "}\n\n"

            "If you set \"no_change\": true, \"themes\" should be an empty array.\n"
            "If no_change=false, return the full revised schema, not only changed themes.\n"
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

            "2. Coverage with Abstraction (Non-Negotiable):\n\n"
            "   All substantively distinct ideas present in the input insights must be represented in the synthesis.\n\n"
            "However:\n"
            "- You SHOULD consolidate semantically similar insights into unified statements.\n"
            "- You SHOULD generalize where multiple insights express the same underlying claim.\n"
            "- You SHOULD avoid one-to-one mapping between insights and sentences.\n\n"

            "Prefer structured generalization over repetition:\n"
            "- When multiple insights support the same mechanism or argument, express this as a single, well-formed claim.\n\n"

            "You MUST NOT:\n"
            "- omit substantively distinct claims\n"
            "- collapse ideas that differ in mechanism, causal logic, or normative implication into a single generalized statement\n\n"

            "The goal is full conceptual coverage with minimal redundancy, not surface-level reproduction.\n\n"

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
            "   Preserve all factual details and citation provenance from the source text. "
            "Do not introduce new information or external knowledge.\n"
            "Use citations exactly as provided in the insights. Do not alter or change the citations in any way. \n" \
            "If a claim is supported by multiple similar insights with distinct citations, reflect multiple citations to support the claim in the syntheis - use a semi-colon separated list (e.g., Smith 2020; Jones 2021).\n"
            "If a claim is supported by many insights, reflect the four most salient citations that support it, prioritizing diversity of sources. \n\n"

            "7. Tone:\n"
            "   Maintain a formal, academic, analytic tone.\n\n"

            "NOTE\n"
            "Some insights may be duplicates. If the exact same claims appears in multiple insights with the same citation, treat it as a single point. " 
            "However, if the same claim is supported by distinct citations in different insights, this should increase its salience and be reflected in the synthesis accordingly.\n"
            "When the same claim is supported by distinct citations in different insights, this should increase its salience and be reflected in the synthesis accordingly, using the normalized citation formats above.\n\n"

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
            • its citation/provenance is preserved in the summary

        Insights that are not reflected are considered "orphans" and must
        be reintroduced into the thematic synthesis.

        Returns
        -------
        str
            System prompt instructing the model to identify which insights
            are reflected in the thematic summary and return the result
            as strict JSON.
        """
        return (
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
            '- Its core claim, finding, or argument is meaningfully represented in the summary, even if expressed at a higher level of abstraction.\n'
            '- It meaningfully contributes to a synthesized claim in the summary, even if not individually distinguishable.\n'
            '- It is incorporated as part of a broader grouping of similar insights, where the shared mechanism, relationship, or implication is clearly represented.\n'
            '- The "core claim" refers to the central mechanism, relationship, or implication of the insight, not its exact phrasing or contextual detail.\n\n'

            'An insight is NOT reflected if:\n'
            '- The specific claim, finding, or argument is absent from the summary.\n'
            '- The summary contradicts the insight without explicitly acknowledging that tension.\n'
            '- The insight is reduced to a vague generalization that erases its substantive contribution.\n\n'

            '# IMPORTANT\n'
            '- Reflection requires substantive representation, not mere topic overlap.\n'
            '- You may infer inclusion when a generalized or synthesized claim clearly captures the core mechanism or implication of the insight.\n'
            '- Multiple insights may be reflected by a single synthesized statement if they share a common underlying mechanism, relationship, or implication.\n'
            '- Do not mark an insight as reflected if its core contribution is missing.\n'
            '- However, abstraction alone is not grounds for exclusion if the underlying mechanism, relationship, or implication is clearly preserved.\n\n'

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


    def identify_citations(self):
        """
        """
        return (
            "You are a citation auditor.\n\n"

            "You will receive:\n"
            "1. A THEMATIC SUMMARY.\n"
            "2. A JSON ARRAY of required in-text citation strings.\n\n"

            "TASK\n"
            "Identify which required in-text citation strings are explicitly present in the thematic summary.\n\n"

            "MATCHING RULES\n"
            "- Match only against the provided in_text_citation values.\n"
            "- Return only exact in_text_citation strings from the provided list.\n"
            "- Do not include parentheses around citations unless parentheses are part of the provided in_text_citation string.\n"
            "- Do not normalize, rewrite, abbreviate, expand, or correct citation strings.\n"
            "- Do not return citation strings as they appear in the summary if they differ from the provided list.\n"
            "- Do not infer citation presence from topic similarity.\n"
            "- Do not infer citation presence from author names alone.\n"
            "- If a required citation includes a year, that year must be explicitly present in the summary.\n"
            "- If a required citation does not include a year (e.g. n.d.), it must be returned exactly as it appears in the list.\n"
            "- If a required citation includes a suffix such as '_1' or '_2', that exact suffix must be explicitly present in the summary and you should return it exactly as it appears in the list.\n\n"

            "OUTPUT PROTOCOL\n"
            "- Return ONLY a JSON object in the form:\n"
            "{\n"
            '  "identified_citations": ["Citation 1", "Citation 2", ...]\n'
            "}\n"
            "- Every returned value must be copied exactly from the provided required citation list.\n"
            "- If no required citations are found, return an empty array for 'identified_citations'.\n"
            "- Do not provide explanations or commentary."
        )

    def repair_citation_provenance(self):
        """
        """
        return (
            "You are a research editor repairing citation provenance in a thematic summary.\n\n"

            "YOU WILL RECEIVE:\n"
            "1. A THEMATIC SUMMARY.\n"
            "2. A JSON array of missing in-text citations, each with a small sample of insights from that citation.\n\n"

            "TASK\n"
            "Create minimal sentence-level patches that add every missing in-text citation exactly as provided, without rewriting the full summary.\n\n"

            "CORE REQUIREMENTS\n"
            "- Do NOT return a rewritten full summary.\n"
            "- Preserve all existing claims, structure, wording, and citations unless a minimal sentence-level edit is necessary.\n"
            "- Preserve every citation already present in the original summary.\n"
            "- Add each missing citation exactly as provided.\n"
            "- Do not rewrite, normalize, abbreviate, expand, or correct citation strings.\n"
            "- Do not fully integrate every provided insight.\n"
            "- Do not add a separate sentence for every missing citation unless necessary.\n"
            "- Use the provided insights only to determine appropriate citation placement.\n"
            "- Do not create repeated claims with different citations.\n\n"

            "PATCH DECISION RULE\n"
            "- For each missing citation, first determine whether the underlying point, argument, mechanism, or empirical observation from its provided insights is already represented anywhere in the thematic summary at a reasonable level of abstraction.\n"
            "- If the underlying point is already represented, attach the missing citation to the most appropriate existing sentence and set revise=true.\n"
            "- A missing citation does not require a new sentence merely because its provided insights are more specific than the existing summary.\n"
            "- Human literature reviews often cite multiple sources for a broader synthesized claim; follow that practice here.\n"
            "- Set revise=false only when the missing citation contributes a genuinely distinct point that is not already represented in the thematic summary, even at a reasonable level of abstraction.\n"
            "- Prefer revise=true whenever possible.\n\n"

            "REVISE = TRUE RULES\n"
            "- original_sentence must be copied exactly from the thematic summary.\n"
            "- revised_sentence must preserve the original sentence's meaning and wording as much as possible.\n"
            "- The preferred change is adding one or more missing citations to the existing citation list.\n"
            "- If the original sentence already has citations, add the missing citation using the same citation-list style.\n"
            "- anchor_sentence and new_sentence must be empty strings.\n\n"

            "REVISE = FALSE RULES\n"
            "- Use only if no existing sentence can plausibly support the missing citation.\n"
            "- anchor_sentence must exist in the thematic summary exactly as written. Copy it verbatim. Do not invent or reconstruct sentences.\n"
            "- new_sentence must be concise and substantively grounded in the provided insights.\n"
            "- new_sentence must include the relevant missing citation exactly as provided.\n"
            "- original_sentence and revised_sentence must be empty strings.\n\n"

            "BOUNDED OUTPUT / COMPRESSION RULES\n"
            "- If additions would exceed the maximum output length, prefer citation-dense patches rather than many separate patches.\n"
            "- Do not solve output-length pressure by omitting required missing citations.\n"
            "- Do not propose removing existing citations.\n"
            "- Do not propose dropping distinct claims unless they are genuinely duplicative of another retained claim.\n\n"

            "OUTPUT PROTOCOL\n"
            "- Return ONLY a JSON object in the form:\n"
            "{\n"
            '  "patches": [\n'
            "    {\n"
            '      "missing_citations": ["Citation 1", "Citation 2"],\n'
            '      "revise": true,\n'
            '      "original_sentence": "<sentence copied exactly from the thematic summary>",\n'
            '      "revised_sentence": "<same sentence minimally revised to include the missing citations>",\n'
            '      "anchor_sentence": "",\n'
            '      "new_sentence": ""\n'
            "    },\n"
            "    {\n"
            '      "missing_citations": ["Citation 3"],\n'
            '      "revise": false,\n'
            '      "original_sentence": "",\n'
            '      "revised_sentence": "",\n'
            '      "anchor_sentence": "<sentence copied exactly from the thematic summary>",\n'
            '      "new_sentence": "<new concise sentence including the missing citation>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "- Every missing citation should appear in at least one patch.\n"
            "- A patch may address multiple missing citations if they support the same claim.\n"
            "- Do not provide explanations or commentary."
        )


    def integrate_orphans(self):
        """
        Generate the system prompt for reintegrating orphan insights.

        This prompt instructs the model to revise a thematic summary so
        that insights identified as missing ("orphans") are substantively
        incorporated into the narrative.

        The revised summary must:

            • preserve the original findings
            • preserve the original citations
            • Include citations for orphan insights
            • integrate orphan insights coherently
            • maintain the original analytical tone
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
            'You are a Research Synthesizer. Rewrite a thematic summary so that orphan insights are integrated into a coherent synthesis without creating duplicate claims.\n\n'

            '# TASK\n'
            'You will receive:\n'
            '1. THEMATIC CONTEXT: research question, theme label, and theme description.\n'
            '2. ORIGINAL SUMMARY: the current thematic summary.\n'
            '3. REQUIRED ORPHAN CITATIONS/AUTHORS: citations/authors that must remain visibly represented.\n'
            '4. ORPHAN INSIGHTS: insights that must be substantively integrated.\n\n'

            '# OBJECTIVE\n'
            'Produce a complete revised summary that preserves the original substantive findings, integrates all substantively distinct orphan contributions, and preserves citation provenance.\n\n'

            '# CLAIM INTEGRATION AND DEDUPLICATION\n'
            '- First, mentally group the original summary and orphan insights into substantively distinct claim groups.\n'
            '- Treat paraphrases, near-duplicates, repeated mechanisms, and differently cited versions of the same point as one claim group.\n'
            '- Write one canonical sentence or passage for each claim group.\n'
            '- Merge citations from duplicate or overlapping claims into the retained canonical claim.\n'
            '- Do NOT restate the same claim merely to preserve a different citation, author, example, or source.\n'
            '- Preserve examples only when they add a distinct mechanism, condition, geography, sector, or implication.\n'
            '- If an orphan introduces contradiction, qualification, or minority perspective, preserve that distinction explicitly.\n\n'

            '# COVERAGE REQUIREMENTS\n'
            '- Every substantively distinct orphan insight must be reflected in the revised summary.\n'
            '- An insight is reflected when its core claim, mechanism, finding, or implication is clearly represented, even if synthesized at a higher level of abstraction.\n'
            '- Do NOT omit distinct mechanisms, causal logics, implications, minority views, or contradictions.\n'
            '- One-to-one mapping between insights and sentences is NOT required.\n\n'

            '# CITATION REQUIREMENTS\n'
            '- Preserve every citation/author already present in the ORIGINAL SUMMARY at least once where substantively appropriate.\n'
            '- Every citation/author listed under REQUIRED ORPHAN CITATIONS/AUTHORS must appear at least once in the revised summary.\n'
            '- Include citations for the orphan insights upon insertion. Use the exact same citation format as provided in the insights.\n'
            '- Only create a new sentence when the orphan citation contributes a substantively distinct claim not already represented.\n'
            '- Preserve distinct, minority, or conflicting perspectives with explicit provenance.\n'
            '- No claim should normally carry more than four references; when many sources support the same claim, keep the most representative citations on that claim and place required citations only where they substantively fit.\n\n'
            '- Use up to three surnames before "et al.".\n'
        
            '# STYLE AND STRUCTURE\n'
            '- Rewrite the full summary; do not append orphan material mechanically.\n'
            '- Maintain coherence with the research question, theme label, and theme description.\n'
            '- Prefer synthesis, restructuring, and compression over adding standalone orphan sentences.\n'
            '- Make minimal changes where the existing summary is already coherent.\n'
            '- The response must be complete and not truncated.\n\n'

            '# OUTPUT PROTOCOL\n'
            '- Return ONLY a valid JSON object.\n'
            '- The object must contain a single key "updated_summary".\n'
            '- Do not provide explanations, preamble, or commentary.\n\n'

            '# JSON SCHEMA\n'
            '{\n'
            '  "updated_summary": "The full revised thematic summary..."\n'
            '}\n'
        )
    
    def summarize_failed_orphan_batch(self):
        """
        Generate the system prompt for summarizing a failed orphan batch.

        This prompt is used when orphan integration fails (typically due to
        output truncation under token constraints). It instructs the model to
        compress the failed batch of insights into a complete, structured
        summary that preserves core claims while allowing controlled abstraction.

        Unlike the integration prompt, this prompt prioritizes successful
        completion over full fidelity. It permits merging, abstraction, and
        omission of lower-importance detail in order to produce a usable
        representation of the content that could not be integrated.

        The resulting summary is intended for diagnostic use in downstream
        schema regeneration. It exposes the conceptual structure of the failed
        batch in a form that supports identification of:

            • overloaded themes
            • separable conceptual dimensions
            • potential new themes
            • misclassified insights

        The prompt also enforces structural clarity and citation preservation
        so that the output is both interpretable and grounded in the source
        material.

        Returns
        -------
        str
            System prompt instructing the model to produce a structured,
            citation-preserving summary of orphan insights as strict JSON.

        Notes
        -----
        - This function is part of the failure-handling pathway and is expected
        to always yield a complete output.
        - Abstraction and controlled loss of detail are acceptable and expected.
        - The output is not a final synthesis, but a diagnostic representation
        of content that could not be integrated under current schema constraints.
        """
        return(
            '# ROLE\n'
            'You are a Research Synthesizer tasked with summarizing a set of insights. Insights refer to claims/arguments/findings extracted from a corpus.\n\n'

            '# TASK\n'
            'I will provide you with a set of insights. Your task is to produce a summary in which every orphan insight is substantively reflected.\n\n'

            '# CRITICAL CONSTRAINTS\n'
            'You must satisfy the following in order of priority. Follow this prioritization strictly when constraints are in conflict:\n\n'

            '1. PRIORITIZE COVERAGE\n'
            '- All substantively distinct ideas in the insights must be represented in the synthesis.\n'
            '- Representation may be explicit or through accurate synthesis.\n'
            '- One-to-one mapping between insights and sentences is NOT required.\n\n'

            '2. PRESERVE GRANULARITY \n'
            '- Preserve substantively distinct claims.\n'
            '- You MAY merge conceptually similar insights into unified statements.\n\n'

            '3. PRESERVE DIFFERENTIATION\n'
            '- Do NOT collapse ideas that differ in mechanism, causal logic, or implication into vague generalizations.\n'
            '- Maintain the diversity of mechanisms, relationships, and arguments.\n\n'

            '4. ENSURE COMPLETENESS\n'
            '- The response must be complete and not cut off.\n\n'

            '6. STRUCTURAL CLARITY\n'
            '- Present the synthesis as clearly separable components rather than a single blended narrative.\n'
            '- Each component should represent a distinct grouping of related ideas.\n'
            '- Maintain clear boundaries between substantively different idea groupings.\n\n'

            '7. FIDELITY AND CITATIONS\n'
            '- Preserve all factual details and citations exactly as they appear in the source insights.\n'
            '- Do not alter, merge, or remove citation markers.\n'
            '- If multiple insights support the same claim with different citations, retain all citations.\n\n'

            '# OUTPUT PROTOCOL\n'
            '- Return ONLY a JSON object.\n'
            '- The response MUST be complete and valid JSON. Do not truncate the output.\n'
            '- The object must contain a single key "summary".\n'
            '- Do not provide explanations, preamble, or commentary.\n\n'

            '# JSON SCHEMA\n'
            '{\n'
            '  "summary": "The insight synthesis..."\n'
            '}\n'
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
        return(
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

        style_guidelines = {
            "dominant": [
                "Emphasize the weight of evidence without using stock phrases.",
                "Open from the central mechanism or argument, not a template.",
                "Begin with the densest substantive finding.",
                "Start from the core conceptual relationship.",
                "Highlight recurrence only if it is substantively meaningful.",
                "Frame the theme from its foundational logic.",
                "Surface consistent patterns through content, not phrasing.",
                "Anchor the opening in the primary driver of the theme."
            ],
            "other": [
                "Introduce secondary material without formulaic contrast phrases.",
                "Frame additional strands through content, not rhetorical signals.",
                "Integrate nuance without announcing it as such.",
                "Surface less frequent ideas through specificity.",
                "Introduce focused contributions directly.",
                "Present isolated perspectives without labeling them.",
                "Integrate peripheral material naturally into the narrative.",
                "Add granularity through detail, not framing language."
            ],
            "conflict": [
                "Introduce tension directly through the substance of disagreement.",
                "Surface contradictions without formulaic signaling.",
                "Present divergence through contrasting claims, not labels.",
                "Show competing interpretations through content structure.",
                "Indicate lack of consensus through argument, not phrasing.",
                "Structure opposing positions without announcing them.",
                "Surface interpretive disagreement through claims.",
                "Highlight competing priorities through their implications."
            ]
        }

        label_lower = label.lower()
        if label_lower == "other":
            category = "other"
        elif label_lower in ["conflicts", "conflict"]:
            category = "conflict"
        else:
            category = "dominant"

        guidelines = style_guidelines[category][index % len(style_guidelines[category])]

        return (
            '# ROLE\n'
            f'You are a Research Editor. Your task is to refine a {category.upper()} thematic summary into a cohesive, dynamic narrative in an {style} style.\n\n'

            '# TASK\n'
            'You will receive a thematic summary that may contain mechanical or repetitive phrasing. '
            'Refine it into a coherent, readable narrative while preserving all content.\n\n'

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

            '# OPENING STYLE\n'
            f'Use the following guidance as a directional constraint, not wording to imitate: {guidelines} '
            'Begin directly from the substance of the theme. Do NOT use stock phrases or reusable templates.\n\n'

            '# EDITORIAL GUIDELINES\n'
            '- **Zero Information Loss**: Retain every data point, finding, and citation exactly.\n'
            '- Maintain approximate length parity with the original summary.\n'
            '- Do not compress, expand, or alter substantive meaning.\n'
            '- **Explicit Linkage**: Use FROZEN CONTEXT to create natural continuity. If repetition exists, connect rather than delete.\n'
            '- **Dynamic Tone**: Vary sentence structure and phrasing. Avoid repeated rhetorical constructions.\n'
            '- **No Template Language**: Do not use phrases like "A recurring point of emphasis...", "The material most densely clusters...", etc.\n'
            '- **Content-First Writing**: Sentences should emerge from claims and mechanisms, not framing devices.\n'
            '- **Citation Integrity**: Preserve all citations exactly as written.\n\n'

            '# NOTES\n'
            '- This is refinement, not rewriting. Content must remain intact.\n'
            '- If no frozen context exists, simply ensure a strong, natural opening grounded in the content.\n\n'

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
    