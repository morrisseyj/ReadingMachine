"""
Rendering layer for ReadingMachine outputs.

This module converts synthesized thematic analyses into presentation-
ready documents. It operates on completed synthesis artifacts stored
in `SummaryState` and produces formatted outputs such as Markdown,
DOCX, and PDF reports.

Unlike the analytical pipeline implemented in `core.py`, the rendering
layer does not modify analytical state. Instead it performs a series of
optional presentation transformations that improve readability and
prepare the synthesis for distribution.

The rendering workflow typically proceeds as follows:

    1. Select summaries to render
       The renderer determines which synthesis artifact should be used
       (cluster summaries, populated themes, or redundancy-reduced
       summaries) based on the current stage of the summarization
       pipeline.

    2. Optional stylistic rewriting
       Theme summaries may be rewritten in a different stylistic voice
       (e.g. academic, journalistic, conversational) while attempting to
       preserve their analytical content.

    3. Question-level summaries
       A concise narrative overview is generated for each research
       question by synthesizing the themes associated with that
       question.

    4. Executive summary and title
       The entire synthesis is summarized into a high-level executive
       overview and document title.

    5. Final render assembly
       All narrative components are integrated into a single ordered
       render DataFrame representing the final document structure.

    6. Document export
       The render DataFrame is converted into Markdown, DOCX, or PDF
       formats for distribution.

Render artifacts are persisted to disk so that expensive LLM operations
(such as stylistic rewrites or executive summary generation) can be
reused across sessions without recomputation.

State Interaction
-----------------

The rendering layer reads from the following state objects:

    SummaryState
        Provides the synthesized thematic summaries.

    CorpusState
        Provides source metadata and traceability information.

Rendering artifacts are intentionally not stored in a dedicated state
object because they represent terminal presentation outputs rather than
iterative analytical state.

Traceability
------------

The module also provides utilities for tracing claims appearing in the
rendered synthesis back to their originating insights and document
chunks. This preserves the pipeline’s core design principle that all
synthesized claims remain auditable against the underlying corpus.

Design Principle
----------------

Rendering separates *analysis* from *presentation*. The analytical
pipeline produces structured thematic knowledge, while the rendering
layer focuses on transforming that structure into readable narrative
outputs without altering the underlying analytical artifacts.
"""

# Import custom libraries and modules
from . import config, utils
from .prompts import Prompts
from .state import SummaryState, CorpusState


# Import standard libraries
from typing import Any, Dict, Optional
import os
import pandas as pd
import unicodedata
import re
from docx import Document
import json
import hashlib
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import shutil


class Render:
    def __init__(
        self, 
        llm_client: Any,
        ai_model: str,
        summary_state: SummaryState = None, 
        corpus_state: CorpusState = None,
        force: bool = False,
        render_object: tuple[str, int] = None,
        render_path: str = config.RENDER_SAVE_LOCATION,
        render_hash = config.summary_hash,
        output_save_location: str = os.path.join(config.OUTPUT_SAVE_LOCATION)
        ):
        """
        Rendering and presentation layer for ReadingMachine outputs.

        The Render class converts synthesized thematic summaries into
        presentation-ready documents and provides optional LLM-assisted
        formatting tools for improving narrative readability.

        Unlike the analytical pipeline stages, rendering artifacts are not
        part of the core analytical state. Instead they represent terminal
        presentation outputs derived from `SummaryState`. These artifacts are
        persisted separately so that expensive LLM-assisted formatting steps
        (such as stylistic rewrites or executive summary generation) can be
        reused across sessions without recomputation.

        The rendering pipeline supports several optional post-processing
        operations:

            • stylistic rewriting of theme summaries
            • question-level narrative summaries
            • executive summary generation
            • title generation
            • final document assembly
            • export to Markdown, DOCX, or PDF

        Rendering also provides a traceability utility that allows users to
        locate the original insights and document chunks associated with a
        claim appearing in the final synthesis.

        Initialization performs three main steps:

            1. Determine which synthesis artifact should be rendered based on
               the current stage of the summarization pipeline.
            2. Compute a deterministic hash of the selected summary object to
               ensure rendering operations correspond to a stable analytical
               state.
            3. Recover any previously generated rendering artifacts if the
               stored render hash matches the current summary content.

        Parameters
        ----------
        llm_client : Any
            Client used to call the LLM for rendering-related operations
            including stylistic rewriting, question summaries, and
            executive summary generation.

        ai_model : str
            Model identifier used for rendering-related LLM calls.

        summary_state : SummaryState
            Object containing the synthesized thematic summaries and related
            synthesis artifacts produced by the `Summarize` pipeline stage.

        corpus_state : CorpusState
            Object containing corpus metadata and insight information used
            for traceability and claim verification during rendering.

        force : bool, default=False
            If True, bypass stage validation checks and allow rendering even
            if the synthesis pipeline has not completed the orphan or
            redundancy passes. This should be used cautiously because earlier
            synthesis stages may contain unresolved omissions or redundancy.

        render_object : tuple[str, int], optional
            Explicitly specify which synthesis artifact to render and which
            index to use. For example:

                ("cluster_summary", 0)
                ("populated_theme", -1)
                ("redundancy", 0)

            This option requires `force=True`.

        render_path : str
            Directory where persistent rendering artifacts are stored. These
            include stylistic rewrites, question summaries, executive
            summaries, and the final render DataFrame.

        render_hash : str
            Filename used to store a deterministic hash of the summary object
            being rendered. This hash ensures that previously generated render
            artifacts are only reused when the underlying summaries have not
            changed.

        output_save_location : str
            Directory where exported documents (Markdown, DOCX, or PDF) will
            be written.

        Notes
        -----
        Rendering operations are guarded by a summary hash mechanism that
        prevents accidental reuse of formatting artifacts generated from a
        different synthesis state.

        If a previous render hash is detected but does not match the current
        summary object, the user is prompted to either:

            (1) abort rendering to avoid overwriting prior results
            (2) clear previous render artifacts and start a new render pass
        """
        # Check that we can instantiate this class - i.e. summaries have been generated
        if summary_state.cluster_summary_list == [] and summary_state.populated_theme_list == []:
            raise ValueError("SummaryState appears to be empty. Please run the summarization process before instantiating the Render class.")
        
        if render_object is not None:
            if not isinstance(render_object, tuple) or len(render_object) != 2:
                raise ValueError("render_object must be a tuple of length 2.")
            if render_object[0] not in ["cluster_summary", "populated_theme", "redundancy"]:
                raise ValueError(f"Invalid render_object '{render_object[0]}'. Valid options are 'cluster_summary', 'populated_theme', or 'redundancy'.")

        self.summary_state = summary_state
        self.corpus_state = corpus_state
        self.render_object = render_object
        self.force = force
        self.llm_client = llm_client
        self.ai_model = ai_model
        self.render_hash = render_hash
        self.render_path = render_path
        self.output_save_location: str = output_save_location

        # Get the exact content that we want to render on, depending on whether the use ran the redundancy pass or stopped at the final orphan pass ()
        self.summary_to_render = self._get_summaries_for_render(force=force)

        # Ensure sort for subsequent operations
        self.summary_to_render = self.summary_to_render.sort_values(by=["question_id", "theme_id"]).reset_index(drop=True)
        # Hash the summary to eiher check against previous renders or to set if this is the first time. This guards against changes in summary objects during rendering pass
        current_summary_hash = self._compute_df_hash(self.summary_to_render)

        # Handle check and recovery of pervious renders
        self._reinitialize_render(current_summary_hash=current_summary_hash)

    def _get_summaries_for_render(self, force) -> pd.DataFrame:
        """
        Select the summary object that should be rendered.

        This method determines which synthesis artifact should be used as the
        source for rendering based on the current stage of the summarization
        pipeline.

        Rendering normally occurs after either:

            - the orphan integration pass (fidelity-first output), or
            - the redundancy reduction pass (readability-first output).

        If `force=True`, earlier stages such as cluster summaries or populated
        themes may be rendered directly.

        If `render_object` is explicitly provided, that object is returned
        regardless of pipeline stage.

        Returns
        -------
        pd.DataFrame
            The summary DataFrame selected for rendering.
        """
        # First check whether the user has specified a render object, if so make sure they have set flag to true
        if self.render_object is not None and force == False:
            raise ValueError("You have specified a render_object but force is set to False. Please set force to True to render the specified object, or set render_object to None to automatically determine which summaries to render based on the current stage of the summarization process.")
        # If render object and force flag are set return the specified object at the specified stage, make sure index is within range
        if self.render_object is not None and force == True:
            obj_name, index = self.render_object
            if obj_name == "cluster_summary":
                if not (-len(self.summary_state.cluster_summary_list) <= index < len(self.summary_state.cluster_summary_list)):
                    raise ValueError(f"Index {index} out of range for cluster_summary_list with length {len(self.summary_state.cluster_summary_list)}.")
                return self.summary_state.cluster_summary_list[index]
            if obj_name == "populated_theme":
                if not (-len(self.summary_state.populated_theme_list) <= index < len(self.summary_state.populated_theme_list)):
                    raise ValueError(f"Index {index} out of range for populated_theme_list with length {len(self.summary_state.populated_theme_list)}.")
                return self.summary_state.populated_theme_list[index]
            else:
                obj_name = "redundancy" # This is not needed, just here for explicit clarity
                if not (-len(self.summary_state.redundancy_list) <= index < len(self.summary_state.redundancy_list)):
                    raise ValueError(f"Index {index} out of range for redundancy_list with length {len(self.summary_state.redundancy_list)}.")
                return self.summary_state.redundancy_list[index]


        # If render object not set, automatically get the summary object. First get the standard approach - redundancy or orphan - if they are complete. Otherwise check the flaf force option and return other options
        status_diagnostic = self.summary_state.status(diagnostic=True)
        max_stage, last_stage = status_diagnostic["max_stage"], status_diagnostic["stage"]
        # If the last stage is mapped themes or theme schema, we have to check whether a populated theme was generated - i.e. is max stage > 1 (i.e. a full pass has happened). If so the last available render is on the last populated_theme
        if last_stage in ["theme_schema", "mapped_theme"] and max_stage > 1:
            last_stage = "populated_theme"
        elif last_stage in ["theme_schema", "mapped_theme", "cluster_summary"] and max_stage <= 1:
            last_stage = "cluster_summary"
        else:
            last_stage = last_stage
        
        
        if last_stage == "redundancy":
            print("Summaries have gone through the redundancy pass and been checked for orphans. This is a readability first output, but has a risk of insight ommission. If fidelity is your priority you should run the orphan pass again before rendering. SummaryState.handle_orphans().")
            return(self.summary_state.redundancy_list[0])
        if last_stage == "orphan":
            print("Summaries have gone through the orphan pass, but not been checked for redundancy. This is a fidelity first output and may contain redundancies. If you want to handle redundancies before rendering, please run the redundancy pass: SummaryState.address_redundancy().")
            return(self.summary_state.populated_theme_list[-1])
        if last_stage not in ["redundancy", "orphan"] and not force:
            raise ValueError(f"Your summary state is currently at stage '{last_stage}'. You can only render summaries if you have completed the orphan pass and/or the redundancy pass. Use SummaryState.status() to determine your current stage. " 
                                "If you know what you are doing and want to render with the current summaries, please instantiate the Render class with force=True to override this check.")
        if last_stage in ["cluster_summary", "populated_theme"] and force:
            print(f"WARNING Force rendering with summaries from stage '{last_stage}'. Be aware that the summaries may not be fully processed and may contain redundancies or unhandled orphans.")
            if last_stage == "populated_theme":
                return(self.summary_state.populated_theme_list[-1])
            else:
                return(self.summary_state.cluster_summary_list[0])
            
    @staticmethod
    def _compute_df_hash(df) -> str:
        """
        Compute a deterministic SHA-256 hash of a DataFrame.

        The DataFrame is normalized prior to hashing by:

            - filling NaN values
            - sorting columns alphabetically
            - sorting rows deterministically

        This ensures the resulting hash is stable across runs and can be used
        to detect whether the underlying summary content has changed between
        render passes.

        Returns
        -------
        str
            SHA-256 hash representing the normalized DataFrame.
        """

        if df is None or df.empty:
            return "EMPTY"

        # Defensive copy
        df_normalized = df.copy()

        # Normalize NaNs
        df_normalized = df_normalized.fillna("__NULL__")

        # Sort columns alphabetically for consistency
        df_normalized = df_normalized.sort_index(axis=1)

        # Sort rows deterministically by all columns
        df_normalized = df_normalized.sort_values(
            by=list(df_normalized.columns)
        ).reset_index(drop=True)

        # Convert to stable JSON representation
        json_bytes = df_normalized.to_json().encode("utf-8")

        # Hash
        return hashlib.sha256(json_bytes).hexdigest()          
                   
    def _reinitialize_render(self, current_summary_hash) -> None:
        """
        Initialize or recover rendering artifacts.

        This method ensures that rendering operations correspond to a stable
        summary state by comparing the current summary hash with any previously
        stored render hash.

        Behavior
        --------

        If no previous render hash exists:
            A new hash file is created and rendering proceeds normally.

        If a previous render hash exists and matches:
            Previously generated rendering artifacts are loaded and attached
            to the current Render instance.

        If a previous render hash exists but differs:
            The user is prompted to either:

                1. Abort rendering to prevent overwriting previous outputs.
                2. Delete previous render artifacts and proceed with a new render.

        This mechanism prevents accidental reuse of rendering artifacts that
        were generated from a different synthesis state.

        """
        # Check if any previous render objects exist:
        os.makedirs(self.render_path, exist_ok=True) # Ensure the render path exists

        if os.path.exists(os.path.join(self.render_path, self.render_hash)):
            saved_hash_df = pd.read_parquet(os.path.join(self.render_path, self.render_hash))
            saved_hash = saved_hash_df["summary_hash"].iloc[0]
            if saved_hash != current_summary_hash:
                restart = None
                while restart not in ["1", "2"]:
                    restart = input(
                        "A previous render hash was found that does not match the current summary hash. This suggests that the summaries have changed since the last render. " \
                        "Do you want to proceed with rendering and overwrite the previous render?\n"
                        "(1) Yes, proceed with rendering and overwrite the previous render.\n" 
                        "(2) No, stop the rendering process to avoid overwriting the previous render (this will abort render and require the correct summary_state be passed to Render init).\n"
                        "(1/2):\n"
                        ).strip()

                if restart == "2":
                    raise ValueError("Rendering process aborted by user to avoid overwriting previous render. Please check your summary_state and ensure it is correct before re-instantiating the Render class.")
                else:
                    # Clean up the render artifacts and start new render
                    shutil.rmtree(self.render_path) # This will remove all the previous render artifacts, which is important to avoid confusion and ensure a clean slate for the new render. We can be aggressive here with the cleanup because we have already confirmed with the user that they want to proceed with overwriting the previous render.
                    os.makedirs(self.render_path, exist_ok=True) # Re-create the render path after cleanup
                    # Create and save the new hash file
                    new_hash_df = pd.DataFrame({"summary_hash": [current_summary_hash]})
                    new_hash_df.to_parquet(os.path.join(self.render_path, self.render_hash), index=False)
                    print("Previous render artifacts deleted. Hash set to current summary_to_render.")
                    return(None)
                
            else: # if the hash matches we load the old dataframe and create all the attributes that we can from it.
                print("A previous render with the same summary hash was found. Loading the previously generated cosmetic artifacts. You can overwrite these by recalling the generation methods.\n")
                path = Path(self.render_path)
                loaded_artefacts = []
                for file in path.glob("*.parquet"):
                    if file.stem == config.render_prefix["render_title_exec_summary_df"]:
                        self.title_exec_summary_df = pd.read_parquet(file)
                        loaded_artefacts.append("title_summary_df")
                    elif file.stem == config.render_prefix["render_question_summary_df"]:
                        self.question_summary_df = pd.read_parquet(file)
                        loaded_artefacts.append("question_summary_df")
                    else:
                        if file.stem == config.render_prefix["render_stylized_rewrite_df"]:
                            self.stylized_rewrite_df = pd.read_parquet(file)
                            loaded_artefacts.append("stylized_rewrite_df")
                
                # If all three artefacts are loaded, we can integrate the cosmetic changes into a render_df that can be used for the final output generation. If some but not all cosmetic artefacts are loaded, we do not automatically integrate as the render_df would be incomplete, but we do inform the user of which artefacts were loaded and that they can be integrated by calling integrate_cosmetic_changes() once they have added any missing artefacts. If no cosmetic artefacts are loaded, we do not attempt to integrate and just inform the user that no cosmetic artefacts were found and they can proceed with generating new cosmetic artefacts or check their render save location if they believe they should be there.
                if len(loaded_artefacts) == 3:
                    final_render_df = self.integrate_cosmetic_changes() # I don't call render_df here (self.final_render_df gets set in integrate_cosmetic_changes), but mention it to be explicit
                    print("All cosmetic artefacts loaded and integrated. You can proceed with generating the final outputs.")
                
                if not hasattr(self, "final_render_df"):
                    print(f"{len(loaded_artefacts)} cosmetic artifacts loaded: {loaded_artefacts}. Final render_df not found. You should add any missing artifacts and/or call integrate_cosmetic_changes() to construct the final_render_df)")


        else: # if no previous file exists, save the hash and create the render df from the summary_to_render
            # Create and save the new hash file
            os.makedirs(self.render_path, exist_ok=True) # Ensure the render path exists
            new_hash_df = pd.DataFrame({"summary_hash": [current_summary_hash]})
            new_hash_df.to_parquet(os.path.join(self.render_path, self.render_hash), index=False)
            return(None)

    
    
    def stylistic_rewrite(self, style: str = "academic") -> pd.DataFrame:
        """
        Apply a stylistic rewrite to theme summaries.

        This optional rendering step rewrites the thematic summaries in a
        specified stylistic register (e.g., academic, journalistic,
        conversational) while attempting to preserve the original meaning
        and analytical content.

        The rewrite operates sequentially within each research question.
        Previously rewritten themes are passed to the model as frozen context
        to maintain stylistic consistency and narrative continuity across
        themes.

        Parameters
        ----------
        style : str, default="academic"
            Target writing style for the rewritten summaries. The prompt
            template determines how this style is interpreted by the model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing rewritten theme summaries with columns:

                - stylized_text
                - question_id
                - theme_id

        Behavior
        --------
        If a stylistic rewrite already exists, the user is prompted to either:

            (1) regenerate the rewrite
            (2) reuse the existing rewrite

        If question summaries have already been generated, the user is warned
        that stylistic rewriting ideally occurs before question-level summaries
        so that stylistic consistency propagates through later rendering stages.

        Notes
        -----
        This step prioritizes readability and narrative flow. Although prompts
        attempt to preserve analytical fidelity, any rewrite carries some risk
        of information loss or semantic drift. For maximum fidelity, users may
        prefer to skip this step. 
        """
        # First check wheter a stylistic rewrite has already been done, if so offer reload or regenerate options
        if hasattr(self, "stylized_rewrite_df"): 
            rerun = None
            while rerun not in ["1", "2"]:
                rerun = input(
                    "A stylistic rewrite of the themes has already been generated. Do you want to regenerate the stylistic rewrite?\n"
                    "(1) Yes, regenerate the stylistic rewrite.\n" 
                    "(2) No, keep the existing stylistic rewrite.\n"
                    "(1/2):\n"
                ).strip()
                
            if rerun == "2":
                print("Keeping existing stylistic rewrite.")
                return self.stylized_rewrite_df
            else:                
                print("Regenerating stylistic rewrite.")
        
        # Then check whether the summaries have already been generated. If they have, send a warning that it is better to run the stylistic re-write prior to summarizing so that the tone is carried through
        if hasattr(self, "question_summary_df"):
            abort = None
            while abort not in ["1", "2"]:
                abort = input(
                    "You have already generated question summaries. It is generally recommended to do the stylistic rewrite before generating the question summaries so that the style is carried through to the question summaries. Do you want to proceed with the stylistic rewrite even though you have already generated question summaries?\n"
                    "(1) Yes, proceed with the stylistic rewrite even though I have already generated question summaries.\n" 
                    "(2) No, I want to do a stylistic rewrite before generating question summaries.\n"
                    "(1/2):\n"
                ).strip()

                if abort == "2":
                    print("Aborting stylistic rewrite so that you can do it before generating question summaries. Please run stylistic_rewrite() and then re-run gen_question_summaries() to generate the question summaries with the rewritten themes.")
                    return None
        
        stylized_content_df_out = pd.DataFrame(columns=["stylized_text", "question_id", "theme_id"])

        total_themes = self.summary_to_render.shape[0]
        count = 1

        for qid in self.summary_to_render["question_id"].unique():
            question_df = self.summary_to_render[self.summary_to_render["question_id"] == qid].copy()
            # Place frozen content here so that it resets for each rq
            frozen_content = ""
            for idx, row in question_df.iterrows():
                print(f"Stylistic rewrite for theme {count} of {total_themes}")
                question_text = row["question_text"]
                content = row["thematic_summary"]
                theme_label = row["theme_label"]
                theme_id = row["theme_id"]
                sys_prompt = Prompts().stylistic_rewrite(style=style, index=idx, label=theme_label)
                user_prompt = (
                    f"CURRENT RESEARCH QUESTION: {question_text}\n"
                    "FROZEN CONTENT:\n"
                    f"{frozen_content}\n\n"
                    f"CURRENT THEME LABEL: {theme_label}\n"
                    "THEMATIC SUMMARY TO STYLE:\n"
                    f"{content}\n\n"
                )
                fall_back = {"refined_summary": ""}

                json_schema = {
                    "name": "stylistic_rewrite_generator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "refined_summary": {
                                "type": "string",
                                "description": "Refined thematic summary text."
                            }
                        },
                        "required": ["refined_summary"],
                        "additionalProperties": False
                    }
                }
                response = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    return_json=True,
                    json_schema=json_schema,
                    fall_back=fall_back
                )

                # Get the data, transform to df
                stylized_content = response["refined_summary"]
                stylized_content_df = pd.DataFrame({
                    "stylized_text": [stylized_content],
                    "question_id": [qid],
                    "theme_id": [theme_id],
                    })
                
                # Append to the output df
                stylized_content_df_out = pd.concat([stylized_content_df_out, stylized_content_df], ignore_index=True)
                frozen_content += f"{row['theme_label']}\n\n{stylized_content}\n\n"
                count += 1
        
        self.stylized_rewrite_df = stylized_content_df_out
        self.stylized_rewrite_df.to_parquet(os.path.join(self.render_path, config.render_prefix["render_stylized_rewrite_df"] + ".parquet"), index=False)
        return(stylized_content_df_out)
    
    
    def _gen_question_payload(self, qid) -> str:
        """
        Construct the thematic payload for a research question.

        This helper method concatenates all theme summaries associated with
        a specific research question into a single structured text block.
        Each theme is clearly labeled and separated with boundary markers
        so that the LLM can distinguish between themes when generating
        higher-level summaries.

        The resulting string is used as the input payload for:

            - question-level summaries
            - executive summaries

        Parameters
        ----------
        qid : str or int
            Identifier of the research question whose themes should be
            aggregated.

        Returns
        -------
        str
            Concatenated thematic content formatted as:

                Theme: <theme_label>
                <thematic_summary>
                --- END THEME ---

        Notes
        -----
        This structured format improves model comprehension by explicitly
        separating themes while preserving their ordering within the
        research question.
        """
        question_df = self.summary_to_render[self.summary_to_render["question_id"] == qid].copy()
        output = ""
        for _, row in question_df.iterrows():
            output += f"Theme: {row['theme_label']}\n"
            output += f"{row['thematic_summary']}\n"
            output += "--- END THEME ---\n"

        return(output)
    
    
    def gen_question_summaries(self) -> pd.DataFrame:

        """
        Generate question-level summaries from theme summaries.

        This method synthesizes the theme summaries associated with each
        research question into a concise overview paragraph. The goal is
        to provide a high-level narrative summary that captures the
        thematic structure identified during synthesis.

        Each research question is processed independently. All theme
        summaries belonging to that question are combined into a structured
        payload before being sent to the LLM for summarization.

        Behavior
        --------
        If question summaries already exist, the user is prompted to either:

            (1) regenerate the summaries
            (2) keep the existing summaries

        If a stylistic rewrite has not been performed, the user is warned
        that stylistic rewriting can be applied prior to question summary
        generation so that stylistic changes propagate through the summary.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the generated summaries with columns:

                - content
                - doc_attr
                - question_id
                - question_text

        Notes
        -----
        Question summaries serve as an intermediate narrative layer between
        individual theme summaries and higher-level outputs such as
        executive summaries or final report sections.
        """
        
        # First check whether question summaries already exist, if so offer reload or regenerate options
        if hasattr(self, "question_summary_df"):
            rerun = None
            while rerun not in ["1", "2"]:
                rerun = input(
                    "Question summaries already exist. Do you want to regenerate the question summaries?\n"
                    "(1) Yes, regenerate the question summaries.\n" 
                    "(2) No, keep the existing question summaries.\n"
                    "(1/2):\n"
                ).strip()
            if rerun == "2":
                print("Keeping existing question summaries.")
                return self.question_summary_df
            else:                
                print("Regenerating question summaries.")

        # Then check whether a stylistic re-write has been done. Flag for the user that if they want a stylistic re-write its likely better to do it after the summaries have been generated, but they can proceed as they like
        if not hasattr(self, "stylized_rewrite_df"):
            abort = None
            while abort not in ["1", "2"]:
                abort = input(
                    "You have not done a stylistic rewrite of the themes. This is not neccesary to proceed, "
                    "however if you intend to do a stylistic rewrite and want  the style of the re-write to appear in your summaries it is reccomended to do the re-write first. "
                    "Do you want to proceed with generating question summaries without doing a stylistic rewrite?\n"
                    "(1) Yes, proceed with generating question summaries without doing a stylistic rewrite.\n" 
                    "(2) No, I want to do a stylistic rewrite before generating question summaries.\n"
                    "(1/2):\n"
                ).strip()

                if abort == "2":
                    print("Aborting question summary generation so that you can do a stylistic rewrite first. Please run stylistic_rewrite() and then re-run gen_question_summaries() to generate the question summaries with the rewritten themes.")
                    return None
        
        # loop over question id to get the question payload
        question_summaries_df_list = []

        working_render_df = self.summary_to_render.copy()

        total_questions = working_render_df["question_id"].nunique()
        count = 1

        for qid in working_render_df["question_id"].unique():
            print(f"Generating summary for question {count} of {total_questions}")
            question_id = qid
            question_text = working_render_df[working_render_df["question_id"] == qid]["question_text"].iloc[0]
            question_payload = self._gen_question_payload(qid=question_id)

        # Prepare the full payload for the llm call
            sys_prompt = Prompts().question_summaries()
            user_prompt = (
            f"Research question: {question_text}\n\n"
            "CONTENT TO SUMMARIZE:\n"
            f"{question_payload}\n\n"
            )

            fall_back = {"summary": ""}
            json_schema = {
                "name": "question_summary_generator",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A single-paragraph thematic overview (3-5 sentences) synthesizing the themes for the research question."
                        }
                    },
                    "required": ["summary"],
                    "additionalProperties": False
                }
            }
            
            # Call the llm
            response = utils.call_chat_completion(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                return_json=True,
                json_schema=json_schema,
                fall_back=fall_back
            )

            question_summary_txt = response["summary"]
            question_summary_df = pd.DataFrame({
                "content": [question_summary_txt],
                "doc_attr": ["question_summary"],
                "question_id": [qid],
            })

            question_summaries_df_list.append(question_summary_df)
            count += 1

        question_summaries_df_out = pd.concat(question_summaries_df_list, ignore_index=True).sort_values(by=["question_id"]).reset_index(drop=True)
        question_summaries_df_out = (
            question_summaries_df_out
            .merge(self.corpus_state.questions[["question_id", "question_text"]],
                   on="question_id",
                   how="left")
        )


        self.question_summary_df = question_summaries_df_out
        self.question_summary_df.to_parquet(os.path.join(self.render_path, config.render_prefix["render_question_summary_df"] + ".parquet"), index=False)       
        return(self.question_summary_df)
    
    
    def _gen_exec_summary_payload(self) -> str:
        """
        Construct the payload used to generate the executive summary.

        This method aggregates the thematic summaries for all research
        questions into a single structured text block. Each research
        question section contains its associated theme summaries, and
        questions are clearly separated with boundary markers.

        The resulting payload provides the LLM with the full synthesized
        thematic structure of the corpus so it can generate a coherent
        executive overview of the findings.

        Returns
        -------
        str
            Structured text payload containing all theme summaries grouped
            by research question in the following format:

                Research Question: <question_text>
                Theme: <theme_label>
                <thematic_summary>
                --- END THEME ---
                ...
                === END QUESTION ===

        Notes
        -----
        This payload structure ensures the model has visibility into the
        complete synthesis output while preserving the boundaries between
        themes and research questions.
        """

        temp_summary_df = self.summary_to_render.copy()
        temp_summary_df = temp_summary_df.sort_values(by=["question_id", "theme_id"]).reset_index(drop=True)

        # Loop over the questions
        output = ""
        for qid in temp_summary_df["question_id"].unique():
            output += f"Research Question: {temp_summary_df[temp_summary_df['question_id'] == qid]['question_text'].iloc[0]}\n"
            question_output = self._gen_question_payload(qid)
            output += question_output
            output += "\n=== END QUESTION ===\n\n"

        return output
    

    def gen_exec_summary(self, word_count: int = 500) -> str:
        """
        Generate the executive summary and document title.

        This method produces a high-level narrative overview of the entire
        synthesis by summarizing the thematic results across all research
        questions. It also generates a concise title describing the overall
        synthesis.

        The LLM receives the aggregated theme summaries for all research
        questions and produces:

            • an executive summary in continuous prose
            • a concise title for the document

        Parameters
        ----------
        word_count : int, default=500
            Target length for the executive summary in words.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the generated title and executive summary
            with columns:

                - content
                - doc_attr

            Where `doc_attr` indicates whether the row represents the
            document title or the executive summary.

        Behavior
        --------
        If an executive summary and title already exist, the user is
        prompted to either:

            (1) regenerate them
            (2) keep the existing results

        Notes
        -----
        The generated outputs are stored in `self.title_exec_summary_df`
        and persisted to disk so that expensive LLM calls do not need to
        be repeated across render sessions.
        """

        if hasattr(self, "title_exec_summary_df"):
            rerun = None
            while rerun not in ["1", "2"]:
                rerun = input(
                    "An executive summary and title have already been generated. Do you want to regenerate the executive summary and title?\n"
                    "(1) Yes, regenerate the executive summary and title.\n" 
                    "(2) No, keep the existing executive summary and title.\n"
                    "(1/2):\n"
                ).strip()
                
            if rerun == "2":
                print("Keeping existing executive summary and title.")
                return self.title_exec_summary_df
            else:                
                print("Regenerating executive summary and title.")

        exec_summary_payload = self._gen_exec_summary_payload()
        
        sys_prompt = Prompts().exec_summary(word_count=word_count)
        user_prompt = exec_summary_payload
        fall_back = {"executive_summary": "", "title": ""}
        json_schema = {
            "name": "executive_summary_generator",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "executive_summary": {
                        "type": "string",
                        "description": "The final executive summary text in continuous prose."
                    },
                    "title": {
                        "type": "string",
                        "description": "A concise descriptive title, maximum 12 words, no subtitle."
                    }
                },
                "required": ["executive_summary", "title"],
                "additionalProperties": False
            }
        }

        response = utils.call_chat_completion(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            llm_client=self.llm_client,
            ai_model=self.ai_model,
            return_json=True,
            json_schema=json_schema,
            fall_back=fall_back
        )

        exec_summary = response["executive_summary"]
        title = response["title"]
        
        title_exec_summary_df = pd.DataFrame({
            "content": [title, exec_summary],
            "doc_attr": ["title", "exec_summary"]
            })

        self.title_exec_summary_df = title_exec_summary_df

        self.title_exec_summary_df.to_parquet(os.path.join(self.render_path, config.render_prefix["render_title_exec_summary_df"] + ".parquet"), index=False)

        return self.title_exec_summary_df

    
    def _add_question_summaries_to_render_df(self, render_df, question_summaries: list) -> pd.DataFrame:
        """
        Insert question-level summaries into the render DataFrame.

        This helper method integrates question summaries into the
        theme-level render structure. Each question summary is inserted
        before the themes associated with that research question so that
        the final rendered output follows a clear narrative hierarchy:

            research question
            → question summary
            → theme summaries

        Parameters
        ----------
        render_df : pd.DataFrame
            DataFrame containing theme summaries prepared for rendering.

        question_summaries : list[str]
            List of generated summaries for each research question.

        Returns
        -------
        pd.DataFrame
            Updated render DataFrame containing question summaries inserted
            ahead of their corresponding themes.

        Notes
        -----
        This method only inserts the summaries into the rendering structure.
        Integration with the title and executive summary occurs later during
        `integrate_cosmetic_changes()`.
        """
        working_render_df = render_df.copy()

        # Separate the working_render_df into a list of dataframes, one for each question
        list_of_qdfs = []
        for qid in self.corpus_state.questions["question_id"].to_list():
            question_df = working_render_df[working_render_df["question_id"] == qid].copy()
            list_of_qdfs.append(question_df)
        
        # Concat the question summaries with the corresponding question dfs and create a new list of dataframes with summaries
        list_of_qdfs_with_summaries = []
        for qdf, q_summary in zip(list_of_qdfs, question_summaries):
            question_summary_df = pd.DataFrame({
                "content": [q_summary],
                "doc_attr": ["question_summary"],
                "question_id": [qdf["question_id"].iloc[0]]
            })
            qdf_with_summary = pd.concat([question_summary_df, qdf], ignore_index=True)
            list_of_qdfs_with_summaries.append(qdf_with_summary)

        # Concat the list of question dfs with summaries into a single dataframe
        questions_with_summaries_df = pd.concat(list_of_qdfs_with_summaries, ignore_index=True)
        questions_with_summaries_df = (
            questions_with_summaries_df
            .drop(columns=["question_text"], errors="ignore")
            .merge(self.corpus_state.questions[["question_id", "question_text"]],
                   on="question_id",
                   how="left")
        )
        # Return the resulting dataframe
        return questions_with_summaries_df


    def integrate_cosmetic_changes(self) -> pd.DataFrame:
        """
        Construct the final render DataFrame by integrating cosmetic artifacts.

        This method combines the synthesized thematic summaries with
        optional presentation-layer artifacts produced during the rendering
        stage. These artifacts include:

            - question summaries
            - stylistic rewrites
            - title and executive summary

        The resulting DataFrame represents the fully assembled document
        structure used for final export (Markdown, DOCX, or PDF).

        Returns
        -------
        pd.DataFrame
            Final render DataFrame containing all narrative components and
            rendering metadata.

        Workflow
        --------
        The method performs the following steps:

            1. Validate that the summary hash matches the render hash
            recorded during initialization.
            2. Construct the base render DataFrame from theme summaries.
            3. Insert question summaries if available.
            4. Merge stylistic rewrites if present.
            5. Prepend the title and executive summary.
            6. Assign a deterministic document order.
            7. Persist the final render DataFrame.

        Raises
        ------
        ValueError
            If the underlying summary data has changed since the Render
            object was initialized.

        Notes
        -----
        The hash validation step ensures that cosmetic artifacts are only
        applied to the exact synthesis output from which they were
        generated, preventing inconsistencies if summaries are modified
        after rendering begins.
        """

        # --- Hash validation ---
        current_summary_hash = self._compute_df_hash(self.summary_to_render)
        saved_hash_df = pd.read_parquet(os.path.join(self.render_path, self.render_hash))
        saved_hash = saved_hash_df["summary_hash"].iloc[0]

        if current_summary_hash != saved_hash:
            raise ValueError(
                "The summary_to_render has changed since Render initialization. "
                "Reinitialize Render before integrating artifacts."
            )

        # --- Base render df ---
        render_df = (
            self.summary_to_render
            .sort_values(by=["question_id", "theme_id"])
            .copy()
        )

        render_df.rename(columns={"thematic_summary": "content"}, inplace=True)
        render_df["doc_attr"] = "thematic_summary"

        # --- Question summaries ---
        if hasattr(self, "question_summary_df"):
            render_df = self._add_question_summaries_to_render_df(
                render_df,
                self.question_summary_df["content"].to_list()
            )

        # --- Stylized rewrite ---
        if hasattr(self, "stylized_rewrite_df"):
            render_df = render_df.merge(
                self.stylized_rewrite_df[["question_id", "theme_id", "stylized_text"]],
                on=["question_id", "theme_id"],
                how="left"
            )
            render_df["stylized_text"] = render_df["stylized_text"].fillna(render_df["content"])

        # --- Title + Exec summary ---
        if hasattr(self, "title_exec_summary_df"):
            title_exec_df = self.title_exec_summary_df.copy()
            # Check if there is a stylized rewrite, in which case copy the content column to the stylized text so we have something to render
            if "stylized_text" in render_df.columns:
                title_exec_df["stylized_text"] = title_exec_df["content"] # This is to ensure that if there is a stylized rewrite for the title and exec summary, which otherwise are under content.
            # use safe concat with schema to ensure that schema is adopted - to keep ordering dtypes correct, take schema from render.
            render_df = utils.concat_with_schema(title_exec_df, render_df, schema_from="bottom")



        # --- Final ordering ---
        render_df["doc_order"] = range(len(render_df))
        render_df.drop(columns = ["allocated_length", "current_length", "perc_of_max_length", "length_flag"], inplace=True, errors="ignore") # We don't need this column in the render df and it can cause issues if it is left in with null values, so we drop it if it exists. We set errors to ignore just in case it doesn't exist, which can happen if the summary_to_render was generated with an older version of the code that didn't have allocated_length in the summary df.

        # --- Save ---
        self.final_render_df = render_df
        self.final_render_df.to_parquet(
            os.path.join(self.render_path, config.render_prefix["final_render_df"] + ".parquet"),
            index=False
        )
        return self.final_render_df
    
    def render_output(
        self,
        output_type: str = "md",
        use_stylized: bool = False,
        filename: str = "rendered_output"
    ) -> None:
        """
        Export the rendered synthesis to a document format.

        This method converts the final render DataFrame into a structured
        document and writes it to disk in the requested format.

        Supported formats include:

            - Markdown (.md)
            - Microsoft Word (.docx)
            - PDF (.pdf)

        Parameters
        ----------
        output_type : str, default="md"
            Output format to generate. Must be one of:

                "md"
                "docx"
                "pdf"

        use_stylized : bool, default=False
            If True, use the stylistically rewritten text (`stylized_text`)
            when available. If False, the original synthesized summaries
            (`content`) are used.

        filename : str, default="rendered_output"
            Base filename for the exported document (without extension).

        Raises
        ------
        ValueError
            If `final_render_df` has not yet been created or if the
            specified output type is unsupported.

        Notes
        -----
        This method dispatches rendering to format-specific helper methods:

            - `_render_to_markdown`
            - `_render_to_docx`
            - `_render_to_pdf`
        """

        if not hasattr(self, "final_render_df"):
            raise ValueError(
                "final_render_df not found. Please run integrate_cosmetic_changes() first."
            )

        if output_type not in ["md", "docx", "pdf"]:
            raise ValueError("output_type must be one of: 'md', 'docx', 'pdf'.")
        
        os.makedirs(self.output_save_location, exist_ok=True)

        df = self.final_render_df.copy()

        # Choose text source
        if use_stylized and "stylized_text" in df.columns:
            df["final_text"] = df["stylized_text"].fillna(df["content"])
        else:
            df["final_text"] = df["content"]

        # Dispatch
        if output_type == "md":
            self._render_to_markdown(df, filename)

        elif output_type == "docx":
            self._render_to_docx(df, filename)

        elif output_type == "pdf":
            self._render_to_pdf(df, filename)

        print(f"Render complete: {filename}.{output_type}")


    def _render_to_markdown(self, df, filename):

        """
        Render the synthesis document as Markdown.

        This method converts the final render DataFrame into a Markdown
        document with hierarchical headings corresponding to the narrative
        structure of the synthesis.

        Structure
        ---------
        The resulting Markdown document is organized as:

            Title
            Executive Summary
            Research Question
                Question Overview
                Theme Summaries

        Parameters
        ----------
        df : pd.DataFrame
            Render DataFrame containing ordered narrative components.

        filename : str
            Output filename (without extension).
        """

        lines = []

        for _, row in df.sort_values("doc_order").iterrows():

            text = row["final_text"]
            attr = row["doc_attr"]
            question_text = row["question_text"]

            if attr == "title":
                lines.append(f"# {text}\n")

            elif attr == "exec_summary":
                lines.append(f"## EXECUTIVE SUMMARY\n{text}\n")

            elif attr == "question_summary":
                lines.append(f"## {question_text}\n## Question Overview\n{text}\n")

            elif attr == "thematic_summary":
                lines.append(f"### {row.get('theme_label','')}\n{text}\n")

        output_path = os.path.join(self.output_save_location, f"{filename}.md")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _render_to_docx(self, df, filename):
        """
        Render the synthesis document as a Microsoft Word file.

        This method converts the render DataFrame into a structured DOCX
        document using hierarchical headings to represent the synthesis
        narrative.

        Parameters
        ----------
        df : pd.DataFrame
            Render DataFrame containing ordered narrative components.

        filename : str
            Output filename (without extension).

        Notes
        -----
        The document structure mirrors the Markdown renderer but uses
        Word heading levels to create the hierarchy:

            Title → Heading 0
            Executive Summary → Heading 1
            Research Questions → Heading 1
            Themes → Heading 2
        """

        document = Document()

        for _, row in df.sort_values("doc_order").iterrows():

            text = row["final_text"]
            attr = row["doc_attr"]
            question_text = row["question_text"]

            if attr == "title":
                document.add_heading(text, level=0)

            elif attr == "exec_summary":
                document.add_heading("EXECUTIVE SUMMARY", level=1)
                document.add_paragraph(text)

            elif attr == "question_summary":
                document.add_heading(question_text, level=1)
                document.add_heading("Question Overview", level=1)
                document.add_paragraph(text)

            elif attr == "thematic_summary":
                document.add_heading(row.get("theme_label",""), level=2)
                document.add_paragraph(text)

        output_path = os.path.join(self.output_save_location, f"{filename}.docx")
        document.save(output_path)

    def _render_to_pdf(self, df, filename):
        """
        Render the synthesis document as a PDF.

        This method converts the render DataFrame into a PDF document using
        the ReportLab library.

        Paragraph formatting is normalized to ensure that line breaks in the
        synthesized summaries are preserved in the final PDF.

        Parameters
        ----------
        df : pd.DataFrame
            Render DataFrame containing ordered narrative components.

        filename : str
            Output filename (without extension).

        Notes
        -----
        Markdown-style line breaks are converted into HTML-style break tags
        so that the ReportLab paragraph renderer correctly interprets them.
        """

        def normalize_for_pdf(text):
            # Convert double newlines into paragraph breaks
            text = re.sub(r"\n\s*\n", "<br/><br/>", text)
            # Convert remaining single newlines
            text = text.replace("\n", "<br/>")
            return text

        output_path = os.path.join(self.output_save_location, f"{filename}.pdf")
        doc = SimpleDocTemplate(output_path)

        styles = getSampleStyleSheet()
        elements = []

        for _, row in df.sort_values("doc_order").iterrows():

            text = row["final_text"]
            text = normalize_for_pdf(text)
            attr = row["doc_attr"]
            question_text = row["question_text"]

            if attr == "title":
                elements.append(Paragraph(f"<b>{text}</b>", styles["Title"]))
                elements.append(Spacer(1, 0.3 * inch))

            elif attr == "exec_summary":
                elements.append(Paragraph("<b>EXECUTIVE SUMMARY</b>", styles["Heading1"]))
                elements.append(Paragraph(text, styles["BodyText"]))
                elements.append(Spacer(1, 0.2 * inch))

            elif attr == "question_summary":
                elements.append(Paragraph(f"<b>{question_text}</b>", styles["Heading1"]))
                elements.append(Paragraph("<b>Question Overview</b>", styles["Heading1"]))
                elements.append(Paragraph(text, styles["BodyText"]))
                elements.append(Spacer(1, 0.2 * inch))

            elif attr == "thematic_summary":
                elements.append(Paragraph(f"<b>{row.get('theme_label','')}</b>", styles["Heading2"]))
                elements.append(Paragraph(text, styles["BodyText"]))
                elements.append(Spacer(1, 0.2 * inch))

        doc.build(elements)

    def trace_claim(self, 
                    question_text: str, 
                    theme_label:str, 
                    citation_lastname: list, 
                    citation_year: int):
        
        """
        Trace a claim in the rendered synthesis back to its source text.

        This method allows users to retrieve the original insights and
        document chunks that contributed to a specific claim appearing in
        the rendered thematic summaries.

        The lookup proceeds through the pipeline traceability chain:

            theme
            → mapped insights
            → source insight records
            → source document chunks

        Parameters
        ----------
        question_text : str
            Text of the research question containing the claim.

        theme_label : str
            Theme label under which the claim appears in the rendered output.

        citation_lastname : list[str]
            List of author last names used to identify the source paper.
            Matching is fuzzy and will return results if any of the provided
            names appear in the author field.

        citation_year : int
            Publication year used to filter the source paper.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the candidate insights and document chunks
            associated with the specified claim. Columns include:

                - insight_id
                - insight
                - chunk_id
                - chunk_text
                - paper_title
                - paper_author
                - paper_date

        Notes
        -----
        This method provides a lightweight traceability mechanism for
        verifying synthesized claims against the underlying corpus.

        Future versions of the pipeline may allow direct retrieval via
        `insight_id` rather than citation metadata.

        """
        
        # Get the theme_id for the question and theme label
        question_id = self.final_render_df[self.final_render_df["question_text"] == question_text]["question_id"].iloc[0]
        theme_id = self.final_render_df[
            (self.final_render_df["question_id"] == question_id) & (self.final_render_df["theme_label"] == theme_label)
        ]["theme_id"].iloc[0]

        # use theme id to get the insights that were mapped to the theme in the summary state
        possible_insights_df = self.summary_state.mapped_theme_list[-1][
            (self.summary_state.mapped_theme_list[-1]["theme_id"] == theme_id)
            & (self.summary_state.mapped_theme_list[-1]["question_id"] == question_id)
        ].copy()

        possible_insights_lst = possible_insights_df["insight_id"].tolist()

        # Get all the insights that match the citation info and the theme_id, merge with chunks to get the chunk_text
        possible_insights_df = self.corpus_state.insights[self.corpus_state.insights["insight_id"].isin(possible_insights_lst)].copy()
        regex_pattern = "|".join(citation_lastname)
        possible_insights_df = (
            possible_insights_df[
            possible_insights_df["paper_author"].str.contains(
                pat = regex_pattern, case=False, na=False, regex=True
                ) 
                & (possible_insights_df["paper_date"] == citation_year)
            ].merge(
                self.corpus_state.chunks[["paper_id", "chunk_id", "chunk_text"]], 
                how="left",
                on=["chunk_id","paper_id"]
            ).copy()
        )
        # Return the relevant columns for the user to parse. Allows them to confirm they have the correct citation.
        possible_insights_df = possible_insights_df[["insight_id", "insight", "chunk_id", "chunk_text", "paper_title", "paper_author", "paper_date"]]
        return(possible_insights_df)
        