# multi_tool_agent/agent.py
import os
import json
from typing import Dict, Any, List, Union
import traceback
import uuid

from dotenv import load_dotenv

try:
    from google.adk.agents import Agent
except ImportError:
    print("CRITICAL Error: google-adk library not found. Please install it: pip install google-adk")
    exit(1)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("CRITICAL Error: google-generativeai library not found. Please install it: pip install google-generativeai")
    exit(1)

from repo_fetcher import clone_repository, cleanup_repository, get_file_content
from code_analyzer import get_repository_structure_string, detect_python_dependencies, identify_code_files_for_rag
from rag_processor import RAGProcessor

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("CRITICAL Error: GEMINI_API_KEY not found in .env (multi_tool_agent/agent.py).")
    exit(1)

AGENT_LLM_MODEL_NAME = "gemini-2.0-flash"

try:
    rag_processor_instance_for_tools = RAGProcessor(gemini_api_key=GEMINI_API_KEY)
    print("Global RAGProcessor instance initialized for agent tools (multi_tool_agent/agent.py).")
except Exception as e:
    print(f"CRITICAL Error initializing RAGProcessor in multi_tool_agent/agent.py: {e}.")
    traceback.print_exc()
    rag_processor_instance_for_tools = None

_session_active_repo_paths: Dict[str, str] = {}

def create_session_id() -> str:
    return str(uuid.uuid4())

def analyze_github_repository(github_url: str, session_id: str) -> Dict[str, Any]:
    print(f"\n[TOOL_CALL] analyze_github_repository: URL='{github_url}', Session='{session_id[:8]}'")
    global _session_active_repo_paths
    if session_id in _session_active_repo_paths:
        old_path = _session_active_repo_paths[session_id]
        print(f"  Cleaning up previous repository for session {session_id[:8]} at {old_path}")
        cleanup_repository(old_path)
        del _session_active_repo_paths[session_id]

    local_repo_path, repo_obj_for_cleanup = clone_repository(github_url)
    if not local_repo_path:
        return {"status": "error", "summary": f"Failed to clone repository: {github_url}"}

    _session_active_repo_paths[session_id] = local_repo_path
    print(f"  Repository cloned to: {local_repo_path}")

    analysis_results: Dict[str, Any] = {
        "status": "pending",
        "summary": "Analysis in progress.",
        "project_name": "",
        "local_repo_path": local_repo_path,
        "file_structure_string": "",
        "dependencies": [],
        "files_selected_for_rag": [],
        "rag_collection_name": "",
        "rag_ingestion_summary": "RAG ingestion not attempted."
    }

    try:
        project_name = github_url.split('/')[-1].replace('.git', '')
        analysis_results["project_name"] = project_name
        analysis_results["file_structure_string"] = get_repository_structure_string(local_repo_path, max_depth=3, max_items=40)
        analysis_results["dependencies"] = detect_python_dependencies(local_repo_path, get_file_content)
        files_for_rag = identify_code_files_for_rag(local_repo_path, max_files=20)
        analysis_results["files_selected_for_rag"] = files_for_rag

        if files_for_rag and rag_processor_instance_for_tools:
            rag_collection_name = f"repo_{project_name.replace('-', '_').lower()}_{session_id[:8]}"
            analysis_results["rag_collection_name"] = rag_collection_name

            print(f"  Setting up RAG collection: '{rag_collection_name}'")
            rag_processor_instance_for_tools.setup_collection(rag_collection_name, delete_existing=True)
            print(f"  Ingesting {len(files_for_rag)} files into RAG for collection '{rag_collection_name}'...")
            ingest_status = rag_processor_instance_for_tools.ingest_files_content(
                repo_local_path=local_repo_path,
                relative_file_paths=files_for_rag,
                get_file_content_func=get_file_content
            )
            analysis_results["rag_ingestion_details"] = ingest_status
            if ingest_status.get("status") == "success":
                analysis_results["rag_ingestion_summary"] = f"RAG processed {ingest_status.get('chunks_added', 0)} chunks from {ingest_status.get('files_processed',0)} files."
            else:
                analysis_results["rag_ingestion_summary"] = f"RAG ingestion issue: {ingest_status.get('message', 'Unknown error')}."
        else:
            analysis_results["rag_ingestion_summary"] = "RAG ingestion skipped (no files or RAG processor issue)."
        
        analysis_results["status"] = "success"
        analysis_results["summary"] = (
            f"Successfully analyzed repository: {project_name}. "
            f"{analysis_results['rag_ingestion_summary']} "
            "All analysis data is included in this tool's output. The agent must remember it."
        )
        return analysis_results

    except Exception as e:
        print(f"  Error during repository analysis: {e}")
        traceback.print_exc()
        cleanup_repository(local_repo_path, repo_obj_for_cleanup)
        if session_id in _session_active_repo_paths:
            del _session_active_repo_paths[session_id]
        analysis_results["status"] = "error"
        analysis_results["summary"] = f"Error during analysis: {e}"
        return analysis_results

def retrieve_code_or_document_context(query: str, rag_collection_name: str, n_results: Union[int, None] = 3) -> Dict[str, Any]:
    print(f"\n[TOOL_CALL] retrieve_code_or_document_context: Query='{query}', Collection='{rag_collection_name}'")
    if n_results is None or not isinstance(n_results, int) or n_results <= 0: n_results = 3

    if not rag_processor_instance_for_tools:
        return {"status": "error", "summary": "RAG system not available.", "retrieved_context": ""}
    
    if not rag_collection_name:
        return {"status": "error", "summary": "RAG collection name not provided. Cannot query.", "retrieved_context": ""}
    
    try:
        if rag_processor_instance_for_tools.collection is None or rag_processor_instance_for_tools.collection.name != rag_collection_name:
            print(f"  Switching RAG processor to collection: {rag_collection_name}")
            rag_processor_instance_for_tools.collection = rag_processor_instance_for_tools.chroma_client.get_collection(name=rag_collection_name)
    except Exception as e:
        return {"status": "error", "summary": f"Could not access RAG collection '{rag_collection_name}': {e}", "retrieved_context": ""}
    
    retrieved_text = rag_processor_instance_for_tools.query_vector_db(query, n_results=n_results)
    if "CONTEXT_NOT_AVAILABLE" in retrieved_text:
        return {"status": "no_results", "summary": f"No specific context found for query: '{query}'.", "retrieved_context": ""}
    else:
        return {"status": "success", "summary": f"Retrieved relevant context for query: '{query}'.", "retrieved_context": retrieved_text}

readme_agent_instructions = f"""
Run this create_session_id to create a unique session ID for this interaction and use this session ID for all tool calls.
You are "ReadmeGen", an AI assistant specialized in generating comprehensive README.md files for GitHub projects.
Your primary goal is to help the user create an excellent README by interactively building sections.

**Core Workflow:**
1.  **Greeting & Repo Request:** Start by greeting the user and asking for the GitHub repository URL.
2.  **Analyze Repository (Mandatory First Step for New Repo):**
    *   When the user provides a GitHub URL, YOU **MUST** call the `analyze_github_repository` function.
    *   The tool will return a detailed dictionary containing: 'project_name', 'local_repo_path', 'file_structure_string', 'dependencies', 'rag_collection_name', and 'rag_ingestion_summary'.
    *   YOU **MUST REMEMBER** these details, especially 'project_name', 'file_structure_string', 'dependencies', and 'rag_collection_name' for subsequent operations.
    *   Report the 'summary' from the tool's output to the user.
3.  **Generating README Sections:**
    *   The user will ask you to generate specific sections (e.g., "Overview", "Tech Stack", "Installation", "Usage").
    *   For each section:
        a.  **Recall Analysis Details:** Use the 'project_name', 'file_structure_string', and 'dependencies' that you remembered from the output of `analyze_github_repository`.
        b.  **Gather Specific Context (RAG):** Before writing, YOU **SHOULD** call the `retrieve_code_or_document_context` function. You MUST pass the 'rag_collection_name' (that you remembered from `analyze_github_repository` output) to this tool, along with a relevant query for the section.
        c.  **Synthesize and Generate:** Combine the remembered static analysis data and the RAG context from `retrieve_code_or_document_context` to write the Markdown content.
        d.  Present the generated Markdown to the user.
4.  **Handling User Feedback & Iteration:** Incorporate user feedback. If the user mentions an image for file structure (you cannot process images), ask for a text description.
5.  **Switching Repositories:** If the user provides a new GitHub URL, re-run the `analyze_github_repository` function for the new URL. This will provide new analysis data you must then remember.

**Function Usage Guidelines:**
*   `analyze_github_repository(github_url)`: Use ONCE per new repository URL. Carefully note ALL its output fields for later use.
*   `retrieve_code_or_document_context(query, rag_collection_name, n_results)`: Use FREQUENTLY before writing detailed content. You MUST provide the `rag_collection_name` from the analysis step.

**Interaction Style:** Be conversational, clear, and ask clarifying questions. State limitations if info isn't found. Output clean Markdown.
"""

try:
    root_agent = Agent(
        model=AGENT_LLM_MODEL_NAME,
        tools=[
            analyze_github_repository,
            retrieve_code_or_document_context,
            create_session_id
        ],
        instruction=readme_agent_instructions,
        name="readme_gen_chat_agent",
        description="An interactive AI assistant for generating README.md files for GitHub projects."
    )
    print(f"Google ADK Agent '{root_agent.name}' defined successfully in agent.py.")
except Exception as e:
    print(f"CRITICAL Error defining Google ADK Agent in agent.py: {e}")
    traceback.print_exc()
    exit(1)