# rag_processor.py
import os
import json
import chromadb
from tqdm import tqdm
from pathlib import Path
import hashlib
import shutil
import tempfile

from dotenv import load_dotenv

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    langchain_chunker = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
except ImportError:
    print("Warning: langchain library not found. Using basic splitter. pip install langchain")
    class BasicTextSplitter:
        def __init__(self, chunk_size=700, chunk_overlap=100):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text):
            if not text: return []
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
    langchain_chunker = BasicTextSplitter()

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    raise ImportError("google-generativeai library not found. Please install it: pip install google-generativeai")

DEFAULT_CHROMA_DB_PATH = "./chroma_db_readme_agent"
DEFAULT_GEMINI_EMBEDDER_MODEL_ID = "models/text-embedding-004"

class RAGProcessor:
    _gemini_client = None
    _current_embedding_model_id = None

    def __init__(self,
                 gemini_api_key: str,
                 chroma_db_path: str = DEFAULT_CHROMA_DB_PATH,
                 embedding_model_id: str = DEFAULT_GEMINI_EMBEDDER_MODEL_ID):

        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for RAGProcessor.")

        self._initialize_gemini_client_if_needed(gemini_api_key, embedding_model_id)

        self.embedding_model_id = embedding_model_id
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = None

    @classmethod
    def _initialize_gemini_client_if_needed(cls, api_key: str, model_id: str):
        if cls._gemini_client and cls._current_embedding_model_id == model_id:
            return True

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Cannot initialize Gemini client.")

        cls._current_embedding_model_id = model_id
        print(f"Initializing Gemini Client for embedding model: {model_id}...")

        try:
            cls._gemini_client = genai.Client(api_key=api_key)
            test_result = cls._gemini_client.models.embed_content(
                model=model_id,
                contents=["test initialization"],
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            if not (hasattr(test_result, 'embeddings') and test_result.embeddings and hasattr(test_result.embeddings[0], 'values')):
                test_single_result = cls._gemini_client.models.embed_content(
                    model=model_id, content="test single",
                    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                if not (hasattr(test_single_result, 'embedding') and hasattr(test_single_result.embedding, 'values')):
                     print(f"Warning: Test embedding response structure not as expected for model {model_id}. Proceeding with caution. Batch response: {test_result}, Single response: {test_single_result}")
                else:
                     print(f"Gemini Client configured successfully for model {model_id} (single content test).")
            else:
                print(f"Gemini Client configured successfully for model {model_id} (batch content test).")
            return True
        except AttributeError as e:
            raise ConnectionError(f"Failed to initialize Gemini Client (AttributeError): {e}. Check google-generativeai version compatibility with genai.Client().")
        except Exception as e:
            detailed_error = str(e)
            if hasattr(e, 'message'): detailed_error = e.message
            elif hasattr(e, 'args') and e.args: detailed_error = str(e.args[0])

            if "API_KEY_INVALID" in detailed_error.upper() or "PERMISSION_DENIED" in detailed_error.upper():
                 error_msg = f"API key issue or permissions with model '{model_id}': {detailed_error}"
            elif "QUOTA" in detailed_error.upper():
                 error_msg = f"Quota exceeded for model '{model_id}'. Details: {detailed_error}"
            elif "404" in detailed_error or "NOT_FOUND" in detailed_error.upper():
                 error_msg = f"Embedding model '{model_id}' not found or unavailable: {detailed_error}"
            else:
                 error_msg = f"Failed to initialize/test Gemini Embedder with model '{model_id}': {detailed_error}"
            raise ConnectionError(error_msg)

    def setup_collection(self, collection_name: str, delete_existing: bool = False):
        if delete_existing:
            try:
                self.chroma_client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass

        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"embedding_model": self.embedding_model_id}
            )
            print(f"Using ChromaDB collection: {self.collection.name} (Count: {self.collection.count()})")
        except Exception as e:
            raise RuntimeError(f"Failed to setup ChromaDB collection '{collection_name}': {e}")

    def _generate_embeddings_batch(self, texts: list[str], task_type="RETRIEVAL_DOCUMENT"):
        if not RAGProcessor._gemini_client:
            raise RuntimeError("Gemini client not initialized.")
        try:
            response = RAGProcessor._gemini_client.models.embed_content(
                model=RAGProcessor._current_embedding_model_id,
                contents=texts,
                config=genai_types.EmbedContentConfig(task_type=task_type)
            )
            if hasattr(response, 'embeddings') and response.embeddings:
                if all(hasattr(emb, 'values') for emb in response.embeddings):
                    return [list(emb.values) for emb in response.embeddings]
                else:
                    raise ValueError("Unexpected structure within response.embeddings: item missing 'values'.")
            else:
                if len(texts) == 1 and hasattr(response, 'embedding') and hasattr(response.embedding, 'values'):
                    return [list(response.embedding.values)]
                raise ValueError(f"Unexpected embedding response structure from client. Expected 'embeddings' list. Got: {response}")
        except Exception as e:
            print(f"Error generating Gemini embeddings batch with client: {e}")
            return None

    def ingest_files_content(self,
                             repo_local_path: str,
                             relative_file_paths: list[str],
                             get_file_content_func,
                             max_chunks_per_file=20):
        if not self.collection:
            raise RuntimeError("ChromaDB collection is not set up. Call setup_collection first.")

        all_chunk_texts = []
        all_metadatas = []
        all_ids = []
        processed_files = 0

        for rel_path in tqdm(relative_file_paths, desc="Processing files for RAG", unit="file"):
            content = get_file_content_func(repo_local_path, rel_path)
            if not content or content.isspace():
                continue

            chunks = langchain_chunker.split_text(content)
            if not chunks:
                continue

            chunks = chunks[:max_chunks_per_file]
            file_hash = hashlib.md5(rel_path.encode()).hexdigest()[:8]

            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{file_hash}_c{i}"
                metadata = {
                    "source_file": rel_path, "chunk_index": i,
                    "file_content_length": len(content), "chunk_length": len(chunk_text)
                }
                all_chunk_texts.append(chunk_text)
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)
            processed_files +=1

        if not all_chunk_texts:
            print("No text chunks generated from provided files.")
            return {"status": "no_action", "files_processed": processed_files, "chunks_added": 0}

        batch_size = 100
        all_embeddings = []
        for i in tqdm(range(0, len(all_chunk_texts), batch_size), desc="Generating embeddings", unit="batch"):
            batch_texts_to_embed = all_chunk_texts[i : i + batch_size]
            batch_embeddings = self._generate_embeddings_batch(batch_texts_to_embed)
            if batch_embeddings and len(batch_embeddings) == len(batch_texts_to_embed):
                all_embeddings.extend(batch_embeddings)
            else:
                msg = f"Embedding batch failed or mismatch for texts {i} to {i+len(batch_texts_to_embed)-1}."
                print(msg)
                if batch_embeddings is None: print("  Reason: _generate_embeddings_batch returned None (likely an API error).")
                else: print(f"  Reason: Expected {len(batch_texts_to_embed)} embeddings, got {len(batch_embeddings)}.")
                return {"status": "error", "message": msg, "files_processed": processed_files, "chunks_added": 0}

        db_batch_size = 2000
        chunks_added_to_db = 0
        for i in tqdm(range(0, len(all_ids), db_batch_size), desc="Adding to ChromaDB", unit="batch"):
            ids_b, embeds_b, metas_b, docs_b = (
                all_ids[i : i + db_batch_size], all_embeddings[i : i + db_batch_size],
                all_metadatas[i : i + db_batch_size], all_chunk_texts[i : i + db_batch_size]
            )
            try:
                self.collection.add(ids=ids_b, embeddings=embeds_b, metadatas=metas_b, documents=docs_b)
                chunks_added_to_db += len(ids_b)
            except Exception as e:
                msg = f"Error adding batch to ChromaDB (IDs {i} to {i+len(ids_b)-1}): {e}"
                print(msg)
                return {"status": "error", "message": msg, "files_processed": processed_files, "chunks_added": chunks_added_to_db}

        print(f"Ingestion complete. Files processed: {processed_files}. Total chunks added to DB: {chunks_added_to_db}.")
        return {"status": "success", "files_processed": processed_files, "chunks_added": chunks_added_to_db, "collection_total_count": self.collection.count()}

    def query_vector_db(self, query_text: str, n_results: int = 3, where_filter: dict = None):
        if not self.collection or self.collection.count() == 0:
            return "No context available from RAG (collection empty or not ready)."

        query_embedding_list = self._generate_embeddings_batch([query_text], task_type="RETRIEVAL_QUERY")
        if not query_embedding_list or not query_embedding_list[0]:
            return "Failed to generate query embedding for RAG."
        query_embedding = query_embedding_list[0]

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter, include=['documents', 'metadatas', 'distances']
            )
            if results and results['documents'] and results['documents'][0]:
                context_str_parts = [
                    f"Source: {meta.get('source_file', 'N/A')} (Chunk {meta.get('chunk_index', 'N/A')}, Dist: {dist:.4f}):\n{doc}"
                    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
                ]
                return "\n\n---\n\n".join(context_str_parts)
            else:
                return "No relevant documents found in RAG."
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return f"Error during RAG query: {e}"

if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env. Exiting RAG test.")
        exit()

    class MockFileFetcher:
        def __init__(self, base_path_str: str):
            self.base_path = Path(base_path_str)
            os.makedirs(self.base_path / "src", exist_ok=True)
            with open(self.base_path / "src" / "main.py", "w", encoding="utf-8") as f:
                f.write("def hello():\n    print('Hello from main.py')\n\nclass Test:\n    pass\n" * 10)
            with open(self.base_path / "README.md", "w", encoding="utf-8") as f:
                f.write("# Project Title\n\nThis is a test project.\nIt does amazing things.\n" * 5)
            with open(self.base_path / "empty.txt", "w", encoding="utf-8") as f:
                f.write("   ")

        def get_content(self, repo_path_ignored, relative_file_path):
            file_path = self.base_path / relative_file_path
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            return None
        def cleanup(self):
            shutil.rmtree(self.base_path)

    temp_repo_dir_str = tempfile.mkdtemp(prefix="rag_test_repo_")
    mock_fetcher = MockFileFetcher(temp_repo_dir_str)

    test_db_path_str = "./test_chroma_db_rag"
    test_db_path = Path(test_db_path_str)

    rag_proc_instance = None

    if test_db_path.exists():
        try:
            shutil.rmtree(test_db_path)
            print(f"Cleaned up pre-existing {test_db_path_str} before test.")
        except Exception as e:
            print(f"Warning: Could not pre-clean {test_db_path_str}: {e}")

    try:
        rag_proc_instance = RAGProcessor(gemini_api_key=api_key, chroma_db_path=test_db_path_str)
        test_collection_name = "my_test_repo_docs_client_pattern"
        rag_proc_instance.setup_collection(test_collection_name, delete_existing=True)

        files_to_ingest_rel = ["src/main.py", "README.md", "non_existent.py", "empty.txt"]
        ingest_result = rag_proc_instance.ingest_files_content(
            repo_local_path=temp_repo_dir_str,
            relative_file_paths=files_to_ingest_rel,
            get_file_content_func=mock_fetcher.get_content
        )
        print("\nIngestion Result:")
        print(json.dumps(ingest_result, indent=2))

        if ingest_result.get("status") == "success" and ingest_result.get("chunks_added", 0) > 0:
            print("\nQuerying RAG:")
            query1 = "What is this project about?"
            query_result1 = rag_proc_instance.query_vector_db(query1, n_results=2)
            print(f"\nQuery 1: {query1}\n{query_result1}")

            query2 = "Code example for hello function"
            query_result2 = rag_proc_instance.query_vector_db(query2, n_results=1)
            print(f"\nQuery 2: {query2}\n{query_result2}")
        else:
            print("\nSkipping RAG query due to ingestion issues or no chunks added.")

    except Exception as e:
        print(f"Error in RAGProcessor standalone test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mock_fetcher.cleanup()

        if rag_proc_instance and rag_proc_instance.chroma_client:
            try:
                print(f"Resetting ChromaDB client for {test_db_path_str}...")
                rag_proc_instance.chroma_client.reset()
                print("ChromaDB client reset.")
            except Exception as e:
                print(f"Error resetting ChromaDB client: {e}. Manual cleanup of {test_db_path_str} might be needed.")

        if test_db_path.exists():
            try:
                shutil.rmtree(test_db_path)
                print(f"Cleaned up {test_db_path_str}.")
            except Exception as e:
                print(f"Failed to clean up {test_db_path_str} after client reset: {e}")
                print("You might need to delete it manually or ensure no other process is using it.")