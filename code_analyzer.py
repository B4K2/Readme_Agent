# code_analyzer.py
import os
import ast
import toml

def _get_local_file_content(full_file_path):
    if os.path.isfile(full_file_path):
        try:
            with open(full_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return None
    return None

def get_repository_structure_string(repo_path, max_depth=2, ignore_dirs=None, max_items=30):
    if ignore_dirs is None:
        ignore_dirs = ['.git', '.venv', '__pycache__', 'node_modules', '.DS_Store', 'dist', 'build', 'docs', 'examples', 'tests', 'test']

    structure_lines = []
    item_count = 0
    repo_root_name = os.path.basename(repo_path)

    for root, dirs, files in os.walk(repo_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')] # Filter ignored and hidden

        relative_root = os.path.relpath(root, repo_path)
        level = 0 if relative_root == "." else relative_root.count(os.sep) + 1

        if level > max_depth or item_count >= max_items:
            if level <= max_depth and item_count >= max_items and not structure_lines[-1].endswith("..."):
                 structure_lines.append("  " * level + "...")
            dirs[:] = []
            continue

        current_dir_name = os.path.basename(root) if relative_root != "." else repo_root_name
        structure_lines.append("  " * level + f"{current_dir_name}/")
        item_count +=1

        # Sort files and dirs for consistent output (optional)
        files.sort()
        # dirs.sort() # Already sorted by os.walk by default on some systems but explicit can be good

        for f_name in files:
            if item_count >= max_items:
                if not structure_lines[-1].endswith("..."):
                    structure_lines.append("  " * (level + 1) + "...")
                break
            structure_lines.append("  " * (level + 1) + f_name)
            item_count += 1
        if item_count >= max_items: break

    return "\n".join(structure_lines)

def detect_python_dependencies(repo_path, get_file_content_func):
    """
    Detects Python dependencies from requirements.txt and pyproject.toml.
    :param repo_path: Path to the cloned repository.
    :param get_file_content_func: A function like repo_fetcher.get_file_content.
    """
    dependencies = set()
    # requirements.txt
    content_req = get_file_content_func(repo_path, "requirements.txt")
    if content_req:
        for line in content_req.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                dep_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].split('!=')[0].split('~=')[0].split('[')[0].strip()
                if dep_name:
                    dependencies.add(dep_name)
    # pyproject.toml
    content_pyproject = get_file_content_func(repo_path, "pyproject.toml")
    if content_pyproject:
        try:
            data = toml.loads(content_pyproject)
            if "project" in data and "dependencies" in data["project"] and isinstance(data["project"]["dependencies"], list):
                for dep in data["project"]["dependencies"]:
                    dependencies.add(dep.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].strip())
            if "tool" in data and "poetry" in data["tool"] and \
               "dependencies" in data["tool"]["poetry"] and isinstance(data["tool"]["poetry"]["dependencies"], dict):
                for dep_name in data["tool"]["poetry"]["dependencies"].keys():
                    if dep_name.lower() != "python":
                        dependencies.add(dep_name)
        except toml.TomlDecodeError:
            print("Warning: Could not parse pyproject.toml")
        except Exception as e:
            print(f"Warning: Error processing pyproject.toml: {e}")
    return sorted(list(dependencies))

def identify_code_files_for_rag(repo_path, lang_exts=None, max_files=15, common_code_dirs=None):
    """
    Identifies code files and other relevant text files for RAG.
    Prioritizes common code directories.
    """
    if lang_exts is None:
        lang_exts = ['.py', '.js', '.ts', '.java', '.go', '.rs', '.rb', '.php'] # Common code extensions
    text_exts = ['.md', '.txt', '.rst'] # Other text files
    all_relevant_exts = lang_exts + text_exts

    if common_code_dirs is None:
        common_code_dirs = ['src', 'app', 'lib', 'pkg', 'cmd'] # Common source directories

    relevant_files = []
    
    # First pass: common code directories
    for common_dir in common_code_dirs:
        dir_path = os.path.join(repo_path, common_dir)
        if os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                if len(relevant_files) >= max_files: break
                for file in files:
                    if any(file.endswith(ext) for ext in all_relevant_exts):
                        relative_path = os.path.relpath(os.path.join(root, file), repo_path)
                        if relative_path not in relevant_files: # Avoid duplicates
                            relevant_files.append(relative_path)
                            if len(relevant_files) >= max_files: break
                if len(relevant_files) >= max_files: break
        if len(relevant_files) >= max_files: break

    # Second pass: root directory and other directories if still under max_files
    if len(relevant_files) < max_files:
        for root, dirs, files in os.walk(repo_path, topdown=True):
            # Skip already processed common dirs and typical ignored dirs
            dirs[:] = [d for d in dirs if d not in common_code_dirs and 
                       d not in ['.git', '.venv', '__pycache__', 'node_modules', 'dist', 'build', 'docs', 'examples', 'tests', 'test'] and
                       not d.startswith('.')]
            if len(relevant_files) >= max_files: break
            for file in files:
                if any(file.endswith(ext) for ext in all_relevant_exts):
                    relative_path = os.path.relpath(os.path.join(root, file), repo_path)
                    # Check if it's from a common_dir path already added (less likely with dirs[:] filtering)
                    # or if it's already in the list
                    is_in_common_dir_path = any(relative_path.startswith(cd + os.sep) for cd in common_code_dirs)
                    if not is_in_common_dir_path and relative_path not in relevant_files:
                        relevant_files.append(relative_path)
                        if len(relevant_files) >= max_files: break
            if len(relevant_files) >= max_files: break
            
    # Ensure README.md or similar is included if present and not already picked
    readme_candidates = ['README.md', 'README.rst', 'readme.md']
    for rc in readme_candidates:
        if len(relevant_files) >= max_files: break
        if os.path.exists(os.path.join(repo_path, rc)) and rc not in relevant_files:
            relevant_files.append(rc)
            
    return relevant_files[:max_files]

if __name__ == '__main__':
    print("Testing code_analyzer (requires a local repo path for full test)")
    class MockRepoFetcher:
        def get_file_content(self, repo_path, relative_file_path):
            if relative_file_path == "requirements.txt":
                return "flask==2.0\nrequests\n#comment\n"
            if relative_file_path == "pyproject.toml":
                return """
[project]
dependencies = ["django>=3.0", "numpy"]
[tool.poetry.dependencies]
python = "^3.8"
pandas = "^1.2"
"""
            return None

    mock_fetcher = MockRepoFetcher()
    deps = detect_python_dependencies(".", mock_fetcher.get_file_content)
    print(f"Detected dependencies (mock): {deps}")