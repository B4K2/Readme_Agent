# repo_fetcher.py
import os
import shutil
import tempfile
import git
from urllib.parse import urlparse, urlunparse
import stat

def normalize_github_url_for_cloning(url):
    parsed_url = urlparse(url)
    path = parsed_url.path.strip('/')
    if not path.endswith('.git'):
        path += '.git'
    if path and not path.startswith('/'):
        path = '/' + path
    clonable_url_parts = (parsed_url.scheme, parsed_url.netloc, path, '', '', '')
    clonable_url = urlunparse(clonable_url_parts)
    return clonable_url

def clone_repository(repo_url):
    clonable_url = normalize_github_url_for_cloning(repo_url)
    print(f"Normalized URL for cloning: {clonable_url}")
    repo_object = None
    try:
        temp_dir = tempfile.mkdtemp()
        # print(f"Cloning {clonable_url} into {temp_dir}...") # Less verbose for library use
        repo_object = git.Repo.clone_from(clonable_url, temp_dir)
        # print(f"Successfully cloned to: {temp_dir}")
        return temp_dir, repo_object
    except git.exc.GitCommandError as e:
        print(f"Error cloning repository '{repo_url}': {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during cloning '{repo_url}': {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None

def get_file_content(repo_path, relative_file_path):
    full_file_path = os.path.join(repo_path, relative_file_path)
    if os.path.isfile(full_file_path):
        try:
            with open(full_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {full_file_path}: {e}")
            return None
    return None

def handle_remove_readonly(func, path, exc_info):
    exc_type, exc_value, exc_tb = exc_info
    if isinstance(exc_value, PermissionError):
        try:
            os.chmod(path, stat.S_IWUSR)
            func(path)
        except Exception as e:
            # print(f"Still failed to remove {path} after chmod and retry: {e}")
            raise
    else:
        raise exc_value

def cleanup_repository(repo_path, repo_object=None):
    if repo_object:
        try:
            repo_object.close()
        except Exception: # nosec
            pass # Ignore errors during close, primary goal is deletion
    if repo_path and os.path.exists(repo_path):
        # print(f"\nCleaning up temporary directory: {repo_path}")
        try:
            shutil.rmtree(repo_path, onerror=handle_remove_readonly)
            # print("Cleanup successful.")
        except Exception as e:
            print(f"Error during shutil.rmtree of '{repo_path}': {e}")

if __name__ == '__main__':
    # Simple test
    test_url = "https://github.com/pallets/flask.git"
    print(f"Testing repo_fetcher with {test_url}")
    path, repo_obj = clone_repository(test_url)
    if path:
        print(f"Cloned to: {path}")
        readme = get_file_content(path, "README.rst") # Flask uses .rst
        if readme:
            print(f"README content (first 100 chars): {readme[:100].strip()}...")
        cleanup_repository(path, repo_obj)
        print("Test cleanup complete.")
    else:
        print("Test clone failed.")