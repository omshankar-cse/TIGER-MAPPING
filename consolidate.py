import os
from pathlib import Path

# Directories to ignore
IGNORED_DIRS = {
    '.git',
    '.github',
    'venv',
    'env',
    '.venv',
    '.env',
    '__pycache__',
    'node_modules',
    '.idea',
    '.vscode',
    'dist',
    'build',
    '.pytest_cache',
    '.mypy_cache',
    'target',
    'bin',
    'obj',
}

# Supported source/text extensions
ALLOWED_EXTENSIONS = {
    # Python
    '.py', '.pyw', '.ipynb',
    # Web / Frontend
    '.js', '.jsx', '.ts', '.tsx', '.html', '.htm', '.css', '.scss', '.sass', '.vue', '.svelte',
    # Data & Config formats
    '.json', '.yaml', '.yml', '.toml', '.xml', '.ini', '.cfg', '.conf', '.env.example',
    # Documentation & Text
    '.md', '.txt', '.rst', '.csv',  # Note: text files
    # Backend / System languages
    '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.php', '.rb', '.sh', '.bat', '.ps1', '.sql',
}

# Specifically excluded extensions (binary, cache, images, large datasets)
EXCLUDED_EXTENSIONS = {
    '.pyc', '.pyo', '.pyd',
    '.sqlite', '.sqlite3', '.db',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
    '.zip', '.tar', '.gz', '.7z', '.rar',
    '.exe', '.dll', '.so', '.dylib', '.bin',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.mp4', '.mp3', '.wav', '.mov', '.avi',
    '.pkl', '.pickle', '.parquet', '.h5', '.hdf5', '.pt', '.pth', '.onnx',
}

# Output file name and script name to exclude from the dump
OUTPUT_FILE = 'project_code_dump.txt'
SCRIPT_FILE = 'consolidate.py'


def is_text_file(filepath: Path) -> bool:
    """Check if a file should be included based on extension."""
    ext = filepath.suffix.lower()
    
    if ext in EXCLUDED_EXTENSIONS:
        return False
    
    # If it's explicitly in allowed extensions, include it
    if ext in ALLOWED_EXTENSIONS:
        return True
    
    # Check known dotfiles without suffix (e.g., .gitignore, Dockerfile, Makefile)
    special_names = {'dockerfile', 'makefile', 'license', 'readme', '.gitignore', '.dockerignore'}
    if filepath.name.lower() in special_names:
        return True
        
    return False


def consolidate_code(root_dir: str = '.'):
    root_path = Path(root_dir).resolve()
    output_path = root_path / OUTPUT_FILE
    
    print(f"Traversing directory: {root_path}")
    print(f"Output will be written to: {output_path}")

    files_processed = 0
    files_skipped = 0

    with open(output_path, 'w', encoding='utf-8', errors='replace') as outfile:
        # Write summary header
        outfile.write("=" * 80 + "\n")
        outfile.write(f"PROJECT CODE CONSOLIDATION DUMP\n")
        outfile.write(f"Root: {root_path.name}\n")
        outfile.write("=" * 80 + "\n\n")

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune ignored directories in-place
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith('.git')]

            for filename in filenames:
                # Exclude output file and this script itself
                if filename in (OUTPUT_FILE, SCRIPT_FILE):
                    continue

                file_path = Path(dirpath) / filename
                
                # Check if file matches our code criteria
                if not is_text_file(file_path):
                    files_skipped += 1
                    continue

                try:
                    rel_path = file_path.relative_to(root_path).as_posix()
                except ValueError:
                    rel_path = str(file_path)

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                        content = infile.read()

                    # Write file header and content
                    separator = f"\n{'=' * 80}\n--- File: {rel_path} ---\n{'=' * 80}\n"
                    outfile.write(separator)
                    outfile.write(content)
                    outfile.write("\n")

                    files_processed += 1
                    print(f"[INCLUDED] {rel_path}")

                except Exception as e:
                    print(f"[ERROR] Could not read file {rel_path}: {e}")
                    files_skipped += 1

    print("\n" + "=" * 40)
    print(f"Consolidation Complete!")
    print(f"Files Processed: {files_processed}")
    print(f"Files Skipped:   {files_skipped}")
    print(f"Saved To:        {output_path}")
    print("=" * 40)


if __name__ == '__main__':
    consolidate_code('.')
