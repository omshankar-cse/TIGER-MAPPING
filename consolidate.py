import os
from pathlib import Path

# Directories to ignore by their folder name (ignores anywhere in the project)
IGNORED_DIRS = {
    '.git', '.github', 'vebnv', 'env', '.venv', '.env', '__pycache__',
    'node_modules', '.idea', '.vscode', 'dist', 'build', '.pytest_cache',
    '.mypy_cache', 'target', 'bin', 'obj','venv'
}

# Specific absolute or relative paths to ignore
IGNORED_PATHS = [
    r"D:\SIS INTERNSHIP\TIGER PROJECT\TIGER TO MOVE\outputs\predictions"
]

# Supported source/text extensions
ALLOWED_EXTENSIONS = {
    # Python
    '.py', '.pyw', '.ipynb',
    # Web / Frontend
    '.js', '.jsx', '.ts', '.tsx', '.html', '.htm', '.css', '.scss', '.sass', '.vue', '.svelte',
    # Data & Config formats
    '.json', '.yaml', '.yml', '.toml', '.xml', '.ini', '.cfg', '.conf', '.env.example',
    # Documentation & Text
    '.md', '.txt', '.rst', '.csv',  
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
OUTPUT_FILE = 'project_code_dump_v3.txt'
SCRIPT_FILE = 'consolidate.py'


def is_text_file(filepath: Path) -> bool:
    """Check if a file should be included based on extension."""
    ext = filepath.suffix.lower()
    
    if ext in EXCLUDED_EXTENSIONS:
        return False
    
    if ext in ALLOWED_EXTENSIONS:
        return True
    
    special_names = {'dockerfile', 'makefile', 'license', 'readme', '.gitignore', '.dockerignore'}
    if filepath.name.lower() in special_names:
        return True
        
    return False


def consolidate_code(root_dir: str = '.'):
    root_path = Path(root_dir).resolve()
    output_path = root_path / OUTPUT_FILE
    
    # Resolve all specific ignore paths to their absolute forms for accurate comparison
    resolved_ignored_paths = [Path(p).resolve() for p in IGNORED_PATHS]
    
    # Delete the old file to guarantee a fresh start
    if output_path.exists():
        try:
            output_path.unlink()
            print(f"Deleted old dump file: {OUTPUT_FILE}")
        except Exception as e:
            print(f"[WARNING] Could not delete old file. It might be open in another program: {e}")
            
    print(f"Traversing directory: {root_path}")
    print(f"Output will be written to: {output_path}")

    files_processed = 0
    files_skipped = 0

    with open(output_path, 'w', encoding='utf-8', errors='replace') as outfile:
        outfile.write("=" * 80 + "\n")
        outfile.write(f"PROJECT CODE CONSOLIDATION DUMP\n")
        outfile.write(f"Root: {root_path.name}\n")
        outfile.write("=" * 80 + "\n\n")

        for dirpath, dirnames, filenames in os.walk(root_path):
            current_dir = Path(dirpath).resolve()
            
            # Prune directories we want to ignore so os.walk doesn't even search them
            valid_dirs = []
            for d in dirnames:
                # 1. Ignore by folder name
                if d in IGNORED_DIRS or d.startswith('.git'):
                    continue
                
                # 2. Ignore by specific exact path
                dir_full_path = (current_dir / d).resolve()
                skip_this_dir = False
                for ig_path in resolved_ignored_paths:
                    # If this directory is the ignored path (or inside it), skip it
                    if str(dir_full_path).startswith(str(ig_path)):
                        skip_this_dir = True
                        break
                
                if not skip_this_dir:
                    valid_dirs.append(d)

            # Update dirnames in-place to control os.walk behavior
            dirnames[:] = valid_dirs

            for filename in filenames:
                if filename in (OUTPUT_FILE, SCRIPT_FILE):
                    continue

                file_path = Path(dirpath) / filename
                
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