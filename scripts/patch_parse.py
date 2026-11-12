import re

path = r"D:\k8s-rag\ingestion\parse.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Find the should_include function and replace it
old_func = '''def should_include(file_path: Path, repo_root: Path) -> bool:
    try:
        relative = file_path.relative_to(repo_root)
    except ValueError:
        return False

    parts = relative.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False

    if file_path.name in EXCLUDED_FILES:
        return False

    return True'''

new_func = '''def should_include(file_path: Path, repo_root: Path) -> bool:
    # Use string-based comparison to avoid Windows Path.relative_to quirks
    file_str = str(file_path).replace("\\\\", "/")
    root_str = str(repo_root).replace("\\\\", "/")

    if not file_str.startswith(root_str):
        return False

    relative = file_str[len(root_str):].lstrip("/")

    # Must be under content/en/docs/
    prefix = "content/en/docs/"
    if not relative.startswith(prefix):
        return False

    rest = relative[len(prefix):]
    subdir = rest.split("/")[0] if "/" in rest else ""

    if subdir in EXCLUDED_DIRS:
        return False

    if file_path.name in EXCLUDED_FILES:
        return False

    return True'''

if old_func not in content:
    print("ERROR: old function not found")
    exit(1)

content = content.replace(old_func, new_func)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("patched OK")