path = r"D:\k8s-rag\ingestion\parse.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = """    md = mistune.create_markdown(renderer=None, plugins=["table"])
    ast_tokens = md.parse(body)

    tokens = _walk_ast(ast_tokens, [])"""

new = """    md = mistune.create_markdown(renderer=None, plugins=["table"])
    ast_result = md.parse(body)

    # mistune 3.x returns a tuple of (tokens, block_state).
    # We only want the tokens.
    if isinstance(ast_result, tuple):
        ast_tokens = ast_result[0]
    else:
        ast_tokens = ast_result

    tokens = _walk_ast(ast_tokens, [])"""

if old not in content:
    print("ERROR: target block not found")
    exit(1)

content = content.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("patched OK")