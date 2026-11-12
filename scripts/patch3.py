path = r"D:\k8s-rag\ingestion\parse.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add a regex for Hugo heading anchors after the HTML_COMMENT_PATTERN definition
old_regex_block = '''HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)'''

new_regex_block = '''HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_ANCHOR_PATTERN = re.compile(r"\\s*\\{#[^}]+\\}\\s*$")
EMPHASIS_MARKER_PATTERN = re.compile(r"\\*\\*|\\*|__|_")'''

if old_regex_block not in content:
    print("ERROR: regex block not found")
    exit(1)
content = content.replace(old_regex_block, new_regex_block)

# 2. Add a helper function to clean heading text
old_walk_start = '''def _walk_ast(tokens, section_path: list[str]) -> list[Token]:'''

new_walk_start = '''def _clean_heading_text(text: str) -> str:
    """Strip Hugo heading anchors like {#anchor-id}."""
    return HEADING_ANCHOR_PATTERN.sub("", text).strip()


def _walk_ast(tokens, section_path: list[str]) -> list[Token]:'''

if old_walk_start not in content:
    print("ERROR: _walk_ast definition not found")
    exit(1)
content = content.replace(old_walk_start, new_walk_start)

# 3. Apply _clean_heading_text to heading extraction
old_heading = '''        if token_type == "heading":
            level = token.get("attrs", {}).get("level", 2)
            text = _extract_text(token.get("children", []))
            if text.strip():
                current_path = current_path[: level - 1] + [text.strip()]
                result.append(
                    Token(
                        type="heading",
                        text=text.strip(),
                        level=level,
                        section_path=list(current_path),
                    )
                )'''

new_heading = '''        if token_type == "heading":
            level = token.get("attrs", {}).get("level", 2)
            text = _clean_heading_text(_extract_text(token.get("children", [])))
            if text:
                current_path = current_path[: level - 1] + [text]
                result.append(
                    Token(
                        type="heading",
                        text=text,
                        level=level,
                        section_path=list(current_path),
                    )
                )'''

if old_heading not in content:
    print("ERROR: heading block not found")
    exit(1)
content = content.replace(old_heading, new_heading)

# 4. Add emphasis stripping to paragraph text extraction
old_para = '''        elif token_type == "paragraph":
            text = _extract_text(token.get("children", []))
            if text.strip():
                result.append(
                    Token(
                        type="paragraph",
                        text=text.strip(),
                        section_path=list(current_path),
                    )
                )'''

new_para = '''        elif token_type == "paragraph":
            raw_text = _extract_text(token.get("children", []))
            text = EMPHASIS_MARKER_PATTERN.sub("", raw_text).strip()
            if text:
                result.append(
                    Token(
                        type="paragraph",
                        text=text,
                        section_path=list(current_path),
                    )
                )'''

if old_para not in content:
    print("ERROR: paragraph block not found")
    exit(1)
content = content.replace(old_para, new_para)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("patched OK")