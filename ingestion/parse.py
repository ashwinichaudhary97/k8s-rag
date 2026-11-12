"""
Markdown parser for Kubernetes documentation.

Parses a single Markdown file into a structured document with typed tokens:
headings, paragraphs, code blocks, and tables. Handles front matter, Hugo
shortcodes, and HTML comments.

The output is a flat token stream that the chunker consumes. The parser does
not chunk; it only parses.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import mistune
import yaml


EXCLUDED_DIRS = {
    "contribute",
    "doc-contributor-tools",
    "home",
    "images",
}

EXCLUDED_FILES = {
    "test.md",
}


STRIP_ENTIRELY_SHORTCODES = {
    "feature-state",
    "codenew",
    "glossary_definition",
    "tabs",
    "tab",
    "capture",
}

KEEP_BODY_SHORTCODES = {
    "note",
    "caution",
    "warning",
    "important",
    "tip",
}

SHORTCODE_BODY_PATTERN = re.compile(
    r"\{\{[<%]\s*(note|caution|warning|important|tip)\s*[>%]\}\}"
    r"(.*?)"
    r"\{\{[<%]\s*/\s*\1\s*[>%]\}\}",
    re.DOTALL,
)

GLOSSARY_TOOLTIP_PATTERN = re.compile(
    r"\{\{<\s*glossary_tooltip\s+text=\"([^\"]*)\"[^>]*>\}\}"
)

GENERIC_SHORTCODE_PATTERN = re.compile(
    r"\{\{[<%]\s*/?\s*([a-zA-Z0-9_-]+)[^>%]*[>%]\}\}"
)

HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_ANCHOR_PATTERN = re.compile(r"\s*\{#[^}]+\}\s*$")
EMPHASIS_MARKER_PATTERN = re.compile(r"\*\*|\*|__|_")


@dataclass
class Token:
    type: Literal["heading", "paragraph", "code", "table"]
    text: str
    level: int = 0
    language: str = ""
    section_path: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    file_path: str
    title: str
    doc_type: str
    tokens: list[Token]


def _extract_front_matter(content: str) -> tuple[dict, str]:
    if not content.startswith("---"):
        return {}, content

    end = content.find("\n---", 3)
    if end == -1:
        return {}, content

    front_matter_text = content[3:end].strip()
    remaining = content[end + 4:].lstrip("\n")

    try:
        front_matter = yaml.safe_load(front_matter_text) or {}
    except yaml.YAMLError:
        front_matter = {}

    return front_matter, remaining


def _handle_hugo_shortcodes(content: str) -> str:
    content = GLOSSARY_TOOLTIP_PATTERN.sub(r"\1", content)
    content = SHORTCODE_BODY_PATTERN.sub(r"\2", content)
    content = GENERIC_SHORTCODE_PATTERN.sub("", content)
    return content


def _clean_content(content: str) -> str:
    content = HTML_COMMENT_PATTERN.sub("", content)
    content = _handle_hugo_shortcodes(content)
    return content


def _clean_heading_text(text: str) -> str:
    """Strip Hugo heading anchors like {#anchor-id}."""
    return HEADING_ANCHOR_PATTERN.sub("", text).strip()


def _walk_ast(tokens, section_path: list[str]) -> list[Token]:
    """Walk mistune AST tokens, producing a flat list of typed tokens.

    mistune sometimes nests lists of tokens inside the token list (e.g., from
    list items or block quotes). We handle both dicts and lists.
    """
    result = []
    current_path = list(section_path)

    if not isinstance(tokens, list):
        return result

    for token in tokens:
        # Handle nested lists
        if isinstance(token, list):
            result.extend(_walk_ast(token, current_path))
            continue

        if not isinstance(token, dict):
            continue

        token_type = token.get("type", "")

        if token_type == "heading":
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
                )

        elif token_type == "block_code":
            text = token.get("raw", "")
            info = token.get("attrs", {}).get("info", "") or ""
            language = info.strip().split()[0] if info.strip() else ""
            result.append(
                Token(
                    type="code",
                    text=text.rstrip(),
                    language=language,
                    section_path=list(current_path),
                )
            )

        elif token_type == "table":
            text = _extract_table_text(token)
            result.append(
                Token(
                    type="table",
                    text=text,
                    section_path=list(current_path),
                )
            )

        elif token_type == "paragraph":
            raw_text = _extract_text(token.get("children", []))
            text = EMPHASIS_MARKER_PATTERN.sub("", raw_text).strip()
            if text:
                result.append(
                    Token(
                        type="paragraph",
                        text=text,
                        section_path=list(current_path),
                    )
                )

        elif token_type in ("list", "block_quote", "list_item"):
            result.extend(_walk_ast(token.get("children", []), current_path))

        elif token_type in ("block_html", "thematic_break", "blank_line"):
            continue

        else:
            if "children" in token:
                result.extend(_walk_ast(token.get("children", []), current_path))

    return result


def _extract_text(children) -> str:
    if not isinstance(children, list):
        if isinstance(children, str):
            return children
        return ""

    parts = []
    for child in children:
        if isinstance(child, str):
            parts.append(child)
            continue

        if isinstance(child, list):
            parts.append(_extract_text(child))
            continue

        if not isinstance(child, dict):
            continue

        child_type = child.get("type", "")

        if child_type == "text":
            parts.append(child.get("raw", ""))
        elif child_type in ("codespan", "code_inline"):
            parts.append("`" + child.get("raw", "") + "`")
        elif child_type == "strong":
            parts.append("**" + _extract_text(child.get("children", [])) + "**")
        elif child_type == "emphasis":
            parts.append("*" + _extract_text(child.get("children", [])) + "*")
        elif child_type == "link":
            parts.append(_extract_text(child.get("children", [])))
        elif child_type == "image":
            alt = _extract_text(child.get("children", []))
            parts.append(f"[image: {alt}]" if alt else "")
        elif child_type == "linebreak":
            parts.append(" ")
        else:
            if "children" in child:
                parts.append(_extract_text(child.get("children", [])))
            elif "raw" in child:
                parts.append(child.get("raw", ""))

    return "".join(parts)


def _extract_table_text(token: dict) -> str:
    raw = token.get("raw", "")
    if raw:
        return raw.rstrip()

    lines = []
    for child in token.get("children", []):
        if not isinstance(child, dict):
            continue
        if child.get("type") == "table_head":
            cells = _extract_row_cells(child)
            lines.append("| " + " | ".join(cells) + " |")
        elif child.get("type") == "table_body":
            for row in child.get("children", []):
                if not isinstance(row, dict):
                    continue
                cells = _extract_row_cells(row)
                lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _extract_row_cells(row_token: dict) -> list[str]:
    cells = []
    for cell in row_token.get("children", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("type") == "table_cell":
            cells.append(_extract_text(cell.get("children", [])).strip())
    return cells


def should_include(file_path: Path, repo_root: Path) -> bool:
    # Use string-based comparison to avoid Windows Path.relative_to quirks
    file_str = str(file_path).replace("\\", "/")
    root_str = str(repo_root).replace("\\", "/")

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

    return True


def parse_file(file_path: Path, repo_root: Path) -> ParsedDocument | None:
    if not should_include(file_path, repo_root):
        return None

    content = file_path.read_text(encoding="utf-8")
    front_matter, body = _extract_front_matter(content)
    body = _clean_content(body)

    md = mistune.create_markdown(renderer=None, plugins=["table"])
    ast_result = md.parse(body)

    # mistune 3.x returns a tuple of (tokens, block_state).
    # We only want the tokens.
    if isinstance(ast_result, tuple):
        ast_tokens = ast_result[0]
    else:
        ast_tokens = ast_result

    tokens = _walk_ast(ast_tokens, [])

    if not tokens:
        return None

    relative = file_path.relative_to(repo_root)
    title = front_matter.get("title", "") or file_path.stem
    doc_type = front_matter.get("content_type", "") or "unknown"

    return ParsedDocument(
        file_path=str(relative).replace("\\", "/"),
        title=title,
        doc_type=doc_type,
        tokens=tokens,
    )


def iter_files(repo_root: Path):
    docs_root = repo_root / "content" / "en" / "docs"
    for md_file in docs_root.rglob("*.md"):
        yield md_file