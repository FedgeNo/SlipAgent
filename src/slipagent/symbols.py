"""Offline syntax-tree outlines and references for repository orientation."""

from __future__ import annotations

import re

from tree_sitter import Node, Parser
from tree_sitter_language_pack import SupportedLanguage, get_language

LANGUAGES: dict[str, SupportedLanguage] = {
    ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".hpp": "cpp", ".java": "java", ".php": "php", ".rb": "ruby",
    ".cs": "csharp", ".swift": "swift", ".kt": "kotlin", ".sh": "bash",
}
DEFINITIONS = frozenset({
    "function_declaration", "function_definition", "function_item", "method_definition",
    "method_declaration", "method_signature", "method", "singleton_method",
    "class_declaration", "class_definition", "class", "interface_declaration",
    "interface_definition", "struct_item", "enum_item", "trait_item", "type_alias_declaration",
    "type_spec", "struct_specifier", "enum_specifier", "namespace_definition",
    "object_declaration", "protocol_declaration", "typealias_declaration",
})


def _name(node: Node) -> Node | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return name
    declarator = node.child_by_field_name("declarator")
    while declarator is not None:
        if declarator.type in {"identifier", "field_identifier", "qualified_identifier"}:
            return declarator
        declarator = declarator.child_by_field_name("declarator")
    return None


def outline(raw: bytes, suffix: str) -> tuple[str, set[str]]:
    """Return bounded declaration headers and graph tags without executing code."""
    language = LANGUAGES.get(suffix)
    if language is None:
        return "", set()
    parser = Parser(get_language(language))
    tree = parser.parse(raw)
    declarations: list[str] = []
    references: set[str] = set()
    definitions: set[str] = set()
    stack = [tree.root_node]
    visited = 0
    while stack and visited < 1_000_000:
        node = stack.pop()
        visited += 1
        named = _name(node)
        value = node.child_by_field_name("value")
        callable_variable = (node.type == "variable_declarator" and value is not None
                             and value.type in {"arrow_function", "function_expression"})
        if (node.type in DEFINITIONS or callable_variable) and named is not None:
            name = raw[named.start_byte:named.end_byte].decode("utf-8", errors="replace")
            definitions.add(name)
            if len(declarations) < 1000:
                signature_node = value if callable_variable and value is not None else node
                body = signature_node.child_by_field_name("body")
                if body is None:
                    body = next((child for child in signature_node.named_children
                                 if child.type.endswith("_body") or child.type in {
                                     "body_statement", "declaration_list", "field_declaration_list",
                                     "block", "statement_block", "compound_statement",
                                 }), None)
                end = body.start_byte if body is not None else signature_node.end_byte
                header = raw[node.start_byte:min(end, node.start_byte + 16000)].decode("utf-8", errors="replace")
                header = re.sub(r"\s+", " ", header).strip().rstrip(";{ ")[:4000]
                # Tuple indexing avoids Tree-sitter 0.26.0's borrowed-reference Point.row getter.
                declarations.append(f"{node.start_point[0] + 1}: {header}")
        if (node.type in {"identifier", "type_identifier", "field_identifier", "property_identifier", "name", "constant"}
                and len(references) < 100_000):
            # Exclude declaration names, but retain their separate usages.
            parent = node.parent
            declaration_name = _name(parent) if parent is not None and parent.type in DEFINITIONS else None
            if declaration_name != node:
                references.add(raw[node.start_byte:node.end_byte].decode("utf-8", errors="replace"))
        stack.extend(reversed(node.named_children))
    tags = {"def:" + name for name in definitions} | {"ref:" + name for name in references}
    if tree.root_node.has_error:
        declarations.append("[partial outline: syntax errors; read source]")
    if stack:
        declarations.append("[partial outline: node limit]")
    return "\n".join(declarations), tags
