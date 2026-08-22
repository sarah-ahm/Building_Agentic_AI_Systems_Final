"""Symbol extraction with tree-sitter (RepoScribe's ground-truth oracle).

We parse source *text* (no compiler, no type-checker) into a ``SymbolTable`` of every
declaration and its ``file:line`` location. This table is what the output guardrail
checks generated docs against, which is what makes groundedness measurable.

Supported grammars: TypeScript / TSX / JavaScript / Python. Unknown node types are
skipped rather than raising, so parsing degrades gracefully.
"""

from __future__ import annotations

from functools import lru_cache

from .models import Language, SourceFile, Symbol, SymbolTable

# extension -> (our Language label, tree-sitter grammar name)
_EXT_GRAMMAR = {
    ".ts": ("typescript", "typescript"),
    ".tsx": ("typescript", "tsx"),
    ".js": ("javascript", "javascript"),
    ".jsx": ("javascript", "javascript"),
    ".mjs": ("javascript", "javascript"),
    ".py": ("python", "python"),
}

# Declaration node types we treat as documentable symbols (TS/JS).
_TS_KINDS = {
    "function_declaration": "function",
    "class_declaration": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type_alias",
    "enum_declaration": "enum",
}


def detect_language(path: str) -> Language:
    for ext, (lang, _) in _EXT_GRAMMAR.items():
        if path.endswith(ext):
            return lang  # type: ignore[return-value]
    return "other"


def _grammar_for(path: str) -> str | None:
    for ext, (_, grammar) in _EXT_GRAMMAR.items():
        if path.endswith(ext):
            return grammar
    return None


@lru_cache(maxsize=8)
def _parser(grammar: str):
    from tree_sitter_language_pack import get_parser

    return get_parser(grammar)


def extract_symbols(files: list[SourceFile]) -> SymbolTable:
    """Parse every source file into a combined SymbolTable."""
    table = SymbolTable()
    for f in files:
        grammar = _grammar_for(f.path)
        if grammar is None:
            continue
        src = f.text.encode("utf-8", errors="replace")
        tree = _parser(grammar).parse(src)
        if f.language == "python":
            _extract_python(tree.root_node, src, f, table)
        else:
            _extract_ts(tree.root_node, src, f, table)
    return table


# --- helpers -------------------------------------------------------------------

def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _lines(node) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _signature(node, src: bytes) -> str | None:
    """Header text of a declaration: up to its body/value, whitespace-collapsed."""
    body = node.child_by_field_name("body") or node.child_by_field_name("value")
    raw = src[node.start_byte : body.start_byte] if body else src[node.start_byte : node.end_byte]
    sig = " ".join(raw.decode("utf-8", errors="replace").split())
    sig = sig.rstrip(" {=").strip()
    return sig[:200] or None


def _leading_jsdoc(node, src: bytes) -> str | None:
    """A ``/** ... */`` block comment immediately preceding this declaration."""
    prev = node.prev_sibling
    if prev is not None and prev.type == "comment":
        txt = _text(prev, src).strip()
        if txt.startswith("/**"):
            return txt
    return None


# --- TypeScript / JavaScript ---------------------------------------------------

def _extract_ts(root, src: bytes, f: SourceFile, table: SymbolTable) -> None:
    for child in root.children:
        if child.type == "export_statement":
            _handle_ts_export(child, src, f, table)
        elif child.type in _TS_KINDS or child.type == "lexical_declaration":
            _add_ts_decl(child, src, f, table, exported=False, doc_node=child)


def _handle_ts_export(node, src: bytes, f: SourceFile, table: SymbolTable) -> None:
    """An ``export ...`` statement: either a re-export barrel entry or a declaration."""
    clause = next((c for c in node.children if c.type == "export_clause"), None)
    if clause is not None:
        # `export { a, b } from '...'` / `export type { A } from '...'` -> public surface.
        for spec in clause.children:
            if spec.type == "export_specifier":
                alias = spec.child_by_field_name("alias") or spec.child_by_field_name("name")
                if alias is not None:
                    name = _text(alias, src)
                    if name not in table.reexports:
                        table.reexports.append(name)
        return

    for c in node.children:
        if c.type in _TS_KINDS or c.type == "lexical_declaration":
            _add_ts_decl(c, src, f, table, exported=True, doc_node=node)


def _add_ts_decl(node, src: bytes, f: SourceFile, table: SymbolTable, *, exported: bool, doc_node) -> None:
    docstring = _leading_jsdoc(doc_node, src)

    if node.type == "lexical_declaration":
        decl = next((c for c in node.children if c.type == "variable_declarator"), None)
        if decl is None:
            return
        name_node = decl.child_by_field_name("name")
        value = decl.child_by_field_name("value")
        kind = "function" if value is not None and value.type in ("arrow_function", "function", "function_expression") else "constant"
        if name_node is None:
            return
        table.symbols.append(
            Symbol(
                name=_text(name_node, src),
                kind=kind,
                language=f.language,
                file=f.path,
                start_line=_lines(node)[0],
                end_line=_lines(node)[1],
                signature=_signature(decl, src),
                docstring=docstring,
                exported=exported,
            )
        )
        return

    kind = _TS_KINDS[node.type]
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _text(name_node, src)
    start, end = _lines(node)
    table.symbols.append(
        Symbol(
            name=name,
            kind=kind,
            language=f.language,
            file=f.path,
            start_line=start,
            end_line=end,
            signature=_signature(node, src),
            docstring=docstring,
            exported=exported,
        )
    )

    # For classes, also record public methods (kind="method").
    if kind == "class":
        body = node.child_by_field_name("body")
        if body is not None:
            for m in body.children:
                if m.type == "method_definition":
                    mn = m.child_by_field_name("name")
                    if mn is None:
                        continue
                    ms, me = _lines(m)
                    table.symbols.append(
                        Symbol(
                            name=_text(mn, src),
                            kind="method",
                            language=f.language,
                            file=f.path,
                            start_line=ms,
                            end_line=me,
                            signature=_signature(m, src),
                            docstring=_leading_jsdoc(m, src),
                            exported=exported,
                        )
                    )


# --- Python --------------------------------------------------------------------

def _extract_python(root, src: bytes, f: SourceFile, table: SymbolTable) -> None:
    for child in root.children:
        node = child
        if node.type == "decorated_definition":
            node = node.child_by_field_name("definition") or node
        if node.type in ("function_definition", "class_definition"):
            _add_python_decl(node, src, f, table)


def _add_python_decl(node, src: bytes, f: SourceFile, table: SymbolTable) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _text(name_node, src)
    kind = "function" if node.type == "function_definition" else "class"
    start, end = _lines(node)
    table.symbols.append(
        Symbol(
            name=name,
            kind=kind,
            language="python",
            file=f.path,
            start_line=start,
            end_line=end,
            signature=_signature(node, src),
            docstring=_python_docstring(node, src),
            exported=not name.startswith("_"),  # convention: leading underscore == internal
        )
    )


def _python_docstring(node, src: bytes) -> str | None:
    body = node.child_by_field_name("body")
    if body is None or not body.children:
        return None
    first = body.children[0]
    # Some grammar versions wrap it in expression_statement, others expose the string directly.
    if first.type == "expression_statement" and first.children:
        first = first.children[0]
    if first.type == "string":
        return _text(first, src).strip()
    return None
