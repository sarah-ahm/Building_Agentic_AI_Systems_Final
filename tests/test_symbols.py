"""Tree-sitter symbol extraction: the ground-truth oracle."""

from __future__ import annotations

from reposcribe.models import SourceFile
from reposcribe.symbols import extract_symbols

PUBLIC_API = {
    "resolveModule",
    "RegistryType",
    "RegistryEntry",
    "ModuleRecord",
    "AliasModuleRecord",
    "DirModuleRecord",
    "NpmModuleRecord",
    "ModuleResolverConfig",
}


def test_public_api_matches_barrel_reexports(table):
    assert table.public_names() == PUBLIC_API


def test_resolve_module_has_signature_and_docstring(table):
    sym = table.get("resolveModule")
    assert sym is not None
    assert sym.kind == "function"
    assert sym.exported is True
    assert sym.file.endswith("resolve-module.ts")
    assert sym.signature and "importee" in sym.signature
    assert sym.docstring and "Resolves LWC modules" in sym.docstring


def test_registry_type_is_const_and_type_alias(table):
    kinds = {s.kind for s in table.symbols if s.name == "RegistryType"}
    assert kinds == {"constant", "type_alias"}


def test_internal_helpers_are_not_public(table):
    # Exported within their file, but NOT re-exported by the barrel -> not public API.
    assert table.get("getModuleEntry").exported is True
    assert "getModuleEntry" not in table.public_names()
    # A truly private helper is not even exported.
    assert table.get("readJson").exported is False


def test_typescript_inline_kinds():
    src = """
export interface Foo { a: number; }
export enum Color { red, green }
export type Bar = Foo | Color;
function helper(x: number): number { return x; }
"""
    t = extract_symbols([SourceFile(path="x.ts", language="typescript", size_bytes=len(src), text=src)])
    by_name = {s.name: s for s in t.symbols}
    assert by_name["Foo"].kind == "interface" and by_name["Foo"].exported
    assert by_name["Color"].kind == "enum"
    assert by_name["Bar"].kind == "type_alias"
    assert by_name["helper"].exported is False  # no export keyword


def test_python_extraction():
    src = '''
def public_fn(x):
    """Docstring here."""
    return x

def _private_fn():
    return 1

class Thing:
    pass
'''
    t = extract_symbols([SourceFile(path="m.py", language="python", size_bytes=len(src), text=src)])
    by_name = {s.name: s for s in t.symbols}
    assert by_name["public_fn"].kind == "function" and by_name["public_fn"].exported
    assert by_name["public_fn"].docstring and "Docstring" in by_name["public_fn"].docstring
    assert by_name["_private_fn"].exported is False  # leading underscore == internal
    assert by_name["Thing"].kind == "class"
