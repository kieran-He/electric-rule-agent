from app.services.ingest.path_resolver import build_missing_path_hint, resolve_docs_path


def test_resolve_province_path_with_default_root():
    path = resolve_docs_path(
        kb_scope="province",
        province_code="SN",
        docs_path=None,
        docs_root=None,
        default_docs_root="./data/docs",
    )
    assert path.replace("\\", "/").endswith("data/docs/SN")


def test_resolve_global_path_with_default_root():
    path = resolve_docs_path(
        kb_scope="global",
        province_code=None,
        docs_path=None,
        docs_root=None,
        default_docs_root="./data/docs",
    )
    assert path.replace("\\", "/").endswith("data/docs/global")


def test_explicit_docs_path_overrides_defaults():
    path = resolve_docs_path(
        kb_scope="province",
        province_code="SN",
        docs_path="E:/custom/docs",
        docs_root="./data/docs",
        default_docs_root="./data/docs",
    )
    assert path == "E:/custom/docs"


def test_missing_path_hint_contains_expected_location():
    hint = build_missing_path_hint("./data/docs/SN", "province", "SN")
    assert "data/docs/SN" in hint
    assert "explicit docs_path" in hint

