from app.core.repository import DeterministicEmbedder


def test_deterministic_embedder_shape():
    embedder = DeterministicEmbedder(dimension=16)
    vectors = embedder.encode(["abc", "def"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 16


def test_deterministic_embedder_is_stable():
    embedder = DeterministicEmbedder(dimension=32)
    a = embedder.encode(["陕西电力交易规则"])[0]
    b = embedder.encode(["陕西电力交易规则"])[0]
    assert a == b
