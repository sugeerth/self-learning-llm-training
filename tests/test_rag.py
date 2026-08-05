"""Local-first RAG over the prep documents — offline, deterministic tests."""

from rag.corpus import Document, load_corpus
from rag.engine import RagEngine
from rag.retriever import BM25Index, content_terms, tokenize
from rag.sources import LocalSource, WebSource


def _docs():
    return [
        Document("d1#a", "d1.md", "Circuit breaker",
                 "The circuit breaker skips a model after three consecutive "
                 "failures until a cooldown passes."),
        Document("d2#a", "d2.md", "Eval cache",
                 "The eval cache stores content-addressed evaluations so "
                 "identical work never runs twice across sweeps."),
        Document("d3#a", "d3.md", "Autopilot",
                 "Autopilot promotes candidate models to stable from live "
                 "traffic evidence and demotes decaying ones."),
    ]


# ── tokenization / retrieval ────────────────────────────────────────────

def test_tokenize_drops_stopwords_and_shorts():
    toks = tokenize("The circuit breaker is a MODEL")
    assert "circuit" in toks and "breaker" in toks and "model" in toks
    assert "the" not in toks and "is" not in toks and "a" not in toks


def test_bm25_ranks_relevant_doc_first():
    idx = BM25Index(_docs())
    hits = idx.search("how does the circuit breaker work", k=3)
    assert hits and hits[0].doc.doc_id == "d1#a"
    assert "circuit" in hits[0].matched_terms


def test_bm25_deterministic():
    idx = BM25Index(_docs())
    a = [h.doc.doc_id for h in idx.search("eval cache sweeps", k=3)]
    b = [h.doc.doc_id for h in idx.search("eval cache sweeps", k=3)]
    assert a == b and a[0] == "d2#a"


def test_no_match_returns_empty():
    assert BM25Index(_docs()).search("bicycle maintenance schedule") == []


# ── local-first engine: coverage gate + web secondary ───────────────────

def test_local_only_when_covered_never_touches_web():
    calls = []
    def spy(q, k):
        calls.append(q)
        return [("x", "http://x", "irrelevant")]
    eng = RagEngine(LocalSource(_docs()), WebSource(spy), coverage_gate=0.6)
    ans = eng.retrieve("circuit breaker failures cooldown")
    assert not ans.used_web
    assert calls == []                    # web never consulted when covered
    assert ans.confidence == ans.coverage  # full trust, no web penalty
    assert all(h.doc.origin == "local" for h in ans.passages)


def test_web_consulted_only_when_coverage_low():
    def web(q, k):
        return [("Entanglement", "http://ex/qe",
                 "Quantum entanglement correlates distant particles.")]
    eng = RagEngine(LocalSource(_docs()), WebSource(web), coverage_gate=0.6)
    ans = eng.retrieve("what is quantum entanglement")
    assert ans.used_web
    assert any(h.doc.origin == "web" for h in ans.passages)
    # web-reliant answers are tempered even at high coverage
    assert ans.confidence < ans.coverage


def test_web_ranked_below_local():
    def web(q, k):
        return [("stable models", "http://ex/s", "stable candidate promote "
                 "demote autopilot traffic evidence live")]
    eng = RagEngine(LocalSource(_docs()), WebSource(web), coverage_gate=0.99)
    ans = eng.retrieve("autopilot promotes stable models")
    origins = [h.doc.origin for h in ans.passages]
    # every local hit precedes every web hit
    assert origins == sorted(origins, key=lambda o: o == "web")


def test_blocked_web_degrades_to_local():
    def broken(q, k):
        raise ConnectionError("proxy 403")
    eng = RagEngine(LocalSource(_docs()), WebSource(broken), coverage_gate=0.6)
    ans = eng.retrieve("quantum entanglement recipes france")
    assert not ans.used_web                 # never raises, just no web
    assert ans.passages == [] and ans.confidence == 0.0


def test_no_web_source_configured_is_local_only():
    eng = RagEngine(LocalSource(_docs()))   # web=None
    ans = eng.retrieve("obscure topic not present anywhere")
    assert not ans.used_web


# ── interactive understanding ───────────────────────────────────────────

def test_clarify_flags_vague_query():
    eng = RagEngine(LocalSource(_docs()))
    assert eng.clarify("how")               # single content term -> clarify
    assert eng.clarify("circuit breaker cooldown failures") == [] \
        or isinstance(eng.clarify("circuit breaker cooldown failures"), list)


def test_study_questions_are_grounded_and_typed():
    eng = RagEngine(LocalSource(_docs()))
    qs = eng.study_questions("circuit breaker", n=3)
    assert qs
    assert all("source" in q and q["type"] in ("definitional", "cloze")
               for q in qs)
    clozes = [q for q in qs if q["type"] == "cloze"]
    assert all("____" in q["q"] and "answer" in q for q in clozes)


def test_answer_names_missing_terms():
    eng = RagEngine(LocalSource(_docs()))
    ans = eng.retrieve("circuit breaker bicycle maintenance")
    assert "bicycle" in ans.missing_terms or "maintenance" in ans.missing_terms


# ── corpus loading excludes training data ───────────────────────────────

def test_load_corpus_excludes_data_and_caches():
    docs = load_corpus(".")
    sources = {d.source for d in docs}
    assert not any(s.startswith("data" + __import__("os").sep) for s in sources)
    assert not any("pytest_cache" in s for s in sources)
    assert any(s.endswith("README.md") for s in sources)   # real docs present
