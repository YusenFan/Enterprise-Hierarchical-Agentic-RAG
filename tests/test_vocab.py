from src.metadata.vocab import ProjectVocabulary


def test_match_aliases_case_and_boundaries():
    vocab = ProjectVocabulary(projects={"perf-canary": ["perf canary", "perf-canary"]},
                              entities={"Triton": ["Triton"], "H100": ["H100"]})
    projects, entities = vocab.match("The PERF-CANARY service uses Triton on H100-80GB; tritonic is not Triton.")
    assert projects == ["perf-canary"]
    assert entities == ["Triton"]          # "H100-80GB" is hyphen-joined, so H100 is not word-bounded


def test_empty_and_missing_file(tmp_path):
    assert ProjectVocabulary().match("anything") == ([], [])
    assert ProjectVocabulary.load(str(tmp_path / "missing.json")).is_empty()
    assert ProjectVocabulary.load(None).is_empty()


def test_load_json(tmp_path):
    path = tmp_path / "v.json"
    path.write_text('{"projects": {"Apollo": ["Project Apollo"]}, "entities": {"Model-X": "Model-X"}}')
    vocab = ProjectVocabulary.load(str(path))
    assert vocab.match("Project Apollo ships model-x") == (["Apollo"], ["Model-X"])
