from studio import writer


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.asks = []

    def generate(self, system, ask, schema, temperature=0.7, **_):
        self.asks.append(ask)
        return self.replies.pop(0)


def para(n):
    return " ".join(["calm"] * n)


def test_extend_appends_until_close_to_target():
    llm = FakeLLM([{"paragraphs": [para(150), para(150)], "scenes": ["a"]},
                   {"paragraphs": [para(140)], "scenes": ["b"]}])
    paragraphs, scenes = writer._extend(llm, "brief", "Ch", [para(150)] * 4, ["s1", "s2"], 1030, 9, False, "src")
    assert sum(len(p.split()) for p in paragraphs) == 1040
    assert scenes == ["s1", "s2", "a", "b"]
    assert "CONTINUE" in llm.asks[0]


def test_extend_stops_when_long_enough_or_empty():
    llm = FakeLLM([{"paragraphs": [], "scenes": []}])
    paragraphs, _ = writer._extend(llm, "brief", "Ch", [para(150)] * 3, [], 1030, 9, True, "src")
    assert len(paragraphs) == 3
    llm = FakeLLM([])
    paragraphs, _ = writer._extend(llm, "brief", "Ch", [para(150)] * 6, [], 1030, 9, True, "src")
    assert len(paragraphs) == 6 and not llm.asks
