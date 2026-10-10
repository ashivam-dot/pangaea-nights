from studio import gates

SOURCE = ("Anomalocaris was a large predator of the Cambrian seas, reaching perhaps 38 centimetres. "
          "It lived alongside trilobites in the Burgess Shale.")


def report_for(text):
    r = gates.Report()
    gates.check_facts(text, gates.Sources([SOURCE]), r)
    return r


def test_supported_names_and_numbers_pass():
    assert report_for("Anomalocaris drifted above the trilobites, perhaps 38 centimetres long.").ok


def test_invented_taxon_is_caught():
    r = report_for("Beside it swam Zorbulonia, a gentle filter feeder.")
    assert any("Zorbulonia" in p for p in r.problems)
    assert r.bad_sentences == ["Beside it swam Zorbulonia, a gentle filter feeder."]


def test_invented_number_is_caught():
    r = report_for("Anomalocaris grew to 90 centimetres.")
    assert any("unsupported number" in p for p in r.problems)


def style(paragraphs, opening=False, hedges=1, target=60):
    r = gates.Report()
    gates.check_style(paragraphs, r, opening, hedges, target)
    return r.problems


CALM = ("You might picture the water, still and clear, as you rest here. Scientists think the light was soft. "
        "Your breathing slows, and you drift a little further down, past the slow animals of the reef, "
        "while the sea moves gently above you and the sand lies quiet below.")


def test_calm_paragraph_passes_style():
    assert style([CALM]) == []


def test_style_catches_exclamation_alarm_and_calls():
    problems = style([CALM + " What a terrifying hunter! Subscribe for more."])
    assert any(p == "exclamation" for p in problems)
    assert any("alarm word" in p for p in problems)
    assert any("call to action" in p for p in problems)


def test_one_quiet_call_is_allowed_in_the_opening_only():
    text = CALM + " If you enjoy these slow journeys, you can subscribe."
    assert not any("call to action" in p for p in style([text], opening=True))
    assert any("call to action" in p for p in style([text], opening=False))


def test_style_wants_hedges_and_second_person():
    problems = style(["The sea was calm. The light was soft. The sand was pale and the reef was quiet and old "
                      "and the animals moved slowly over it in the long warm afternoon of the ancient world."],
                     hedges=2)
    assert any("hedges" in p for p in problems)
    assert any("second person" in p for p in problems)


def test_prompts_reject_people_and_text():
    good = ("A wide calm underwater view of a Cambrian reef at dusk, soft blue light falling through clear "
            "water onto sponges and drifting trilobites, deep shadows, still and quiet.")
    assert gates.check_prompt(good) == []
    assert any("diver" in i for i in gates.check_prompt(good + " A diver watches."))
    assert any("text" in i for i in gates.check_prompt(good + " With a text label."))
    assert "too short to describe a scene" in gates.check_prompt("A reef.")
    assert any("leaf-like" in i for i in gates.check_prompt(good + " Its leaf-like ridges sway."))
    assert any("resembling" in i for i in gates.check_prompt(good + " A disc resembling a flower."))


def test_drop_sentences_removes_only_bad_ones():
    out = gates.drop_sentences(["Keep this. Drop this. Keep that.", "Drop this."], ["Drop this."])
    assert out == ["Keep this. Keep that."]
