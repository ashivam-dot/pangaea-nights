from studio import branding


def test_the_about_text_covers_both_formats_and_the_disclosure():
    d = branding.DESCRIPTION
    assert "Survive the Night" in d and "Tuesday and Friday" in d
    assert "AI generated" in d and "AI image tools" in d
    assert len(d) <= 1000 and len(branding.KEYWORDS) <= 500


def test_the_plan_keeps_the_other_settings():
    got = branding.plan({"description": "old", "country": "IN", "title": "Pangaea Nights", "unsubscribedTrailer": "x"})
    assert got["country"] == "IN" and got["description"] == branding.DESCRIPTION
    assert "title" not in got and "unsubscribedTrailer" not in got
