from studio import numbers


def vals(text):
    return [q.value for q in numbers.quantities(text)]


def test_digits_words_and_units_agree():
    assert vals("66 million years") == [66e6]
    assert vals("sixty-six million years") == [66e6]
    assert vals("about 66 Ma") == [66e6]


def test_scale_phrases():
    assert vals("a hundred million years") == [1e8]
    assert vals("half a billion years") == [5e8]
    assert vals("five hundred million") == [5e8]
    assert vals("1,200 kilometres") == [1200]


def test_small_counting_words_are_exempt_but_others_are_not():
    one, four = numbers.quantities("one animal"), numbers.quantities("four animals")
    assert one[0].exempt and not four[0].exempt


def test_codes_are_not_numbers():
    assert vals("C4 grasses") == []


def test_scale_word_after_digits_is_not_a_second_number():
    src = numbers.values("The extinction happened 66 Ma.")
    assert numbers.unsupported("It happened 66 million years ago.", src) == []
    assert vals("150 hundred million") == [150 * 100 * 1e6]


def test_unsupported_against_sources():
    src = numbers.values("The Cambrian began 538.8 Ma, and the explosion lasted about 20 million years.")
    assert numbers.unsupported("It began about 538.8 million years ago.", src) == []
    assert numbers.unsupported("Twenty million years of change.", src) == []
    bad = numbers.unsupported("It began 600 million years ago.", src)
    assert [q.value for q in bad] == [600e6]
