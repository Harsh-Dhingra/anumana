import anumana


def test_version():
    assert isinstance(anumana.__version__, str)
    assert anumana.__version__.count(".") == 2
