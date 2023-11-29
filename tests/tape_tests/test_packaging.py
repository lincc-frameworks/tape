import tape


def test_version():
    """Check to see that the version property returns something"""
    assert tape.__version__ is not None
