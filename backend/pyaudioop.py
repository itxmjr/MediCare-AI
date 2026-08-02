try:
    import audioop as _audioop
except ImportError:
    try:
        import audioop_lts as _audioop
    except ImportError:
        _audioop = None

if _audioop:
    from audioop_lts import *
else:
    raise ImportError("Could not import audioop or audioop_lts")
