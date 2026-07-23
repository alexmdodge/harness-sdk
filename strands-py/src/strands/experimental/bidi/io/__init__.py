"""IO channel implementations for bidirectional streaming."""


def __getattr__(name):
    """Lazy imports to avoid hard dependency on pyaudio/prompt_toolkit/ivsrealtime at import time."""
    if name == "BidiAudioIO":
        from .audio import BidiAudioIO

        return BidiAudioIO
    if name == "BidiTextIO":
        from .text import BidiTextIO

        return BidiTextIO
    if name == "BidiIvsIO":
        from .ivs import BidiIvsIO

        return BidiIvsIO
    if name == "BidiWebRtcIO":
        from .webrtc import BidiWebRtcIO

        return BidiWebRtcIO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BidiAudioIO", "BidiTextIO", "BidiIvsIO", "BidiWebRtcIO"]
