"""Signaling provider protocol and implementations for WebRTC IO adapters.

A SignalingProvider encapsulates the full lifecycle of a single WebRTC peer
connection — how it's established (provider-specific signaling) and how
media/data is exchanged once connected (uniform interface).

Implementations:
    - IvsSignalingProvider: Amazon IVS Real-Time Streaming
"""

from .provider import MediaKind, SignalingProvider

__all__ = ["MediaKind", "SignalingProvider"]
