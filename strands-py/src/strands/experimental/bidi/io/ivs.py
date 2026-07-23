"""Convenience wrapper: IVS Real-Time Streaming IO for Strands BidiAgent.

Combines IvsSignalingProvider + BidiWebRtcIO into a single constructor.
This is the recommended entry point for IVS-based voice applications.

Usage:
    from strands.experimental.bidi.io.ivs import BidiIvsIO

    ivs_io = BidiIvsIO(token=participant_token)
    agent = BidiAgent(model=BidiNovaSonicModel(), tools=[...])
    await agent.run(inputs=[ivs_io.input()], outputs=[ivs_io.output()])
"""

from __future__ import annotations

from typing import Any

from ..types.io import BidiInput, BidiOutput
from .signaling.ivs import IvsSignalingProvider
from .webrtc import BidiWebRtcIO


class BidiIvsIO:
    """Bridges IVS Real-Time Streaming to Strands BidiInput/BidiOutput.

    IVS handles the full WebRTC stack (signaling, ICE, Opus encode/decode).
    This adapter shuttles PCM between IVS callbacks and Strands bidi events.

    Internally creates an IvsSignalingProvider and wraps it with BidiWebRtcIO
    for audio normalization and real-time pacing.

    Args:
        token: IVS Stage participant token (must have PUBLISH + SUBSCRIBE).
        callbacks: Optional StageEventCallbacks subclass for lifecycle events.
    """

    def __init__(self, token: str, callbacks: Any = None) -> None:
        """Create IVS IO adapter with the given participant token.

        Args:
            token: IVS Stage participant token (must have PUBLISH + SUBSCRIBE).
            callbacks: Optional StageEventCallbacks subclass for lifecycle events.
        """
        self._signaling = IvsSignalingProvider(token=token, callbacks=callbacks)
        self._webrtc = BidiWebRtcIO(signaling=self._signaling)

    def input(self) -> BidiInput:
        """Return IVS audio input adapter (browser audio → agent)."""
        return self._webrtc.input()

    def output(self) -> BidiOutput:
        """Return IVS audio output adapter (agent audio → browser)."""
        return self._webrtc.output()
