"""SignalingProvider protocol — abstracts WebRTC peer connection lifecycle.

This is to WebRTC infrastructure what BidiModel is to AI model providers:
each managed service (IVS, KVS, LiveKit) has its own implementation, but
the interface is the same. BidiWebRtcIO consumes this protocol and never
knows which service negotiated the connection.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

MediaKind = Literal["audio", "video"]


@runtime_checkable
class SignalingProvider(Protocol):
    """Full WebRTC lifecycle for a single peer connection.

    One instance per client session. Handles signaling, connection setup,
    and media/data exchange. Implementations own all provider-specific
    concerns (token management, SDK initialization, thread safety).

    Audio format contract:
        - receive_media() returns raw PCM int16 mono/stereo bytes at the
          provider's native delivery rate (e.g., 48kHz for IVS/Opus).
        - send_media() accepts raw PCM int16 bytes at the provider's
          expected input rate. The provider handles codec encoding.
        - Resampling between provider rate and model rate is NOT the
          provider's responsibility — BidiWebRtcIO handles that.

    Properties:
        input_sample_rate: Sample rate of audio returned by receive_media().
        input_channels: Number of channels in receive_media() output.
        output_sample_rate: Sample rate expected by send_media().
        output_channels: Number of channels expected by send_media().
    """

    @property
    def input_sample_rate(self) -> int:
        """Sample rate of audio delivered by receive_media()."""
        ...

    @property
    def input_channels(self) -> int:
        """Channel count of audio delivered by receive_media()."""
        ...

    @property
    def output_sample_rate(self) -> int:
        """Sample rate expected by send_media()."""
        ...

    @property
    def output_channels(self) -> int:
        """Channel count expected by send_media()."""
        ...

    async def start(self) -> None:
        """Establish the peer connection.

        Negotiate SDP/ICE, complete connection setup, and return only
        when the connection is ready for media/data exchange.
        """
        ...

    async def stop(self) -> None:
        """Close the peer connection and release all resources.

        Must be safe to call multiple times. Should not block the event
        loop for extended periods (use background threads for slow teardown).
        """
        ...

    async def receive_media(self, kind: MediaKind = "audio") -> bytes:
        """Receive the next media frame from the remote peer.

        Blocks until a frame is available or the connection ends.
        Returns empty bytes on connection close.

        Args:
            kind: Media type to receive (currently only "audio" is required).

        Returns:
            Raw PCM int16 bytes at self.input_sample_rate / self.input_channels.
        """
        ...

    async def send_media(self, data: bytes, kind: MediaKind = "audio") -> None:
        """Send a media frame to the remote peer.

        Args:
            data: Raw PCM int16 bytes at self.output_sample_rate / self.output_channels.
            kind: Media type to send.
        """
        ...

    def send_media_sync(self, data: bytes, kind: MediaKind = "audio") -> None:
        """Synchronous variant of send_media for use from pacer threads.

        Must be thread-safe. Default implementations may wrap the async version,
        but providers with thread-safe push APIs (like IVS) should override directly.

        Args:
            data: Raw PCM int16 bytes at self.output_sample_rate / self.output_channels.
            kind: Media type to send.
        """
        ...

    async def receive_data(self) -> str:
        """Receive the next data channel message from the remote peer.

        Blocks until a message is available or the connection ends.
        Returns empty string on connection close.

        Returns:
            JSON-encoded string message.
        """
        ...

    async def send_data(self, message: str) -> None:
        """Send a data channel message to the remote peer.

        Args:
            message: JSON-encoded string to send.
        """
        ...
