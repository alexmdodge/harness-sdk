"""Provider-agnostic WebRTC IO adapter for Strands BidiAgent.

BidiWebRtcIO consumes a SignalingProvider and translates between WebRTC
media/data and Strands bidi events. It handles:
    - Stereo→mono downmix (WebRTC typically delivers stereo)
    - Resampling between provider rate (48kHz) and model rate (16kHz/24kHz)
    - Base64 encoding/decoding for bidi audio events
    - Real-time paced audio output via a dedicated thread
    - Interruption handling (buffer clear)
    - Event logging (transcripts, connection state)

This class never changes regardless of which SignalingProvider is used.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading

import numpy as np

from ..agent.agent import BidiAgent
from ..types.events import (
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
)
from ..types.io import BidiInput, BidiOutput
from .signaling.provider import SignalingProvider

logger = logging.getLogger(__name__)


def _resample_pcm_s16(data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM int16 mono audio between sample rates.

    Uses simple integer-ratio decimation/interpolation with numpy.
    For 48kHz→16kHz (÷3) and 48kHz→24kHz (÷2) this is exact.
    """
    if from_rate == to_rate:
        return data

    samples = np.frombuffer(data, dtype=np.int16)
    gcd = np.gcd(from_rate, to_rate)
    up = to_rate // gcd
    down = from_rate // gcd

    if up == 1:
        resampled = samples[::down]
    elif down == 1:
        resampled = np.repeat(samples, up)
    else:
        upsampled = np.repeat(samples, up)
        resampled = upsampled[::down]

    return resampled.astype(np.int16).tobytes()


class _WebRtcInput(BidiInput):
    """Reads audio from a SignalingProvider, normalizes, and emits bidi events."""

    def __init__(self, parent: BidiWebRtcIO) -> None:
        self._parent = parent
        self._signaling = parent._signaling
        self._frame_count: int = 0

    async def start(self, agent: BidiAgent) -> None:
        """Start the signaling provider and configure audio format from agent model."""
        audio_config = agent.model.config["audio"]
        self._channels = audio_config["channels"]
        self._format = audio_config["format"]
        self._model_rate = audio_config["input_rate"]

        logger.info(
            "WebRTC input starting | provider_rate=%dHz→model_rate=%dHz, channels=%d",
            self._signaling.input_sample_rate,
            self._model_rate,
            self._channels,
        )

        # Start the underlying signaling provider
        await self._signaling.start()
        logger.info("WebRTC input started — signaling provider connected")

    async def stop(self) -> None:
        """Stop the signaling provider."""
        logger.info("WebRTC input stopping (received %d frames)", self._frame_count)
        await self._signaling.stop()
        logger.info("WebRTC input stopped")

    async def __call__(self) -> BidiAudioInputEvent:
        """Read audio from signaling provider, normalize, and return bidi event.

        Handles stereo→mono downmix and resampling from provider rate to model rate.
        """
        data = await self._signaling.receive_media("audio")

        if not data:
            # Empty data signals shutdown
            await asyncio.sleep(float("inf"))

        self._frame_count += 1

        # Downmix stereo to mono if provider delivers multi-channel
        if self._signaling.input_channels == 2:
            samples = np.frombuffer(data, dtype=np.int16).reshape(-1, 2)
            mono = (samples[:, 0].astype(np.int32) + samples[:, 1].astype(np.int32)) // 2
            data = mono.astype(np.int16).tobytes()

        # Resample from provider rate to model rate
        data = _resample_pcm_s16(data, self._signaling.input_sample_rate, self._model_rate)

        return BidiAudioInputEvent(
            audio=base64.b64encode(data).decode("utf-8"),
            channels=self._channels,
            format=self._format,
            sample_rate=self._model_rate,
        )


class _WebRtcOutput(BidiOutput):
    """Sends agent audio to a SignalingProvider at real-time pace.

    Buffers audio from the model and drains it via a dedicated pacer thread
    to maintain precise timing. Handles interruption by clearing the buffer.
    """

    _FRAME_DURATION_MS = 10

    def __init__(self, parent: BidiWebRtcIO) -> None:
        self._parent = parent
        self._signaling = parent._signaling
        self._output_buffer: bytearray = bytearray()
        self._output_frame_count: int = 0
        self._pacer_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

    async def start(self, agent: BidiAgent) -> None:
        """Configure output format and start the pacer thread."""
        audio_config = agent.model.config["audio"]
        self._model_rate = audio_config["output_rate"]
        self._channels = audio_config["channels"]
        self._frames_per_push = (self._model_rate * self._FRAME_DURATION_MS) // 1000
        self._bytes_per_push = self._frames_per_push * 2 * self._channels

        self._running = True
        self._pacer_thread = threading.Thread(target=self._pacer_loop, daemon=True, name="webrtc-audio-pacer")
        self._pacer_thread.start()

        logger.info(
            "WebRTC output started | model_rate=%dHz, pacer=%dms (%d frames/push)",
            self._model_rate,
            self._FRAME_DURATION_MS,
            self._frames_per_push,
        )

    async def stop(self) -> None:
        """Stop the pacer thread."""
        self._running = False
        if self._pacer_thread:
            self._pacer_thread.join(timeout=1.0)
        logger.info("WebRTC output stopped (pushed %d chunks)", self._output_frame_count)

    def _pacer_loop(self) -> None:
        """Drain audio buffer at real-time rate using monotonic clock scheduling."""
        import time

        interval = self._FRAME_DURATION_MS / 1000.0
        next_push = time.monotonic()
        late_count = 0

        while self._running:
            next_push += interval
            now = time.monotonic()
            sleep_time = next_push - now

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                late_count += 1
                if sleep_time < -0.05:
                    next_push = time.monotonic()

            with self._lock:
                if len(self._output_buffer) < self._bytes_per_push:
                    continue
                chunk = bytes(self._output_buffer[: self._bytes_per_push])
                del self._output_buffer[: self._bytes_per_push]

            # Send directly — SignalingProvider.send_media is sync-safe for
            # the pacer thread in IVS (audio_input.push is thread-safe).
            # For providers that need async, override with a threadsafe wrapper.
            self._signaling.send_media_sync(chunk)
            self._output_frame_count += 1

            if self._output_frame_count == 1:
                logger.info(
                    "First paced audio push | frames=%d, buffer_remaining=%d bytes",
                    self._frames_per_push,
                    len(self._output_buffer),
                )
            elif self._output_frame_count % 1000 == 0:
                logger.info(
                    "Paced audio pushes: %d (buffer=%d bytes, late=%d)",
                    self._output_frame_count,
                    len(self._output_buffer),
                    late_count,
                )

    async def __call__(self, event: BidiOutputEvent) -> None:
        """Process output events from the agent."""
        if isinstance(event, BidiAudioStreamEvent):
            pcm = base64.b64decode(event["audio"])
            with self._lock:
                self._output_buffer.extend(pcm)

        elif isinstance(event, BidiInterruptionEvent):
            logger.info("Interruption received (reason=%s) — clearing output buffer", event["reason"])
            with self._lock:
                self._output_buffer.clear()
            self._output_frame_count = 0

        else:
            # Log non-audio events
            if hasattr(event, "__getitem__"):
                event_type = event.get("type", "")
                if event_type == "bidi_transcript_stream":
                    if event.get("is_final"):
                        logger.info(
                            "Transcript [%s]: %s",
                            event.get("role", "?"),
                            event.get("current_transcript", event.get("text", "")),
                        )
                    return
                elif event_type == "bidi_connection_start":
                    logger.info("Model connected (connection_id=%s)", event.get("connection_id", "?"))
                    return
                elif event_type == "bidi_response_start":
                    logger.info("Model response started")
                    return
                elif event_type == "bidi_response_complete":
                    logger.info("Model response complete (reason=%s)", event.get("stop_reason", "?"))
                    return
                elif event_type == "bidi_connection_close":
                    logger.info("Model connection closed (reason=%s)", event.get("reason", "?"))
                    return
            logger.debug("Output event: %s", type(event).__name__)


class BidiWebRtcIO:
    """Provider-agnostic WebRTC IO adapter.

    Bridges a SignalingProvider to BidiInput/BidiOutput by handling audio
    normalization (downmix, resample, base64) and real-time pacing.

    Usage:
        signaling = IvsSignalingProvider(token=token)
        webrtc_io = BidiWebRtcIO(signaling=signaling)
        await agent.run(inputs=[webrtc_io.input()], outputs=[webrtc_io.output()])

    Args:
        signaling: A SignalingProvider implementation.
    """

    def __init__(self, signaling: SignalingProvider) -> None:
        """Create a WebRTC IO adapter with the given signaling provider.

        Args:
            signaling: A SignalingProvider implementation for media transport.
        """
        self._signaling = signaling

    def input(self) -> _WebRtcInput:
        """Return WebRTC audio input adapter (remote peer → agent)."""
        return _WebRtcInput(self)

    def output(self) -> _WebRtcOutput:
        """Return WebRTC audio output adapter (agent → remote peer)."""
        return _WebRtcOutput(self)
