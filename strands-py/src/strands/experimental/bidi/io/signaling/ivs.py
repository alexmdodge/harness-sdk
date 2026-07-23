"""IVS Real-Time Streaming implementation of SignalingProvider.

Uses the IVS Python SDK (C-backed) which handles the full WebRTC stack
internally: signaling, ICE/DTLS, Opus codec, jitter buffer, AEC. This
provider exposes the SDK's audio callback and AudioInput push interfaces
through the SignalingProvider protocol.

The IVS SDK delivers decoded PCM at 48kHz stereo (2ch, int16 interleaved)
and accepts PCM at any rate for publishing (it resamples internally to
48kHz for Opus encoding).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any

from .provider import MediaKind, SignalingProvider

try:
    from ivsrealtime import (
        AudioBuffer,
        AudioInput,
        CallbackStrategy,
        PCMFormat,
        Stage,
        StageEventCallbacks,
        SubscribeType,
    )
except ImportError:
    AudioBuffer = None  # type: ignore[assignment, misc]
    AudioInput = None  # type: ignore[assignment, misc]
    CallbackStrategy = None  # type: ignore[assignment, misc]
    PCMFormat = None  # type: ignore[assignment, misc]
    Stage = None  # type: ignore[assignment, misc]
    StageEventCallbacks = None  # type: ignore[assignment, misc]
    SubscribeType = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class IvsSignalingProvider:
    """IVS Real-Time Streaming implementation of SignalingProvider.

    Joins an IVS Stage as a server-side participant using the IVS C SDK
    for the full WebRTC stack. IVS manages signaling, STUN, and TURN.

    Audio delivery:
        - Input: 48kHz, stereo (2ch), int16 interleaved, ~480 frames/10ms
        - Output: Any sample rate accepted (SDK resamples to 48kHz internally)

    Args:
        token: IVS Stage participant token (must have PUBLISH + SUBSCRIBE).
        output_sample_rate: Sample rate for audio sent via send_media().
            The SDK accepts any rate and resamples internally.
        output_channels: Channel count for send_media() audio.
        callbacks: Optional StageEventCallbacks for lifecycle events.
    """

    def __init__(
        self,
        token: str,
        output_sample_rate: int = 16000,
        output_channels: int = 1,
        callbacks: Any = None,
    ) -> None:
        self._token = token
        self._callbacks = callbacks
        self._output_sample_rate = output_sample_rate
        self._output_channels = output_channels

        self._stage: Stage | None = None
        self._audio_input: AudioInput | None = None
        self._strategy: CallbackStrategy | None = None
        self._audio_buffer: queue.Queue[bytes] = queue.Queue()
        self._frame_count: int = 0
        self._pts: int = 0

    # --- SignalingProvider properties ---

    @property
    def input_sample_rate(self) -> int:
        """IVS delivers at 48kHz (Opus decode rate)."""
        return 48000

    @property
    def input_channels(self) -> int:
        """IVS delivers stereo (2 channels, interleaved)."""
        return 2

    @property
    def output_sample_rate(self) -> int:
        """Rate for send_media() — SDK resamples internally."""
        return self._output_sample_rate

    @property
    def output_channels(self) -> int:
        """Channels for send_media()."""
        return self._output_channels

    # --- Lifecycle ---

    async def start(self) -> None:
        """Join the IVS Stage and establish the peer connection."""
        logger.info("IVS signaling: starting (token length=%d)", len(self._token))

        self._audio_input = AudioInput(
            samples_per_sec=self._output_sample_rate,
            num_channels=self._output_channels,
            format=PCMFormat.S16_INTERLEAVED,
        )

        self._strategy = CallbackStrategy(
            subscription_types=lambda p: SubscribeType.AUDIO,
            should_publish=lambda p: p.is_local,
            streams_to_publish=lambda p: [self._audio_input],
        )

        self._stage = Stage(
            token=self._token,
            strategy=self._strategy,
            callbacks=self._callbacks,
        )
        self._stage.set_audio_callback(self._on_audio)
        self._stage.join()

        logger.info("IVS signaling: stage joined, waiting for remote audio")

    async def stop(self) -> None:
        """Leave the IVS Stage and release resources (fire-and-forget).

        raw_stage_destroy blocks indefinitely in some environments (see
        THREADING.md). Teardown runs in a daemon thread to avoid stalling
        the event loop.
        """
        logger.info("IVS signaling: stopping (received %d audio frames)", self._frame_count)

        # Capture references before clearing
        stage = self._stage
        audio_input = self._audio_input
        strategy = self._strategy

        # Clear immediately so a new provider can be created
        self._stage = None
        self._audio_input = None
        self._strategy = None

        # Unblock any waiting receive_media()
        self._audio_buffer.put(b"")

        # Fire-and-forget teardown in daemon thread
        def _teardown():
            try:
                if stage is not None:
                    logger.info("IVS teardown: calling stage.leave()...")
                    stage.leave()
                    logger.info("IVS teardown: stage.leave() returned")

                    if hasattr(stage, "_audio_reader") and stage._audio_reader is not None:
                        logger.info("IVS teardown: calling audio_reader.close()...")
                        stage._audio_reader.close()
                        stage._audio_reader = None
                        logger.info("IVS teardown: audio_reader.close() returned")

                    if stage._handle is not None:
                        logger.info("IVS teardown: calling raw_stage_destroy()...")
                        from ivsrealtime._native_gen import raw_stage_destroy

                        raw_stage_destroy(stage._handle)
                        stage._handle = None
                        logger.info("IVS teardown: raw_stage_destroy() returned")

                if audio_input is not None:
                    audio_input.close()
                if strategy is not None:
                    strategy.close()
                logger.info("IVS teardown: completed successfully")
            except Exception as teardown_error:
                logger.warning("IVS teardown error (non-fatal): %s", teardown_error)

        threading.Thread(target=_teardown, name="ivs-teardown", daemon=True).start()
        logger.info("IVS signaling: stopped (teardown in background)")

    # --- Media ---

    async def receive_media(self, kind: MediaKind = "audio") -> bytes:
        """Receive next audio frame from IVS (blocks until available).

        Returns raw PCM int16 bytes at 48kHz stereo as delivered by the SDK.
        Returns empty bytes when the connection is closed.
        """
        data = await asyncio.to_thread(self._audio_buffer.get)
        return data

    async def send_media(self, data: bytes, kind: MediaKind = "audio") -> None:
        """Send PCM audio to IVS for publishing to remote peers.

        The SDK handles resampling to 48kHz and Opus encoding.

        Args:
            data: Raw PCM int16 bytes at self.output_sample_rate.
        """
        self.send_media_sync(data, kind)

    def send_media_sync(self, data: bytes, kind: MediaKind = "audio") -> None:
        """Thread-safe synchronous audio push to IVS.

        AudioInput.push() is thread-safe per the IVS SDK.
        Called directly from the pacer thread for minimal latency.
        """
        if self._audio_input is None:
            return
        num_frames = len(data) // (2 * self._output_channels)
        self._audio_input.push(data, num_frames, self._pts)
        self._pts += num_frames

    # --- Data channel (not supported by IVS SDK currently) ---

    async def receive_data(self) -> str:
        """Not supported — IVS SDK doesn't expose data channels."""
        await asyncio.sleep(float("inf"))
        return ""

    async def send_data(self, message: str) -> None:
        """Not supported — IVS SDK doesn't expose data channels."""
        pass

    # --- Internal ---

    def _on_audio(self, buf: "AudioBuffer") -> None:
        """IVS audio callback — fires on the SDK's audio pump thread.

        Enqueues raw PCM bytes for consumption by receive_media().
        """
        if not buf.data:
            return

        self._frame_count += 1
        if self._frame_count == 1:
            logger.info(
                "IVS signaling: first audio frame | rate=%dHz, channels=%d, frames=%d",
                buf.samples_per_sec,
                buf.num_channels,
                buf.num_frames,
            )
        elif self._frame_count % 500 == 0:
            logger.info(
                "IVS signaling: audio frames received: %d (queue size: %d)",
                self._frame_count,
                self._audio_buffer.qsize(),
            )

        self._audio_buffer.put(buf.data)
