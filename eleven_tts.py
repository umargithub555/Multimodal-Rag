import asyncio
import threading
import traceback
from typing import Any, AsyncGenerator
from elevenlabs import ElevenLabs
from queue import Queue, Empty, Full
class ElevenTTS:
    """
    Streams audio from ElevenLabs using a non-blocking producer-consumer model.
    """
    def __init__(self, client, voice: str = "CwhRBWXzGAHq8TQ4Fs17", model_id: str = "eleven_multilingual_v2"):
        self.client = client
        self.voice = voice
        self.model_id = model_id
    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        print(f":speaking_head_in_silhouette:  Streaming TTS: {text[:80]}...")
        chunk_queue = Queue[Any](maxsize=5)  # Reduced queue size for better responsiveness
        producer_thread = None
        stop_event = threading.Event()
        # 1. Define the producer (runs in a separate thread)
        def _producer():
            """
            Producer function that calls the blocking SDK.
            """
            stream_obj = None
            try:
                # Get text_to_speech service
                tts_obj = self.client.text_to_speech
                if not tts_obj or not hasattr(tts_obj, "stream"):
                    raise RuntimeError("TextToSpeechClient.stream not found in ElevenLabs SDK.")
                # Call stream method with correct parameters
                stream_obj = tts_obj.stream(
                    text=text,
                    voice_id=self.voice,
                    model_id=self.model_id,
                    output_format="pcm_22050"  # PCM format for 22050 Hz
                )
                # Stream the audio chunks
                for item in stream_obj:
                    # Check if we should stop
                    if stop_event.is_set():
                        break
                    # Extract bytes from the response
                    if hasattr(item, 'audio'):
                        chunk = bytes(item.audio)
                    elif isinstance(item, bytes):
                        chunk = item
                    else:
                        chunk = _extract_bytes_from_response(item)
                    if chunk:
                        try:
                            chunk_queue.put(chunk, timeout=1.0)  # Block until queue has space or timeout
                        except Full:
                            # Queue is full, skip this chunk
                            continue
            except GeneratorExit:
                print("TTS producer: GeneratorExit (stream was cancelled by consumer).")
            except Exception as e:
                if "text must not be empty" in str(e):
                    print(f"TTS Producer Warning: {e}")
                else:
                    print(f":x: TTS Producer Error: {e}")
                    traceback.print_exc()
            finally:
                stop_event.set()  # Signal to stop
                if stream_obj and hasattr(stream_obj, 'close'):
                    try:
                        stream_obj.close()
                    except:
                        pass
                try:
                    chunk_queue.put(None, timeout=0.5)  # Signal end of stream
                except:
                    pass
                print("TTS producer: finished and put None.")
        # 2. Start the producer thread
        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()
        # 3. The consumer (this async generator)
        loop = asyncio.get_running_loop()
        try:
            while True:
                # Asynchronously get items from the sync queue with timeout
                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(None, chunk_queue.get),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Check if thread is still alive
                    if not producer_thread.is_alive():
                        # Thread died, check for final None
                        try:
                            chunk = chunk_queue.get_nowait()
                        except:
                            break
                    else:
                        # Thread still alive, continue waiting
                        continue
                if chunk is None:
                    break # End of stream signal
                yield chunk
        except asyncio.CancelledError:
            print("TTS consumer: stream cancelled.")
            # Signal producer to stop
            stop_event.set()
            try:
                chunk_queue.put(None, timeout=0.1)
            except:
                pass
            raise
        finally:
            # Clean up the producer thread
            stop_event.set()  # Signal thread to stop
            if producer_thread and producer_thread.is_alive():
                try:
                    # Wait briefly for thread to finish
                    await asyncio.wait_for(
                        loop.run_in_executor(None, producer_thread.join, 0.1),
                        timeout=0.5
                    )
                except:
                    pass
            print("TTS stream consumer finished.")
# Helper function (no changes)
def _extract_bytes_from_response(res: Any) -> bytes:
    try:
        if res is None:
            return b""
        if isinstance(res, (bytes, bytearray, memoryview)):
            return bytes(res)
        if isinstance(res, dict):
            for k in ("audio", "audio_content", "content", "data"):
                if k in res and isinstance(res[k], (bytes, bytearray, memoryview)):
                    return bytes(res[k])
            for v in res.values():
                if isinstance(v, (bytes, bytearray, memoryview)):
                    return bytes(v)
        for attr in ("content", "audio", "audio_content", "chunk"):
            if hasattr(res, attr):
                val = getattr(res, attr)
                if isinstance(val, (bytes, bytearray, memoryview)):
                    return bytes(val)
        if isinstance(res, str):
            return res.encode("utf-8")
    except Exception:
        pass
    return b""