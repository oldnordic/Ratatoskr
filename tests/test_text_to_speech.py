import sys
import types

# Provide dummy modules so import succeeds
sys.modules['sounddevice'] = types.SimpleNamespace(play=lambda *a, **k: None, wait=lambda: None)
sys.modules['numpy'] = types.SimpleNamespace(ndarray=list)
sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

class DummyTTSClass:
    def __init__(self, *args, **kwargs):
        class Synth:
            output_sample_rate = 22050
        self.synthesizer = Synth()

    def to(self, device):
        pass

    def tts(self, text):
        return [0.0, 0.0, 0.0]

sys.modules['TTS.api'] = types.SimpleNamespace(TTS=DummyTTSClass)

import voice.text_to_speech as tts_mod


def test_speak(monkeypatch):
    playback = []

    class DummyThread:
        def __init__(self, target, args, daemon=False):
            target(*args)

        def start(self):
            pass

    def dummy_play(data, samplerate):
        playback.append((len(data), samplerate))

    def dummy_wait():
        playback.append("wait")

    monkeypatch.setattr(tts_mod.threading, "Thread", DummyThread)
    monkeypatch.setattr(tts_mod.sd, "play", dummy_play)
    monkeypatch.setattr(tts_mod.sd, "wait", dummy_wait)

    tts_mod.speak("hi")
    assert playback == [(3, 22050), "wait"]
