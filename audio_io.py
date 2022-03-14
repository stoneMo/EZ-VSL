import av
# import torchaudio
import numpy as np
from fractions import Fraction


# def load_audio_torchaudio(fn):
#     data, sr = torchaudio.load(fn)
#     return data, sr


def open_audio_av(path):
    container = av.open(path)
    for stream in container.streams.video:
        stream.codec_context.thread_type = av.codec.context.ThreadType.NONE
        stream.codec_context.thread_count = 1
    for stream in container.streams.audio:
        stream.codec_context.thread_type = av.codec.context.ThreadType.NONE
        stream.codec_context.thread_count = 1
    return container


def load_audio_av(path=None, container=None, rate=None, start_time=None, duration=None, layout="mono"):
    if container is None:
        container = av.open(path)
    audio_stream = container.streams.audio[0]

    # Parse metadata
    _ss = audio_stream.start_time * audio_stream.time_base if audio_stream.start_time is not None else 0.
    _dur = audio_stream.duration * audio_stream.time_base
    _ff = _ss + _dur
    _rate = audio_stream.rate

    if rate is None:
        rate = _rate
    if start_time is None:
        start_time = _ss
    if duration is None:
        duration = _ff - start_time
    duration = min(duration, _ff - start_time)
    end_time = start_time + duration

    resampler = av.audio.resampler.AudioResampler(format="s16p", layout=layout, rate=rate)

    # Read data
    chunks = []
    container.seek(int(start_time * av.time_base))
    for frame in container.decode(audio=0):
        chunk_start_time = frame.pts * frame.time_base
        chunk_end_time = chunk_start_time + Fraction(frame.samples, frame.rate)
        if chunk_end_time < start_time:   # Skip until start time
            continue
        if chunk_start_time > end_time:       # Exit if clip has been extracted
            break

        try:
            frame.pts = None
            if resampler is not None:
                chunks.append((chunk_start_time, resampler.resample(frame).to_ndarray()))
            else:
                chunks.append((chunk_start_time, frame.to_ndarray()))
        except AttributeError:
            break

    # Trim for frame accuracy
    audio = np.concatenate([af[1] for af in chunks], 1)
    ss = int((start_time - chunks[0][0]) * rate)
    t = int(duration * rate)
    if ss < 0:
        audio = np.pad(audio, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
        ss = 0
    audio = audio[:, ss: ss+t]

    # Normalize to [-1, 1]
    audio = audio / np.iinfo(audio.dtype).max

    return audio, rate


def audio_info_av(inpt, audio=None, format=None):
    container = inpt
    if isinstance(inpt, str):
        try:
            container = av.open(inpt, format=format)
        except av.AVError:
            return None, None

    audio_stream = container.streams.audio[audio]
    time_base = audio_stream.time_base
    duration = audio_stream.duration * time_base
    start_time = audio_stream.start_time * time_base
    channels = audio_stream.channels
    fps = audio_stream.rate
    chunk_size = audio_stream.frame_size
    chunks = audio_stream.frames
    meta = {'channels': channels,
            'fps': fps,
            'start_time': start_time,
            'duration': duration,
            'chunks': chunks,
            'chunk_size': chunk_size}
    return meta
