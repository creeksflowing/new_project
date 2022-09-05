import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from numba import jit
from scipy import signal
from scipy.io import wavfile

# from maad import sound
# from maad import util

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main():
    st.header("WebRTC demo")

    pages = {
        "WebRTC is sendonly and audio frames are visualized with matplotlib (sendonly)": app_sendonly_audio,
        # noqa: E501
        "Plot audio representation with scikit-maad": app_room_measurements,
    }
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_sendonly_audio():
    """A sample to use WebRTC in sendonly mode to transfer audio frames
    from the browser to the server and visualize them with matplotlib
    and `st.pyplot`."""
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True},
    )

    fig_place = st.empty()

    fig, [ax_time, ax_freq] = plt.subplots(
        2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2}
    )

    sound_window_len = 5000  # 5s
    sound_window_buffer = None
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk
                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]

            if sound_window_buffer:
                # Ref: https://own-search-and-study.xyz/2017/10/27/python%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E9%9F%B3%E5%A3%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%82%92%E4%BD%9C/  # noqa
                sound_window_buffer = sound_window_buffer.set_channels(
                    1
                )  # Stereo to mono
                sample = np.array(sound_window_buffer.get_array_of_samples())

                ax_time.cla()
                times = (np.arange(-len(sample), 0)) / sound_window_buffer.frame_rate
                ax_time.plot(times, sample)
                ax_time.set_xlabel("Time")
                ax_time.set_ylabel("Magnitude")

                spec = np.fft.fft(sample)
                freq = np.fft.fftfreq(sample.shape[0], 1.0 / sound_chunk.frame_rate)
                freq = freq[: int(freq.shape[0] / 2)]
                spec = spec[: int(spec.shape[0] / 2)]
                spec[0] = spec[0] / 2

                ax_freq.cla()
                ax_freq.plot(freq, np.abs(spec))
                ax_freq.set_xlabel("Frequency")
                ax_freq.set_yscale("log")
                ax_freq.set_ylabel("Magnitude")

                fig_place.pyplot(fig)
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break


def app_room_measurements():
    audio_files_path = r"data/audio_files"
    sweep_string = ""
    inv_filter_string = ""
    ir_string = ""

    @jit(nopython=True)
    def fade(data, gain_start, gain_end):
        """
        Create a fade on an input object

        Parameters
        ----------
        :param data: The input array
        :param gain_start: The fade starting point
        :param gain_end: The fade ending point

        Returns
        -------
        data : object
            An input array with the fade applied
        """
        gain = gain_start
        delta = (gain_end - gain_start) / (len(data) - 1)
        for i in range(len(data)):
            data[i] = data[i] * gain
            gain = gain + delta

        return data

    @jit(nopython=True)
    def generate_exponential_sweep(
        sweep_duration, sr, starting_frequency, ending_frequency
    ):
        """
        Generate an exponential sweep using Farina's log sweep theory

        Parameters
        ----------
        :param sweep_duration: The duration of the excitement signal (in seconds)
        :param sr: The sampling frequency
        :param starting_frequency: The starting frequency of the excitement signal
        :param ending_frequency: The ending frequency of the excitement signal

        Returns
        -------
        exponential_sweep : array
            An array with the fade() function applied
        """
        time_in_samples = sweep_duration * sr
        exponential_sweep = np.zeros(time_in_samples, dtype=np.double)
        for n in range(time_in_samples):
            t = n / sr
            exponential_sweep[n] = np.sin(
                (2.0 * np.pi * starting_frequency * sweep_duration)
                / np.log(ending_frequency / starting_frequency)
                * (
                    np.exp(
                        (t / sweep_duration)
                        * np.log(ending_frequency / starting_frequency)
                    )
                    - 1.0
                )
            )

        number_of_samples = 50
        exponential_sweep[-number_of_samples:] = fade(
            exponential_sweep[-number_of_samples:], 1, 0
        )

        return exponential_sweep

    @jit(nopython=True)
    def generate_inverse_filter(
        sweep_duration, sr, exponential_sweep, starting_frequency, ending_frequency
    ):
        """
        Generate an inverse filter using Farina's log sweep theory

        Parameters
        ----------
        :param sweep_duration: The duration of the excitement signal (in seconds)
        :param sr: The sampling frequency
        :param exponential_sweep: The resulting array of the generate_exponential_sweep() function
        :param starting_frequency: The starting frequency of the excitement signal
        :param ending_frequency: The ending frequency of the excitement signal

        Returns
        -------
        inverse_filter : array
             The array resulting from applying an amplitude envelope to the exponential_sweep array
        """
        time_in_samples = sweep_duration * sr
        amplitude_envelope = np.zeros(time_in_samples, dtype=np.double)
        inverse_filter = np.zeros(time_in_samples, dtype=np.double)
        for n in range(time_in_samples):
            amplitude_envelope[n] = pow(
                10,
                (
                    (-6 * np.log2(ending_frequency / starting_frequency))
                    * (n / time_in_samples)
                )
                * 0.05,
            )
            inverse_filter[n] = exponential_sweep[-n] * amplitude_envelope[n]

        return inverse_filter

    def deconvolve(ir_sweep, ir_inverse):
        """
        A deconvolution of the exponential sweep and the relative inverse filter

        Parameters
        ----------
        :param ir_sweep: The resulting array of the generate_exponential_sweep() function
        :param ir_inverse: The resulting array of the generate_inverse_filter() function

        Returns
        -------
        normalized_ir : array
             An N-dimensional array containing a subset of the discrete linear deconvolution of ir_sweep with ir_inverse
        """
        impulse_response = signal.fftconvolve(
            ir_sweep, ir_inverse, mode="full"
        )  # Convolve two N-dimensional arrays using FFT

        normalized_ir = impulse_response * (1.0 / np.max(abs(impulse_response)))

        return normalized_ir

    sample_rate_option = st.selectbox("Select the desired sample rate", (44100, 48000))
    sweep_duration_option = st.selectbox("Select the duration of the sweep", (3, 7, 14))
    max_reverb_option = st.selectbox(
        "Select the expected maximum reverb decay time", (1, 2, 3, 5, 10)
    )

    st.caption(
        """
                Note that longer sweeps provide more accuracy,
                but even short sweeps can be used to measure long decays
                """
    )

    def write_wav_file(file_name, rate, data):
        save_file_path = os.path.join(audio_files_path, file_name)
        wavfile.write(save_file_path, rate, data)
        st.success(f"File successfully written to audio_files_path as:>> {file_name}")

    def play_sweep(wavefile_name):
        read_file_path = os.path.join(audio_files_path, wavefile_name)
        # Extract data and sampling rate from file
        sample_rate, data = wavfile.read(read_file_path)

        stop_button = st.button("Stop")

        if "stop_button_state" not in st.session_state:
            st.session_state.stop_button_state = False

        sd.play(data, sample_rate)

        if stop_button or st.session_state.stop_button_state:
            st.session_state.stop_button_state = True

            sd.stop()

        else:
            sd.wait()  # Wait until file is done playing

    user_input = str(st.text_input("Name your file: "))

    if user_input:
        sweep_string = user_input + "_exponential_sweep_.wav"
        inv_filter_string = user_input + "_inverse_filter_.wav"
        ir_string = user_input + "_impulse_response_.wav"

        st.write(sweep_string)

        play_button = st.button("Play")

        if "play_button_state" not in st.session_state:
            st.session_state.play_button_state = False

        if play_button or st.session_state.play_button_state:
            st.session_state.play_button_state = True

            sweep = generate_exponential_sweep(
                sweep_duration_option, sample_rate_option, 20, 24000
            )
            inv_filter = generate_inverse_filter(
                sweep_duration_option, sample_rate_option, sweep, 20, 24000
            )

            write_wav_file(file_name=sweep_string, rate=sample_rate_option, data=sweep)
            write_wav_file(
                file_name=inv_filter_string, rate=sample_rate_option, data=inv_filter
            )

            play_sweep(sweep_string)


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
