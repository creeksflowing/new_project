from __future__ import print_function
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sounddevice as sd
from pydub import AudioSegment
from pydub.silence import split_on_silence
from numba import jit
from scipy import signal
from scipy.io import wavfile


import PyOctaveBand

st.set_option("deprecation.showPyplotGlobalUse", False)

HERE = Path(__file__).parent
AUDIO_FILES_PATH = r"data/audio_files"


def write_wav_file(file_name, rate, data):
    save_file_path = os.path.join(AUDIO_FILES_PATH, file_name)
    wavfile.write(save_file_path, rate, data.astype(np.float32))


def load_wav_file(file_name):
    save_file_path = os.path.join(AUDIO_FILES_PATH, file_name)
    rate, data = wavfile.read(save_file_path)
    return data


def main():
    st.header("Wave(r)")


def app_room_measurements():
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

    def select_option():
        global sample_rate_option
        global sweep_duration_option
        global max_reverb_option
        sample_rate_option = st.selectbox(
            "Select the desired sample rate", (44100, 48000)
        )
        sweep_duration_option = st.selectbox(
            "Select the duration of the sweep", (3, 7, 14)
        )
        max_reverb_option = st.selectbox(
            "Select the expected maximum reverb decay time", (1, 2, 3, 5, 10)
        )

        st.caption(
            """
                    Note that longer sweeps provide more accuracy,
                    but even short sweeps can be used to measure long decays
                    """
        )

    def add_zeros(array, seconds_to_add, sr):
        num_zeros = seconds_to_add * sr
        resulting_array = np.pad(array, (0, num_zeros), "constant")

        return resulting_array

    def remove_silence(wavefile_name):
        read_file_path = os.path.join(AUDIO_FILES_PATH, wavefile_name)
        sound = AudioSegment.from_file(read_file_path, format="wav")
        audio_chunks = split_on_silence(
            sound, min_silence_len=100, silence_thresh=-90, keep_silence=50
        )

        combined = AudioSegment.empty()

        for chunk in audio_chunks:
            combined += chunk

        combined.export(f"./{AUDIO_FILES_PATH}/{wavefile_name}", format="wav")

    def play():

        sweep = generate_exponential_sweep(
            sweep_duration_option,
            sample_rate_option,
            20,
            24000,
        )
        inv_filter = generate_inverse_filter(
            sweep_duration_option,
            sample_rate_option,
            sweep,
            20,
            24000,
        )

        write_wav_file(
            file_name=sweep_string,
            rate=sample_rate_option,
            data=add_zeros(sweep, max_reverb_option + 1, sample_rate_option),
        )
        write_wav_file(
            file_name=inv_filter_string,
            rate=sample_rate_option,
            data=add_zeros(inv_filter, max_reverb_option + 1, sample_rate_option),
        )

        read_file_path = os.path.join(AUDIO_FILES_PATH, sweep_string)
        # Extract data and sampling rate from file
        sample_rate, data = wavfile.read(read_file_path)

        user_sweep = sd.playrec(data, sample_rate, channels=1, blocking=True)

        write_wav_file(
            file_name=user_sweep_string, rate=sample_rate_option, data=user_sweep
        )

        time.sleep(1)

        remove_silence(user_sweep_string)
        remove_silence(inv_filter_string)

        user_sweep = load_wav_file(user_sweep_string)
        inverse_filter = load_wav_file(inv_filter_string)

        write_wav_file(
            file_name=ir_string,
            rate=sample_rate_option,
            data=deconvolve(user_sweep, inverse_filter),
        )

    def plot_waveform(file):
        data = load_wav_file(file)
        sig = np.frombuffer(data, dtype=np.float32)
        sig = sig[:]

        plt.figure(1)

        plot_a = plt.subplot(211)
        plot_a.plot(sig)
        plot_a.set_xlabel("sample rate * time")
        plot_a.set_ylabel("Energy")

        plt.show()

    def plot_spectrogam(file):
        data = load_wav_file(file)
        sig = np.frombuffer(data, dtype=np.float32)
        sig = sig[:]

        plt.figure(1)

        plot_b = plt.subplot(212)
        plot_b.specgram(sig, NFFT=1024, Fs=sample_rate_option, noverlap=900)
        plot_b.set_xlabel("Time")
        plot_b.set_ylabel("Frequency")

        plt.show()

    def plot_spectrum(file):
        data = load_wav_file(file)
        # Sample rate and duration
        duration = sweep_duration_option  # In seconds

        # Time array
        x = np.arange(np.round(sample_rate_option * duration)) / sample_rate_option

        # Filter (only octave spectra)
        spl, freq = PyOctaveBand.octavefilter(
            data, fs=sample_rate_option, fraction=3, order=6, limits=[12, 20000], show=1
        )

        # Filter (get spectra and signal in bands)
        splb, freqb, xb = PyOctaveBand.octavefilter(
            data,
            fs=sample_rate_option,
            fraction=3,
            order=6,
            limits=[12, 20000],
            show=0,
            sigbands=1,
        )

        fig, ax = plt.subplots()
        ax.semilogx(freq, spl, "b")
        ax.grid(which="major")
        ax.grid(which="minor", linestyle=":")
        ax.set_xlabel(r"Frequency [Hz]")
        ax.set_ylabel("Level [dB]")
        plt.xlim(11, 25000)
        ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        ax.set_xticklabels(["63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
        plt.show()

    def user_input():
        global sweep_string
        global inv_filter_string
        global user_sweep_string
        global ir_string
        user_input = st.text_input("Name your file: ")

        if user_input:
            sweep_string = user_input + "_exponential_sweep.wav"
            inv_filter_string = user_input + "_inverse_filter.wav"
            user_sweep_string = user_input + "_user_exponential_sweep.wav"
            ir_string = user_input + "_impulse_response.wav"

            play_button = st.button("Play")

            if "play_button_state" not in st.session_state:
                st.session_state.play_button_state = False

            if play_button or st.session_state.play_button_state:
                st.session_state.play_button_state = True

                play()

                st.pyplot(plot_waveform(ir_string))
                st.pyplot(plot_spectrogam(ir_string))
                st.pyplot(plot_spectrum(ir_string))

    select_option()
    user_input()


if __name__ == "__main__":
    import os

    main()
    app_room_measurements()
