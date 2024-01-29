import scipy.io
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np


##################### CONSTANTS #####################
SAMPLING_FREQUENCY = 2148  # Hz


##################### READ DATA #####################
def read_mat_file_one_sensor(file_path: str) -> np.ndarray:
    """
    Read a MATLAB file and return the EMG data for one sensor.

    Args:
        file_path: Path to the .mat file.

    Returns:
        EMG data array for the sensor.
    """
    data = scipy.io.loadmat(file_path)
    emg_signal = np.array(data["Trigno_Avanti_Data"][0][0][0])  # Convert to NumPy array
    return emg_signal


def read_mat_file_two_sensors(file_path: str) -> tuple:
    """
    Read a MATLAB file and return the EMG data for two sensors.

    Args:
        file_path: Path to the .mat file.

    Returns:
        A tuple containing EMG data arrays for the two sensors.
    """
    data = scipy.io.loadmat(file_path)
    emg_sensor1 = np.array(data["Trigno_Avanti_Data"][0][0][0])
    emg_sensor2 = np.array(data["Trigno_Avanti_Data"][0][2][0])
    return emg_sensor1, emg_sensor2


##################### PROCESS DATA #####################


def process_data(emg_data: np.ndarray) -> np.ndarray:
    """
    Process the EMG data.
    """
    # Example: Normalize data
    processed_data = emg_data / emg_data.max()
    return processed_data


def compute_median_frequency(emg_signal: np.ndarray) -> float:
    """
    Compute the median frequency of the EMG signal.

    Args:
        emg_signal: The EMG signal.

    Returns:
        The median frequency of the EMG signal.
    """
    fft_values = np.fft.rfft(emg_signal)
    power_spectrum = np.abs(fft_values) ** 2
    total_power = np.sum(power_spectrum)
    cumulative_power = np.cumsum(power_spectrum)
    median_index = np.where(cumulative_power >= total_power / 2)[0][0]
    fft_frequency_bins = np.fft.rfftfreq(len(emg_signal), d=1 / SAMPLING_FREQUENCY)
    return fft_frequency_bins[median_index]


def compute_sliding_window_median_frequency(
    emg_data: np.ndarray, window_size: int, step_size: int = 1
) -> np.ndarray:
    """
    Compute the median frequency for each sliding window of the EMG signal.

    Args:
        emg_data: The EMG signal.
        window_size: The size of the sliding window.
        step_size: The step size between each window.

    Returns:
        The median frequency for each sliding window of the EMG signal.
    """
    median_frequencies = []
    for start in range(0, len(emg_data) - window_size + 1, step_size):
        window_data = emg_data[start : start + window_size]
        median_freq = compute_median_frequency(window_data)
        median_frequencies.append(median_freq)
    return np.array(median_frequencies)


##################### PLOT DATA #####################


def create_plotly_graphs(folder_path: str):
    """
    Create Plotly graphs for each exercise in the folder, for two sensors,
    with an empty row after each exercise for separation.
    """
    exercises = [file for file in os.listdir(folder_path) if file.endswith(".mat")]

    # Custom sort function to sort by the numerical part of the file name
    def sort_key(name: str) -> int:
        return int(name.split("_")[1].split(".")[0])

    # Sort exercises by numerical part of file name
    exercises.sort(key=sort_key)

    rows = len(exercises) * 3
    fig = make_subplots(
        rows=rows,
        cols=3,
        subplot_titles=["EMG Data", "Processed Data", "Median Frequency"],
        vertical_spacing=0.001,  # Adjusted spacing between rows
    )

    for i, exercise in enumerate(exercises, start=1):
        emg_sensor1, emg_sensor2 = read_mat_file_two_sensors(
            os.path.join(folder_path, exercise)
        )
        for sensor_index, emg_data in enumerate([emg_sensor1, emg_sensor2], start=1):
            row = (i - 1) * 3 + sensor_index  # Adjust row index for the empty row

            processed_data = process_data(emg_data)
            median_frequencies = compute_sliding_window_median_frequency(
                emg_data, window_size=1000
            )

            # Plot raw EMG data
            fig.add_trace(
                go.Scatter(
                    y=emg_data,
                    mode="lines",
                    name=f"Exercise {i} Sensor {sensor_index} Raw",
                ),
                row=row,
                col=1,
            )

            # Plot processed EMG data
            fig.add_trace(
                go.Scatter(
                    y=processed_data,
                    mode="lines",
                    name=f"Exercise {i} Sensor {sensor_index} Processed",
                ),
                row=row,
                col=2,
            )

            # Plot median frequency
            fig.add_trace(
                go.Scatter(
                    y=median_frequencies,
                    mode="lines",
                    name=f"Exercise {i} Sensor {sensor_index} Median Frequency",
                ),
                row=row,
                col=3,
            )

    fig.update_layout(height=200 * rows, title_text="Workout Data Analysis")
    fig.show()


# Usage
create_plotly_graphs("workout_data")
