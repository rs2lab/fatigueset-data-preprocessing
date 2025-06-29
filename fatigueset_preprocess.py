#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import multiprocessing as mp

from glob import glob


INTERESTING_SENSORS_KIND = {
    'ear_ppg_left.csv': ['green', 'ir', 'red'],
    'ear_ppg_right.csv': ['green', 'ir', 'red'],
    'forehead_eeg_alpha_abs.csv': ['TP9', 'AF7', 'AF8', 'TP10'],
    'forehead_eeg_beta_abs.csv': ['TP9', 'AF7', 'AF8', 'TP10'],
    'forehead_eeg_delta_abs.csv': ['TP9', 'AF7', 'AF8', 'TP10'],
    'forehead_eeg_gamma_abs.csv': ['TP9', 'AF7', 'AF8', 'TP10'],
    'forehead_eeg_raw.csv': ['TP9', 'AF7', 'AF8', 'TP10'],
    'forehead_eeg_theta_abs.csv': ['TP9', 'AF7', 'AF8', 'TP10'],
    'chest_raw_breathing.csv': ['breathing_waveform'],
    'chest_raw_ecg.csv': ['ecg_waveform'], 
    'wrist_eda.csv': ['eda'], # Eletrodermal activity
    'wrist_skin_temperature.csv': ['temp'],
    'wrist_bvp.csv': ['bvp'], # PPG Blood volume pulse
}

STATS = ['min', 'max', 'std', 'mean', 'median']


def get_sliding_secs_stats(df: pd.DataFrame, columns: list, secs: int) -> pd.DataFrame:
    """
    Get tumbling window statistics for a given DataFrame using pandas resample.
    This is more efficient and robust than the manual loop implementation.
    
    Args:
        df: pandas DataFrame containing the data, MUST have a DatetimeIndex.
        columns: list of column names to calculate statistics for
        secs: size of the window in seconds
    """
    # Create the aggregation dictionary dynamically
    # {'TP9': ['min', 'max', ...], 'AF7': ['min', 'max', ...]}
    agg_dict = {col: STATS for col in columns}
    
    # Resample creates time-based bins (windows) and applies aggregations to each.
    # The '.T.stack().to_frame().T' part is a trick to flatten the multi-level columns.
    windowed_df = df[columns].resample(f'{secs}s').agg(agg_dict)
    
    # Flatten the multi-level column index ('TP9'/'min') into a single level ('TP9_min')
    windowed_df.columns = [f'{col}_{stat}' for col, stat in windowed_df.columns]
    
    return windowed_df


def load_user_data_from_path(user_session_path: str, window_secs: int, drop_first_30_secs: bool = True) -> pd.DataFrame:
    """
    Load user data from a given path and return a DataFrame with sliding window statistics.
    This revised version correctly aligns data from all sensors by their timestamps.
    """
    sensor_dfs = []

    for sensor_file, columns in INTERESTING_SENSORS_KIND.items():
        file_path = os.path.join(user_session_path, sensor_file)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime and set as index for resampling
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        # Use the revised, robust function
        df_stats = get_sliding_secs_stats(df, columns, window_secs)
        
        # Add a prefix to distinguish sensor columns
        sensor_prefix = sensor_file.replace('.csv', '')
        df_stats = df_stats.rename(columns=lambda c: f"{sensor_prefix}_{c}")
        sensor_dfs.append(df_stats)

    if not sensor_dfs:
        return pd.DataFrame()

    # NOTE: This automatically handles different start/end times and missing data between sensors.
    # Any timestamp that exists in one DataFrame but not another will result in NaNs, which is correct.
    # TODO: Consider using a different approach to handle missing data, or remove the fields with a high
    # percentage of NaNs.
    user_data = pd.concat(sensor_dfs, axis=1)

    # Sort by time, just in case
    user_data = user_data.sort_index()

    # Handle the case where the DataFrame might be empty after processing.
    if user_data.empty:
        return pd.DataFrame()

    first_timestamp = user_data.index[0]
    user_data["from_start_timestamp_secs"] = (user_data.index - first_timestamp).total_seconds()

    if drop_first_30_secs:
        # Note: The first row represents the [0, window_secs] interval.
        # Its 'from_start_timestamp_secs' will be `window_secs`.
        mask = user_data["from_start_timestamp_secs"] > 30
        user_data = user_data[mask]

    # Reset index to a standard 0,1,2... index for the final CSV
    user_data = user_data.reset_index(drop=True)

    return user_data


def load_user_data_with_psycho_fatigue(user_session_path: str, window_secs: float, fatigue_median: float, drop_first_30_secs: bool = True) -> pd.DataFrame:
    """
    Load user data from a given path and return a DataFrame with psychological fatigue levels included
    
    Args:
        user_session_path: path to the user session directory
        window_secs: size of the sliding window in seconds
        fatigue_median: median of the psychological fatigue scores
        drop_first_30_secs: whether to drop the first 30 seconds of data
    """
    user_df = load_user_data_from_path(user_session_path, window_secs, drop_first_30_secs=drop_first_30_secs)
    user_reported_data = pd.read_csv(os.path.join(user_session_path, "exp_fatigue.csv"))

    mask = user_df['from_start_timestamp_secs'] != user_df['from_start_timestamp_secs']

    user_df["psycho_fatigue"] = 0
    lastMentalFatigueScore = 0
    lastFatigueRecordStartTime = 0

    for row in user_reported_data.itertuples():
        if row.mentalFatigueScore > fatigue_median:
            mask |= (user_df['from_start_timestamp_secs'] > lastFatigueRecordStartTime) & (user_df['from_start_timestamp_secs'] < row.mentalFatigueAnswerTime)

        lastFatigueRecordStartTime = row.mentalFatigueAnswerTime
        lastMentalFatigueScore = row.mentalFatigueScore

    if lastMentalFatigueScore > fatigue_median and lastFatigueRecordStartTime < user_df['from_start_timestamp_secs'].max():
        mask |= user_df['from_start_timestamp_secs'] > lastFatigueRecordStartTime

    user_df.loc[mask, "psycho_fatigue"] = 1 # Set Self reported psychological fatigue levels to 1 = HIGH
    return user_df


def get_user_fatigue_median(user_dir: str) -> float:
    """
    Get the median of the psychological fatigue scores for a given user
    
    Args:
        user_dir: path to the user directory
    """
    user_data_01 = pd.read_csv(os.path.join(user_dir, "01", "exp_fatigue.csv"))
    user_data_03 = pd.read_csv(os.path.join(user_dir, "03", "exp_fatigue.csv"))
    uu = pd.concat([user_data_01, user_data_03], axis=0)
    return uu['mentalFatigueScore'].median()


def process_session(user_path: str, session_id: str, window_secs: int, fatigue_median: float, data_save_dir: str) -> None:
    """
    Process a user session and save the data to a file
    
    Args:
        user_path: path to the user directory
        session_id: session ID (01 or 03)
        window_secs: size of the sliding window in seconds
        fatigue_median: median of the psychological fatigue scores
        data_save_dir: path to save the data
    """
    # TODO: needs to better adapt for windows
    uid = user_path.rstrip('/').split('/')[-1]
    try:
        session_data = load_user_data_with_psycho_fatigue(
            user_session_path=os.path.join(user_path, session_id),
            window_secs=window_secs,
            fatigue_median=fatigue_median,
        )
        session_data['physio_fatigue'] = 0 if session_id == '01' else 1
        file_path = os.path.join(data_save_dir, f'{uid}_{session_id}.csv')
        session_data.to_csv(file_path, index=False)
        print(f"Saved {file_path}")
    except KeyboardInterrupt as e:
        raise
    except Exception as e:
        print(f"Error processing {user_path} {session_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--window_secs", type=int, required=True, help="Window size in seconds")
    parser.add_argument("--drop_first_30_secs", type=bool, required=False, default=True, help="Drop first 30 seconds of data")
    parser.add_argument("--data_save_dir", type=str, required=True, help="Path to save the data")
    parser.add_argument("--num_cores", type=int, default=2, help="Number of CPU cores to use")
    args = parser.parse_args()

    save_dir = os.path.join(args.data_save_dir, f"{args.window_secs}s")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_cores = min(mp.cpu_count(), args.num_cores)

    with mp.Pool(processes=num_cores) as pool:
        tasks = []
        for user_path in glob(os.path.join(args.data_path, '[0-9]*')):
            if not os.path.isdir(user_path):
                continue
            per_user_fatigue_median = get_user_fatigue_median(user_path)

            tasks.append((user_path, '01', args.window_secs, per_user_fatigue_median, save_dir))
            tasks.append((user_path, '03', args.window_secs, per_user_fatigue_median, save_dir))

        # Execute tasks in parallel
        pool.starmap(process_session, tasks)

