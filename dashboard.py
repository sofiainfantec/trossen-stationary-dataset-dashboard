from pathlib import Path
from datetime import timedelta, datetime
import os

import pandas as pd
import streamlit as st
import plotly.express as px


DEFAULT_DATASET_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "lerobot"
    / "YourUser"
    / "trossen_ai_stationary_pro1"
)

DATASET_PATH = Path(
    os.getenv("DATASET_PATH", str(DEFAULT_DATASET_PATH))
)

OUTPUT_FILE = (
    Path.home()
    / "Documents"
    / "robotics_reports"
    / "stationary_dataset_summary.csv"
)


def format_duration(seconds):
    if pd.isna(seconds):
        return "N/A"
    return str(timedelta(seconds=round(seconds)))


def get_duration_from_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)

    if "timestamp" in df.columns:
        return df["timestamp"].max() - df["timestamp"].min()

    if "time" in df.columns:
        return df["time"].max() - df["time"].min()

    return None


def get_camera_status(videos_path, episode):
    camera_dirs = sorted((videos_path / "chunk-000").glob("observation.images.*"))

    camera_status = {}
    video_count = 0

    for camera_dir in camera_dirs:
        camera_name = camera_dir.name.replace("observation.images.", "")
        video_file = camera_dir / f"{episode}.mp4"

        exists = video_file.exists()
        camera_status[camera_name] = exists

        if exists:
            video_count += 1

    return camera_status, video_count, len(camera_dirs)


@st.cache_data
def load_dataset(dataset_path):
    data_path = dataset_path / "data"
    videos_path = dataset_path / "videos"

    parquet_files = sorted(data_path.glob("chunk-*/*.parquet"))

    records = []

    for parquet_file in parquet_files:
        episode = parquet_file.stem
        duration = get_duration_from_parquet(parquet_file)

        if duration is not None:
            modified_time = datetime.fromtimestamp(parquet_file.stat().st_mtime)
            estimated_end_time = modified_time
            estimated_start_time = modified_time - timedelta(seconds=float(duration))

            camera_status, video_count, expected_camera_count = get_camera_status(
                videos_path,
                episode,
            )

            record = {
                "episode": episode,
                "duration_seconds": float(duration),
                "duration": format_duration(duration),
                "estimated_start_time": estimated_start_time,
                "estimated_end_time": estimated_end_time,
                "video_count": video_count,
                "expected_camera_count": expected_camera_count,
                "camera_complete": video_count == expected_camera_count,
            }

            for camera_name, exists in camera_status.items():
                record[f"camera_{camera_name}"] = exists

            records.append(record)

    df = pd.DataFrame(records)

    if not df.empty:
        df = df.sort_values("estimated_start_time").reset_index(drop=True)

        df["previous_episode"] = df["episode"].shift(1)
        df["previous_end_time"] = df["estimated_end_time"].shift(1)

        df["setup_time_seconds"] = (
            df["estimated_start_time"] - df["previous_end_time"]
        ).dt.total_seconds()

        df.loc[df["setup_time_seconds"] < 0, "setup_time_seconds"] = 0
        df["setup_time"] = df["setup_time_seconds"].apply(format_duration)

    return df


st.set_page_config(
    page_title="Trossen Stationary Dataset Dashboard",
    layout="wide",
)

st.title("Trossen Stationary Dataset Dashboard")

df = load_dataset(DATASET_PATH)

if df.empty:
    st.warning("No data found. Check DATASET_PATH.")
    st.stop()


if st.button("Export CSV"):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    st.success(f"CSV exported to: {OUTPUT_FILE}")


total_recordings = len(df)
total_time = df["duration_seconds"].sum()
median_duration = df["duration_seconds"].median()

longest = df.loc[df["duration_seconds"].idxmax()]
shortest = df.loc[df["duration_seconds"].idxmin()]

valid_setup_df = df.dropna(subset=["setup_time_seconds"])

avg_setup_time = valid_setup_df["setup_time_seconds"].mean()
total_setup_time = valid_setup_df["setup_time_seconds"].sum()
total_session_time = total_time + total_setup_time

recording_utilization = (
    total_time / total_session_time * 100
    if total_session_time > 0
    else 0
)

complete_camera_episodes = df["camera_complete"].sum()
camera_completeness_rate = complete_camera_episodes / total_recordings * 100


st.subheader("Recording Summary")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Recordings", total_recordings)
col2.metric("Total Recording Time", format_duration(total_time))
col3.metric("Typical Duration", format_duration(median_duration))
col4.metric("Longest Episode", longest["duration"])
col5.metric("Shortest Episode", shortest["duration"])

st.divider()


st.subheader("Operational Efficiency")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Setup Time", format_duration(avg_setup_time))
col2.metric("Total Setup Time", format_duration(total_setup_time))
col3.metric("Total Session Time", format_duration(total_session_time))
col4.metric("Recording Utilization", f"{recording_utilization:.1f}%")

st.divider()


st.subheader("Camera Completeness")

col1, col2, col3 = st.columns(3)

col1.metric("Expected Cameras", int(df["expected_camera_count"].max()))
col2.metric("Complete Episodes", int(complete_camera_episodes))
col3.metric("Camera Completeness Rate", f"{camera_completeness_rate:.1f}%")

camera_columns = [
    col for col in df.columns
    if col.startswith("camera_") and col != "camera_complete"
]

camera_summary = []

for camera_col in camera_columns:
    camera_summary.append({
        "camera": camera_col.replace("camera_", ""),
        "episodes_with_video": int(df[camera_col].sum()),
        "total_episodes": total_recordings,
        "completeness_rate": f"{df[camera_col].mean() * 100:.1f}%",
    })

st.dataframe(
    pd.DataFrame(camera_summary),
    use_container_width=True,
)

st.divider()


st.subheader("Episode Duration Distribution")

fig = px.histogram(
    df,
    x="duration_seconds",
    nbins=20,
    title="Distribution of Episode Duration",
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("Longest Episodes")

st.dataframe(
    df.sort_values("duration_seconds", ascending=False)
    [["episode", "duration"]]
    .head(10),
    use_container_width=True,
)


st.subheader("Shortest Episodes")

st.dataframe(
    df.sort_values("duration_seconds", ascending=True)
    [["episode", "duration"]]
    .head(10),
    use_container_width=True,
)


st.subheader("Longest Setup Times")

st.dataframe(
    valid_setup_df.sort_values("setup_time_seconds", ascending=False)
    [["previous_episode", "episode", "setup_time"]]
    .head(10),
    use_container_width=True,
)


st.subheader("Shortest Setup Times")

st.dataframe(
    valid_setup_df.sort_values("setup_time_seconds", ascending=True)
    [["previous_episode", "episode", "setup_time"]]
    .head(10),
    use_container_width=True,
)


st.subheader("Episodes With Missing Camera Videos")

missing_camera_df = df[df["camera_complete"] == False]

if missing_camera_df.empty:
    st.success("All episodes have videos for every detected camera.")
else:
    display_cols = ["episode", "video_count", "expected_camera_count"] + camera_columns

    st.dataframe(
        missing_camera_df[display_cols],
        use_container_width=True,
    )


st.subheader("All Episodes")

all_episode_columns = [
    "episode",
    "duration",
    "setup_time",
    "estimated_start_time",
    "estimated_end_time",
    "video_count",
    "expected_camera_count",
    "camera_complete",
] + camera_columns

st.dataframe(
    df[all_episode_columns],
    use_container_width=True,
)
