import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from typing import Dict, Callable
from .utils import filename, str2bool, write_srt


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir", type=str,
        help="Path to directory containing video files to transcribe (processed recursively)"
    )
    parser.add_argument(
        "--ext", type=str, default=".avi",
        help="File extension to search for (e.g. .avi or avi)"
    )
    parser.add_argument(
        "--model", default="small",
        choices=whisper.available_models(),
        help="Name of the Whisper model to use"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str,
        default=".", help="Directory to save the SRT files"
    )
    parser.add_argument(
        "--verbose", type=str2bool, default=False,
        help="Whether to print out Whisper progress/debug messages"
    )
    parser.add_argument(
        "--task", type=str, default="transcribe",
        choices=["transcribe", "translate"],
        help="Perform X->X transcription or X->English translation"
    )
    parser.add_argument(
        "--language", type=str, default="auto",
        choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs",
                 "ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo",
                 "fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is",
                 "it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt",
                 "lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn",
                 "no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl",
                 "sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl",
                 "tr","tt","uk","ur","uz","vi","yi","yo","zh"],
        help="Origin language of the video, or auto-detect"
    )
    # --- Timing control (configurable) ---
    parser.add_argument(
        "--reading_speed", type=float, default=15.0,
        help="Reading speed in characters per second for retiming long segments"
    )
    parser.add_argument(
        "--min_duration", type=float, default=1.0,
        help="Minimum duration (seconds) when adjusting long segments"
    )
    parser.add_argument(
        "--max_duration", type=float, default=6.0,
        help="Maximum duration (seconds) when adjusting long segments"
    )
    parser.add_argument(
        "--long_threshold", type=float, default=6.0,
        help="Only segments longer than this (seconds) will be adjusted"
    )

    args = parser.parse_args().__dict__

    input_dir: str = args.pop("input_dir")
    file_ext_raw: str = args.pop("ext").lower()
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    language: str = args.pop("language")

    # Timing params
    reading_speed: float = args.pop("reading_speed")
    min_duration: float = args.pop("min_duration")
    max_duration: float = args.pop("max_duration")
    long_threshold: float = args.pop("long_threshold")

    # Normalize extension (accept ".avi" or "avi")
    file_ext = file_ext_raw if file_ext_raw.startswith(".") else f".{file_ext_raw}"

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection."
        )
        args["language"] = "en"
    elif language != "auto":
        args["language"] = language

    # Load Whisper model
    model = whisper.load_model(model_name)

    # Collect all files matching extension
    video_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(file_ext):
                video_files.append(os.path.join(root, f))

    if not video_files:
        print(f"No files with extension '{file_ext}' found in {input_dir}")
        return

    # Extract audio for each video
    audios = get_audio(video_files)

    # Generate SRTs
    get_subtitles(
        audio_paths=audios,
        output_dir=output_dir,
        base_dir=input_dir,
        transcribe=lambda audio_path: model.transcribe(audio_path, **args),
        reading_speed=reading_speed,
        min_duration=min_duration,
        max_duration=max_duration,
        long_threshold=long_threshold,
    )


def get_audio(paths):
    temp_dir = tempfile.gettempdir()
    audio_paths: Dict[str, str] = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths


def adjust_segments_threshold_right_anchored(
    segments,
    *,
    reading_speed: float,
    min_duration: float,
    max_duration: float,
    long_threshold: float
):
    """
    Retimes only segments whose duration exceeds `long_threshold`.
    For those, we estimate duration from reading speed and ANCHOR TO THE SEGMENT'S END:
        new_start = min(end, next_start) - estimated_duration
        new_end   = min(end, next_start)
    We also ensure no overlap with the previous segment's end.
    """
    adjusted = []
    prev_end = 0.0
    EPS = 0.001  # 1 ms safety to avoid zero/negative durations

    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        text = (seg.get("text") or "").strip()

        next_start = float(segments[i + 1]["start"]) if i + 1 < len(segments) else float("inf")
        duration = end - start

        if duration > long_threshold:
            # Cap the effective right boundary by next segment's start
            right_edge = min(end, next_start)

            # Estimate desired duration from reading speed, clamped
            est_dur = len(text) / max(reading_speed, 1e-6)
            est_dur = max(min_duration, min(est_dur, max_duration))

            # Anchor to the right edge
            new_end = right_edge
            new_start = right_edge - est_dur

            # Respect previous segment end and timeline start
            new_start = max(new_start, prev_end)

            # If there's no room, collapse to a tiny duration instead of overlapping
            if new_end - new_start < EPS:
                new_start = max(prev_end, new_end - min_duration)
                if new_end - new_start < EPS:
                    new_start = max(prev_end, new_end - EPS)

            start, end = new_start, new_end
        else:
            # Keep original, but still prevent overlap with previous due to prior adjustments
            if start < prev_end:
                start = prev_end
                if end < start + EPS:
                    end = start + EPS

            # Also ensure we don't run into the next segment start
            if end > next_start:
                end = next_start
                if end < start + EPS:
                    start = max(prev_end, end - EPS)

        adjusted.append({
            "start": start,
            "end": end,
            "text": text
        })
        prev_end = end

    return adjusted


def get_subtitles(
    audio_paths: Dict[str, str],
    output_dir: str,
    base_dir: str,
    transcribe: Callable[[str], dict],
    *,
    reading_speed: float,
    min_duration: float,
    max_duration: float,
    long_threshold: float,
):
    for path, audio_path in audio_paths.items():
        # Build output filename from the path relative to base_dir, lowercased, with separators -> "_"
        rel_path = os.path.relpath(path, start=base_dir)
        normalized = rel_path.replace("\\", "_").replace("/", "_").lower()
        srt_filename = f"{normalized}.srt"
        srt_path = os.path.join(output_dir, srt_filename)

        print(f"Generating subtitles for {filename(path)}... This might take a while.")

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        # Adjust only long segments, anchoring to their end (fixes 0→20s type cases to (20-D)→20)
        result["segments"] = adjust_segments_threshold_right_anchored(
            result["segments"],
            reading_speed=reading_speed,
            min_duration=min_duration,
            max_duration=max_duration,
            long_threshold=long_threshold,
        )

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)

        print(f"Saved SRT to {os.path.abspath(srt_path)}")


if __name__ == '__main__':
    main()
