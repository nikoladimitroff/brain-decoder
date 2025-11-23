import json
import subprocess
import numpy as np
from pathlib import Path
from entries import *


def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')


def extract_braindata_for_stimulus(brain_data, stimulus: StimulusEntry) -> list[BrainDataEntry]:
    entries = []
    for entry_idx in range(len(brain_data["time"])):
        time_point = brain_data["time"][entry_idx]
        if time_point > stimulus.end_time:
            break
        if time_point < stimulus.start_time:
            continue
        data_entry = BrainDataEntry(brain_data, entry_idx)
        entries.append(data_entry)
    return entries


cache_dir = Path(getGitRoot()) / "data/cache"
cache_dir.mkdir(exist_ok=True)

def extract_activation_for_stimulus(session_data: SessionData, stimulus: StimulusEntry) -> list[ConceptActivationEntry]:
    # Create cache filename
    cache_filename = f"{session_data.idx}_{stimulus.label}.json"
    cache_path = cache_dir / cache_filename
    
    # Check if cached file exists
    if cache_path.exists():
        print(f"Loading cached data from {cache_filename}")
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
            return [ConceptActivationEntry.from_dict(entry) for entry in cached_data]
    
    # If no cache, compute normally
    print(f"Computing activation data for session {session_data.idx}, stimulus '{stimulus.label}'")
    entries = extract_braindata_for_stimulus(session_data.brain_data, stimulus)
    max_x = session_data.max_x
    max_y = session_data.max_y
    max_z = session_data.max_z
    max_t_brain = session_data.max_t
    max_t_stimuli = session_data.stimulus_data[-1].end_time
    print(f"Dimensions: x={max_x}, y={max_y}, z={max_z}, t_brain={max_t_brain}, t_stimuli={max_t_stimuli}\n")

    activation_entries = []
    for entry in entries:
        for (x, y, z) in zip(range(max_x), range(max_y), range(max_z)):
            hbo_value = entry.get_hbo(x, y, z)
            hbr_value = entry.get_hbr(x, y, z)
            if hbo_value != 0.0 or hbr_value != 0.0:
                activation_entries.append(ConceptActivationEntry(x, y, z, hbo_value, hbr_value))
    
    # Save to cache
    print(f"Saving {len(activation_entries)} entries to cache: {cache_filename}")
    with open(cache_path, 'w') as f:
        json.dump([entry.to_dict() for entry in activation_entries], f, separators=(',', ':'))
    
    return activation_entries


def parse_session_protocol(session_path: Path) -> list[StimulusEntry]:
    protocol_file = list(session_path.glob("*.jsonl"))[0]
    events = []
    with open(protocol_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            entry = json.loads(line)
            if entry["event"]["event_type"] in ['stimulus']:
                stimulus = entry["event"]["label"]
                # brain data time is in seconds, protocol time is in milliseconds
                start_time = float(entry["event"]["start_ms"]) / 1000.0
                end_time = float(entry["event"]["end_ms"]) / 1000.0
                #wordcloud_file = entry["event"]["file"]
                events.append(StimulusEntry(stimulus, start_time, end_time))#, wordcloud_file))

            #if entry["event"]["event_type"] in ['stimulus', 'fixation']:
            #    stimulus = entry["event"]["label"]
            #    start_time = entry["event"]["start_ms"]
            #    end_time = entry["event"]["end_ms"]
            #    wordcloud_file = entry["event"]["file"]
            #    events.append({"stimulus": stimulus, "start_time": start_time, "end_time": end_time, "wordcloud_file": wordcloud_file})
    return events
    

def parse_all_sessions() -> list[SessionData]:
    data_dir = Path(getGitRoot()) / "data/lys_trial_data"
    session_paths = data_dir.glob("session-*/")
    sessions = []
    for idx, session in enumerate(session_paths):
        brain_data = np.load(session / "reconstruction_data.npz")
        protocol = parse_session_protocol(session)
        sessions.append(SessionData(idx, brain_data, protocol))
    return sessions
