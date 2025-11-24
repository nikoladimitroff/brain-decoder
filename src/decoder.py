import json
import subprocess
from matplotlib.pylab import cast
import numpy as np
from pathlib import Path
from entries import *


def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')


# Implement this in whatever fashion required to average signals
def average_signal_in_same_stimulus(signal_values: np.ndarray) -> np.float64:
    if len(signal_values) == 0:
        return np.float64(0.0)
    return cast(np.float64, np.average(signal_values))

def average_signal_across_stimuli(signal_values: np.ndarray) -> np.float64:
    if len(signal_values) == 0:
        return np.float64(0.0)
    return cast(np.float64, np.average(signal_values))

# Implement this to define how important changes are from rested to the stimulus
def signal_delta(stimulus_value: np.float64, rest_value: np.float64) -> np.float64:
    # todo: may be take the sqrt to emphasize changes more?
    return (stimulus_value - rest_value)

def extract_average_stimulus_delta(sessions: list[SessionData], stimulus_label: str) -> list[ConceptActivationEntry]:
    aggregated_activations: dict[tuple[int, int, int], list[ConceptActivationEntry]] = {}
    for session in sessions:
        print(f"Searching session {session.idx} for stimulus '{stimulus_label}'")
        for stimulus_idx, entry in enumerate(session.stimulus_data):
            if entry.label != stimulus_label:
                continue
            print(f"  Found stimulus at index {stimulus_idx} ({entry})")
            stimulus_deltas = extract_stimulus_delta(session, stimulus_idx)
            for entry in stimulus_deltas:
                coords = (entry.x, entry.y, entry.z)
                if aggregated_activations.get(coords) is None:
                    aggregated_activations[coords] = []
                aggregated_activations[coords].append(entry)

    assert(len(aggregated_activations) > 0, f"No activations found for stimulus label '{stimulus_label}' across all sessions")

    # Now average the activations across sessions
    averaged_activations: list[ConceptActivationEntry] = []
    for coords, entries in aggregated_activations.items():
        avg_hbo = average_signal_across_stimuli(np.array([e.hbo for e in entries]))
        avg_hbr = average_signal_across_stimuli(np.array([e.hbr for e in entries]))
        averaged_activations.append(ConceptActivationEntry(coords[0], coords[1], coords[2], avg_hbo, avg_hbr))

    assert(len(averaged_activations) > 0, "No activations found for the given stimulus label")
    return averaged_activations


__default_entry = ConceptActivationEntry(0,0,0, np.float64(0.0), np.float64(0.0))
def extract_stimulus_delta(session_data: SessionData, stimulus_idx: int) -> list[ConceptActivationEntry]:
    t = session_data.brain_data["time"]
    # get the indices corresponding to the stimulus time range
    # nonzero returns a tuple, we only care about the first step
    fixation = session_data.stimulus_data[stimulus_idx - 1]
    assert(fixation.label == "rest")
    stimulus = session_data.stimulus_data[stimulus_idx]
    stimulus_activations = extract_activations_for_stimulus(session_data, stimulus)
    fixation_activations = extract_activations_for_stimulus(session_data, fixation)

    def entry_diff(coords: tuple[int, int, int], stimul_entry: ConceptActivationEntry) -> ConceptActivationEntry:
        rest_entry = fixation_activations.get(coords, __default_entry)
        return ConceptActivationEntry(coords[0], coords[1], coords[2],
                                      signal_delta(stimul_entry.hbo, rest_entry.hbo),
                                      signal_delta(stimul_entry.hbr, rest_entry.hbr))

    diff_activations = []
    for entry in stimulus_activations.values():
        coords = (entry.x, entry.y, entry.z)
        diff = entry_diff(coords, entry)
        if diff.hbo != 0.0 or diff.hbr != 0.0:
            diff_activations.append(diff)
    return diff_activations


def extract_activations_for_stimulus(session_data: SessionData, stimulus: StimulusEntry) -> dict[tuple[int, int, int],ConceptActivationEntry]:
    t = session_data.brain_data["time"]
    # get the indices corresponding to the stimulus time range
    # nonzero returns a tuple, we only care about the first step
    valid_indices = np.nonzero((t >= stimulus.start_time) & (t <= stimulus.end_time))[0]
    hbo = session_data.brain_data["hbo"]
    # mask the hbo data to only include the valid time indicess
    timed_hbo = hbo[:,:,:,valid_indices]
    hbr = session_data.brain_data["hbr"]
    # mask the timed hbo data to only include non-zero entries
    nonzero_hbo = np.nonzero(timed_hbo)
    # extract the indices along each dimension
    x = nonzero_hbo[0]
    y = nonzero_hbo[1]
    z = nonzero_hbo[2]
    # Note: since we masked the time dimension, nonzero_hbo[3] will by definition contain only
    # 0...len(valid_indices)-1, so we need to map it back to the original time indices
    t = nonzero_hbo[3] + valid_indices[0]

    all_activations: dict[tuple[int, int, int], list[tuple[np.float64, np.float64]]] = {}
    for (x_idx, y_idx, z_idx, t_idx) in zip(x, y, z, t):
        hbo_value = cast(np.float64, hbo[x_idx][y_idx][z_idx][t_idx])
        hbr_value = cast(np.float64, hbr[x_idx][y_idx][z_idx][t_idx])
        if all_activations.get((int(x_idx), int(y_idx), int(z_idx))) is None:
            all_activations[(int(x_idx), int(y_idx), int(z_idx))] = []
        all_activations[(int(x_idx), int(y_idx), int(z_idx))].append((hbo_value, hbr_value))
    
    activations: dict[tuple[int, int, int], ConceptActivationEntry] = {}
    for coords, values in all_activations.items():
        hbo_values = np.array([v[0] for v in values])
        hbr_values = np.array([v[1] for v in values])
        avg_hbo = average_signal_in_same_stimulus(hbo_values)
        avg_hbr = average_signal_in_same_stimulus(hbr_values)
        activations[coords] = ConceptActivationEntry(coords[0], coords[1], coords[2], avg_hbo, avg_hbr)
    return activations

def parse_session_protocol(session_path: Path) -> list[StimulusEntry]:
    protocol_file = list(session_path.glob("*.jsonl"))[0]
    events = []
    with open(protocol_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            entry = json.loads(line)
            if entry["event"]["event_type"] in ["stimulus", "fixation"]:
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
