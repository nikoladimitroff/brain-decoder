import json
import subprocess
from matplotlib.pylab import cast
import numpy as np
from pathlib import Path
from entries import *


def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')


def extract_activations_for_stimulus_2(session_data: SessionData, stimulus: StimulusEntry) -> list[ConceptActivationEntry]:
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

    activations = []
    for (x_idx, y_idx, z_idx, t_idx) in zip(x, y, z, t):
        hbo_value = cast(np.float64, hbo[x_idx][y_idx][z_idx][t_idx])
        hbr_value = cast(np.float64, hbr[x_idx][y_idx][z_idx][t_idx])
        activations.append(ConceptActivationEntry(int(x_idx), int(y_idx), int(z_idx), hbo_value, hbr_value))
    return activations

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
