from pathlib import Path
import json
import sys
import numpy as np

class StimulusEntry:
    def __init__(self, label: str, start_time: float, end_time: float):#, wordcloud_file: str):
        self.label = label
        self.start_time = start_time
        self.end_time = end_time
        #self.wordcloud_file = wordcloud_file

    @property
    def duration(self):
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"{self.label}: start_time={self.start_time / 1000:.2f}, duration={self.duration / 1000:.2f}"


class BrainDataEntry:
    def __init__(self, session_data, entry_idx: int):
        self.session_data = session_data
        self.entry_idx = entry_idx

    @property
    def time(self):
        return self.session_data["time"][self.entry_idx]
    
    def get_hbo(self, x: int, y: int, z: int):
        return self.session_data["hbo"][x][y][z][self.entry_idx]

    def get_hbr(self, x: int, y: int, z: int):
        return self.session_data["hbr"][x][y][z][self.entry_idx]


def parse_session_protocol(session_path: Path):
    protocol_file = list(session_path.glob("*.jsonl"))[0]
    events = []
    with open(protocol_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            entry = json.loads(line)
            if entry["event"]["event_type"] in ['stimulus']:
                stimulus = entry["event"]["label"]
                start_time = float(entry["event"]["start_ms"])
                end_time = float(entry["event"]["end_ms"])
                #wordcloud_file = entry["event"]["file"]
                events.append(StimulusEntry(stimulus, start_time, end_time))#, wordcloud_file))

            #if entry["event"]["event_type"] in ['stimulus', 'fixation']:
            #    stimulus = entry["event"]["label"]
            #    start_time = entry["event"]["start_ms"]
            #    end_time = entry["event"]["end_ms"]
            #    wordcloud_file = entry["event"]["file"]
            #    events.append({"stimulus": stimulus, "start_time": start_time, "end_time": end_time, "wordcloud_file": wordcloud_file})
    return events
    

def get_glove_embedding(stimulus, embeddings_dict):
    return embeddings_dict[stimulus]
    

def get_embeddings_dict(glove_path, embedding_dim=300):
    """
    Download the GloVe embeddings from:
        wget http://nlp.stanford.edu/data/glove.840B.300d.zip
        unzip glove.840B.300d.zip
    """
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ', embedding_dim)
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def map_concept_braindata(session_data, stimulus: StimulusEntry):
    entries = []
    for entry_idx in range(len(session_data["time"])):
        time_point = session_data["time"][entry_idx]
        if time_point > stimulus.end_time:
            break
        if time_point < stimulus.start_time:
            continue
        data_entry = BrainDataEntry(session_data, entry_idx)
        entries.append(data_entry)
    return entries

if __name__ == '__main__':
    data_dir = Path('C:\\Dev\\brain-decoder\\data\\lys_trial_data')
    session_paths = data_dir.glob("session-*/")
    sessions = {}
    stimuli = []
    for session in session_paths:
        session_idx = int(session.name.split("\\")[-1].split("-")[1])
        brain_data = np.load(session / "reconstruction_data.npz")
        protocol = parse_session_protocol(session)
        sessions[session_idx] = {
            "brain_data": brain_data,
            "protocol": protocol
        }


    for (session_idx, session_data) in sessions.items():
        print(session_data["protocol"])
        break
    
    # keys: hbo, hbr, time
    for (session_idx, session_data) in sessions.items():
        with open('output.txt', 'w') as f:
            np.set_printoptions(threshold=sys.maxsize)
            
            max_x = len(session_data["brain_data"]["hbo"])
            max_y = len(session_data["brain_data"]["hbo"][0])
            max_z = len(session_data["brain_data"]["hbo"][0][0])

            print(f"Dimensions: x={max_x}, y={max_y}, z={max_z}", file=f)
            for entry_idx in range(10, 11):
                brain_data_entry = BrainDataEntry(session_data["brain_data"], entry_idx)

                for x in range(20, 22):
                    for y in range(20, 22):
                        for z in range(20, 22):
                            hbo_value = brain_data_entry.get_hbo(x, y, z)
                            hbr_value = brain_data_entry.get_hbr(x, y, z)
                            print(f"Entry idx: {entry_idx} [{x}, {y}, {z}] - HBO: {hbo_value}, HBR: {hbr_value}", file=f)

                
                #print(f"# entries: {len(session_data["brain_data"]["time"])}")
                #print(f"# entries: {len(session_data["brain_data"]["hbo"])}")
                #print(f"# entries: {len(session_data["brain_data"]["hbr"])}")
                #print(f"HBO length: {len(session_data["brain_data"]["hbo"][entry_idx])}")
                #print(f"HBO[0] length: {len(session_data["brain_data"]["hbo"][entry_idx][0])}")
                #print(f"HBO[0][0] length: {len(session_data["brain_data"]["hbo"][entry_idx][0][0])}")
                #
                #print(f"hbr length: {len(session_data["brain_data"]["hbr"][entry_idx])}")
                #print(f"hbr[0] length: {len(session_data["brain_data"]["hbr"][entry_idx][0])}")
                #print(f"hbr[0][0] length: {len(session_data["brain_data"]["hbr"][entry_idx][0][0])}")


                #print(f"{entry_idx} --- {session_data["brain_data"]["time"][entry_idx]}:", file=f)
                #for j in range(30, 31):
                #    for k in range(0, 1):
                #        print(f"hbo[{entry_idx}][{j}][{k}]={session_data["brain_data"]["hbo"][entry_idx][j][k]}\n", file=f)
                #        print(f"hbr[{entry_idx}][{j}][{k}]={session_data["brain_data"]["hbr"][entry_idx][j][k]}\n", file=f)
                #print("\n", file=f)
        break

    #glove_path = 'C:\\Dev\\brain-decoder\\data\\glove\\glove.840B.300d.txt'
    #embeddings_dict = get_embeddings_dict(glove_path)

# [time, hbo, hbr]
# hbo: []