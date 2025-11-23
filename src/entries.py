import numpy as np

class StimulusEntry:
    def __init__(self, label: str, start_time: float, end_time: float):#, wordcloud_file: str):
        self.label = label
        self.start_time = np.float32(start_time)
        self.end_time = np.float32(end_time)
        #self.wordcloud_file = wordcloud_file

    @property
    def duration(self) -> np.float32:
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"{self.label}: start_time={self.start_time:.2f}, duration={self.duration:.2f}"


class BrainDataEntry:
    def __init__(self, session_data, entry_idx: int):
        self.session_data = session_data
        self.entry_idx = entry_idx

    @property
    def time(self) -> np.float32:
        return self.session_data["time"][self.entry_idx]
    
    def get_hbo(self, x: int, y: int, z: int) -> np.float32:
        return self.session_data["hbo"][x][y][z][self.entry_idx]

    def get_hbr(self, x: int, y: int, z: int) -> np.float32:
        return self.session_data["hbr"][x][y][z][self.entry_idx]

class SessionData:
    def __init__(self, idx: int, brain_data: np.ndarray, stimulus_data: list[StimulusEntry]):
        self.idx = idx
        self.brain_data = brain_data
        self.stimulus_data = stimulus_data

    @property
    def max_x(self) -> int:
        return len(self.brain_data["hbo"])

    @property
    def max_y(self) -> int:
        return len(self.brain_data["hbo"][0])

    @property
    def max_z(self) -> int:
        return len(self.brain_data["hbo"][0][0])
    
    @property
    def max_t(self) -> np.float32:
        return np.max(self.brain_data["time"])


class ConceptActivationEntry:
    def __init__(self, x: int, y: int, z: int, hbo: np.float32, hbr: np.float32):
        self.x = x
        self.y = y
        self.z = z
        self.hbo = hbo
        self.hbr = hbr
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'x': int(self.x),
            'y': int(self.y), 
            'z': int(self.z),
            'hbo': float(self.hbo),
            'hbr': float(self.hbr)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConceptActivationEntry':
        """Create from dictionary for JSON deserialization"""
        return cls(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            hbo=np.float32(data['hbo']),
            hbr=np.float32(data['hbr'])
        )