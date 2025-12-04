import numpy as np

class Vec:
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

class StimulusEntry:
    def __init__(self, label: str, start_time: float, end_time: float):#, wordcloud_file: str):
        self.label = label
        self.start_time = np.float64(start_time)
        self.end_time = np.float64(end_time)
        #self.wordcloud_file = wordcloud_file

    @property
    def duration(self) -> np.float64:
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"{self.label}: start_time={self.start_time:.2f}, duration={self.duration:.2f}"

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


class ConceptActivationEntry:
    def __init__(self, x: int, y: int, z: int, hbo: np.float64, hbr: np.float64):
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
            hbo=np.float64(data['hbo']),
            hbr=np.float64(data['hbr'])
        )
    
    def __repr__(self):
        return f"(x={self.x}, y={self.y}, z={self.z}, hbo={self.hbo}, hbr={self.hbr})"

