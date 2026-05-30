"""
Core Data Models
-----------
Defines type-safe data structures for the application state.
Standardizes the format of detections, session metrics, and hardware reports.
"""

from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

class AlertTier(Enum):
    """
    Defines the escalation level of a detection track.
    - NONE: Below threshold or decayed.
    - CONCERN: Non-critical artifact (e.g., stringing) has met the streak requirement.
    - WARNING: Critical failure (e.g., spaghetti) has met the warning streak but not yet the pause streak.
    - PAUSE: Critical failure has met the final pause streak requirement.
    """
    NONE = auto()
    CONCERN = auto()
    WARNING = auto()
    PAUSE = auto()

@dataclass
class Detection:
    """Represents a single AI detection result."""
    class_name: str
    confidence: float
    # Matches the user-facing exclusion zone format: [x1, y1, x2, y2] in absolute pixels
    bbox: List[float] 

    def to_list(self) -> List:
        return [self.class_name, self.confidence, self.bbox]

@dataclass
class HeuristicTrack:
    """Represents the evaluation state for a single classification bucket (Critical or Artifact)."""
    tier: AlertTier
    max_confidence: float
    primary_class: str
    new_streak: int

@dataclass
class HeuristicResult:
    """The dual-track result returned by the HeuristicService after evaluating a frame."""
    critical: HeuristicTrack
    artifact: HeuristicTrack

@dataclass
class AlertPayload:
    """
    Strongly-typed structure for Home Assistant MQTT alerts.
    Prevents malformed dicts from breaking the JSON serialization layer.
    """
    state: str                 # "ON" or "OFF"
    streak: int
    confidence: float          # Percentage format (0.0 to 100.0)
    image_url: Optional[str]
    run_id: str
    primary_type: str
    all_types: Dict[str, float]

@dataclass
class SessionData:
    """
    Metrics and identifiers for the current print session.
    These fields represent the volatile state that changes during a print.
    We track critical failures (spaghetti) independently from artifacts (stringing)
    so one does not mask the other.
    """
    active: bool = False
    print_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    frames_analyzed: int = 0
    fail_frames_detected: int = 0
    critical_streak: int = 0
    artifact_streak: int = 0
    max_confidence_seen: float = 0.0

@dataclass
class AlertState:
    """
    Tracks which alerts have been published to the MQTT broker.
    Used to prevent redundant 'ON' messages during a continuous failure streak.
    """
    broker_warning_on: bool = False
    broker_pause_on: bool = False
    broker_concern_on: bool = False

@dataclass
class HardwareStats:
    """
    Detailed status for a specific component's hardware acceleration.
    'active' indicates if the acceleration path was successfully initialized.
    """
    mode: str = "cpu"
    device: str = "unknown"
    active: bool = False
    note: str = "Initializing..."

@dataclass
class HardwareReport:
    """
    Aggregated hardware status for both the Logic and ML containers.
    Since they may reside on different hosts, they are tracked independently.
    """
    logic: HardwareStats = field(default_factory=HardwareStats)
    ml: HardwareStats = field(default_factory=HardwareStats)