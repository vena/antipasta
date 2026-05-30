"""
Heuristic Service
-----------------
Evaluates filtered detections against confidence and streak thresholds.
Splits detections into Critical (Pause-worthy) and Artifact (Warning-worthy) 
tracks, evaluating their independent 'leaky bucket' streak mechanisms simultaneously.
"""

from typing import List, Dict
from core.models import Detection, AlertTier, HeuristicTrack, HeuristicResult

class HeuristicService:
    def __init__(
        self, 
        confidence_threshold: float, 
        warning_streak: int, 
        pause_streak: int, 
        concern_streak: int, 
        critical_classes: List[str]
    ):
        self.conf_threshold = confidence_threshold
        self.warning_streak = warning_streak
        self.pause_streak = pause_streak
        self.concern_streak = concern_streak
        
        # We normalize the list to lowercase to ensure robust matching against the AI output
        self.critical_classes = [c.lower() for c in critical_classes]

    def evaluate(
        self, 
        detections: List[Detection], 
        current_critical_streak: int, 
        current_artifact_streak: int
    ) -> HeuristicResult:
        """
        Analyzes detections, splitting them into critical and artifact buckets.
        Returns a complete dual-track HeuristicResult.
        """
        critical_dets = []
        artifact_dets = []
        
        for det in detections:
            if det.class_name.lower() in self.critical_classes:
                critical_dets.append(det)
            else:
                artifact_dets.append(det)

        # The thresholds dict acts as a flexible mapping for the specific track
        critical_track = self._evaluate_track(
            critical_dets, 
            current_critical_streak, 
            {"pause": self.pause_streak, "warning": self.warning_streak}
        )
        
        artifact_track = self._evaluate_track(
            artifact_dets, 
            current_artifact_streak, 
            {"concern": self.concern_streak}
        )

        return HeuristicResult(critical=critical_track, artifact=artifact_track)

    def _evaluate_track(self, detections: List[Detection], current_streak: int, thresholds: Dict[str, int]) -> HeuristicTrack:
        """
        Evaluates a single bucket of detections. 
        Applies the 'leaky bucket' pattern: Increment streak if the highest confidence 
        detection exceeds the threshold, otherwise decay the streak by 1.
        """
        if not detections:
            # Decay the streak if there are no relevant detections in this frame
            return HeuristicTrack(
                tier=AlertTier.NONE, 
                max_confidence=0.0, 
                primary_class="none", 
                new_streak=max(0, current_streak - 1)
            )
        
        max_det = max(detections, key=lambda x: x.confidence)
        new_streak = current_streak
        tier = AlertTier.NONE

        if max_det.confidence >= self.conf_threshold:
            new_streak += 1
            
            # Tier escalation evaluates from most severe to least severe
            if 'pause' in thresholds and new_streak >= thresholds['pause']:
                tier = AlertTier.PAUSE
            elif 'warning' in thresholds and new_streak >= thresholds['warning']:
                tier = AlertTier.WARNING
            elif 'concern' in thresholds and new_streak >= thresholds['concern']:
                tier = AlertTier.CONCERN
        else:
            # Detection exists but is below threshold; treat as a miss and decay
            new_streak = max(0, current_streak - 1)

        return HeuristicTrack(
            tier=tier, 
            max_confidence=max_det.confidence, 
            primary_class=max_det.class_name, 
            new_streak=new_streak
        )