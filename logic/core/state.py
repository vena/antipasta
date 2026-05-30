"""
State Manager
-------------
Provides a thread-safe interface for reading and writing volatile application state.
Now includes hardware reporting for system diagnostics.
"""

import threading
from dataclasses import asdict
from typing import Any, Dict
from core.models import SessionData, AlertState, HardwareReport, HardwareStats

class StateManager:
    def __init__(self):
        """
        Initializes the state containers. Static configuration is intentionally 
        excluded here to maintain config.py as the single source of truth.
        """
        self._lock = threading.Lock()
        self.session = SessionData()
        self.alerts = AlertState()
        self.hardware = HardwareReport()

    def get_session(self) -> SessionData:
        """Returns a thread-safe copy of the current session data."""
        with self._lock:
            return SessionData(**asdict(self.session))

    def update_session(self, **kwargs):
        """Atomically updates session fields via keyword arguments."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.session, key):
                    setattr(self.session, key, value)

    def get_alerts(self) -> AlertState:
        """Returns a thread-safe copy of the current alert states."""
        with self._lock:
            return AlertState(**asdict(self.alerts))

    def update_alerts(self, **kwargs):
        """Atomically updates alert fields via keyword arguments."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.alerts, key):
                    setattr(self.alerts, key, value)

    def get_hardware_report(self) -> HardwareReport:
        """Returns a thread-safe copy of the current hardware status."""
        with self._lock:
            # We recreate the report using snapshots of the nested logic/ml stats.
            # This ensures the caller receives a point-in-time copy that is safe to read.
            return HardwareReport(
                logic=HardwareStats(**asdict(self.hardware.logic)),
                ml=HardwareStats(**asdict(self.hardware.ml))
            )

    def update_hardware(self, component: str, **kwargs):
        """
        Updates hardware stats for either 'logic' or 'ml'.
        Args:
            component: 'logic' or 'ml'
            kwargs: Fields of HardwareStats to update
        """
        with self._lock:
            target = getattr(self.hardware, component, None)
            if target:
                for key, value in kwargs.items():
                    if hasattr(target, key):
                        setattr(target, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the volatile state.
        This structured dict is consumed by the /stats/ API endpoint.
        """
        with self._lock:
            return {
                "session": asdict(self.session),
                "alerts": asdict(self.alerts),
                "hardware": asdict(self.hardware)
            }

# Singleton instance for the application
APP_STATE = StateManager()