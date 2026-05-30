"""
Session Manager
---------------
Handles initialization and teardown of print sessions.
Triggers archival cleanup when a new print starts and ensures
all semantic streak tracking is fully reset.
"""

import uuid
import logging
from datetime import datetime
from core.state import StateManager
from handlers.local_storage_handler import cleanup_failures

logger = logging.getLogger("AntiPasta.Core.Session")

class SessionManager:
    def __init__(self, state_manager: StateManager):
        self.state = state_manager

    def start_new_session(self):
        """Resets metrics and generates a new unique ID for a print job."""
        
        # Cleanup is performed at the START of a session to ensure the 
        # failures directory is lean before the next run begins.
        cleanup_failures()

        new_id = str(uuid.uuid4())
        self.state.update_session(
            active=True,
            print_id=new_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            frames_analyzed=0,
            fail_frames_detected=0,
            critical_streak=0,
            artifact_streak=0,
            max_confidence_seen=0.0
        )
        
        # Ensure all semantic alert states are reset for the new job
        self.state.update_alerts(
            broker_warning_on=False, 
            broker_pause_on=False,
            broker_concern_on=False
        )
        
        logger.info(f"New session started: {new_id}")
        return new_id

    def end_session(self):
        """Finalizes the current session metrics."""
        self.state.update_session(
            active=False,
            end_time=datetime.now().isoformat(),
            critical_streak=0,
            artifact_streak=0
        )
        self.state.update_alerts(
            broker_warning_on=False, 
            broker_pause_on=False,
            broker_concern_on=False
        )
        logger.info("Session ended.")