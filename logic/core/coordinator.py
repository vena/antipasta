"""
AntiPasta Coordinator
---------------------
Orchestrates the monitoring cycle. 
Uses Dependency Injection to decouple the core loop from service implementations.
Handles the dual-track routing of critical failures vs. non-critical artifacts.
"""

import time
import uuid
import logging
import os
from threading import Lock
from typing import Dict

from core.state import APP_STATE
from core.models import Detection, AlertTier, AlertPayload
import config
from handlers.inference_client import perform_inference
from handlers.annotation_handler import draw_detections
from handlers.local_storage_handler import archive_failure, save_latest_frame

logger = logging.getLogger("AntiPasta.Core.Coordinator")

class AntiPastaCoordinator:
    def __init__(self, mqtt_client, camera_client, session_manager, detection_filter, heuristic_service):
        self.mqtt = mqtt_client
        self.camera = camera_client
        # Prevents race conditions between the heartbeat loop and the /frame.jpg 
        # API endpoint when accessing the shared camera driver.
        self.camera_lock = Lock()
        self.session_mgr = session_manager
        self.filter = detection_filter
        self.heuristics = heuristic_service
        
        # Tracks if we have already alerted the user that we are waiting for the AI.
        # This prevents log-spam during the model synchronization phase.
        self._waiting_logged = False

    def run(self):
        """Main loop that evaluates the printer state and triggers processing."""
        logger.info("Coordinator started.")
        
        try:
            while True:
                # Using time.monotopic() instead of time.time() because
                # time.time() is subject to system clock adjustments.
                loop_start = time.monotonic()
                
                session = APP_STATE.get_session()
                p_state = self.mqtt.printer_state
                
                # Using fullmatch to evaluate the regexes compiled in config.py
                is_printing = bool(config.STATUS_PRINTING_REGEX.fullmatch(p_state))
                is_paused = bool(config.STATUS_PAUSED_REGEX.fullmatch(p_state))
                
                # Session Lifecycle Management
                if is_printing and not session.active:
                    run_id = self.session_mgr.start_new_session()
                    self._reset_mqtt(run_id)
                    session = APP_STATE.get_session()

                if is_printing:
                    self._process_cycle(session)
                elif is_paused:
                    # Somebody paused the printer (e.g., to clear a tangle or swap filament).
                    # Human Intervention Reset: We explicitly clear the streaks and alerts
                    # to acknowledge the user is handling the issue. This prevents the system from
                    # immediately re-triggering a pause when the print resumes if a stray false 
                    # positive occurs immediately after un-pausing.
                    self._waiting_logged = False
                    if session.active and (session.critical_streak > 0 or session.artifact_streak > 0):
                        logger.info("Printer pause detected. Resetting failure streaks.")
                        APP_STATE.update_session(critical_streak=0, artifact_streak=0)
                        self._reset_mqtt(session.print_id)
                else:
                    # Terminal or unknown state (idle, finish, stop, offline).
                    # Reset waiting flag so we log it once per future job start.
                    self._waiting_logged = False
                    
                    if session.active:
                        self.session_mgr.end_session()
                        self._reset_mqtt(session.print_id)

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0, config.FRAME_INTERVAL - elapsed))
                
        except Exception as e:
            # A hard exit is required here. If we let the background thread die silently, 
            # Waitress will continue to serve requests on the main thread, resulting in a 
            # zombie container that never processes frames but appears perfectly healthy to Docker.
            logger.critical("Fatal error in coordinator thread. Terminating application.", exc_info=True)
            os._exit(1)

    def _process_cycle(self, session):
        """Executes a single capture-analyze-dispatch cycle."""
        with self.camera_lock:
            frame_bytes = self.camera.get_frame()

        if not frame_bytes: return

        save_latest_frame(frame_bytes)

        # perform_inference gracefully handles "Initializing" states.
        raw_results, inf_time, status_msg = perform_inference(frame_bytes)
        
        if status_msg:
            # If the engine is initializing, we log it once and skip the cycle.
            if "Initializing" in status_msg or "Waiting" in status_msg:
                if not self._waiting_logged:
                    logger.info(f"Monitoring active, but {status_msg}")
                    self._waiting_logged = True
            else:
                # This is a real error (Unauthorized, 500, etc.)
                logger.warning(f"Inference cycle skipped: {status_msg}")
            return

        # Engine is ready; clear the waiting flag for future interruptions.
        if self._waiting_logged:
            logger.info("AI Engine connected and ready. Resuming analysis.")
            self._waiting_logged = False

        detections = [Detection(d[0], d[1], d[2]) for d in raw_results]
        filtered = self.filter.apply(detections, frame_bytes)
        
        # The heuristic engine handles splitting the detections into the two parallel tracks
        eval_result = self.heuristics.evaluate(filtered, session.critical_streak, session.artifact_streak)

        inf_ms = int(inf_time * 1000)
        overall_max_conf = max(eval_result.critical.max_confidence, eval_result.artifact.max_confidence)
        
        # Tally a failure frame if EITHER track exceeded the raw confidence threshold
        is_fail_frame = overall_max_conf >= config.CONFIDENCE_THRESHOLD
        
        APP_STATE.update_session(
            frames_analyzed=session.frames_analyzed + 1,
            max_confidence_seen=max(session.max_confidence_seen, overall_max_conf),
            fail_frames_detected=session.fail_frames_detected + (1 if is_fail_frame else 0),
            critical_streak=eval_result.critical.new_streak,
            artifact_streak=eval_result.artifact.new_streak
        )

        self._dispatch(session, eval_result, filtered, inf_ms, frame_bytes)

    def _generate_alert_payload(self, session, track, all_types: Dict[str, float], frame_bytes, detections, inf_ms) -> AlertPayload:
        """
        Helper method to archive a failure frame and generate the strictly-typed alert payload.
        Ensures consistent UUID assignment between the disk archive and the MQTT message.
        """
        event_id = str(uuid.uuid4())
        try:
            annotated = draw_detections(frame_bytes, detections, inf_ms / 1000, config.EXCLUSION_ZONES)
            archive_failure(event_id, annotated)
        except Exception as e:
            logger.error(f"Archive failed for event {event_id}: {e}")
            
        return AlertPayload(
            state="ON",
            streak=track.new_streak,
            confidence=round(track.max_confidence * 100, 1),
            image_url=f"{config.CONTROLLER_EXTERNAL_URL}/failure_frame.jpg?event={event_id}",
            run_id=session.print_id,
            primary_type=track.primary_class,
            all_types=all_types
        )

    def _dispatch(self, session, eval_result, detections, inf_ms, frame_bytes):
        """Publishes the results of the evaluation to the MQTT broker via independent tracks."""
        alerts = APP_STATE.get_alerts()
        all_types = {d.class_name: round(d.confidence * 100, 1) for d in detections}
        
        # Telemetry updates (Uses the overall highest confidence across both tracks)
        overall_conf_pct = round(max(eval_result.critical.max_confidence, eval_result.artifact.max_confidence) * 100, 1)
        primary_class = eval_result.critical.primary_class if eval_result.critical.max_confidence >= eval_result.artifact.max_confidence else eval_result.artifact.primary_class
        
        self.mqtt.publish_confidence(overall_conf_pct, inf_ms, {
            "run_id": session.print_id,
            "primary_type": primary_class,
            "all_types": all_types
        })

        # --- CRITICAL TRACK (Spaghetti, Failure) ---
        c_track = eval_result.critical
        if c_track.tier == AlertTier.WARNING and not alerts.broker_warning_on:
            payload = self._generate_alert_payload(session, c_track, all_types, frame_bytes, detections, inf_ms)
            self.mqtt.publish_warning(payload)
            APP_STATE.update_alerts(broker_warning_on=True)
            
        elif c_track.tier == AlertTier.PAUSE and not alerts.broker_pause_on:
            payload = self._generate_alert_payload(session, c_track, all_types, frame_bytes, detections, inf_ms)
            self.mqtt.publish_pause(payload)
            APP_STATE.update_alerts(broker_pause_on=True)
            
        elif c_track.new_streak == 0:
            # The 'leaky bucket' has fully decayed back to 0. Clear MQTT states.
            off_payload = AlertPayload("OFF", 0, 0.0, None, session.print_id, "none", all_types)
            if alerts.broker_warning_on:
                self.mqtt.publish_warning(off_payload)
                APP_STATE.update_alerts(broker_warning_on=False)
            if alerts.broker_pause_on:
                self.mqtt.publish_pause(off_payload)
                APP_STATE.update_alerts(broker_pause_on=False)

        # --- ARTIFACT TRACK (Stringing, Zits) ---
        a_track = eval_result.artifact
        if a_track.tier == AlertTier.CONCERN and not alerts.broker_concern_on:
            payload = self._generate_alert_payload(session, a_track, all_types, frame_bytes, detections, inf_ms)
            self.mqtt.publish_concern(payload)
            APP_STATE.update_alerts(broker_concern_on=True)
            
        elif a_track.new_streak == 0:
            # The 'leaky bucket' has fully decayed back to 0. Clear MQTT states.
            if alerts.broker_concern_on:
                off_payload = AlertPayload("OFF", 0, 0.0, None, session.print_id, "none", all_types)
                self.mqtt.publish_concern(off_payload)
                APP_STATE.update_alerts(broker_concern_on=False)

    def _reset_mqtt(self, run_id):
        """Ensures MQTT sensors are strictly typed and cleared at the start/end of a session."""
        baseline = AlertPayload(
            state="OFF", 
            streak=0, 
            confidence=0.0, 
            image_url=None, 
            run_id=run_id, 
            primary_type="none", 
            all_types={}
        )
        
        self.mqtt.publish_warning(baseline)
        self.mqtt.publish_pause(baseline)
        self.mqtt.publish_concern(baseline)
        
        APP_STATE.update_alerts(broker_warning_on=False, broker_pause_on=False, broker_concern_on=False)