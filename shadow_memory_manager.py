"""
ShadowMemory: Privacy-preserving, ephemeral memory system for passive observation
and behavior modulation.

This module provides a volatile, in-memory system for observing and analyzing
behavioral patterns without persisting sensitive data.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from collections import defaultdict
import re

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import os
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShadowMemory:
    """
    ShadowMemory is a volatile, ephemeral memory for passive observation and behavior modulation.
    All observed data is encrypted in-memory and never written to persistent storage.
    """

    def __init__(self, 
                context_window_seconds: int = 3600,
                recency_weight: float = 0.9,
                encryption_level: str = "high"):
        """
        Initialize ShadowMemory system with encryption and configuration parameters.
        
        Args:
            context_window_seconds: How many seconds of data to consider relevant (default: 1 hour)
            recency_weight: Weight for recency bias in analysis (0.0-1.0)
            encryption_level: Encryption strength ("medium" or "high")
        """
        # AES encryption key (session-specific, volatile only)
        self.encryption_level = encryption_level
        self._generate_encryption_keys()
        
        # Memory configuration
        self.context_window_seconds = context_window_seconds
        self.recency_weight = recency_weight
        self.memory_store = []  # List to store encrypted, ephemeral memory
        
        # Analysis trackers
        self.pattern_detectors = {
            "loop_detection": self._detect_behavioral_loops,
            "escalation_detection": self._detect_escalation,
            "anomaly_detection": self._detect_anomalies,
        }
        
        # Session metadata
        self.session_start_time = time.time()
        self.observation_count = 0
        
        logger.info("ShadowMemory initialized with encryption level: %s", encryption_level)

    def _generate_encryption_keys(self):
        """Generate encryption keys based on specified security level"""
        if self.encryption_level == "high":
            # Generate a random salt
            salt = os.urandom(16)
            # Generate a strong password
            password = os.urandom(32)
            # Use PBKDF2 to derive a key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.fernet_key = key
        else:
            # Standard Fernet key generation
            self.fernet_key = Fernet.generate_key()
            
        self.fernet = Fernet(self.fernet_key)
        
    def observe(self, data: dict):
        """
        Passively observes and encrypts input data (e.g., sensor, behavior data).

        Args:
            data (dict): Raw metadata to be stored (e.g., sensor readings).

        Schema example for `data`:
        {
            "sensor_type": "audio",
            "emotion_detected": "joy",
            "tags": ["notification", "interaction"],
            "intensity": 0.8
        }
        """
        timestamp = time.time()
        
        # Add metadata before encryption
        data["_meta"] = {
            "observation_id": self.observation_count,
            "relative_time": timestamp - self.session_start_time
        }
        
        # Encrypt the data
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode("utf-8"))
        
        # Store encrypted data along with timestamp
        self.memory_store.append({
            "timestamp": timestamp, 
            "encrypted": encrypted_data
        })
        
        # Update observation count
        self.observation_count += 1
        
        # Optional: Prune old memories outside context window
        self._prune_old_memories()
            
    def _prune_old_memories(self):
        """Remove memories outside the context window to manage memory usage"""
        if not self.memory_store:
            return
            
        current_time = time.time()
        cutoff_time = current_time - self.context_window_seconds
        
        # Keep only memories within the context window
        self.memory_store = [
            memory for memory in self.memory_store
            if memory["timestamp"] >= cutoff_time
        ]

    def inform_behavior(self) -> dict:
        """
        Analyzes observed data to produce statistical summaries for behavior modulation.
        Behavior summary examples: emotional trends, stability, reaction patterns.

        Returns:
            dict: Statistical summary of observed data.
        Example output:
        {
            "emotional_trend": "volatile",
            "average_intensity": 0.65,
            "behavior_loops_detected": ["repeating_phrases"],
            "trigger_tags": ["stress", "dog_interaction"]
        }
        """
        if not self.memory_store:
            return {"status": "no_data"}

        # Extract and decrypt all memories within context window
        memories = self._get_recent_memories()
        
        # Initialize analysis variables
        total_intensity = 0
        emotion_count = 0
        tags = []
        sensor_types = {}
        emotions = {}
        
        # Process each memory
        for memory in memories:
            # Apply recency weighting - more recent observations have higher weight
            age_factor = (time.time() - memory["timestamp"]) / self.context_window_seconds
            recency_weight = self.recency_weight ** age_factor
            
            # Analyze intensity with recency weighting
            if "intensity" in memory["data"]:
                weighted_intensity = memory["data"]["intensity"] * recency_weight
                total_intensity += weighted_intensity
                emotion_count += 1

            # Collect tags (with duplicates to measure frequency)
            if "tags" in memory["data"]:
                tags.extend(memory["data"]["tags"])
                
            # Track sensor types
            if "sensor_type" in memory["data"]:
                sensor_type = memory["data"]["sensor_type"]
                sensor_types[sensor_type] = sensor_types.get(sensor_type, 0) + 1
                
            # Track emotions
            if "emotion_detected" in memory["data"]:
                emotion = memory["data"]["emotion_detected"]
                emotions[emotion] = emotions.get(emotion, 0) + recency_weight
        
        # Calculate emotional trend
        average_intensity = total_intensity / emotion_count if emotion_count > 0 else 0
        emotional_trend = self._determine_emotional_trend(average_intensity, emotions)
        
        # Calculate tag frequency and significance
        tag_frequency = defaultdict(int)
        for tag in tags:
            tag_frequency[tag] += 1
            
        # Sort tags by frequency for significance
        significant_tags = sorted(
            tag_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]  # Top 5 tags
        
        # Get top tags
        top_tags = [tag for tag, _ in significant_tags]
        
        # Run pattern detectors
        behavioral_patterns = {}
        for detector_name, detector_func in self.pattern_detectors.items():
            behavioral_patterns[detector_name] = detector_func(memories)
        
        # Generate behavior loops list
        behavior_loops = []
        if behavioral_patterns["loop_detection"]["detected"]:
            behavior_loops.extend(behavioral_patterns["loop_detection"]["patterns"])
            
        # Assemble the complete behavior summary
        behavior_summary = {
            "emotional_trend": emotional_trend,
            "average_intensity": average_intensity,
            "behavior_loops_detected": behavior_loops,
            "trigger_tags": top_tags,
            "dominant_sensors": max(sensor_types.items(), key=lambda x: x[1])[0] if sensor_types else "none",
            "observation_count": len(memories),
            "escalation_risk": behavioral_patterns["escalation_detection"]["risk_level"],
            "confidence": self._calculate_confidence(memories)
        }
        
        return behavior_summary
        
    def _get_recent_memories(self):
        """Extract and decrypt memories within context window with proper structure"""
        current_time = time.time()
        cutoff_time = current_time - self.context_window_seconds
        
        # Filter, decrypt, and structure memories
        recent_memories = []
        for memory in self.memory_store:
            if memory["timestamp"] >= cutoff_time:
                try:
                    # Decrypt the data
                    decrypted_data = self.fernet.decrypt(memory["encrypted"]).decode("utf-8")
                    data = json.loads(decrypted_data)
                    
                    # Add to recent memories with proper structure
                    recent_memories.append({
                        "timestamp": memory["timestamp"],
                        "age": current_time - memory["timestamp"],
                        "data": data
                    })
                except Exception as e:
                    # Skip corrupted memories
                    logger.warning("Failed to decrypt memory: %s", str(e))
                    
        # Sort by timestamp (oldest first)
        return sorted(recent_memories, key=lambda x: x["timestamp"])
    
    def _determine_emotional_trend(self, average_intensity, emotions):
        """Determine emotional trend based on intensity and detected emotions"""
        # First check if we have specific emotions to consider
        if emotions:
            # Find the dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Map emotions to trends
            negative_emotions = ["anger", "fear", "anxiety", "frustration", "sadness"]
            positive_emotions = ["joy", "excitement", "happiness", "contentment"]
            
            if dominant_emotion in negative_emotions and average_intensity > 0.6:
                return "volatile"
            elif dominant_emotion in positive_emotions and average_intensity > 0.6:
                return "excited"
            elif dominant_emotion in negative_emotions and average_intensity < 0.4:
                return "concerned"
            elif dominant_emotion in positive_emotions and average_intensity < 0.4:
                return "content"
        
        # Default intensity-based categorization
        if average_intensity > 0.7:
            return "volatile"
        elif average_intensity < 0.3:
            return "calm"
        else:
            return "neutral"
    
    def _detect_behavioral_loops(self, memories):
        """Detect repetitive behavior patterns in the memory stream"""
        # Initialize result
        result = {
            "detected": False,
            "patterns": [],
            "confidence": 0.0
        }
        
        if len(memories) < 3:
            return result
            
        # Look for repetitive actions or tags
        action_sequence = []
        tag_sequence = []
        
        for memory in memories:
            # Extract the relevant behavioral indicators
            data = memory["data"]
            
            # Track sensor types and tags for pattern detection
            if "sensor_type" in data:
                action_sequence.append(data["sensor_type"])
                
            if "tags" in data:
                tag_sequence.extend(data["tags"])
        
        # Detect repeating subsequences in actions (e.g., click, pause, click, pause)
        action_patterns = self._find_repeating_subsequences(action_sequence)
        
        # Detect repeating tag patterns
        tag_patterns = self._find_repeating_subsequences(tag_sequence)
        
        # Add detected patterns to result
        if action_patterns:
            result["detected"] = True
            result["patterns"].extend([f"repeating_{pattern}" for pattern in action_patterns])
            
        if tag_patterns:
            result["detected"] = True
            result["patterns"].extend([f"repetitive_{pattern}" for pattern in tag_patterns])
            
        # Calculate confidence based on number and length of detected patterns
        if result["patterns"]:
            total_pattern_length = sum(len(pattern.split('_')) for pattern in result["patterns"])
            result["confidence"] = min(0.9, total_pattern_length / 20)
        
        return result
    
    def _find_repeating_subsequences(self, sequence):
        """Helper function to find repeating subsequences in a list"""
        if not sequence or len(sequence) < 4:
            return []
            
        patterns = []
        # Look for sequences of length 2-4 that repeat at least twice
        for length in range(2, min(5, len(sequence) // 2 + 1)):
            for i in range(len(sequence) - length*2 + 1):
                candidate = tuple(sequence[i:i+length])
                
                # Check if this sequence repeats elsewhere
                repeats = 0
                for j in range(i + length, len(sequence) - length + 1, length):
                    if tuple(sequence[j:j+length]) == candidate:
                        repeats += 1
                    else:
                        break
                        
                if repeats >= 1:  # Pattern repeats at least once (appears twice total)
                    pattern_str = "_".join(str(x) for x in candidate)
                    if pattern_str not in patterns:
                        patterns.append(pattern_str)
        
        return patterns
                
    def _detect_escalation(self, memories):
        """Detect escalating intensity or concerning trends in behavior"""
        result = {
            "risk_level": "low",
            "indicators": [],
            "confidence": 0.0
        }
        
        if len(memories) < 3:
            return result
            
        # Track intensity over time to detect escalation
        intensity_trend = []
        for memory in memories:
            data = memory["data"]
            if "intensity" in data:
                intensity_trend.append(data["intensity"])
        
        # Check for rising intensity trend
        if len(intensity_trend) >= 3:
            # Simple escalation check: is the moving average increasing?
            window_size = min(3, len(intensity_trend))
            moving_avgs = []
            
            for i in range(len(intensity_trend) - window_size + 1):
                window = intensity_trend[i:i+window_size]
                moving_avgs.append(sum(window) / window_size)
            
            # Check if moving average is consistently increasing
            is_escalating = all(moving_avgs[i] < moving_avgs[i+1] for i in range(len(moving_avgs)-1))
            
            if is_escalating and moving_avgs[-1] > 0.6:
                result["risk_level"] = "high"
                result["indicators"].append("rising_intensity")
                result["confidence"] = 0.8
            elif is_escalating:
                result["risk_level"] = "moderate"
                result["indicators"].append("gradual_escalation")
                result["confidence"] = 0.6
        
        # Check for concerning tags
        concerning_tags = ["alert", "warning", "danger", "threat", "stress", "anxiety", "fear"]
        detected_concerning_tags = []
        
        for memory in memories:
            data = memory["data"]
            if "tags" in data:
                for tag in data["tags"]:
                    if tag in concerning_tags and tag not in detected_concerning_tags:
                        detected_concerning_tags.append(tag)
        
        if detected_concerning_tags:
            # Upgrade risk level if concerning tags found
            if result["risk_level"] == "low":
                result["risk_level"] = "moderate"
            result["indicators"].extend(detected_concerning_tags)
            result["confidence"] = max(result["confidence"], 0.7)
            
        return result
    
    def _detect_anomalies(self, memories):
        """Detect unusual or anomalous patterns in user behavior"""
        result = {
            "anomalies_detected": False,
            "anomaly_types": [],
            "confidence": 0.0
        }
        
        if len(memories) < 5:
            return result
            
        # Get sensor type frequencies
        sensor_counts = defaultdict(int)
        for memory in memories:
            data = memory["data"]
            if "sensor_type" in data:
                sensor_counts[data["sensor_type"]] += 1
        
        # Check timing patterns for anomalies
        timestamps = [memory["timestamp"] for memory in memories]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        # Detect unusual gaps or bursts
        avg_interval = sum(intervals) / len(intervals)
        anomalous_intervals = [
            interval for interval in intervals 
            if interval > avg_interval * 3 or interval < avg_interval * 0.2
        ]
        
        if anomalous_intervals:
            result["anomalies_detected"] = True
            if anomalous_intervals[0] > avg_interval * 3:
                result["anomaly_types"].append("unusual_pauses")
            else:
                result["anomaly_types"].append("activity_bursts")
            
        # Detect unusual intensity patterns
        intensities = []
        for memory in memories:
            data = memory["data"]
            if "intensity" in data:
                intensities.append(data["intensity"])
                
        if intensities:
            avg_intensity = sum(intensities) / len(intensities)
            intensity_variance = sum((i - avg_intensity) ** 2 for i in intensities) / len(intensities)
            
            # High variance indicates erratic behavior
            if intensity_variance > 0.1:
                result["anomalies_detected"] = True
                result["anomaly_types"].append("erratic_intensity")
                
        # Set confidence based on number of detected anomalies
        if result["anomaly_types"]:
            result["confidence"] = min(0.8, 0.4 + 0.2 * len(result["anomaly_types"]))
            
        return result
    
    def _calculate_confidence(self, memories):
        """Calculate confidence in behavioral analysis based on data quality and quantity"""
        # Base confidence on number of observations
        if not memories:
            return 0.0
            
        # Confidence increases with more observations, up to a limit
        observation_factor = min(0.7, len(memories) / 20)
        
        # Confidence increases with sensor diversity
        sensor_types = set()
        for memory in memories:
            data = memory["data"]
            if "sensor_type" in data:
                sensor_types.add(data["sensor_type"])
                
        sensor_factor = min(0.2, len(sensor_types) / 10)
        
        # Total confidence capped at 0.95 to acknowledge inherent uncertainty
        return min(0.95, observation_factor + sensor_factor)

    def purge(self):
        """
        Securely purges the ephemeral memory by clearing the in-memory storage and
        regenerating the session-specific encryption key.
        """
        # Log purge event (without details)
        logger.info("Purging ShadowMemory with %d observations", len(self.memory_store))
        
        # Clear all memory
        self.memory_store.clear()
        
        # Regenerate the encryption key
        self._generate_encryption_keys()
        
        # Reset session metadata
        self.session_start_time = time.time()
        self.observation_count = 0

        return {"status": "memory_purged"}
    
    def get_memory_stats(self) -> dict:
        """
        Get statistics about the current memory state without exposing content.
        Safe to expose to users/admins for diagnostic purposes.
        """
        if not self.memory_store:
            return {
                "status": "empty",
                "observation_count": 0,
                "memory_size_bytes": 0,
                "session_duration": time.time() - self.session_start_time
            }
        
        # Calculate total memory size in bytes
        total_bytes = sum(len(memory["encrypted"]) for memory in self.memory_store)
        
        # Get time range
        oldest_timestamp = min(memory["timestamp"] for memory in self.memory_store)
        newest_timestamp = max(memory["timestamp"] for memory in self.memory_store)
        time_range = newest_timestamp - oldest_timestamp
        
        return {
            "status": "active",
            "observation_count": len(self.memory_store),
            "memory_size_bytes": total_bytes,
            "session_duration": time.time() - self.session_start_time,
            "memory_time_range": time_range,
            "observations_per_minute": len(self.memory_store) / (time_range / 60) if time_range > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    shadow = ShadowMemory()

    # Simulate observing data
    shadow.observe({
        "sensor_type": "audio",
        "emotion_detected": "joy",
        "tags": ["interaction", "greeting"],
        "intensity": 0.6,
    })
    shadow.observe({
        "sensor_type": "proximity",
        "emotion_detected": "stress",
        "tags": ["crowded", "alert"],
        "intensity": 0.9,
    })

    # Inform behavior summary
    print(shadow.inform_behavior())

    # Securely purge memory
    print(shadow.purge())