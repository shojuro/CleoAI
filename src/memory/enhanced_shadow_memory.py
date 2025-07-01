"""
Enhanced ShadowMemory module with domain-specific extensions for different applications.

This module extends the base ShadowMemory system with specialized capabilities for:
- Educational contexts (student engagement, learning patterns)
- Dating and relationship contexts (social dynamics, safety)
- Personal security (environment assessment, risk detection)
- Healthcare (wellness patterns, adherence)
"""

import time
import json
import logging
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
import re
import random

from shadow_memory_manager import ShadowMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EducationalShadowMemory(ShadowMemory):
    """
    Enhanced ShadowMemory for educational contexts. Adds specialized detection
    for learning patterns, engagement levels, and educational emotional states.
    """
    
    def __init__(self, **kwargs):
        """Initialize EducationalShadowMemory with base ShadowMemory parameters"""
        super().__init__(**kwargs)
        self.learning_patterns = {
            "disengagement_count": 0,
            "self_monologue_detected": False,
            "struggle_indicators": [],
            "last_interaction_timestamp": time.time()
        }
        
        # Educational context thresholds - customize based on age groups
        self.disengagement_threshold = 30  # seconds without interaction
        self.frustration_threshold = 0.65  # intensity level
        
    def observe_educational(self, data: dict):
        """
        Specialized observation function for educational contexts.
        
        Args:
            data (dict): Educational context data
            
        Example:
        {
            "sensor_type": "interaction",
            "engagement_level": 0.4,
            "subject": "mathematics",
            "question_difficulty": 0.7,
            "time_on_question": 45,
            "self_talk_detected": True,
            "tags": ["multiplication", "hesitation"],
            "intensity": 0.8
        }
        """
        # Track learning-specific patterns
        current_time = time.time()
        
        # Check for disengagement
        if "sensor_type" in data and data["sensor_type"] == "interaction":
            self.learning_patterns["last_interaction_timestamp"] = current_time
        elif current_time - self.learning_patterns["last_interaction_timestamp"] > self.disengagement_threshold:
            self.learning_patterns["disengagement_count"] += 1
            
        # Track self-monologue
        if "self_talk_detected" in data and data["self_talk_detected"]:
            self.learning_patterns["self_monologue_detected"] = True
            
        # Add to struggle indicators if difficulty is high and intensity suggests frustration
        if "question_difficulty" in data and "intensity" in data:
            if data["question_difficulty"] > 0.6 and data["intensity"] > self.frustration_threshold:
                if "subject" in data:
                    self.learning_patterns["struggle_indicators"].append(data["subject"])
                    
        # Pass to standard observation function
        self.observe(data)
        
    def inform_educational_behavior(self) -> dict:
        """
        Provides education-specific behavioral insights.
        """
        # Get base behavior analysis
        behavior = self.inform_behavior()
        
        # Enhance with educational insights
        educational_insights = {
            "disengagement_level": "high" if self.learning_patterns["disengagement_count"] > 3 
                                  else "moderate" if self.learning_patterns["disengagement_count"] > 0 
                                  else "low",
            "self_reflection_detected": self.learning_patterns["self_monologue_detected"],
            "struggling_subjects": list(set(self.learning_patterns["struggle_indicators"])),
            "learning_emotional_state": behavior["emotional_trend"],
            "recommended_action": self._determine_educational_action(behavior)
        }
        
        return {**behavior, **educational_insights}
    
    def _determine_educational_action(self, behavior: dict) -> str:
        """
        Determines appropriate educational intervention based on behavior.
        """
        if behavior["emotional_trend"] == "volatile":
            return "reduce_difficulty"
        elif self.learning_patterns["disengagement_count"] > 2:
            return "increase_engagement"
        elif self.learning_patterns["self_monologue_detected"] and behavior.get("average_intensity", 0) > 0.5:
            return "offer_assistance"
        else:
            return "continue_current_approach"
            
    def reset_learning_session(self):
        """Reset learning session statistics while preserving memory"""
        self.learning_patterns = {
            "disengagement_count": 0,
            "self_monologue_detected": False,
            "struggle_indicators": [],
            "last_interaction_timestamp": time.time()
        }


class RelationshipShadowMemory(ShadowMemory):
    """
    Enhanced ShadowMemory for dating and relationship contexts.
    Focused on social dynamics, emotional safety, and connection quality.
    """
    
    def __init__(self, **kwargs):
        """Initialize RelationshipShadowMemory with base parameters"""
        super().__init__(**kwargs)
        self.relationship_metrics = {
            "concern_flags": [],
            "comfort_indicators": [],
            "interaction_quality": 0.5,  # Default neutral
            "recent_sentiment_shift": 0.0,  # No shift initially
            "safety_assessment": "unknown"
        }
        
        # Add relationship-specific pattern detectors
        self.pattern_detectors.update({
            "connection_quality": self._assess_connection_quality,
            "safety_analysis": self._analyze_interaction_safety
        })
        
    def observe_relationship_context(self, data: dict):
        """
        Specialized observation function for relationship contexts.
        
        Args:
            data (dict): Relationship interaction data
            
        Example:
        {
            "sensor_type": "conversation",
            "conversation_type": "dating",
            "sentiment": "positive",
            "topics": ["personal_history", "interests"],
            "conversation_flow": 0.8,
            "engagement_signals": ["follow_up_questions", "laughter"],
            "conversation_duration": 300,
            "intensity": 0.5,
            "tags": ["first_date", "getting_to_know"]
        }
        """
        # Process relationship-specific indicators
        if "sentiment" in data:
            sentiment_value = 0.8 if data["sentiment"] == "positive" else (
                0.4 if data["sentiment"] == "neutral" else 0.1
            )
            
            # Track sentiment shifts
            current_interaction_quality = self.relationship_metrics["interaction_quality"]
            self.relationship_metrics["recent_sentiment_shift"] = sentiment_value - current_interaction_quality
            self.relationship_metrics["interaction_quality"] = (
                current_interaction_quality * 0.7 + sentiment_value * 0.3  # Smooth changes
            )
            
        # Check for comfort indicators
        if "engagement_signals" in data:
            comfort_signals = ["laughter", "self_disclosure", "extended_conversation", "future_plans"]
            for signal in data["engagement_signals"]:
                if signal in comfort_signals and signal not in self.relationship_metrics["comfort_indicators"]:
                    self.relationship_metrics["comfort_indicators"].append(signal)
        
        # Check for concern flags in conversation
        if "flags" in data:
            concern_flags = ["pushy", "uncomfortable", "personal_boundary", "pressure", "unwanted_advance"]
            for flag in data["flags"]:
                if flag in concern_flags and flag not in self.relationship_metrics["concern_flags"]:
                    self.relationship_metrics["concern_flags"].append(flag)
                    
        # Pass to standard observation
        self.observe(data)
    
    def inform_relationship_behavior(self) -> dict:
        """
        Provides relationship-specific behavioral insights.
        """
        # Get base behavior analysis
        behavior = self.inform_behavior()
        
        # Run connection quality assessment
        memories = self._get_recent_memories()
        connection_assessment = self.pattern_detectors["connection_quality"](memories)
        safety_assessment = self.pattern_detectors["safety_analysis"](memories)
        
        # Combine for relationship-specific insights
        relationship_insights = {
            "interaction_quality": self.relationship_metrics["interaction_quality"],
            "comfort_signals": self.relationship_metrics["comfort_indicators"],
            "concern_flags": self.relationship_metrics["concern_flags"],
            "sentiment_trajectory": "improving" if self.relationship_metrics["recent_sentiment_shift"] > 0.2 else (
                "declining" if self.relationship_metrics["recent_sentiment_shift"] < -0.2 else "stable"
            ),
            "connection_strength": connection_assessment["connection_level"],
            "safety_assessment": safety_assessment["safety_level"],
            "recommended_action": self._determine_relationship_action(behavior, connection_assessment, safety_assessment)
        }
        
        return {**behavior, **relationship_insights}
    
    def _assess_connection_quality(self, memories):
        """Assess the quality of connection in relationship interactions"""
        result = {
            "connection_level": "moderate",
            "indicators": [],
            "confidence": 0.5
        }
        
        if len(memories) < 3:
            return result
            
        # Track positive interaction signals
        positive_signals = 0
        total_assessed = 0
        
        for memory in memories:
            data = memory["data"]
            
            if "engagement_signals" in data:
                positive_signals += len(data["engagement_signals"])
                total_assessed += 1
                
            if "conversation_flow" in data:
                if data["conversation_flow"] > 0.7:
                    positive_signals += 1
                total_assessed += 1
        
        # Calculate connection quality
        if total_assessed > 0:
            connection_ratio = positive_signals / (total_assessed * 2)  # Normalize to 0-1 range
            
            if connection_ratio > 0.7:
                result["connection_level"] = "strong"
                result["confidence"] = 0.8
            elif connection_ratio < 0.3:
                result["connection_level"] = "weak"
                result["confidence"] = 0.7
            else:
                result["connection_level"] = "moderate"
                result["confidence"] = 0.6
                
            # Add specific indicators
            if "engagement_signals" in data:
                result["indicators"] = data["engagement_signals"]
        
        return result
    
    def _analyze_interaction_safety(self, memories):
        """Analyze interaction for safety concerns"""
        result = {
            "safety_level": "normal",
            "warning_signals": [],
            "confidence": 0.5
        }
        
        if len(memories) < 2:
            return result
            
        # Safety concerns to look for
        concerning_tags = ["pressure", "uncomfortable", "boundary", "unwanted", "pushy", "invasive"]
        warning_count = 0
        
        for memory in memories:
            data = memory["data"]
            
            # Check for concerning tags
            if "tags" in data:
                for tag in data["tags"]:
                    if any(concern in tag.lower() for concern in concerning_tags):
                        warning_count += 1
                        if tag not in result["warning_signals"]:
                            result["warning_signals"].append(tag)
            
            # Check for explicit flags
            if "flags" in data:
                for flag in data["flags"]:
                    if any(concern in flag.lower() for concern in concerning_tags):
                        warning_count += 2  # Explicit flags weighted more heavily
                        if flag not in result["warning_signals"]:
                            result["warning_signals"].append(flag)
        
        # Determine safety level
        if warning_count >= 3:
            result["safety_level"] = "concerning"
            result["confidence"] = 0.8
        elif warning_count > 0:
            result["safety_level"] = "caution"
            result["confidence"] = 0.7
        else:
            result["safety_level"] = "normal"
            result["confidence"] = 0.6
            
        return result
    
    def _determine_relationship_action(self, behavior, connection, safety):
        """Determine appropriate relationship action based on assessments"""
        # Safety always takes highest priority
        if safety["safety_level"] == "concerning":
            return "suggest_boundary_setting"
        
        # Next priority is emotional state
        if behavior["emotional_trend"] == "volatile":
            return "reduce_intensity"
        
        # Finally consider connection quality
        if connection["connection_level"] == "weak" and behavior["emotional_trend"] != "volatile":
            return "boost_engagement"
        
        # Default if everything seems normal
        return "continue_natural_conversation"


class SecurityShadowMemory(ShadowMemory):
    """
    Enhanced ShadowMemory for personal security contexts.
    Focused on environmental risk assessment and safety monitoring.
    """
    
    def __init__(self, **kwargs):
        """Initialize SecurityShadowMemory with security-specific parameters"""
        super().__init__(**kwargs)
        self.security_metrics = {
            "environment_risk_level": "normal",
            "suspicious_patterns": [],
            "trusted_locations": [],
            "trusted_contacts": [],
            "anomalies_detected": [],
            "last_risk_assessment": time.time()
        }
        
        # Add security-specific pattern detectors
        self.pattern_detectors.update({
            "environment_risk": self._assess_environment_risk,
            "anomalous_behavior": self._detect_anomalous_behavior
        })
        
    def observe_security_context(self, data: dict):
        """
        Specialized observation function for security contexts.
        
        Args:
            data (dict): Security and environmental data
            
        Example:
        {
            "sensor_type": "location",
            "location_type": "public",
            "crowd_density": 0.7,
            "time_of_day": "night",
            "area_familiarity": 0.2,
            "accompanied": False,
            "unusual_activity": False, 
            "tags": ["unfamiliar", "transit"],
            "intensity": 0.6
        }
        """
        # Process security-specific indicators
        if "location_type" in data and "area_familiarity" in data:
            # Track known locations
            if data["area_familiarity"] > 0.8 and "location_id" in data:
                if data["location_id"] not in self.security_metrics["trusted_locations"]:
                    self.security_metrics["trusted_locations"].append(data["location_id"])
        
        # Track contacts
        if "contact_info" in data and "trust_level" in data["contact_info"]:
            contact_id = data["contact_info"].get("contact_id", None)
            if contact_id and data["contact_info"]["trust_level"] > 0.8:
                if contact_id not in self.security_metrics["trusted_contacts"]:
                    self.security_metrics["trusted_contacts"].append(contact_id)
        
        # Check for explicitly flagged security concerns
        if "unusual_activity" in data and data["unusual_activity"]:
            if "activity_type" in data:
                anomaly = data["activity_type"]
                if anomaly not in self.security_metrics["anomalies_detected"]:
                    self.security_metrics["anomalies_detected"].append(anomaly)
                    
        # Pass to standard observation
        self.observe(data)
        
        # Perform risk assessment if sufficient time has passed
        current_time = time.time()
        if current_time - self.security_metrics["last_risk_assessment"] > 60:  # Reassess every minute
            self._update_risk_assessment()
            self.security_metrics["last_risk_assessment"] = current_time
    
    def inform_security_behavior(self) -> dict:
        """
        Provides security-specific behavioral insights.
        """
        # Get base behavior analysis
        behavior = self.inform_behavior()
        
        # Get recent memories for assessment
        memories = self._get_recent_memories()
        
        # Run security-specific assessments
        environment_assessment = self.pattern_detectors["environment_risk"](memories)
        anomaly_assessment = self.pattern_detectors["anomalous_behavior"](memories)
        
        # Combine for security-specific insights
        security_insights = {
            "environment_risk_level": self.security_metrics["environment_risk_level"],
            "detected_anomalies": self.security_metrics["anomalies_detected"],
            "suspicious_patterns": self.security_metrics["suspicious_patterns"],
            "in_trusted_location": self._is_in_trusted_location(memories),
            "environment_safety": environment_assessment["risk_level"],
            "behavioral_anomalies": anomaly_assessment["anomaly_types"],
            "recommended_action": self._determine_security_action(behavior, environment_assessment, anomaly_assessment)
        }
        
        return {**behavior, **security_insights}
    
    def _assess_environment_risk(self, memories):
        """Assess environmental risk based on location, time, and other factors"""
        result = {
            "risk_level": "normal",
            "risk_factors": [],
            "confidence": 0.5
        }
        
        if len(memories) < 2:
            return result
            
        # Risk factors to evaluate
        risk_factors = []
        risk_score = 0
        
        for memory in memories:
            data = memory["data"]
            
            # Evaluate location safety
            if "location_type" in data:
                if data["location_type"] == "isolated":
                    risk_factors.append("isolated_location")
                    risk_score += 2
                elif data["location_type"] == "crowd" and data.get("crowd_density", 0) > 0.8:
                    risk_factors.append("dense_crowd")
                    risk_score += 1
            
            # Consider time factors
            if "time_of_day" in data and data["time_of_day"] == "night":
                risk_factors.append("nighttime")
                risk_score += 1
                
            # Consider familiarity
            if "area_familiarity" in data and data["area_familiarity"] < 0.3:
                risk_factors.append("unfamiliar_area")
                risk_score += 1
                
            # Consider accompaniment
            if "accompanied" in data and not data["accompanied"]:
                risk_factors.append("unaccompanied")
                risk_score += 1
        
        # Determine risk level
        if risk_score >= 4:
            result["risk_level"] = "elevated"
            result["confidence"] = 0.8
        elif risk_score >= 2:
            result["risk_level"] = "moderate"
            result["confidence"] = 0.7
        else:
            result["risk_level"] = "normal"
            result["confidence"] = 0.6
        
        result["risk_factors"] = list(set(risk_factors))  # Deduplicate
        return result
    
    def _detect_anomalous_behavior(self, memories):
        """Detect anomalous behavior patterns that might indicate security concerns"""
        result = {
            "anomalies_detected": False,
            "anomaly_types": [],
            "confidence": 0.5
        }
        
        if len(memories) < 3:
            return result
            
        # Signs of potential security concerns
        unusual_patterns = []
        
        # Check for rapid location changes
        locations = []
        timestamps = []
        
        for memory in memories:
            data = memory["data"]
            if "location_id" in data:
                locations.append(data["location_id"])
                timestamps.append(memory["timestamp"])
        
        if len(locations) >= 3 and len(timestamps) >= 3:
            # Check for unusually rapid transitions
            for i in range(len(locations) - 1):
                if locations[i] != locations[i+1]:
                    time_diff = timestamps[i+1] - timestamps[i]
                    if time_diff < 60:  # Less than a minute between location changes
                        unusual_patterns.append("rapid_location_change")
                        break
        
        # Check for unusual activity flags
        for memory in memories:
            data = memory["data"]
            if "unusual_activity" in data and data["unusual_activity"]:
                if "activity_type" in data and data["activity_type"] not in unusual_patterns:
                    unusual_patterns.append(data["activity_type"])
        
        # Set result based on detected patterns
        if unusual_patterns:
            result["anomalies_detected"] = True
            result["anomaly_types"] = unusual_patterns
            result["confidence"] = 0.7
            
        return result
    
    def _is_in_trusted_location(self, memories):
        """Determine if current location is trusted based on recent observations"""
        if not memories:
            return False
            
        # Check most recent memory for location
        latest_memory = memories[-1]["data"]
        if "location_id" in latest_memory:
            return latest_memory["location_id"] in self.security_metrics["trusted_locations"]
            
        return False
        
    def _update_risk_assessment(self):
        """Update overall risk assessment based on recent observations"""
        memories = self._get_recent_memories()
        if not memories:
            return
            
        environment_risk = self.pattern_detectors["environment_risk"](memories)
        anomaly_risk = self.pattern_detectors["anomalous_behavior"](memories)
        
        # Update environment risk level
        self.security_metrics["environment_risk_level"] = environment_risk["risk_level"]
        
        # Update suspicious patterns
        if anomaly_risk["anomalies_detected"]:
            for anomaly in anomaly_risk["anomaly_types"]:
                if anomaly not in self.security_metrics["suspicious_patterns"]:
                    self.security_metrics["suspicious_patterns"].append(anomaly)
    
    def _determine_security_action(self, behavior, environment, anomalies):
        """Determine appropriate security action based on assessments"""
        # Critical risk factors take highest priority
        if environment["risk_level"] == "elevated" and behavior["emotional_trend"] == "volatile":
            return "suggest_leaving_situation"
            
        # Check for concerning anomalies
        if anomalies["anomalies_detected"] and "rapid_location_change" in anomalies["anomaly_types"]:
            return "verify_location_safety"
            
        # Moderate risk situations
        if environment["risk_level"] == "moderate":
            return "increase_awareness"
            
        # Normal situations but unfamiliar
        if "unfamiliar_area" in environment["risk_factors"]:
            return "provide_area_information"
            
        # Everything seems fine
        return "normal_monitoring"


class HealthcareShadowMemory(ShadowMemory):
    """
    Enhanced ShadowMemory for healthcare and wellness contexts.
    Focused on patient well-being, adherence patterns, and health trends.
    """
    
    def __init__(self, **kwargs):
        """Initialize HealthcareShadowMemory with healthcare-specific parameters"""
        super().__init__(**kwargs)
        self.healthcare_metrics = {
            "adherence_patterns": {},
            "wellness_indicators": [],
            "stress_levels": [],
            "routine_consistency": 0.5,
            "health_trends": {}
        }
        
        # Add healthcare-specific pattern detectors
        self.pattern_detectors.update({
            "adherence_analysis": self._analyze_adherence_patterns,
            "wellness_trend": self._analyze_wellness_trends
        })
        
    def observe_healthcare_context(self, data: dict):
        """
        Specialized observation function for healthcare contexts.
        
        Args:
            data (dict): Healthcare and wellness data
            
        Example:
        {
            "sensor_type": "health_event",
            "event_type": "medication",
            "adherence": True,
            "scheduled_time": "08:00:00",
            "actual_time": "08:15:00",
            "wellness_indicators": ["energy", "mood"],
            "wellness_scores": {"energy": 0.7, "mood": 0.8},
            "stress_level": 0.4,
            "tags": ["morning_routine", "medication"],
            "intensity": 0.3
        }
        """
        # Process healthcare-specific indicators
        
        # Track adherence patterns
        if "event_type" in data and "adherence" in data:
            event_type = data["event_type"]
            if event_type not in self.healthcare_metrics["adherence_patterns"]:
                self.healthcare_metrics["adherence_patterns"][event_type] = {
                    "adherence_count": 0,
                    "missed_count": 0,
                    "total_count": 0
                }
                
            self.healthcare_metrics["adherence_patterns"][event_type]["total_count"] += 1
            if data["adherence"]:
                self.healthcare_metrics["adherence_patterns"][event_type]["adherence_count"] += 1
            else:
                self.healthcare_metrics["adherence_patterns"][event_type]["missed_count"] += 1
                
        # Track wellness indicators
        if "wellness_scores" in data:
            for indicator, score in data["wellness_scores"].items():
                if indicator not in self.healthcare_metrics["health_trends"]:
                    self.healthcare_metrics["health_trends"][indicator] = []
                
                self.healthcare_metrics["health_trends"][indicator].append({
                    "timestamp": time.time(),
                    "score": score
                })
                
        # Track stress levels
        if "stress_level" in data:
            self.healthcare_metrics["stress_levels"].append({
                "timestamp": time.time(),
                "level": data["stress_level"]
            })
                    
        # Pass to standard observation
        self.observe(data)
        
    def inform_healthcare_behavior(self) -> dict:
        """
        Provides healthcare-specific behavioral insights.
        """
        # Get base behavior analysis
        behavior = self.inform_behavior()
        
        # Get recent memories for assessment
        memories = self._get_recent_memories()
        
        # Run healthcare-specific assessments
        adherence_assessment = self.pattern_detectors["adherence_analysis"](memories)
        wellness_assessment = self.pattern_detectors["wellness_trend"](memories)
        
        # Calculate current stress level
        current_stress = self._calculate_current_stress()
        
        # Combine for healthcare-specific insights
        healthcare_insights = {
            "adherence_quality": adherence_assessment["adherence_quality"],
            "adherence_consistency": adherence_assessment["consistency"],
            "wellness_trend": wellness_assessment["trend"],
            "current_stress_level": current_stress,
            "routine_consistency": self._calculate_routine_consistency(),
            "recommended_action": self._determine_healthcare_action(behavior, adherence_assessment, wellness_assessment)
        }
        
        return {**behavior, **healthcare_insights}
    
    def _analyze_adherence_patterns(self, memories):
        """Analyze adherence patterns for healthcare routines"""
        result = {
            "adherence_quality": "moderate",
            "missed_events": [],
            "consistency": 0.5,
            "confidence": 0.5
        }
        
        if len(memories) < 3:
            return result
            
        # Analyze adherence events
        adherence_events = 0
        total_events = 0
        missed_types = []
        
        for memory in memories:
            data = memory["data"]
            if "event_type" in data and "adherence" in data:
                total_events += 1
                if data["adherence"]:
                    adherence_events += 1
                else:
                    if data["event_type"] not in missed_types:
                        missed_types.append(data["event_type"])
        
        # Calculate adherence rate
        if total_events > 0:
            adherence_rate = adherence_events / total_events
            
            # Determine quality level
            if adherence_rate > 0.8:
                result["adherence_quality"] = "high"
                result["confidence"] = 0.8
            elif adherence_rate < 0.5:
                result["adherence_quality"] = "low"
                result["confidence"] = 0.7
            else:
                result["adherence_quality"] = "moderate"
                result["confidence"] = 0.6
                
            result["consistency"] = adherence_rate
            result["missed_events"] = missed_types
            
        return result
    
    def _analyze_wellness_trends(self, memories):
        """Analyze wellness trends over time"""
        result = {
            "trend": "stable",
            "indicators": {},
            "confidence": 0.5
        }
        
        if len(memories) < 3:
            return result
            
        # Extract wellness indicators
        indicators = {}
        
        for memory in memories:
            data = memory["data"]
            if "wellness_scores" in data:
                for indicator, score in data["wellness_scores"].items():
                    if indicator not in indicators:
                        indicators[indicator] = []
                    indicators[indicator].append({
                        "timestamp": memory["timestamp"],
                        "score": score
                    })
        
        # Analyze trends for each indicator
        overall_trend = 0
        for indicator, values in indicators.items():
            if len(values) < 3:
                continue
                
            # Sort by timestamp
            values.sort(key=lambda x: x["timestamp"])
            
            # Calculate simple linear trend
            start_score = values[0]["score"]
            end_score = values[-1]["score"]
            trend = end_score - start_score
            
            result["indicators"][indicator] = "improving" if trend > 0.1 else (
                "declining" if trend < -0.1 else "stable"
            )
            
            # Contribute to overall trend
            overall_trend += trend
        
        # Determine overall trend if we have indicators
        if indicators:
            avg_trend = overall_trend / len(indicators)
            result["trend"] = "improving" if avg_trend > 0.1 else (
                "declining" if avg_trend < -0.1 else "stable"
            )
            result["confidence"] = min(0.7, 0.4 + 0.1 * len(indicators))
            
        return result
    
    def _calculate_current_stress(self):
        """Calculate current stress level based on recent measurements"""
        if not self.healthcare_metrics["stress_levels"]:
            return "unknown"
            
        # Get stress levels from last 3 hours
        current_time = time.time()
        recent_stress = [
            entry["level"] for entry in self.healthcare_metrics["stress_levels"]
            if current_time - entry["timestamp"] < 10800  # 3 hours
        ]
        
        if not recent_stress:
            return "unknown"
            
        # Calculate average recent stress
        avg_stress = sum(recent_stress) / len(recent_stress)
        
        # Categorize
        if avg_stress > 0.7:
            return "high"
        elif avg_stress < 0.3:
            return "low"
        else:
            return "moderate"
    
    def _calculate_routine_consistency(self):
        """Calculate consistency of healthcare routines"""
        adherence_patterns = self.healthcare_metrics["adherence_patterns"]
        if not adherence_patterns:
            return 0.5  # Default moderate
            
        # Calculate overall adherence rate
        total_adherence = 0
        total_events = 0
        
        for event_type, counts in adherence_patterns.items():
            total_adherence += counts["adherence_count"]
            total_events += counts["total_count"]
            
        if total_events == 0:
            return 0.5
            
        return total_adherence / total_events
    
    def _determine_healthcare_action(self, behavior, adherence, wellness):
        """Determine appropriate healthcare action based on assessments"""
        # High stress conditions
        if self._calculate_current_stress() == "high":
            return "suggest_stress_management"
            
        # Low adherence
        if adherence["adherence_quality"] == "low":
            return "encourage_adherence"
            
        # Declining wellness
        if wellness["trend"] == "declining":
            return "wellness_check"
            
        # Everything looks good
        if adherence["adherence_quality"] == "high" and wellness["trend"] == "improving":
            return "positive_reinforcement"
            
        # Default moderate approach
        return "maintain_routine"


# Factory class to create the appropriate ShadowMemory based on context
class ShadowMemoryFactory:
    """Factory for creating domain-specific ShadowMemory instances"""
    
    @staticmethod
    def create_shadow_memory(domain: str, **kwargs) -> ShadowMemory:
        """
        Create and return appropriate ShadowMemory subclass for the specified domain.
        
        Args:
            domain: Domain specifier ("education", "relationship", "security", "healthcare")
            **kwargs: Additional parameters to pass to the ShadowMemory constructor
            
        Returns:
            Instance of appropriate ShadowMemory subclass
        """
        if domain.lower() == "education":
            return EducationalShadowMemory(**kwargs)
        elif domain.lower() in ["relationship", "dating"]:
            return RelationshipShadowMemory(**kwargs)
        elif domain.lower() in ["security", "safety"]:
            return SecurityShadowMemory(**kwargs)
        elif domain.lower() in ["healthcare", "health", "wellness"]:
            return HealthcareShadowMemory(**kwargs)
        else:
            # Default to base ShadowMemory
            logger.info(f"Using base ShadowMemory for unrecognized domain: {domain}")
            return ShadowMemory(**kwargs)
