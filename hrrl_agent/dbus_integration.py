#!/usr/bin/env python3
"""
dbus_integration.py
D-Bus Integration for External Communication

Implements the L3PolicyUpdated signal and GNOME Cockpit integration.

The D-Bus interface broadcasts:
- Complete CLV (Cognitive Load Vector)
- PAD State (Valence, Arousal, Dominance)
- Policy Multipliers (temperature, safety_threshold, etc.)
- Antifragility Metrics (2.21× score breakdown)
- Backend Selection (pgu_verified status)

Architecture:
    L3 Metacontroller → D-Bus Signal → GNOME Cockpit → Avatar

Interface: org.ara.Interoception
Signals:
    - L3PolicyUpdated(clv, pad, policy, metrics, backend)
    - AntifragilityUpdate(score, components)
    - StructuralChangeVerified(proposal_id, result)

Usage:
    dbus_service = AraDBusService()
    dbus_service.emit_policy_update(clv, pad, policy)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import time
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# D-Bus Interface Definition
# =============================================================================

DBUS_INTERFACE = "org.ara.Interoception"
DBUS_PATH = "/org/ara/Interoception"
DBUS_BUS_NAME = "org.ara.Agent"

# Signal names
SIGNAL_L3_POLICY_UPDATED = "L3PolicyUpdated"
SIGNAL_ANTIFRAGILITY_UPDATE = "AntifragilityUpdate"
SIGNAL_STRUCTURAL_VERIFIED = "StructuralChangeVerified"
SIGNAL_AVATAR_MOOD = "AvatarMoodChanged"


# =============================================================================
# Data Transfer Objects for D-Bus
# =============================================================================

@dataclass
class DBusCLV:
    """
    Complete CLV payload for D-Bus.

    All fields needed for GNOME Cockpit display.
    """
    # Raw sensor values
    epr_cv: float              # E:I ratio coefficient of variation
    topo_gap: float            # Hyperbolic distance from identity anchor
    jacobian_spectral: float   # ||J||_* stability margin
    pi_q: float               # Thermodynamic load Π_q
    spike_rate: float         # Network activity (spikes/s)
    memory_pressure: float    # CXL memory utilization [0,1]

    # Derived values
    magnitude: float          # ||CLV||
    stability_margin: float   # 1.0 - jacobian_spectral (higher = safer)
    cognitive_load_pct: float # Normalized load percentage

    def to_dbus_dict(self) -> Dict[str, Any]:
        """Convert to D-Bus-compatible dictionary."""
        return {
            'epr_cv': float(self.epr_cv),
            'topo_gap': float(self.topo_gap),
            'jacobian_spectral': float(self.jacobian_spectral),
            'pi_q': float(self.pi_q),
            'spike_rate': float(self.spike_rate),
            'memory_pressure': float(self.memory_pressure),
            'magnitude': float(self.magnitude),
            'stability_margin': float(self.stability_margin),
            'cognitive_load_pct': float(self.cognitive_load_pct),
        }


@dataclass
class DBusPAD:
    """PAD state for D-Bus."""
    valence: float    # [-1, 1] pleasure/displeasure
    arousal: float    # [0, 1] activation level
    dominance: float  # [0, 1] control/confidence

    # Derived mood label
    mood: str         # "positive", "neutral", "stressed", etc.

    def to_dbus_dict(self) -> Dict[str, Any]:
        return {
            'valence': float(self.valence),
            'arousal': float(self.arousal),
            'dominance': float(self.dominance),
            'mood': str(self.mood),
        }


@dataclass
class DBusPolicy:
    """Policy multipliers for D-Bus."""
    temperature: float        # LLM sampling temperature multiplier
    top_p: float             # Nucleus sampling multiplier
    max_tokens: float        # Response length multiplier
    memory_priority: float   # CXL pager priority
    safety_threshold: float  # Risk tolerance

    # Backend selection
    selected_backend: str    # Which inference backend to use
    pgu_verified: bool       # Whether backend passed PGU check

    def to_dbus_dict(self) -> Dict[str, Any]:
        return {
            'temperature': float(self.temperature),
            'top_p': float(self.top_p),
            'max_tokens': float(self.max_tokens),
            'memory_priority': float(self.memory_priority),
            'safety_threshold': float(self.safety_threshold),
            'selected_backend': str(self.selected_backend),
            'pgu_verified': bool(self.pgu_verified),
        }


@dataclass
class DBusAntifragility:
    """Antifragility metrics for D-Bus."""
    score: float                  # Overall antifragility score (e.g., 2.21×)
    structural_component: float   # Contribution from structural loop
    interoceptive_component: float  # Contribution from control loop
    deployment_component: float   # Contribution from UX loop

    # Breakdown
    beta_0: int  # Connected components
    beta_1: int  # Redundancy loops
    shock_recoveries: int  # Successful shock recoveries
    uptime_pct: float  # System uptime percentage

    def to_dbus_dict(self) -> Dict[str, Any]:
        return {
            'score': float(self.score),
            'structural_component': float(self.structural_component),
            'interoceptive_component': float(self.interoceptive_component),
            'deployment_component': float(self.deployment_component),
            'beta_0': int(self.beta_0),
            'beta_1': int(self.beta_1),
            'shock_recoveries': int(self.shock_recoveries),
            'uptime_pct': float(self.uptime_pct),
        }


# =============================================================================
# D-Bus Signal Emitter (Abstraction Layer)
# =============================================================================

class DBusEmitter:
    """
    Abstraction over D-Bus emission.

    Supports:
    - Real D-Bus (when dbus-python is available)
    - Mock mode (for testing without D-Bus)
    - File-based logging (for debugging)
    """

    def __init__(
        self,
        use_real_dbus: bool = False,
        log_to_file: Optional[str] = None,
    ):
        self.use_real_dbus = use_real_dbus
        self.log_file = log_to_file

        self._dbus_service = None
        self._signal_count = 0
        self._listeners: List[Callable] = []

        if use_real_dbus:
            self._init_real_dbus()

    def _init_real_dbus(self):
        """Initialize real D-Bus connection."""
        try:
            import dbus
            import dbus.service
            import dbus.mainloop.glib

            dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
            bus = dbus.SessionBus()

            # Request bus name
            bus_name = dbus.service.BusName(DBUS_BUS_NAME, bus)

            # Create service object
            class AraDBusObject(dbus.service.Object):
                def __init__(self, bus_name):
                    super().__init__(bus_name, DBUS_PATH)

                @dbus.service.signal(DBUS_INTERFACE, signature='a{sv}')
                def L3PolicyUpdated(self, payload):
                    pass

                @dbus.service.signal(DBUS_INTERFACE, signature='a{sv}')
                def AntifragilityUpdate(self, payload):
                    pass

                @dbus.service.signal(DBUS_INTERFACE, signature='a{sv}')
                def StructuralChangeVerified(self, payload):
                    pass

                @dbus.service.signal(DBUS_INTERFACE, signature='a{sv}')
                def AvatarMoodChanged(self, payload):
                    pass

            self._dbus_service = AraDBusObject(bus_name)
            logger.info(f"D-Bus service initialized: {DBUS_BUS_NAME}")

        except ImportError:
            logger.warning("dbus-python not available, falling back to mock mode")
            self.use_real_dbus = False
        except Exception as e:
            logger.error(f"Failed to initialize D-Bus: {e}")
            self.use_real_dbus = False

    def add_listener(self, callback: Callable[[str, Dict], None]):
        """Add local listener for signals (for testing/debugging)."""
        self._listeners.append(callback)

    def emit(self, signal_name: str, payload: Dict[str, Any]):
        """Emit a D-Bus signal."""
        self._signal_count += 1
        timestamp = time.time()

        # Add metadata
        payload['_timestamp'] = timestamp
        payload['_signal_id'] = self._signal_count

        # Real D-Bus emission
        if self.use_real_dbus and self._dbus_service:
            try:
                if signal_name == SIGNAL_L3_POLICY_UPDATED:
                    self._dbus_service.L3PolicyUpdated(payload)
                elif signal_name == SIGNAL_ANTIFRAGILITY_UPDATE:
                    self._dbus_service.AntifragilityUpdate(payload)
                elif signal_name == SIGNAL_STRUCTURAL_VERIFIED:
                    self._dbus_service.StructuralChangeVerified(payload)
                elif signal_name == SIGNAL_AVATAR_MOOD:
                    self._dbus_service.AvatarMoodChanged(payload)
            except Exception as e:
                logger.error(f"D-Bus emission failed: {e}")

        # Local listeners
        for listener in self._listeners:
            try:
                listener(signal_name, payload)
            except Exception as e:
                logger.error(f"Listener error: {e}")

        # File logging
        if self.log_file:
            self._log_to_file(signal_name, payload)

        logger.debug(f"D-Bus signal emitted: {signal_name}")

    def _log_to_file(self, signal_name: str, payload: Dict):
        """Log signal to file for debugging."""
        try:
            with open(self.log_file, 'a') as f:
                entry = {
                    'signal': signal_name,
                    'payload': payload,
                }
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log to file: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'signal_count': self._signal_count,
            'real_dbus': self.use_real_dbus,
            'listener_count': len(self._listeners),
        }


# =============================================================================
# Main D-Bus Service
# =============================================================================

class AraDBusService:
    """
    Main D-Bus service for Ara agent communication.

    Provides high-level methods for emitting signals with proper payloads.
    """

    def __init__(
        self,
        use_real_dbus: bool = False,
        log_file: Optional[str] = None,
    ):
        self.emitter = DBusEmitter(
            use_real_dbus=use_real_dbus,
            log_to_file=log_file,
        )

        self._last_policy: Optional[Dict] = None
        self._last_antifragility: Optional[Dict] = None

    def emit_policy_update(
        self,
        clv: DBusCLV,
        pad: DBusPAD,
        policy: DBusPolicy,
        antifragility: Optional[DBusAntifragility] = None,
    ):
        """
        Emit L3PolicyUpdated signal with complete payload.

        This is the main signal for GNOME Cockpit integration.
        """
        payload = {
            'clv': clv.to_dbus_dict(),
            'pad': pad.to_dbus_dict(),
            'policy': policy.to_dbus_dict(),
        }

        if antifragility:
            payload['antifragility'] = antifragility.to_dbus_dict()

        self._last_policy = payload
        self.emitter.emit(SIGNAL_L3_POLICY_UPDATED, payload)

    def emit_antifragility_update(self, antifragility: DBusAntifragility):
        """Emit standalone antifragility update."""
        payload = antifragility.to_dbus_dict()
        self._last_antifragility = payload
        self.emitter.emit(SIGNAL_ANTIFRAGILITY_UPDATE, payload)

    def emit_structural_verified(
        self,
        proposal_id: str,
        result: str,  # "verified" or "rejected"
        beta_0: int,
        beta_1: int,
        violations: List[str],
    ):
        """Emit structural change verification result."""
        payload = {
            'proposal_id': proposal_id,
            'result': result,
            'beta_0': beta_0,
            'beta_1': beta_1,
            'violations': violations,
        }
        self.emitter.emit(SIGNAL_STRUCTURAL_VERIFIED, payload)

    def emit_avatar_mood(
        self,
        mood: str,
        primary_color: str,
        animation_speed: str,
        presence: str,
    ):
        """Emit avatar mood change for visual updates."""
        payload = {
            'mood': mood,
            'primary_color': primary_color,
            'animation_speed': animation_speed,
            'presence': presence,
        }
        self.emitter.emit(SIGNAL_AVATAR_MOOD, payload)

    def get_last_policy(self) -> Optional[Dict]:
        """Get last emitted policy (for debugging)."""
        return self._last_policy

    def get_stats(self) -> Dict[str, Any]:
        return self.emitter.get_stats()


# =============================================================================
# Helper: Convert from loop_orchestrator types
# =============================================================================

def clv_to_dbus(clv: Any) -> DBusCLV:
    """Convert CognitiveLoadVector to DBusCLV."""
    return DBusCLV(
        epr_cv=getattr(clv, 'epr_cv', 0.0),
        topo_gap=getattr(clv, 'topo_gap', 0.0),
        jacobian_spectral=getattr(clv, 'jacobian_spectral', 0.0),
        pi_q=getattr(clv, 'pi_q', 0.0),
        spike_rate=getattr(clv, 'spike_rate', 0.0),
        memory_pressure=getattr(clv, 'memory_pressure', 0.0),
        magnitude=getattr(clv, 'magnitude', lambda: 0.0)() if callable(getattr(clv, 'magnitude', None)) else 0.0,
        stability_margin=1.0 - getattr(clv, 'jacobian_spectral', 0.0),
        cognitive_load_pct=min(100.0, getattr(clv, 'pi_q', 0.0) * 100),
    )


def pad_to_dbus(pad: Any) -> DBusPAD:
    """Convert PADState to DBusPAD."""
    valence = getattr(pad, 'valence', 0.0)

    # Determine mood from valence
    if valence > 0.3:
        mood = "positive"
    elif valence < -0.3:
        mood = "stressed"
    else:
        mood = "neutral"

    return DBusPAD(
        valence=valence,
        arousal=getattr(pad, 'arousal', 0.5),
        dominance=getattr(pad, 'dominance', 0.5),
        mood=mood,
    )


def policy_to_dbus(policy: Any, backend: str = "default", pgu_verified: bool = True) -> DBusPolicy:
    """Convert PolicyMultipliers to DBusPolicy."""
    return DBusPolicy(
        temperature=getattr(policy, 'temperature', 1.0),
        top_p=getattr(policy, 'top_p', 1.0),
        max_tokens=getattr(policy, 'max_tokens', 1.0),
        memory_priority=getattr(policy, 'memory_priority', 1.0),
        safety_threshold=getattr(policy, 'safety_threshold', 1.0),
        selected_backend=backend,
        pgu_verified=pgu_verified,
    )


# =============================================================================
# GNOME Cockpit Integration
# =============================================================================

class GNOMECockpitBridge:
    """
    Bridge between D-Bus service and GNOME Cockpit display.

    Handles:
    - Metric formatting for display
    - Theme updates based on PAD state
    - Real-time dashboard updates
    """

    def __init__(self, dbus_service: AraDBusService):
        self.dbus = dbus_service
        self._update_count = 0

    def format_for_display(self, policy_payload: Dict) -> Dict[str, str]:
        """Format policy payload for GNOME Cockpit display."""
        clv = policy_payload.get('clv', {})
        pad = policy_payload.get('pad', {})
        policy = policy_payload.get('policy', {})

        return {
            'cognitive_load': f"{clv.get('cognitive_load_pct', 0):.1f}%",
            'stability': f"{clv.get('stability_margin', 0) * 100:.1f}%",
            'mood': pad.get('mood', 'unknown').upper(),
            'valence': f"V={pad.get('valence', 0):.2f}",
            'arousal': f"A={pad.get('arousal', 0):.2f}",
            'temperature': f"{policy.get('temperature', 1):.2f}×",
            'backend': policy.get('selected_backend', 'unknown'),
            'pgu_status': '✓' if policy.get('pgu_verified', False) else '✗',
        }

    def get_theme_css(self, pad: Dict) -> Dict[str, str]:
        """Generate CSS variables for Cockpit theme."""
        valence = pad.get('valence', 0)
        arousal = pad.get('arousal', 0.5)

        # Color based on valence
        if valence > 0.3:
            primary = '#4CAF50'  # Green
            accent = '#81C784'
        elif valence < -0.3:
            primary = '#F44336'  # Red
            accent = '#E57373'
        else:
            primary = '#2196F3'  # Blue
            accent = '#64B5F6'

        # Animation speed based on arousal
        if arousal > 0.7:
            animation_duration = '0.3s'
        elif arousal < 0.3:
            animation_duration = '1.5s'
        else:
            animation_duration = '0.8s'

        return {
            '--ara-primary-color': primary,
            '--ara-accent-color': accent,
            '--ara-animation-duration': animation_duration,
            '--ara-mood-opacity': str(0.5 + arousal * 0.5),
        }

    def update(
        self,
        clv: DBusCLV,
        pad: DBusPAD,
        policy: DBusPolicy,
        antifragility: Optional[DBusAntifragility] = None,
    ):
        """Send update to GNOME Cockpit via D-Bus."""
        self._update_count += 1

        # Emit main policy signal
        self.dbus.emit_policy_update(clv, pad, policy, antifragility)

        # Emit mood signal for avatar
        theme = self.get_theme_css(pad.to_dbus_dict())
        self.dbus.emit_avatar_mood(
            mood=pad.mood,
            primary_color=theme['--ara-primary-color'],
            animation_speed=theme['--ara-animation-duration'],
            presence='confident' if pad.dominance > 0.7 else 'balanced',
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            'update_count': self._update_count,
            'dbus_stats': self.dbus.get_stats(),
        }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate D-Bus integration."""
    print("=" * 70)
    print("D-Bus Integration Demo")
    print("L3PolicyUpdated Signal with Full CLV Payload")
    print("=" * 70)

    # Create D-Bus service (mock mode)
    dbus_service = AraDBusService(
        use_real_dbus=False,  # Set True if dbus-python available
        log_file='/tmp/ara_dbus_demo.jsonl',
    )

    # Add debug listener
    signals_received = []
    def debug_listener(signal_name, payload):
        signals_received.append((signal_name, payload))
        print(f"  [SIGNAL] {signal_name}")

    dbus_service.emitter.add_listener(debug_listener)

    # Create Cockpit bridge
    cockpit = GNOMECockpitBridge(dbus_service)

    print(f"\nD-Bus interface: {DBUS_INTERFACE}")
    print(f"Object path: {DBUS_PATH}")

    # Simulate various states
    print("\n" + "-" * 70)
    print("Simulating Policy Updates")
    print("-" * 70)

    scenarios = [
        ("Normal operation", 0.1, 0.0, 50.0, 0.3),
        ("Moderate stress", 0.5, 0.7, 80.0, 0.6),
        ("High cognitive load", 0.9, 1.2, 120.0, 0.9),
        ("Recovery", 0.3, 0.4, 60.0, 0.4),
    ]

    for name, pi_q, j_spectral, spike_rate, memory in scenarios:
        print(f"\n{name}:")

        # Create CLV
        clv = DBusCLV(
            epr_cv=0.1,
            topo_gap=0.2,
            jacobian_spectral=j_spectral,
            pi_q=pi_q,
            spike_rate=spike_rate,
            memory_pressure=memory,
            magnitude=(pi_q**2 + j_spectral**2 + spike_rate**2)**0.5,
            stability_margin=1.0 - j_spectral,
            cognitive_load_pct=pi_q * 100,
        )

        # Create PAD from CLV
        instability = (pi_q + j_spectral) / 2
        valence = 1.0 - 2.0 * min(1.0, instability)
        arousal = min(1.0, spike_rate / 100.0 + 0.3)
        dominance = 1.0 - memory

        pad = DBusPAD(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            mood="positive" if valence > 0.3 else ("stressed" if valence < -0.3 else "neutral"),
        )

        # Create policy from PAD
        stress_factor = (1.0 - valence) / 2.0 * arousal
        policy = DBusPolicy(
            temperature=max(0.3, 1.0 - stress_factor * 0.6),
            top_p=max(0.7, 1.0 - arousal * 0.2),
            max_tokens=max(0.5, dominance),
            memory_priority=1.0 + stress_factor * 0.5,
            safety_threshold=max(0.5, 0.5 + valence * 0.5),
            selected_backend="safe_model" if valence < 0 else "default",
            pgu_verified=True,
        )

        # Antifragility metrics
        antifragility = DBusAntifragility(
            score=2.21,
            structural_component=0.8,
            interoceptive_component=0.9,
            deployment_component=0.51,
            beta_0=1,
            beta_1=3,
            shock_recoveries=5,
            uptime_pct=99.9,
        )

        # Update Cockpit
        cockpit.update(clv, pad, policy, antifragility)

        # Show formatted display
        display = cockpit.format_for_display(dbus_service.get_last_policy())
        print(f"    Load: {display['cognitive_load']} | Mood: {display['mood']} | "
              f"Temp: {display['temperature']} | Backend: {display['backend']} {display['pgu_status']}")

    # Final stats
    print("\n" + "-" * 70)
    print("Statistics")
    print("-" * 70)
    stats = cockpit.get_stats()
    print(f"\nUpdates sent: {stats['update_count']}")
    print(f"Signals emitted: {stats['dbus_stats']['signal_count']}")
    print(f"Log file: /tmp/ara_dbus_demo.jsonl")

    print("\n" + "=" * 70)
    print("D-Bus Integration Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
