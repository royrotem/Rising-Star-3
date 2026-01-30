"""
System Recommendation Service

Generates AI-powered recommendations for system configuration based on
analyzed data, including system type detection, name suggestion, and
metadata-driven insights.
"""

import re
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


# ─── System-Type Detection Keywords ──────────────────────────────────

_SYSTEM_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "vehicle": [
        "speed", "velocity", "rpm", "engine", "motor", "battery", "fuel", "odometer",
        "gps", "latitude", "longitude", "steering", "brake", "throttle", "gear",
        "wheel", "tire", "acceleration", "can_bus", "obd",
    ],
    "robot": [
        "joint", "axis", "torque", "servo", "gripper", "end_effector", "pose",
        "position", "orientation", "robot", "arm", "actuator", "encoder", "dof",
    ],
    "medical_device": [
        "patient", "heart", "ecg", "ekg", "blood", "pressure", "pulse", "oxygen",
        "saturation", "temperature", "respiration", "mri", "ct", "scan", "dose",
    ],
    "aerospace": [
        "altitude", "airspeed", "heading", "pitch", "roll", "yaw", "thrust",
        "fuel_flow", "engine", "flap", "rudder", "aileron", "flight",
    ],
    "industrial": [
        "pump", "valve", "flow", "pressure", "level", "tank", "motor",
        "conveyor", "plc", "scada", "process", "production", "machine",
        "predictive maintenance", "fault detection", "anomaly", "sensor",
        "vibration", "acoustic", "machine_id", "equipment", "health",
    ],
}

_TYPE_DISPLAY_NAMES: Dict[str, str] = {
    "vehicle": "Vehicle Telemetry System",
    "robot": "Robot Control System",
    "medical_device": "Medical Monitoring System",
    "aerospace": "Flight Data System",
    "industrial": "Industrial Process System",
}

_METADATA_TYPE_PATTERNS: Dict[str, List[str]] = {
    "industrial": [
        "industrial", "machine", "production", "manufacturing",
        "predictive maintenance", "fault detection", "equipment health",
        "sensor network", "iot",
    ],
    "vehicle": ["vehicle", "automotive", "car", "truck", "fleet", "telematics", "driving"],
    "robot": ["robot", "robotic", "automation", "arm", "manipulator"],
    "medical_device": ["medical", "patient", "health", "clinical", "diagnostic"],
    "aerospace": ["aerospace", "flight", "aircraft", "aviation", "drone", "uav"],
}


# ─── Public API ───────────────────────────────────────────────────────

def generate_system_recommendation(
    file_summaries: List[Dict],
    discovered_fields: List[Dict],
    metadata_list: Optional[List[Dict]] = None,
) -> Dict:
    """
    Generate AI recommendations for system configuration based on analyzed data.

    Uses field names, metadata descriptions, and file structure to suggest
    the best system type, name, and description.
    """
    metadata_list = metadata_list or []

    total_records = sum(s.get("record_count", 0) for s in file_summaries)
    all_fields = [f.get("name", "").lower() for f in discovered_fields]
    all_units = [f.get("physical_unit", "") for f in discovered_fields if f.get("physical_unit")]

    metadata_insights = _extract_metadata_insights(metadata_list)

    # Score each system type by keyword matches
    scores = {
        sys_type: sum(1 for f in all_fields if any(k in f for k in keywords))
        for sys_type, keywords in _SYSTEM_TYPE_KEYWORDS.items()
    }

    # Boost from metadata
    detected_type = metadata_insights.get("detected_type")
    if detected_type and detected_type in scores:
        scores[detected_type] += 5

    suggested_type = max(scores, key=scores.get) if max(scores.values()) > 0 else "industrial"
    confidence = (
        min(0.95, max(scores.values()) / max(len(all_fields), 1) + 0.5)
        if all_fields
        else 0.5
    )

    suggested_name = _suggest_name(file_summaries, suggested_type)
    suggested_description = (
        metadata_insights.get("description")
        or _generate_description(suggested_type, discovered_fields, file_summaries, total_records, all_fields)
    )
    reasoning = _build_reasoning(
        suggested_type, scores, all_fields, all_units, metadata_insights
    )

    return {
        "suggested_name": suggested_name,
        "suggested_type": suggested_type,
        "suggested_description": suggested_description,
        "confidence": confidence,
        "reasoning": reasoning,
        "analysis_summary": {
            "files_analyzed": len(file_summaries),
            "total_records": total_records,
            "unique_fields": len(set(all_fields)),
            "detected_units": list(set(all_units))[:10],
            "metadata_found": len(metadata_list) > 0,
        },
    }


def enrich_fields_with_context(
    discovered_fields: List[Dict],
    field_descriptions: Dict[str, str],
    context_texts: List[str],
) -> List[Dict]:
    """
    Second pass: enrich discovered fields with context extracted from all files.

    Updates field meanings based on metadata descriptions found anywhere in the data.
    """
    combined_context = " ".join(context_texts)

    for field in discovered_fields:
        field_name = field.get("name", "")

        # Priority 1: Direct field description from metadata
        if field_name in field_descriptions:
            field["inferred_meaning"] = field_descriptions[field_name]
            field["meaning_source"] = "metadata_description"
            field["confidence"] = min(field.get("confidence", 0.5) + 0.3, 1.0)
            continue

        # Priority 2: Search combined context for mentions of this field
        if combined_context and field.get("inferred_meaning", "").startswith("Unknown"):
            patterns = [
                rf"\b{re.escape(field_name)}\s*[:–-]\s*([^.!?\n\u2B50]+[.!?]?)",
                rf"\b{re.escape(field_name)}\b[^:]*?"
                rf"(?:is|are|represents?|measures?|captures?|records?|indicates?)\s+"
                rf"([^.!?\n]+[.!?]?)",
            ]
            for pattern in patterns:
                match = re.search(pattern, combined_context, re.IGNORECASE)
                if match:
                    desc = re.sub(r"\s+", " ", match.group(1).strip())
                    if 10 < len(desc) < 300:
                        field["inferred_meaning"] = desc
                        field["meaning_source"] = "context_extraction"
                        field["confidence"] = min(field.get("confidence", 0.5) + 0.2, 1.0)
                        break

    return discovered_fields


def build_data_profile(
    records: List[Dict],
    discovered_schema: List[Dict],
) -> Dict[str, Any]:
    """Build a data profile dictionary for AI agents from raw records and schema."""
    if not records:
        return {"record_count": 0, "field_count": 0, "fields": [], "sample_rows": []}

    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    fields = []
    for col in df.columns:
        field_info: Dict[str, Any] = {"name": col, "type": str(df[col].dtype)}
        if col in numeric_cols:
            field_info["mean"] = _safe_float(df[col].mean())
            field_info["std"] = _safe_float(df[col].std())
            field_info["min"] = _safe_float(df[col].min())
            field_info["max"] = _safe_float(df[col].max())
            field_info["median"] = _safe_float(df[col].median())
        else:
            field_info["unique_count"] = int(df[col].nunique())
            top_values = df[col].value_counts().head(5).to_dict()
            field_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        fields.append(field_info)

    correlations = _compute_top_correlations(df, numeric_cols)
    sample_rows = df.head(5).to_dict("records")

    return {
        "record_count": len(df),
        "field_count": len(df.columns),
        "fields": fields,
        "sample_rows": sample_rows,
        "correlations": correlations,
    }


def titles_overlap(title_a: str, title_b: str) -> bool:
    """Check if two anomaly titles refer to the same issue (word-overlap heuristic)."""
    a_words = set(title_a.lower().split())
    b_words = set(title_b.lower().split())
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words)
    return overlap / min(len(a_words), len(b_words)) > 0.5


# ─── Private helpers ──────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None for NaN/inf."""
    try:
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _compute_top_correlations(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> Dict[str, float]:
    """Return correlations above 0.3 between numeric columns."""
    correlations: Dict[str, float] = {}
    if len(numeric_cols) < 2:
        return correlations
    try:
        corr_matrix = df[numeric_cols].corr()
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                val = corr_matrix.loc[col_a, col_b]
                if not np.isnan(val) and abs(val) > 0.3:
                    correlations[f"{col_a} vs {col_b}"] = round(float(val), 3)
    except Exception:
        pass
    return correlations


def _extract_metadata_insights(metadata_list: List[Dict]) -> Dict[str, Any]:
    """Extract insights from dataset metadata descriptions."""
    insights: Dict[str, Any] = {
        "detected_type": None,
        "purpose": None,
        "description": None,
    }

    if not metadata_list:
        return insights

    all_descriptions = " ".join(
        m.get("dataset_description", "") for m in metadata_list if m.get("dataset_description")
    )
    if not all_descriptions:
        return insights

    desc_lower = all_descriptions.lower()

    # Detect system type from metadata
    type_scores = {
        sys_type: sum(1 for p in patterns if p in desc_lower)
        for sys_type, patterns in _METADATA_TYPE_PATTERNS.items()
    }
    if max(type_scores.values()) > 0:
        insights["detected_type"] = max(type_scores, key=type_scores.get)

    # Extract purpose
    purpose_patterns = [
        r"designed to support ([^.]+)",
        r"used for ([^.]+)",
        r"suitable for ([^.]+)",
        r"support[s]? ([^.]*(?:maintenance|detection|monitoring|analysis)[^.]*)",
    ]
    for pattern in purpose_patterns:
        match = re.search(pattern, desc_lower)
        if match:
            insights["purpose"] = match.group(1).strip()[:150]
            break

    # Use first sentence as suggested description
    if len(all_descriptions) > 50:
        first_sentence = re.match(r"^[^.!?]+[.!?]", all_descriptions)
        insights["description"] = (
            first_sentence.group(0).strip() if first_sentence else all_descriptions[:300].strip()
        )

    return insights


def _suggest_name(file_summaries: List[Dict], suggested_type: str) -> str:
    """Generate a suggested system name from filenames or type."""
    file_names = [s.get("filename", "") for s in file_summaries]
    for fn in file_names:
        clean_name = fn.replace("_", " ").replace("-", " ").split(".")[0]
        if len(clean_name) > 3:
            return f"{clean_name.title()} System"
    return _TYPE_DISPLAY_NAMES.get(suggested_type, "Data System")


def _generate_description(
    suggested_type: str,
    discovered_fields: List[Dict],
    file_summaries: List[Dict],
    total_records: int,
    all_fields: List[str],
) -> str:
    """Generate a description for the system based on its type and data."""
    field_sample = ", ".join(all_fields[:3]) if all_fields else ""
    n_fields = len(discovered_fields)
    n_sources = len(file_summaries)

    templates = {
        "vehicle": f"Vehicle telemetry system monitoring {n_fields} parameters including {field_sample}. Data collected from {n_sources} source(s) with {total_records} total records.",
        "robot": f"Robotic system with {n_fields} monitored parameters. Tracking {field_sample} from {n_sources} data source(s).",
        "medical_device": f"Medical device monitoring {n_fields} health parameters from {n_sources} source(s).",
        "aerospace": f"Aerospace system tracking {n_fields} flight parameters from {n_sources} data source(s).",
        "industrial": f"Industrial process system monitoring {n_fields} parameters from {n_sources} source(s).",
    }
    return templates.get(suggested_type, "System monitoring and analysis.")


def _build_reasoning(
    suggested_type: str,
    scores: Dict[str, int],
    all_fields: List[str],
    all_units: List[str],
    metadata_insights: Dict[str, Any],
) -> str:
    """Build a human-readable reasoning string."""
    parts: List[str] = []

    if metadata_insights.get("purpose"):
        parts.append(f"Dataset purpose: {metadata_insights['purpose']}")
    if metadata_insights.get("detected_type"):
        parts.append(f"Metadata indicates {metadata_insights['detected_type']} system")

    keywords = _SYSTEM_TYPE_KEYWORDS.get(suggested_type, [])
    if scores.get(suggested_type, 0) > 0:
        matching = [f for f in all_fields if any(k in f for k in keywords)]
        parts.append(f"Found {scores[suggested_type]} field(s) matching {suggested_type} patterns")
        if matching[:3]:
            parts.append(f"Key indicators: {', '.join(matching[:3])}")

    if all_units:
        parts.append(f"Detected physical units: {', '.join(list(set(all_units))[:5])}")

    return ". ".join(parts) if parts else "Based on general data structure analysis."
