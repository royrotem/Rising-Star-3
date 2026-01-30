"""
Baseline & Historical API for UAIE

Endpoints for capturing snapshots, viewing historical trends,
computing baselines, and comparing current data to baselines.
Additive feature — removing this file + router registration disables it.
"""

from fastapi import APIRouter, HTTPException

from ..services.baseline_store import baseline_store
from ..services.data_store import data_store

router = APIRouter(prefix="/baselines", tags=["baselines"])


@router.post("/systems/{system_id}/snapshot")
async def capture_snapshot(system_id: str):
    """Capture a snapshot of the current system data statistics."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = data_store.get_ingested_records(system_id)

    # Try to load latest analysis
    analysis = None
    try:
        import json
        from pathlib import Path
        import os
        data_dir = os.environ.get("DATA_DIR", "/app/data")
        if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        analysis_path = Path(data_dir) / "analyses" / f"{system_id}.json"
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text())
    except Exception:
        pass

    snapshot = baseline_store.capture_snapshot(system_id, system, records, analysis)
    return {"status": "captured", "snapshot": snapshot}


@router.get("/systems/{system_id}/history")
async def get_history(system_id: str, limit: int = 50):
    """Get historical snapshots for a system (most recent first)."""
    history = baseline_store.get_history(system_id, limit=limit)
    return {"system_id": system_id, "count": len(history), "snapshots": history}


@router.get("/systems/{system_id}/baseline")
async def get_baseline(system_id: str):
    """Compute aggregate baseline from all historical snapshots."""
    baseline = baseline_store.get_baseline(system_id)
    if not baseline:
        raise HTTPException(
            status_code=404,
            detail="No historical snapshots available. Capture snapshots first."
        )
    return baseline


@router.get("/systems/{system_id}/compare")
async def compare_to_baseline(system_id: str):
    """Compare current data against the historical baseline."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = data_store.get_ingested_records(system_id)
    result = baseline_store.compare_to_baseline(system_id, records)
    return result


@router.delete("/systems/{system_id}")
async def clear_history(system_id: str):
    """Clear all historical snapshots for a system."""
    baseline_store.clear(system_id)
    return {"status": "cleared", "system_id": system_id}
