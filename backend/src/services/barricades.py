"""Barricade-gated dispatch derivation.

A barricade is a rectangle laid over the survey board while road works are in
progress.  Two rules follow from it:

* a manhole **inside** the rectangle cannot be reached, so it is deferred and
  receives no work dispatch;
* two manholes closer than the original DBSCAN ``eps`` are **severed** when
  the straight link between them crosses the rectangle.  Sub-zones are
  therefore derived along the remaining neighbour links, using the original
  density threshold ``DBSCAN(eps=26 m, min_samples=3)``.

Derived sub-zones belong to a *partition revision*.  Whenever a barricade is
registered or removed the revision is bumped, sub-zones from the old revision
are invalidated and rebuilt with the remaining barricades.  Ranges that
already sit on a work notice are never touched: ``PointDisposition`` (unique
per point) records the single destination of every manhole, so a barricade
change racing a repair start can only leave a point on its original notice or
return it to the unassigned pool.
"""

from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy import select, text


from ..models import (
    Barricade,
    BarricadeRuleSet,
    PointDisposition,
    RepairZone,
    SurveyBatch,
    SurveyPoint,
    WorkNotice,
)
from ..constants.algorithm import EPS_METERS, MIN_SAMPLES
from .geometry import bounds, overlaps

LOCK_NAMESPACE = 1070
DERIVED_MARKER = "barricade"

# A zone in one of these states is still an open candidate for dispatch; a
# zone already handed to a work notice keeps its original order instead.
DISPATCHABLE_ZONE_STATUSES = {"proposed", "needs_split", "ready_for_notice"}


def _now():
    return datetime.now(timezone.utc)


def lock_batch(db, batch_id: int) -> None:
    """Serialize every rule change and notice/repair transition per batch.

    The advisory lock is transaction scoped, so two concurrent requests —
    removing a barricade while starting a repair, for example — observe one
    consistent view of notices, barricades and dispositions.
    """

    if db.bind is not None and db.bind.dialect.name == "postgresql":
        db.execute(
            text("SELECT pg_advisory_xact_lock(:ns, :batch)"),
            {"ns": LOCK_NAMESPACE, "batch": int(batch_id)},
        )


def active_barricades(db, batch_id: int):
    return db.execute(
        select(Barricade)
        .where(Barricade.batch_id == batch_id, Barricade.status == "active")
        .order_by(Barricade.id)
    ).scalars().all()


def rule_set(db, batch_id: int) -> BarricadeRuleSet:
    row = db.execute(
        select(BarricadeRuleSet).where(BarricadeRuleSet.batch_id == batch_id)
    ).scalar_one_or_none()
    if row is None:
        row = BarricadeRuleSet(batch_id=batch_id, revision=0, updated_at=_now())
        db.add(row)
        db.flush()
    return row


def revision_of(db, batch_id: int) -> int:
    return db.execute(
        select(BarricadeRuleSet.revision).where(BarricadeRuleSet.batch_id == batch_id)
    ).scalar_one_or_none() or 0


def bump_revision(db, batch_id: int) -> int:
    row = rule_set(db, batch_id)
    row.revision += 1
    row.updated_at = _now()
    db.flush()
    return row.revision


# --------------------------------------------------------------------------
# geometry


def normalize_boundary(raw) -> dict:
    try:
        boundary = {key: float(raw[key]) for key in ("min_x", "max_x", "min_y", "max_y")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("围挡边界需要 min_x / max_x / min_y / max_y 四个数值") from exc
    if boundary["min_x"] >= boundary["max_x"] or boundary["min_y"] >= boundary["max_y"]:
        raise ValueError("围挡边界的最小值必须小于最大值")
    return {key: round(value, 2) for key, value in boundary.items()}


def contains(boundary, x, y) -> bool:
    return boundary["min_x"] <= x <= boundary["max_x"] and boundary["min_y"] <= y <= boundary["max_y"]


def segment_crosses(boundary, ax, ay, bx, by) -> bool:
    """Liang–Barsky clipping; the rectangle edge counts as a crossing."""

    dx, dy = bx - ax, by - ay
    low, high = 0.0, 1.0
    for p, q in (
        (-dx, ax - boundary["min_x"]),
        (dx, boundary["max_x"] - ax),
        (-dy, ay - boundary["min_y"]),
        (dy, boundary["max_y"] - ay),
    ):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                if t > high:
                    return False
                low = max(low, t)
            else:
                if t < low:
                    return False
                high = min(high, t)
    return low <= high


def link_severed(a, b, barricades) -> bool:
    return any(segment_crosses(item.boundary, a.x, a.y, b.x, b.y) for item in barricades)


def inside_any(point, barricades):
    return next((item for item in barricades if contains(item.boundary, point.x, point.y)), None)


def _near(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 <= EPS_METERS**2


def point_affected(point, members, barricades) -> bool:
    """Fenced in, or at least one near-neighbour link cut by a barricade."""

    if inside_any(point, barricades):
        return True
    return any(
        other.id != point.id and _near(point, other) and link_severed(point, other, barricades)
        for other in members
    )


def zone_affected(members, barricades) -> bool:
    return any(point_affected(point, members, barricades) for point in members)


# --------------------------------------------------------------------------
# clustering


def cluster_with_severed_links(points, barricades):
    """DBSCAN with the original parameters, severed links set out of reach."""

    if not points:
        return [], []
    size = len(points)
    distances = np.zeros((size, size), dtype=float)
    out_of_reach = EPS_METERS * 1000
    for i in range(size):
        for j in range(i + 1, size):
            gap = float(np.hypot(points[i].x - points[j].x, points[i].y - points[j].y))
            if gap <= EPS_METERS and link_severed(points[i], points[j], barricades):
                gap = out_of_reach
            distances[i, j] = distances[j, i] = gap
    labels = DBSCAN(eps=EPS_METERS, min_samples=MIN_SAMPLES, metric="precomputed").fit_predict(distances)
    clusters = [
        [point for point, current in zip(points, labels) if int(current) == label]
        for label in sorted({int(value) for value in labels})
        if label != -1
    ]
    noise = [point for point, current in zip(points, labels) if int(current) == -1]
    return clusters, noise


def _zone_metrics(members):
    box = bounds([{"x": point.x, "y": point.y} for point in members])
    area = max(1.0, (box["max_x"] - box["min_x"]) * (box["max_y"] - box["min_y"]))
    return {
        "center_x": round(sum(point.x for point in members) / len(members), 2),
        "center_y": round(sum(point.y for point in members) / len(members), 2),
        "average_settlement_mm": round(sum(point.settlement_mm for point in members) / len(members), 2),
        "max_settlement_mm": max(point.settlement_mm for point in members),
        "local_density": round(len(members) / area * 10000, 3),
        "boundary": box,
    }


# --------------------------------------------------------------------------
# derivation


def _is_derived(zone) -> bool:
    return (zone.original_cluster_evidence or {}).get("partition") == DERIVED_MARKER


def _zone_participated(zone) -> bool:
    return "barricade_partition" in (zone.original_cluster_evidence or {})


def _all_zones(db, batch_id: int):
    return db.execute(
        select(RepairZone).where(RepairZone.batch_id == batch_id).order_by(RepairZone.id)
    ).scalars().all()


def _children(zones, parent_id):
    return [zone for zone in zones if zone.split_from_id == parent_id and zone.status != "invalidated"]


def _descendants(zones, root_id):
    found, frontier = [], [root_id]
    while frontier:
        current = frontier.pop()
        for child in _children(zones, current):
            found.append(child)
            frontier.append(child.id)
    return found


def _zone_notice(db, zone):
    return db.execute(
        select(WorkNotice).where(WorkNotice.zone_id == zone.id, WorkNotice.status != "cancelled")
    ).scalar_one_or_none()


def _has_open_order(db, zone) -> bool:
    return _zone_notice(db, zone) is not None


def _invalidate(db, zone, reason) -> None:
    if _zone_notice(db, zone) is not None or zone.status == "split":
        # An ordered range or a supervisor road-split is never discarded.
        return
    zone.status = "invalidated"
    zone.split_reason = reason


def _is_derivation_root(zone, by_id) -> bool:
    """A zone the barricade rules may re-derive.

    Only un-partitioned zones whose whole ancestor chain is made of
    supervisor road splits qualify.  Derived sub-zones are managed by their
    parent, and anything nested under a ``partitioned`` parent is managed by
    that parent's re-derivation too.
    """

    if _is_derived(zone) or zone.status in {"split", "invalidated"}:
        return False
    current = by_id.get(zone.split_from_id) if zone.split_from_id else None
    while current is not None:
        if current.status != "split":
            return False
        current = by_id.get(current.split_from_id) if current.split_from_id else None
    return True


def _pinned_descendants(zones, root_id):
    """Supervisor road-split zones under ``root_id``.

    These are explicit boundary resolutions made by a supervisor, so barricade
    re-derivation keeps them and never pulls their manholes into a new
    sub-zone.
    """

    return [item for item in _descendants(zones, root_id) if not _is_derived(item)]


def _pinned_points(zones, root_id) -> set:
    protected = set()
    for item in _pinned_descendants(zones, root_id):
        protected.update(item.point_ids or [])
    return protected


def _base_status(zone) -> str:
    stored = (zone.original_cluster_evidence or {}).get("barricade_partition") or {}
    status = stored.get("base_status")
    if status in DISPATCHABLE_ZONE_STATUSES:
        return status
    if zone.status in DISPATCHABLE_ZONE_STATUSES:
        return zone.status
    if zone.conflict_notice_id is not None:
        return "needs_split"
    return "proposed"


def _set_partition_evidence(zone, base_status, revision, barricades, summary=None) -> None:
    evidence = dict(zone.original_cluster_evidence or {})
    partition = dict(evidence.get("barricade_partition") or {})
    partition.update(
        {
            "revision": revision,
            "base_status": base_status,
            "barricade_codes": sorted(item.barricade_code for item in barricades),
            "params": {"eps_m": EPS_METERS, "min_samples": MIN_SAMPLES},
            "frozen": False,
        }
    )
    if summary is not None:
        partition["last_summary"] = summary
    evidence["barricade_partition"] = partition
    zone.original_cluster_evidence = evidence


def _clear_partition_evidence(zone) -> None:
    evidence = dict(zone.original_cluster_evidence or {})
    evidence.pop("barricade_partition", None)
    zone.original_cluster_evidence = evidence


def _restore_zone(zone) -> None:
    """Return a parent to its un-partitioned state; children are released."""

    base_status = _base_status(zone)
    _clear_partition_evidence(zone)
    zone.status = base_status
    zone.split_reason = None


def _freeze_zone(db, zone, revision, barricades) -> None:
    """Keep the parent as a container while its ordered sub-zones keep running.

    The parent stays ``partitioned`` (not ``split``) so later barricade
    changes can still re-derive the manholes that are *not* committed to an
    order.
    """

    _set_partition_evidence(zone, _base_status(zone), revision, barricades)
    evidence = dict(zone.original_cluster_evidence or {})
    evidence["barricade_partition"]["frozen"] = True
    zone.original_cluster_evidence = evidence
    zone.status = "partitioned"


def _partition_zone(db, zone, free, barricades, revision) -> dict:
    """Derive sub-zones from ``free`` manholes along un-severed links."""

    base_status = _base_status(zone)
    deferred = [point for point in free if inside_any(point, barricades)]
    reachable = [point for point in free if point not in deferred]
    clusters, noise = cluster_with_severed_links(reachable, barricades)
    notices = db.execute(
        select(WorkNotice).where(WorkNotice.status.in_(["active", "acknowledged"]))
    ).scalars().all()

    zone.status = "partitioned"
    barricade_codes = sorted(item.barricade_code for item in barricades)

    used_codes = {
        code
        for (code,) in db.execute(
            select(RepairZone.zone_code).where(RepairZone.batch_id == zone.batch_id)
        ).all()
    }
    created = []
    for index, cluster in enumerate(clusters, 1):
        metrics = _zone_metrics(cluster)
        conflict = next((item for item in notices if overlaps(metrics["boundary"], item.boundary)), None)
        base_code = f"{zone.zone_code}-B{revision}.{index}"
        candidate, suffix = base_code, 2
        while candidate in used_codes:
            candidate = f"{base_code}-{suffix}"
            suffix += 1
        used_codes.add(candidate)
        child = RepairZone(
            zone_code=candidate,
            batch_id=zone.batch_id,
            cluster_label=zone.cluster_label,
            name=f"{zone.name} / 围挡子片 {index}",
            # A sub-zone that still overlaps an active road notice keeps the
            # road-boundary conflict and needs a supervisor split first;
            # otherwise the barricade partition resolves the constraint.
            status="needs_split" if conflict is not None else "ready_for_notice",
            point_ids=[point.id for point in cluster],
            center_x=metrics["center_x"],
            center_y=metrics["center_y"],
            average_settlement_mm=metrics["average_settlement_mm"],
            max_settlement_mm=metrics["max_settlement_mm"],
            local_density=metrics["local_density"],
            boundary=metrics["boundary"],
            original_cluster_evidence={
                "algorithm": "DBSCAN",
                "partition": DERIVED_MARKER,
                "parent_zone": zone.zone_code,
                "revision": revision,
                "eps_m": EPS_METERS,
                "min_samples": MIN_SAMPLES,
                "severing": "barricade_rectangle",
                "barricade_codes": barricade_codes,
                "point_codes": [point.point_code for point in cluster],
            },
            split_from_id=zone.id,
            conflict_notice_id=conflict.id if conflict is not None else None,
            split_reason=(
                f"围挡规则 rev {revision}：沿未被围挡隔断的近邻，"
                f"按 eps {EPS_METERS:g} m / min_samples {MIN_SAMPLES} 出片"
                + (f"；仍与 {conflict.road_name} 在建边界重叠，需按施工边界拆分" if conflict else "")
            ),
        )
        db.add(child)
        created.append(child)
    db.flush()

    summary = {
        "zone": zone.zone_code,
        "base_status": base_status,
        "children": [child.zone_code for child in created],
        "deferred": [point.point_code for point in deferred],
        "unassigned": [point.point_code for point in noise],
    }
    _set_partition_evidence(zone, base_status, revision, barricades, summary)
    return summary


def _committed_points(db, zones) -> dict:
    """Manholes already on a non-cancelled work notice, keyed by point id."""

    zone_by_id = {zone.id: zone for zone in zones}
    notices = db.execute(
        select(WorkNotice).where(WorkNotice.status != "cancelled", WorkNotice.zone_id.is_not(None))
    ).scalars().all()
    committed = {}
    for notice in notices:
        zone = zone_by_id.get(notice.zone_id)
        if zone is None:
            continue
        for point_id in zone.point_ids or []:
            committed[point_id] = (zone, notice)
    return committed


def _reconcile_root(db, zone, zones, points_by_id, committed, barricades, revision):
    """Re-derive one zone the barricade rules manage.

    Returns a summary when the zone was re-partitioned, otherwise ``None``.
    Only barricade-derived descendants are rebuilt; a supervisor road split
    (a non-derived descendant) reserves its manholes so re-derivation never
    pulls them into a new sub-zone, and any range on an order is untouched.
    """

    members = [points_by_id[point_id] for point_id in zone.point_ids or [] if point_id in points_by_id]
    if not members:
        return None
    if _has_open_order(db, zone):
        # The range itself is on an order: keep it untouched.
        return None
    derived_descendants = [item for item in _descendants(zones, zone.id) if _is_derived(item)]
    ordered_descendants = [item for item in derived_descendants if _has_open_order(db, item)]
    pinned = _pinned_points(zones, zone.id)
    free = [point for point in members if point.id not in committed and point.id not in pinned]
    participated = _zone_participated(zone)
    affected = bool(barricades) and bool(free) and zone_affected(free, barricades)

    if not (participated or affected or ordered_descendants):
        return None

    for item in derived_descendants:
        if item not in ordered_descendants:
            _invalidate(db, item, "围挡规则更新，旧子片失效")

    if not free:
        if ordered_descendants or pinned:
            _freeze_zone(db, zone, revision, barricades)
        else:
            _restore_zone(zone)
        return None
    if not affected and not pinned and not ordered_descendants:
        # No remaining barricade cuts this range: restore the original cluster.
        _restore_zone(zone)
        return None

    summary = _partition_zone(db, zone, free, barricades, revision)
    if (ordered_descendants or pinned) and not summary["children"] and not summary["deferred"]:
        # The remainder is too sparse to form a sub-zone; keep the parent as
        # a frozen record so its committed / road-split ranges stay traceable.
        _freeze_zone(db, zone, revision, barricades)
    return summary


def apply_rules(db, batch_id: int, revision: int) -> dict:
    """Rebuild barricade sub-zones for one batch.  Caller holds the batch lock."""

    batch = db.get(SurveyBatch, batch_id)
    if batch is None:
        raise ValueError("测量批次不存在")
    points = db.execute(
        select(SurveyPoint).where(SurveyPoint.batch_id == batch_id).order_by(SurveyPoint.id)
    ).scalars().all()
    points_by_id = {point.id: point for point in points}
    zones = _all_zones(db, batch_id)
    by_id = {zone.id: zone for zone in zones}
    barricades = active_barricades(db, batch_id)
    committed = _committed_points(db, zones)

    summaries = []
    for zone in zones:
        # Only derivation roots are rebuilt: derived sub-zones are managed by
        # their parent, and non-root road-split children belong to the tree of
        # the root that spawned them.
        if not _is_derivation_root(zone, by_id):
            continue
        summary = _reconcile_root(db, zone, zones, points_by_id, committed, barricades, revision)
        if summary is not None:
            summaries.append(summary)
    db.flush()
    sync_dispositions(db, batch_id)
    return {"revision": revision, "zones": summaries}


def recompute_after_rule_change(db, batch_id: int) -> dict:
    """Bump the partition revision and rebuild the dispatch scope."""

    lock_batch(db, batch_id)
    revision = bump_revision(db, batch_id)
    return apply_rules(db, batch_id, revision)


def apply_current_rules(db, batch_id: int) -> dict:
    lock_batch(db, batch_id)
    current = rule_set(db, batch_id).revision
    return apply_rules(db, batch_id, current)


# --------------------------------------------------------------------------
# per-manhole destination


def sync_dispositions(db, batch_id: int) -> None:
    """Write the single authoritative destination of every surveyed manhole."""

    points = db.execute(
        select(SurveyPoint).where(SurveyPoint.batch_id == batch_id).order_by(SurveyPoint.id)
    ).scalars().all()
    zones = db.execute(select(RepairZone).where(RepairZone.batch_id == batch_id)).scalars().all()
    committed = _committed_points(db, zones)
    barricades = active_barricades(db, batch_id)
    live = [zone for zone in zones if zone.status not in {"partitioned", "split", "invalidated"}]
    parent_ids = {zone.split_from_id for zone in live if zone.split_from_id}
    # Prefer leaf zones: a parent keeps its full historical point list, but a
    # manhole's dispatch destination is always the leaf-most live zone.
    membership = {}
    for zone in live:
        if zone.id in parent_ids:
            continue
        for point_id in zone.point_ids or []:
            membership[point_id] = zone
    existing = {
        row.point_id: row
        for row in db.execute(
            select(PointDisposition).where(PointDisposition.batch_id == batch_id)
        ).scalars().all()
    }
    now = _now()
    for point in points:
        row = existing.get(point.id)
        if row is None:
            row = PointDisposition(point_id=point.id, batch_id=batch_id)
            db.add(row)
        if point.id in committed:
            zone, notice = committed[point.id]
            row.status = "committed"
            row.zone_id = zone.id
            row.notice_id = notice.id
            row.barricade_id = None
            row.reason = f"已留在原联动单 {notice.notice_code}"
        else:
            fence = inside_any(point, barricades)
            if fence is not None:
                row.status = "deferred"
                row.zone_id = None
                row.notice_id = None
                row.barricade_id = fence.id
                row.reason = f"围挡 {fence.barricade_code} 内，暂缓派工"
            elif point.id in membership:
                zone = membership[point.id]
                row.status = "assigned"
                row.zone_id = zone.id
                row.notice_id = None
                row.barricade_id = None
                row.reason = f"待派工 · {zone.zone_code}"
            else:
                row.status = "pending"
                row.zone_id = None
                row.notice_id = None
                row.barricade_id = None
                row.reason = (
                    "尚未入簇，待安排"
                    if point.cluster_label is None
                    else "围挡隔断或密度不足，待重新安排"
                )
        row.batch_id = batch_id
        row.updated_at = now
    db.flush()


def disposition_rows(db, batch_id: int) -> list[dict]:
    """Joined read model: one row per measured manhole in the batch.

    A point without a disposition row has not entered clustering yet and is
    reported as ``pending`` so the page always covers the whole batch.
    """

    rows = db.execute(
        select(SurveyPoint, PointDisposition, RepairZone, WorkNotice, Barricade)
        .outerjoin(PointDisposition, PointDisposition.point_id == SurveyPoint.id)
        .outerjoin(RepairZone, RepairZone.id == PointDisposition.zone_id)
        .outerjoin(WorkNotice, WorkNotice.id == PointDisposition.notice_id)
        .outerjoin(Barricade, Barricade.id == PointDisposition.barricade_id)
        .where(SurveyPoint.batch_id == batch_id)
        .order_by(SurveyPoint.point_code)
    ).all()
    return [
        {
            "point_id": point.id,
            "point_code": point.point_code,
            "manhole_id": point.manhole_id,
            "asset_code": point.manhole.asset_code if point.manhole else None,
            "street_name": point.manhole.street_name if point.manhole else None,
            "status": disposition.status if disposition else "pending",
            "zone_id": zone.id if zone else None,
            "zone_code": zone.zone_code if zone else None,
            "notice_id": notice.id if notice else None,
            "notice_code": notice.notice_code if notice else None,
            "barricade_id": barricade.id if barricade else None,
            "barricade_code": barricade.barricade_code if barricade else None,
            "reason": disposition.reason if disposition else "尚未入簇，待安排",
            "updated_at": disposition.updated_at.isoformat() if disposition and disposition.updated_at else None,
        }
        for point, disposition, zone, notice, barricade in rows
    ]


def barricade_code(db, batch: SurveyBatch) -> str:
    """Time-stamped code that stays unique within the same second."""

    stamp = int(_now().timestamp())
    base = f"BD-{batch.batch_code.replace('SURV-', '')}-{stamp}"
    code, suffix = base, 2
    while db.execute(select(Barricade.id).where(Barricade.barricade_code == code)).first():
        code = f"{base}-{suffix}"
        suffix += 1
    return code


def barricade_view(item) -> dict:
    return {
        "id": item.id,
        "barricade_code": item.barricade_code,
        "batch_id": item.batch_id,
        "zone_id": item.zone_id,
        "label": item.label,
        "boundary": item.boundary,
        "status": item.status,
        "note": item.note,
        "created_by": item.created_by,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "removed_by": item.removed_by,
        "removed_at": item.removed_at.isoformat() if item.removed_at else None,
    }
