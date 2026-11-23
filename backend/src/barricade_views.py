"""Barricade dispatch reconciliation views."""

VIEWS = {
    'barricade-dispatch': (
        '围挡派工去向核对',
        '逐个井盖核对围挡规则下的去向：围挡内暂缓、已留在原联动单、待派工或待重新安排。',
        ['测量批', '井盖', '测量点', '道路', '去向', '关联片区', '关联联动单', '围挡', '说明'],
        """
SELECT b.batch_code,a.asset_code,p.point_code,a.street_name,
       CASE d.status WHEN 'committed' THEN '已留在原联动单'
                     WHEN 'assigned' THEN '待派工'
                     WHEN 'deferred' THEN '围挡内暂缓'
                     ELSE '待重新安排' END,
       COALESCE(z.zone_code,'—'),COALESCE(n.notice_code,'—'),
       COALESCE(f.barricade_code,'—'),COALESCE(d.reason,'')
FROM point_dispositions d
JOIN survey_points p ON p.id=d.point_id
JOIN manhole_assets a ON a.id=p.manhole_id
JOIN survey_batches b ON b.id=d.batch_id
LEFT JOIN repair_zones z ON z.id=d.zone_id
LEFT JOIN work_notices n ON n.id=d.notice_id
LEFT JOIN barricades f ON f.id=d.barricade_id
ORDER BY b.measured_at DESC,a.asset_code
""",
    ),
    'barricade-rule-revisions': (
        '围挡规则版本与子片',
        '对照当前围挡规则版本和各子片来源，确认旧子片是否已失效、已开工范围是否仍挂在原联动单。',
        ['测量批', '规则版本', '片区', '状态', '来源', '围挡规则版本', '点位', '外接框'],
        """
SELECT b.batch_code,COALESCE(r.revision,0),z.zone_code,z.status,
       CASE WHEN z.original_cluster_evidence->>'partition'='barricade' THEN '围挡子片'
            WHEN z.original_cluster_evidence->'barricade_partition' IS NOT NULL THEN '原始聚类（围挡重算）'
            WHEN z.status='invalidated' THEN '已失效子片'
            ELSE '原始聚类' END,
       COALESCE(z.original_cluster_evidence->'barricade_partition'->>'revision','—'),
       json_array_length(COALESCE(z.point_ids,'[]')),
       COALESCE(z.boundary->>'min_x','—')||','||COALESCE(z.boundary->>'min_y','—')||' → '||COALESCE(z.boundary->>'max_x','—')||','||COALESCE(z.boundary->>'max_y','—')
FROM repair_zones z
JOIN survey_batches b ON b.id=z.batch_id
LEFT JOIN barricade_rule_sets r ON r.batch_id=b.id
ORDER BY b.measured_at DESC,z.id
""",
    ),
}
