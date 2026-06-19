# Data Quality Audit Report
**Audit Timestamp:** 2026-06-18 16:09:26
**Total Records Scanned:** 8173

## 1. Summary Metrics
- **Duplicate Records:** 0
- **Invalid Coordinates (outside Bangalore bounds):** 0
- **Chronological Timestamp Inconsistencies:** 48
- **Missing Start Timestamps:** 0

## 2. Missing Value Statistics
| Column Name | Missing Count | Percentage (%) |
|---|---|---|
| `endlatitude` | 169 | 2.07% |
| `endlongitude` | 169 | 2.07% |
| `address` | 3 | 0.04% |
| `end_address` | 7486 | 91.59% |
| `end_datetime` | 7683 | 94.00% |
| `map_file` | 8173 | 100.00% |
| `direction` | 8130 | 99.47% |
| `description` | 1360 | 16.64% |
| `veh_type` | 3286 | 40.21% |
| `veh_no` | 3287 | 40.22% |
| `corridor` | 20 | 0.24% |
| `priority` | 2 | 0.02% |
| `cargo_material` | 7897 | 96.62% |
| `reason_breakdown` | 7897 | 96.62% |
| `age_of_truck` | 7897 | 96.62% |
| `route_path` | 8036 | 98.32% |
| `created_by_id` | 2 | 0.02% |
| `last_modified_by_id` | 3 | 0.04% |
| `assigned_to_police_id` | 8045 | 98.43% |
| `citizen_accident_id` | 8045 | 98.43% |
| `comment` | 8173 | 100.00% |
| `meta_data` | 8173 | 100.00% |
| `kgid` | 259 | 3.17% |
| `resolved_at_address` | 8099 | 99.09% |
| `resolved_at_latitude` | 8099 | 99.09% |
| `resolved_at_longitude` | 8099 | 99.09% |
| `closed_by_id` | 5032 | 61.57% |
| `closed_datetime` | 5032 | 61.57% |
| `resolved_by_id` | 8099 | 99.09% |
| `resolved_datetime` | 8099 | 99.09% |
| `gba_identifier` | 4729 | 57.86% |
| `zone` | 4729 | 57.86% |
| `junction` | 5663 | 69.29% |

## 3. Categorical Distribution Logs
### Event Types:
- **unplanned:** 7706 (94.29%)
- **planned:** 467 (5.71%)
### Priorities:
- **High:** 5030 (61.54%)
- **Low:** 3141 (38.43%)
- **nan:** 2 (0.02%)

## 4. Inconsistency Action Log
- Invalid coordinates will have their cluster mappings assigned to a default cluster center.
- Inconsistent end times are ignored for duration training and binned fallbacks.
- Missing durations are imputed via historical averages of event types.