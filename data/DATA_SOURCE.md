# Data Source and Validation Notes

**Dataset name:** PACE synthetic staffing scenarios
**Type:** Synthetic. An LLM (Claude, Anthropic) wrote a seeded Python simulator to my specification (approved proposal, Section 5).
**Generated:** November 2026, seed `2026`
**Prompt:** [`data_generation_prompt.txt`](data_generation_prompt.txt) (exact prompt)
**Generator:** [`../src/generate_data.py`](../src/generate_data.py)
**Reproduce:** `python src/generate_data.py`. Check reproducibility with `python src/generate_data.py --check`.

## Files

| File | Rows | What it is |
|---|---|---|
| `staffing_scenarios.csv` | 100 | **Analysis dataset.** One row per weekly staffing plan (10 × 10 grid of the two decision variables) with both objectives and supporting metrics |
| `hourly_demand.csv` | 952 | The simulated demand behind the scenarios: 8 weeks × 7 days × 17 open hours of check-ins, on-site occupancy, and childcare demand |
| `pareto_frontier.csv` | — | Written by the notebook: the efficient plans, for Phase 2's decision agent |

## Key columns (`staffing_scenarios.csv`)

| Column | Role | Meaning |
|---|---|---|
| `peak_coverage_pct` | **Decision variable 1** | Staffing in peak windows (weekday 5–9 AM and 4–8 PM, weekend 8 AM–12 PM) as a % of the demand-based target |
| `offpeak_coverage_pct` | **Decision variable 2** | Staffing in all other open hours as a % of the target |
| `labor_cost_per_visit` | **Objective 1 (minimize)** | Weekly flexible labor $ (floor, instructors, childcare, with burden and overtime) ÷ completed visits |
| `member_experience_score` | **Objective 2 (maximize)** | 0–100 composite: 40% floor assistance, 35% class fill-fit, 25% childcare availability |
| `total_labor_hours`, `overtime_hours` | Phase 2 constraints | Weekly hours-budget and labor-rule checks |
| `childcare_turnaways_per_week`, `class_waitlist_per_week` | Phase 2 constraints | Service-level guardrails |
| `cost_per_visit_week_sd`, `mes_week_sd` | Robustness | Week-to-week spread of each objective across the 8 simulated weeks |

## Calibration to public figures

| Input | Value used | Source |
|---|---|---|
| Center memberships | 837,903 | Life Time Q1 2026 earnings release (May 5, 2026) |
| Centers | 190 | Same release |
| Memberships per center | ≈ 4,410 | Derived: 837,903 ÷ 190 |
| Check-ins per membership per week | 3.0 | **Assumption.** A membership often covers a household |
| Center operations expense | $406.7M per quarter (≈ $164.7k per center-week) | Same release, used as a sanity ceiling |

Source: Life Time Group Holdings, Inc. (2026, May 5). *Life Time Reports First Quarter 2026 Financial Results.* https://ir.lifetime.life/news-events/press-releases/detail/1205 (SEC Exhibit 99.1: https://www.sec.gov/Archives/edgar/data/0001869198/000186919826000029/lth-20260331xex991.htm)

Wages ($18/hr floor, $16.50/hr childcare, $45 per class, 1.25× burden) are **assumptions** set at typical hourly fitness-industry levels. The 1:6 childcare ratio is a conservative mixed-age ratio. Actual ratios vary by state and by the children's ages.

## Validation checks (run in Section 2 of the notebook)

1. **Completeness.** 100 rows, no missing values, every grid combination present exactly once.
2. **Volume calibration.** Simulated weekly visits average ≈ 13,230, which matches 4,410 memberships × 3.0 check-ins.
3. **Cost realism.** Flexible labor runs about $11k–$22k per center-week, roughly 7–13% of the $164.7k per center-week center-operations line. That is plausible: the center-ops line also covers management, front desk, maintenance, PT commissions, utilities and supplies.
4. **Hard rules respected by construction.** The childcare ratio is enforced as a capacity, two floor staff are always on duty, and no more than 3 concurrent classes run (studio capacity).
5. **Tradeoff present.** Cost per visit and experience are strongly positively correlated (spending more buys experience). Because cost is minimized, a positive correlation is the tradeoff.
6. **Plausible shape.** Under-staffed peaks produce long queues and sold-out classes. Over-staffed peaks produce half-empty classes, so experience falls while cost rises. Those plans are dominated by design of the physics, not by hand-placement.

## Known limitations of the data

* One representative center. Real centers differ in size, market, amenities (pools, courts, spa) and membership mix.
* Coverage is a single knob per time window shared across all three roles. Real schedules set each role independently, which is a natural Phase 2 extension.
* Member behavior is not adaptive. Members don't shift their visit times in response to crowding.
