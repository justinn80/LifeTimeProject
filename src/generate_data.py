"""
PACE synthetic data generator
=============================

Simulates one representative Life Time center for several synthetic weeks and
evaluates a grid of staffing plans against it. Each plan is defined by two
decision variables:

    peak_coverage_pct     staffing during peak windows, as a % of the
                          demand-based staffing target
    offpeak_coverage_pct  staffing during midday / late-evening hours, as a %
                          of the same target

For every plan the simulator computes the two PACE objectives:

    labor_cost_per_visit     (minimize) flexible labor $ / completed member visit
    member_experience_score  (maximize) 0-100 composite of class fill-fit,
                             floor time-to-assistance, and childcare availability

Design notes
------------
* Common random numbers: every plan is evaluated on the SAME simulated weeks.
  That way, a difference between two plans comes from the staffing decision,
  not from one plan happening to draw a quieter week. This is what makes the
  Pareto dominance comparison fair.
* Demand is random (week, day and hour level noise) so no two weeks and no two
  days are the same, as promised in the proposal.
* Hard rules from the proposal are enforced inside the simulator so every plan
  in the dataset is legal: childcare ratio (1 attendant : 6 children), two-adult
  minimum in the kids area, two floor staff on duty at all times, three studios
  (room capacity), and 1.5x overtime past each role's roster capacity.
* Calibration targets come from Life Time's Q1 2026 results (see
  data/DATA_SOURCE.md). Everything else is an explicit, documented assumption
  in the PARAMS dictionary below.

Usage
-----
    python src/generate_data.py            # writes data/*.csv
    python src/generate_data.py --check    # regenerate in memory, compare to CSV
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Parameters (all assumptions live here so they are easy to audit and tweak)
# --------------------------------------------------------------------------
PARAMS = {
    # Calibration to public figures (Life Time Q1 2026 earnings release)
    "center_memberships_total": 837_903,   # end of Q1 2026
    "centers": 190,                        # as of Mar 31, 2026
    # Assumption: average check-ins per membership per week (a membership can
    # cover a whole household, so this is higher than a single person's rate).
    "visits_per_membership_week": 3.0,

    # Operating window that needs staff (5:00 to 22:00 -> hour index 5..21)
    "open_hour": 5,
    "close_hour": 22,

    # Peak windows: weekday 5-9 AM and 4-8 PM, weekend 8 AM-12 PM
    "weekday_peak_hours": [5, 6, 7, 8, 16, 17, 18, 19],
    "weekend_peak_hours": [8, 9, 10, 11],

    # Day-of-week demand multipliers (Mon..Sun)
    "day_mult": [1.05, 1.03, 1.02, 0.98, 0.90, 0.95, 0.80],

    # Randomness
    "n_weeks": 8,
    "seed": 2026,
    "week_noise_sd": 0.05,
    "day_noise_sd": 0.07,
    "hour_noise_sd": 0.12,

    # Fitness floor
    "avg_stay_hours": 1.25,                # members on site ~= check-ins x stay
    "members_per_floor_staff": 60,         # 100% coverage target ratio
    "min_floor_staff": 2,                  # always 2 on duty (safety rule)
    "assist_requests_per_member_hr": 0.10, # questions, equipment help, spots
    "assist_service_rate_hr": 9.0,         # requests one staffer clears per hr
    "walk_time_min": 1.0,                  # time to physically reach a member
    "wait_cap_min": 15.0,                  # used when the queue is unstable

    # Group fitness
    "class_demand_share": 0.22,            # share of check-ins wanting a class
    "class_capacity": 35,                  # studio capacity (room constraint)
    "studios": 3,                          # max concurrent classes
    "target_fill": 0.85,                   # 100% coverage -> ~85% full classes

    # Childcare (Life Time Kids)
    "kids_per_checkin": 0.06,
    "childcare_ratio": 6,                  # legal max children per attendant
    "min_childcare_staff": 2,              # two-adult rule while open
    "childcare_weekday_hours": list(range(7, 20)),
    "childcare_weekend_hours": list(range(8, 14)),
    "parent_visit_lost_per_turnaway": 0.8, # parent leaves if no childcare spot

    # Wages (base $/hr or $/class) and payroll burden
    "wage_floor": 18.00,
    "wage_childcare": 16.50,
    "pay_per_class": 45.00,
    "burden": 1.25,                        # payroll taxes + benefits
    "overtime_mult": 1.5,
    "floor_roster_hours": 400,             # straight-time hours available/week
    "childcare_roster_hours": 280,

    # Member Experience Score weights
    "w_assist": 0.40,
    "w_class": 0.35,
    "w_childcare": 0.25,

    # Decision-variable grid (10 x 10 = 100 plans)
    "peak_levels": list(range(60, 151, 10)),
    "offpeak_levels": list(range(40, 131, 10)),
}

HOURS = np.arange(PARAMS["open_hour"], PARAMS["close_hour"])
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------
# Demand simulation
# --------------------------------------------------------------------------
def _gauss(h, mu, sd):
    return np.exp(-0.5 * ((h - mu) / sd) ** 2)


def base_hourly_shape(is_weekend: bool) -> np.ndarray:
    """Relative check-in intensity by hour (normalised to sum to 1)."""
    h = HOURS.astype(float)
    if is_weekend:
        shape = 1.0 * _gauss(h, 9.75, 1.8) + 0.35 * _gauss(h, 15.5, 2.5) + 0.05
    else:
        shape = (1.00 * _gauss(h, 6.5, 1.1)      # pre-work rush
                 + 0.45 * _gauss(h, 12.0, 0.9)   # lunch bump
                 + 1.10 * _gauss(h, 17.6, 1.3)   # after-work rush
                 + 0.06)                          # trickle all day
    return shape / shape.sum()


def simulate_demand(p=PARAMS) -> pd.DataFrame:
    """Return one row per (week, day, hour) with check-ins and kids demand."""
    rng = np.random.default_rng(p["seed"])
    memberships_per_center = p["center_memberships_total"] / p["centers"]
    weekly_visits = memberships_per_center * p["visits_per_membership_week"]
    daily_base = weekly_visits / sum(p["day_mult"])

    rows = []
    for w in range(p["n_weeks"]):
        week_mult = rng.normal(1.0, p["week_noise_sd"])
        for d, day in enumerate(DAYS):
            weekend = d >= 5
            day_total = daily_base * p["day_mult"][d] * week_mult * rng.normal(1.0, p["day_noise_sd"])
            shape = base_hourly_shape(weekend)
            hour_noise = rng.lognormal(0.0, p["hour_noise_sd"], size=len(HOURS))
            checkins = day_total * shape * hour_noise
            peak_set = p["weekend_peak_hours"] if weekend else p["weekday_peak_hours"]
            cc_hours = p["childcare_weekend_hours"] if weekend else p["childcare_weekday_hours"]
            kid_noise = rng.lognormal(0.0, 0.15, size=len(HOURS))
            for i, h in enumerate(HOURS):
                rows.append({
                    "week": w + 1,
                    "day": day,
                    "hour": int(h),
                    "is_weekend": weekend,
                    "is_peak": int(h) in peak_set,
                    "checkins": checkins[i],
                    "childcare_open": int(h) in cc_hours,
                    "kids_demand": checkins[i] * p["kids_per_checkin"] * kid_noise[i]
                                   if int(h) in cc_hours else 0.0,
                })
    df = pd.DataFrame(rows)
    # Members on site this hour = this hour's arrivals plus the share of last
    # hour's arrivals who are still working out (avg stay 1.25 h).
    stay = p["avg_stay_hours"]
    prev = df.groupby(["week", "day"])["checkins"].shift(1).fillna(0.0)
    df["occupancy"] = df["checkins"] * min(stay, 1.0) + prev * max(stay - 1.0, 0.0)
    return df


# --------------------------------------------------------------------------
# Queueing model for floor time-to-assistance (Erlang C / M/M/c)
# --------------------------------------------------------------------------
def erlang_c_wait_min(lam: float, mu: float, c: int, cap: float) -> float:
    """Expected wait (minutes) in an M/M/c queue; capped if unstable."""
    if lam <= 0:
        return 0.0
    a = lam / mu                       # offered load (Erlangs)
    rho = a / c
    if rho >= 1.0:
        return cap
    # Erlang C probability that an arriving request must wait
    sum_terms = sum(a ** k / math.factorial(k) for k in range(c))
    last = a ** c / math.factorial(c) / (1 - rho)
    p_wait = last / (sum_terms + last)
    wq_hours = p_wait / (c * mu - lam)
    return min(wq_hours * 60.0, cap)


def class_fit(fill: np.ndarray) -> np.ndarray:
    """1.0 when classes are 75-95% full; drops for empty or sold-out rooms."""
    fit = np.interp(fill, [0.0, 0.25, 0.75, 0.95, 1.0], [0.0, 0.0, 1.0, 1.0, 0.9])
    return fit


# --------------------------------------------------------------------------
# Evaluate one staffing plan on the simulated weeks
# --------------------------------------------------------------------------
def evaluate_plan(demand: pd.DataFrame, peak_pct: float, offpeak_pct: float, p=PARAMS) -> dict:
    cov = np.where(demand["is_peak"], peak_pct / 100.0, offpeak_pct / 100.0)
    occ = demand["occupancy"].to_numpy()
    chk = demand["checkins"].to_numpy()
    kids = demand["kids_demand"].to_numpy()
    cc_open = demand["childcare_open"].to_numpy()

    # ---- Floor staff (staff can be scheduled in half-hour blocks -> 0.5 steps)
    floor_target = occ / p["members_per_floor_staff"]
    floor_staff = np.maximum(p["min_floor_staff"], np.round(cov * floor_target * 2) / 2)
    lam = occ * p["assist_requests_per_member_hr"]
    waits = np.array([
        # Pool fractional staff conservatively: a half-hour shift adds half
        # the service capacity, approximated by scaling mu per whole server.
        erlang_c_wait_min(l, p["assist_service_rate_hr"] * (s / math.ceil(s)), int(math.ceil(s)), p["wait_cap_min"])
        for l, s in zip(lam, floor_staff)
    ])
    tta = waits + p["walk_time_min"]
    assist_score_hr = np.clip(100 * (10.0 - tta) / (10.0 - p["walk_time_min"]), 0, 100)

    # ---- Group fitness classes (room capacity: at most `studios` at once)
    class_demand = chk * p["class_demand_share"]
    class_target = class_demand / (p["class_capacity"] * p["target_fill"])
    classes = np.clip(np.round(cov * class_target), 0, p["studios"])
    seats = classes * p["class_capacity"]
    booked = np.minimum(class_demand, seats)
    fill = np.divide(booked, seats, out=np.zeros_like(booked), where=seats > 0)
    served_share = np.divide(booked, class_demand, out=np.ones_like(booked), where=class_demand > 0)
    class_score_hr = 100 * class_fit(fill) * served_share
    class_score_hr = np.where(classes == 0, 0.0, class_score_hr)
    waitlisted = class_demand - booked

    # ---- Childcare (ratio is a hard rule: capacity = attendants x 6)
    cc_target = kids / p["childcare_ratio"]
    cc_staff = np.where(cc_open, np.maximum(p["min_childcare_staff"], np.round(cov * cc_target * 2) / 2), 0.0)
    cc_capacity = cc_staff * p["childcare_ratio"]
    kids_served = np.minimum(kids, cc_capacity)
    turnaways = kids - kids_served
    cc_score_hr = 100 * np.divide(kids_served, kids, out=np.ones_like(kids), where=kids > 0)

    # ---- Visits actually completed (a parent turned away at childcare leaves)
    lost_visits = turnaways * p["parent_visit_lost_per_turnaway"]
    n_weeks = demand["week"].nunique()
    visits = (chk.sum() - lost_visits.sum()) / n_weeks

    # ---- Weekly labor cost with overtime past roster capacity
    def role_cost(hours_total, wage, roster):
        straight = min(hours_total, roster)
        ot = max(hours_total - roster, 0.0)
        return (straight + ot * p["overtime_mult"]) * wage * p["burden"], ot

    floor_hours = floor_staff.sum() / n_weeks
    cc_hours = cc_staff.sum() / n_weeks
    classes_wk = classes.sum() / n_weeks
    floor_cost, floor_ot = role_cost(floor_hours, p["wage_floor"], p["floor_roster_hours"])
    cc_cost, cc_ot = role_cost(cc_hours, p["wage_childcare"], p["childcare_roster_hours"])
    class_cost = classes_wk * p["pay_per_class"] * p["burden"]
    total_cost = floor_cost + cc_cost + class_cost

    # ---- Aggregate experience (weighted by how many members felt each hour)
    assist_score = np.average(assist_score_hr, weights=chk)
    class_score = np.average(class_score_hr, weights=np.maximum(class_demand, 1e-9))
    cc_score = np.average(cc_score_hr[cc_open], weights=np.maximum(kids[cc_open], 1e-9))
    mes = p["w_assist"] * assist_score + p["w_class"] * class_score + p["w_childcare"] * cc_score

    # Week-to-week spread of the objectives (robustness info for Phase 2)
    wk = demand["week"].to_numpy()
    wk_cost, wk_mes = [], []
    for w in np.unique(wk):
        m = wk == w
        f_c, _ = role_cost(floor_staff[m].sum(), p["wage_floor"], p["floor_roster_hours"])
        c_c, _ = role_cost(cc_staff[m].sum(), p["wage_childcare"], p["childcare_roster_hours"])
        cost_w = f_c + c_c + classes[m].sum() * p["pay_per_class"] * p["burden"]
        vis_w = chk[m].sum() - lost_visits[m].sum()
        wk_cost.append(cost_w / vis_w)
        a_w = np.average(assist_score_hr[m], weights=chk[m])
        k_w = np.average(class_score_hr[m], weights=np.maximum(class_demand[m], 1e-9))
        mm = m & cc_open
        cc_w = np.average(cc_score_hr[mm], weights=np.maximum(kids[mm], 1e-9))
        wk_mes.append(p["w_assist"] * a_w + p["w_class"] * k_w + p["w_childcare"] * cc_w)

    return {
        "peak_coverage_pct": peak_pct,
        "offpeak_coverage_pct": offpeak_pct,
        "labor_cost_per_visit": total_cost / visits,
        "member_experience_score": mes,
        "weekly_labor_cost": total_cost,
        "weekly_visits": visits,
        "floor_staff_hours": floor_hours,
        "childcare_staff_hours": cc_hours,
        "classes_per_week": classes_wk,
        "total_labor_hours": floor_hours + cc_hours + classes_wk,  # 1 instructor-hr per class
        "overtime_hours": floor_ot + cc_ot,
        "assist_score": assist_score,
        "class_fit_score": class_score,
        "childcare_score": cc_score,
        "avg_time_to_assist_min": np.average(tta, weights=chk),
        "peak_time_to_assist_min": np.average(tta[demand["is_peak"].to_numpy()], weights=chk[demand["is_peak"].to_numpy()]),
        "avg_class_fill_rate": booked.sum() / max(seats.sum(), 1e-9),
        "class_waitlist_per_week": waitlisted.sum() / n_weeks,
        "childcare_turnaways_per_week": turnaways.sum() / n_weeks,
        "cost_per_visit_week_sd": float(np.std(wk_cost, ddof=1)),
        "mes_week_sd": float(np.std(wk_mes, ddof=1)),
    }


def build_dataset(p=PARAMS) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = simulate_demand(p)
    plans = [evaluate_plan(demand, pk, op, p)
             for pk in p["peak_levels"] for op in p["offpeak_levels"]]
    scenarios = pd.DataFrame(plans)
    scenarios.insert(0, "plan_id", [f"P{i:03d}" for i in range(1, len(scenarios) + 1)])
    return scenarios.round(4), demand.round(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and confirm it matches data/staffing_scenarios.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scen_path = root / "data" / "staffing_scenarios.csv"
    demand_path = root / "data" / "hourly_demand.csv"
    scenarios, demand = build_dataset()

    if args.check:
        saved = pd.read_csv(scen_path)
        pd.testing.assert_frame_equal(saved, scenarios, check_dtype=False, atol=1e-4)
        print(f"OK: regenerated data matches {scen_path.name} ({len(saved)} rows)")
        return

    scenarios.to_csv(scen_path, index=False)
    demand.to_csv(demand_path, index=False)
    print(f"Wrote {len(scenarios)} staffing plans -> {scen_path.relative_to(root)}")
    print(f"Wrote {len(demand)} simulated center-hours -> {demand_path.relative_to(root)}")


if __name__ == "__main__":
    main()
