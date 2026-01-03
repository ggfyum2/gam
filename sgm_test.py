import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bonus Parlay EV + Breakeven", layout="centered")


# ----------------------------
# Core EV functions
# ----------------------------

def calc_ev(
    wager,
    bonus_val,
    offer1,
    offer2,
    offer3,
    house_edge=0.06,
    true_prob1=None,
    true_prob2=None,
    true_prob3=None,
):
    """
    Your original model:
    - If ALL 3 legs win => payout = offer1*offer2*offer3*wager
    - If exactly 1 leg loses => payout = bonus_val*wager
    - Otherwise => payout = 0
    EV shown as payout EV and net EV (minus wager)
    """
    ps = [true_prob1, true_prob2, true_prob3]
    offers = [offer1, offer2, offer3]
    for i, p in enumerate(ps):
        if p is None:
            ps[i] = 1.0 / (offers[i] + house_edge)
    true_prob1, true_prob2, true_prob3 = ps

    win_all = offer1 * offer2 * offer3 * wager
    lose_1 = bonus_val * wager

    p_wa = true_prob1 * true_prob2 * true_prob3
    p_l1 = (
        true_prob1 * (1 - true_prob2) * true_prob3
        + (1 - true_prob1) * true_prob2 * true_prob3
        + true_prob1 * true_prob2 * (1 - true_prob3)
    )

    ev_payout = p_wa * win_all + p_l1 * lose_1
    ev_net = ev_payout - wager

    return {
        "true_prob1": true_prob1,
        "true_prob2": true_prob2,
        "true_prob3": true_prob3,
        "p_win_all": p_wa,
        "p_lose_1_get_bonus": p_l1,
        "win_all_payout": win_all,
        "lose_1_payout": lose_1,
        "ev_payout": ev_payout,
        "ev_net": ev_net,
    }


def breakeven_true_prob_bet2_equal_bet3(
    wager,
    bonus_val,
    offer1,
    offer23,
    house_edge=0.06,
    true_prob1=None,
):
    """
    Breakeven true probability p for bet2 and bet3 (assumed equal),
    such that EV payout == wager (same EV structure as calc_ev()).
    bet1 fixed. If true_prob1 is None, inferred as 1/(offer1+house_edge).
    """
    q = true_prob1 if true_prob1 is not None else 1.0 / (offer1 + house_edge)
    O = offer1 * (offer23 ** 2)
    B = bonus_val

    # Breakeven: A p^2 + C p - 1 = 0
    A = q * O + (1 - 3 * q) * B
    C = 2 * q * B

    eps = 1e-12
    if abs(A) < eps:
        if abs(C) < eps:
            return None
        p = 1.0 / C
        return p if 0 <= p <= 1 else None

    disc = C * C + 4 * A
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(disc)
    p1 = (-C + sqrt_disc) / (2 * A)
    p2 = (-C - sqrt_disc) / (2 * A)

    candidates = [p for p in (p1, p2) if 0 <= p <= 1]
    if not candidates:
        return None

    def ev_ratio(p):
        p_wa = q * p * p
        p_l1 = 2 * q * p * (1 - p) + (1 - q) * p * p
        return p_wa * O + p_l1 * B  # should equal 1 at breakeven

    return min(candidates, key=lambda p: abs(ev_ratio(p) - 1.0))


def bonus_cash_ev(
    wager,
    odds,
    house_edge=0.06,
    true_prob=None,
):
    """
    Bonus cash bet EV:
    - stake is NOT returned
    - win payout = (odds - 1) * wager
    - loss payout = 0
    """
    p = true_prob if true_prob is not None else 1.0 / (odds + house_edge)
    win_payout = (odds - 1.0) * wager
    ev_payout = p * win_payout
    return {
        "true_prob": p,
        "win_payout": win_payout,
        "ev_payout": ev_payout,
        "ev_net": ev_payout,  # bonus stake => no cash stake returned
    }


def bet_back_ev(
    wager,
    win_odds,
    place_odds,
    house_edge=0.06,
    bonus_val=1.0,
    true_prob_win=None,
    true_prob_place=None,
):
    """
    Single bet with "bet back" as bonus cash if you lose but place (2nd/3rd).

    This version treats place probability as CONDITIONAL on not winning:
      true_prob_place = P(place | not win)

    If your place odds are actually unconditional "Top 3 (includes win)",
    use bet_back_ev_unconditional_place() below.
    """
    p_win = true_prob_win if true_prob_win is not None else 1.0 / (win_odds + house_edge)
    p_place_cond = true_prob_place if true_prob_place is not None else 1.0 / (place_odds + house_edge)

    p_place_not_win = (1.0 - p_win) * p_place_cond
    p_lose_all = (1.0 - p_win) * (1.0 - p_place_cond)

    payout_win = win_odds * wager
    payout_betback = bonus_val * wager

    ev_payout = p_win * payout_win + p_place_not_win * payout_betback

    return {
        "model": "conditional_place_given_not_win",
        "true_prob_win": p_win,
        "true_prob_place_conditional": p_place_cond,
        "p_win": p_win,
        "p_place_not_win": p_place_not_win,
        "p_lose_all": p_lose_all,
        "payout_win": payout_win,
        "payout_betback": payout_betback,
        "ev_payout": ev_payout,
        "ev_net_cash": ev_payout - wager,
        "ev_net_if_bonus_stake": ev_payout,
    }


def bet_back_ev_unconditional_place(
    wager,
    win_odds,
    place_odds,
    house_edge=0.06,
    bonus_val=1.0,
    true_prob_win=None,
    true_prob_place_uncond=None,
):
    """
    Single bet with "bet back" where PLACE means top-3 INCLUDING win.

    Probabilities:
      - P(win) = p_win
      - P(place) = p_place_uncond (includes win)
      - P(place but not win) = max(p_place_uncond - p_win, 0)

    If true probs are None, inferred as 1/(odds + house_edge) (same convention as your other code).
    """
    p_win = true_prob_win if true_prob_win is not None else 1.0 / (win_odds + house_edge)
    p_place = (
        true_prob_place_uncond
        if true_prob_place_uncond is not None
        else 1.0 / (place_odds + house_edge)
    )

    p_place_not_win = max(p_place - p_win, 0.0)
    p_lose_all = max(1.0 - p_place, 0.0)

    payout_win = win_odds * wager
    payout_betback = bonus_val * wager

    ev_payout = p_win * payout_win + p_place_not_win * payout_betback

    return {
        "model": "unconditional_place_top3_includes_win",
        "true_prob_win": p_win,
        "true_prob_place_unconditional": p_place,
        "p_win": p_win,
        "p_place_not_win": p_place_not_win,
        "p_lose_all": p_lose_all,
        "payout_win": payout_win,
        "payout_betback": payout_betback,
        "ev_payout": ev_payout,
        "ev_net_cash": ev_payout - wager,
        "ev_net_if_bonus_stake": ev_payout,
    }


# ----------------------------
# Vig (overround) + de-vigging for WIN + PLACE markets
# ----------------------------

def vig_from_win_place_markets(win_odds_list, place_odds_list, places=3):
    """
    Given odds for N runners (N < 10 ideally; UI supports up to 20),
    compute:

    WIN market:
      implied_win_i = 1 / win_odds_i
      overround_win = sum(implied_win_i) - 1
      house_edge_win = overround_win  (as a fraction)

      de-vig true win probs:
        p_true_win_i = implied_win_i / sum(implied_win)

    PLACE market (Top 'places', e.g. 3):
      implied_place_i = 1 / place_odds_i
      overround_place_abs = sum(implied_place_i) - places
      overround_place_rel = sum(implied_place_i)/places - 1  (handy % measure)

      de-vig true place probs (sum to 'places'):
        p_true_place_i = implied_place_i / sum(implied_place) * places

    Returns dict with edges and de-vigged probs.
    """
    # WIN
    implied_win = [1.0 / o for o in win_odds_list]
    sum_iw = sum(implied_win)
    overround_win = sum_iw - 1.0
    p_true_win = [p / sum_iw for p in implied_win] if sum_iw > 0 else [None] * len(implied_win)

    # PLACE
    implied_place = [1.0 / o for o in place_odds_list]
    sum_ip = sum(implied_place)
    overround_place_abs = sum_ip - float(places)
    overround_place_rel = (sum_ip / float(places) - 1.0) if places > 0 else None
    p_true_place = [p / sum_ip * float(places) for p in implied_place] if sum_ip > 0 else [None] * len(implied_place)

    return {
        "sum_implied_win": sum_iw,
        "overround_win": overround_win,
        "sum_implied_place": sum_ip,
        "overround_place_abs": overround_place_abs,
        "overround_place_rel": overround_place_rel,
        "p_true_win": p_true_win,
        "p_true_place": p_true_place,
    }


# ----------------------------
# UI
# ----------------------------

st.title("Bonus Parlay EV + Breakeven (bet2 = bet3)")

with st.sidebar:
    st.header("Global Inputs")

    wager = st.number_input("Wager ($)", min_value=0.0, value=50.0, step=1.0)
    bonus_val = st.number_input(
        "Bonus value (as $ per $1 bonus)", min_value=0.0, value=1.0, step=0.05
    )
    house_edge = st.number_input(
        "House edge (used to infer probs)", min_value=0.0, value=0.06, step=0.01
    )

    st.subheader("Offers (decimal odds)")
    offer1 = st.number_input("Offer 1", min_value=1.0, value=2.0, step=0.01)
    offer2 = st.number_input("Offer 2", min_value=1.0, value=1.01, step=0.01)
    offer3 = st.number_input("Offer 3", min_value=1.0, value=1.01, step=0.01)

    st.subheader("Optional true probs (leave blank to infer)")
    use_p1 = st.checkbox("Override true_prob1?", value=False)
    true_prob1 = (
        st.number_input("true_prob1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        if use_p1
        else None
    )

    use_p2 = st.checkbox("Override true_prob2?", value=False)
    true_prob2 = (
        st.number_input("true_prob2", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        if use_p2
        else None
    )

    use_p3 = st.checkbox("Override true_prob3?", value=False)
    true_prob3 = (
        st.number_input("true_prob3", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        if use_p3
        else None
    )


# ----------------------------
# Section 1: 3-leg EV model
# ----------------------------

st.subheader("EV for your 3-leg bet (your model)")

res = calc_ev(
    wager=wager,
    bonus_val=bonus_val,
    offer1=offer1,
    offer2=offer2,
    offer3=offer3,
    house_edge=house_edge,
    true_prob1=true_prob1,
    true_prob2=true_prob2,
    true_prob3=true_prob3,
)

col1, col2, col3 = st.columns(3)
col1.metric("EV payout ($)", f"{res['ev_payout']:.4f}")
col2.metric("EV net ($)", f"{res['ev_net']:.4f}")
col3.metric("EV net (%)", f"{(res['ev_net']/wager*100 if wager else 0):.4f}%")

df_true = pd.DataFrame(
    {
        "Leg": ["1", "2", "3"],
        "Offer": [offer1, offer2, offer3],
        "True prob used": [res["true_prob1"], res["true_prob2"], res["true_prob3"]],
    }
)

df_probs = pd.DataFrame(
    {
        "Metric": ["p(win all)", "p(lose exactly 1 & bonus)"],
        "Value": [res["p_win_all"], res["p_lose_1_get_bonus"]],
    }
)

st.write("### Inferred / used true probabilities")
st.dataframe(df_true, use_container_width=True)

st.write("### Outcome probabilities (per your model)")
st.dataframe(df_probs, use_container_width=True)


# ----------------------------
# Section 2: Breakeven p for bet2=bet3
# ----------------------------

st.divider()
st.subheader("Breakeven true probability for bet2 = bet3 (bet1 fixed)")

offer23 = st.number_input(
    "Offer for bet2 and bet3 (same)", min_value=1.0, value=float(offer2), step=0.01
)

p_be = breakeven_true_prob_bet2_equal_bet3(
    wager=wager,
    bonus_val=bonus_val,
    offer1=offer1,
    offer23=offer23,
    house_edge=house_edge,
    true_prob1=true_prob1,
)

if p_be is None:
    st.error("No valid breakeven probability found in [0,1] for these parameters.")
else:
    st.success(f"Breakeven true probability for bet2 = bet3 is: **{p_be:.6f}**")

    if st.button("Sanity check: run calc_ev using breakeven p for legs 2 & 3"):
        check = calc_ev(
            wager=wager,
            bonus_val=bonus_val,
            offer1=offer1,
            offer2=offer23,
            offer3=offer23,
            house_edge=house_edge,
            true_prob1=true_prob1,
            true_prob2=p_be,
            true_prob3=p_be,
        )
        st.write("### Sanity check results (using p_be for leg2 & leg3)")
        st.write(f"EV payout: {check['ev_payout']:.6f} (target = wager {wager:.6f})")
        st.write(f"EV net: {check['ev_net']:.6f} (target = 0)")


# ----------------------------
# Section 3: Bonus cash EV
# ----------------------------

st.divider()
st.subheader("Bonus cash EV (stake not returned)")

st.caption(
    "Bonus cash means you don’t get your stake back. "
    "So wager=50 at odds=3.0 pays (3.0-1)*50=100 on win (not 150)."
)

bc_odds = st.number_input("Bonus cash odds", min_value=1.0, value=3.0, step=0.01, key="bc_odds")
use_bc_p = st.checkbox("Override true probability for bonus cash?", value=False, key="bc_override")
bc_true_prob = (
    st.number_input("Bonus cash true probability", min_value=0.0, max_value=1.0, value=0.33, step=0.01, key="bc_p")
    if use_bc_p
    else None
)

if st.button("Calculate Bonus Cash EV"):
    bc = bonus_cash_ev(
        wager=wager,
        odds=bc_odds,
        house_edge=house_edge,
        true_prob=bc_true_prob,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("True prob used", f"{bc['true_prob']:.6f}")
    c2.metric("Win payout ($)", f"{bc['win_payout']:.4f}")
    c3.metric("EV ($)", f"{bc['ev_net']:.4f}")


# ----------------------------
# Section 4: Bet Back EV (single bet)
# ----------------------------

st.divider()
st.subheader("Bet Back EV (single runner)")

st.caption(
    "Two ways to interpret place odds:\n"
    "- **Unconditional Top-3**: place includes win (typical). We use this below.\n"
    "- Conditional given not win: supported in code, but not used by default."
)

bb_win_odds = st.number_input("Win odds", min_value=1.0, value=3.0, step=0.01, key="bb_win_odds")
bb_place_odds = st.number_input("Place odds (Top-3, includes win)", min_value=1.0, value=1.5, step=0.01, key="bb_place_odds")

bb_override = st.checkbox("Override true probabilities for Bet Back?", value=False, key="bb_override")
if bb_override:
    bb_true_win = st.number_input("True P(win)", min_value=0.0, max_value=1.0, value=0.30, step=0.01, key="bb_pwin")
    bb_true_place = st.number_input("True P(place Top-3)", min_value=0.0, max_value=1.0, value=0.45, step=0.01, key="bb_pplace")
else:
    bb_true_win = None
    bb_true_place = None

if st.button("Calculate Bet Back EV"):
    bb = bet_back_ev_unconditional_place(
        wager=wager,
        win_odds=bb_win_odds,
        place_odds=bb_place_odds,
        house_edge=house_edge,
        bonus_val=bonus_val,
        true_prob_win=bb_true_win,
        true_prob_place_uncond=bb_true_place,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("True P(win)", f"{bb['true_prob_win']:.6f}")
    c2.metric("True P(place)", f"{bb['true_prob_place_unconditional']:.6f}")
    c3.metric("EV payout ($)", f"{bb['ev_payout']:.4f}")

    st.dataframe(
        pd.DataFrame(
            {
                "Outcome": ["Win", "Place but not win", "Lose all"],
                "Probability": [bb["p_win"], bb["p_place_not_win"], bb["p_lose_all"]],
                "Payout ($)": [bb["payout_win"], bb["payout_betback"], 0.0],
            }
        ),
        use_container_width=True,
    )


# ----------------------------
# Section 5: Vig calculator (WIN + PLACE for up to 20 runners) + EV using bet-back wager
# ----------------------------

st.divider()
st.subheader("Vig (Overround) calculator: WIN + PLACE markets (Top-3)")

st.caption(
    "Enter **win odds** and **place odds** for as many runners as you have (1 to N; up to 20). "
    "Assumptions:\n"
    "- Exactly **one** runner wins.\n"
    "- **Place** means **Top-3 (1st/2nd/3rd), includes the winner**.\n"
    "- We compute WIN overround vs 1, and PLACE overround vs 3.\n"
    "- We also **de-vig** by normalizing implied probabilities."
)

N_MAX = 20
rows = []

# A compact grid: 20 rows x 3 cols
for i in range(N_MAX):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input(f"Runner {i+1} name", value="", key=f"runner_name_{i}")
    with c2:
        w = st.text_input(f"Win odds {i+1}", value="", key=f"runner_win_{i}")
    with c3:
        p = st.text_input(f"Place odds {i+1}", value="", key=f"runner_place_{i}")

    # parse
    try:
        win_odds = float(w) if str(w).strip() != "" else None
    except ValueError:
        win_odds = None
    try:
        place_odds = float(p) if str(p).strip() != "" else None
    except ValueError:
        place_odds = None

    if name.strip() == "" and win_odds is None and place_odds is None:
        continue

    rows.append(
        {
            "runner": name.strip() if name.strip() else f"Runner {i+1}",
            "win_odds": win_odds,
            "place_odds": place_odds,
        }
    )

valid = [r for r in rows if (r["win_odds"] is not None and r["place_odds"] is not None and r["win_odds"] >= 1 and r["place_odds"] >= 1)]

if len(rows) == 0:
    st.info("Fill at least one row (runner name optional, but both win odds and place odds are needed to compute vig).")
elif len(valid) == 0:
    st.warning("You have entries, but none have BOTH valid win odds and place odds (>= 1).")
else:
    win_odds_list = [r["win_odds"] for r in valid]
    place_odds_list = [r["place_odds"] for r in valid]

    vig = vig_from_win_place_markets(win_odds_list, place_odds_list, places=3)

    # Show market edges
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WIN sum implied", f"{vig['sum_implied_win']:.6f}")
    c2.metric("WIN overround", f"{vig['overround_win']:.6f}")
    c3.metric("PLACE sum implied", f"{vig['sum_implied_place']:.6f}")
    c4.metric("PLACE overround (abs vs 3)", f"{vig['overround_place_abs']:.6f}")

    st.write("PLACE overround (relative):", f"{(vig['overround_place_rel']*100):.3f}%" if vig["overround_place_rel"] is not None else "N/A")

    # Build table with de-vigged probs
    table = []
    for r, pwin, pplace in zip(valid, vig["p_true_win"], vig["p_true_place"]):
        table.append(
            {
                "Runner": r["runner"],
                "Win odds": r["win_odds"],
                "Place odds": r["place_odds"],
                "De-vig P(win)": pwin,
                "De-vig P(place top3)": pplace,
                "Implied P(win)": 1.0 / r["win_odds"],
                "Implied P(place)": 1.0 / r["place_odds"],
            }
        )
    df = pd.DataFrame(table)
    st.dataframe(df, use_container_width=True)

    # Select one runner to compute Bet Back EV using the *de-vigged* probabilities
    st.write("### Pick a runner to evaluate EV using the Bet Back feature (de-vigged probs)")
    runner_names = [r["runner"] for r in valid]
    pick = st.selectbox("Select runner", runner_names, index=0)

    idx = runner_names.index(pick)
    sel = valid[idx]
    p_win_true = vig["p_true_win"][idx]
    p_place_true = vig["p_true_place"][idx]

    # sanity: place includes win, so place_not_win = place - win
    p_place_not_win = p_place_true - p_win_true
    if p_place_not_win < -1e-9:
        st.warning(
            "For this runner, de-vigged P(place) came out < P(win). "
            "That can happen due to inconsistent markets; EV will clamp place_not_win to 0."
        )
    p_place_not_win = max(p_place_not_win, 0.0)

    # EV with bet-back (unconditional place)
    ev = bet_back_ev_unconditional_place(
        wager=wager,
        win_odds=sel["win_odds"],
        place_odds=sel["place_odds"],
        house_edge=house_edge,  # not used since we pass true probs, but kept for consistency
        bonus_val=bonus_val,
        true_prob_win=p_win_true,
        true_prob_place_uncond=p_place_true,
    )

    st.write("### EV summary (selected runner)")
    c1, c2, c3 = st.columns(3)
    c1.metric("De-vig P(win)", f"{p_win_true:.6f}")
    c2.metric("De-vig P(place)", f"{p_place_true:.6f}")
    c3.metric("EV payout ($)", f"{ev['ev_payout']:.4f}")

    st.dataframe(
        pd.DataFrame(
            {
                "Outcome": ["Win", "Place but not win", "Lose all"],
                "Probability": [ev["p_win"], ev["p_place_not_win"], ev["p_lose_all"]],
                "Payout ($)": [ev["payout_win"], ev["payout_betback"], 0.0],
            }
        ),
        use_container_width=True,
    )

    st.write(f"EV net vs cash stake: **{ev['ev_net_cash']:.4f}**")
    st.write(f"EV if stake itself is bonus (treat wager as free): **{ev['ev_net_if_bonus_stake']:.4f}**")
