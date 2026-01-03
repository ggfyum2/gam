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

    # EV of payout (your model ignores other outcomes' payouts = 0)
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

    # Breakeven condition: q p^2 O + [2 q p(1-p) + (1-q)p^2] B = 1
    # => A p^2 + C p - 1 = 0
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
    "Bonus cash" bet EV:
    - You stake bonus cash, so you do NOT get the stake back.
    - Example: wager=50, odds=3.0 -> profit on win = (odds-1)*wager = 100 (not 150)
    - On loss: payout=0
    EV is computed using true probability.
      - If true_prob is None, infer as 1/(odds+house_edge) (same convention as elsewhere).
    Returns payout EV and net EV (net is same as payout here since stake is bonus).
    """
    p = true_prob if true_prob is not None else 1.0 / (odds + house_edge)
    win_payout = (odds - 1.0) * wager  # profit only, no stake returned
    ev_payout = p * win_payout
    ev_net = ev_payout  # since you didn't pay real cash / no stake returned
    return {
        "true_prob": p,
        "win_payout": win_payout,
        "ev_payout": ev_payout,
        "ev_net": ev_net,
    }


# ----------------------------
# UI
# ----------------------------

st.title("Bonus Parlay EV + Breakeven (bet2 = bet3)")

with st.sidebar:
    st.header("Inputs")

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
# Section 1: Your 3-leg EV model
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
    true_prob1=true_prob1,  # if None, inferred from offer1 + house_edge
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
            true_prob1=true_prob1,  # fixed / inferred
            true_prob2=p_be,
            true_prob3=p_be,
        )
        st.write("### Sanity check results (using p_be for leg2 & leg3)")
        st.write(f"EV payout: {check['ev_payout']:.6f} (target = wager {wager:.6f})")
        st.write(f"EV net: {check['ev_net']:.6f} (target = 0)")


# ----------------------------
# Section 3: Bonus cash EV calculator (new)
# ----------------------------

st.divider()
st.subheader("Bonus cash EV (stake not returned)")

st.caption(
    "Bonus cash means you don’t get your stake back. "
    "So wager=50 at odds=3.0 pays (3.0-1)*50=100 on win (not 150)."
)

bc_odds = st.number_input("Bonus cash odds", min_value=1.0, value=3.0, step=0.01)
use_bc_p = st.checkbox("Override true probability for bonus cash?", value=False)
bc_true_prob = (
    st.number_input("Bonus cash true probability", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
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

    st.write(
        f"If you stake **{wager:.2f}** bonus cash at **{bc_odds:.2f}** odds, "
        f"you win **{bc['win_payout']:.2f}** (profit only) with probability **{bc['true_prob']:.4f}**."
    )


def bet_back_ev(
    wager: float,
    win_odds: float,
    place_odds: float,
    house_edge: float = 0.06,
    bonus_val: float = 1.0,
    true_prob_win: float | None = None,
    true_prob_place: float | None = None,
) -> dict:
    """
    EV for a single bet with a "bet back" feature:

    Outcomes:
      1) WIN:
         - You get normal winnings (assumed stake returned): payout = win_odds * wager
      2) LOSE but FINISH 2nd/3rd ("place" outcome):
         - You get your wager back as BONUS CASH, valued at bonus_val per $1 bonus:
           payout = bonus_val * wager
      3) LOSE and NOT place:
         - payout = 0

    Probabilities:
      - true_prob_win = P(win)
      - true_prob_place = P(place | not win)   (conditional probability among non-wins)

    If true_prob_win is None, inferred as 1/(win_odds + house_edge).
    If true_prob_place is None, inferred as 1/(place_odds + house_edge).

    EV logic:
      P(win) = p_win
      P(place_and_not_win) = (1 - p_win) * p_place
      P(lose_all) = (1 - p_win) * (1 - p_place)

      EV_payout = wager * [ p_win*win_odds + (1-p_win)*p_place*bonus_val ]
      EV_net_cash = EV_payout - wager   (net vs staking real cash)
      EV_net_bonus_stake = EV_payout    (if you treat the wager as bonus stake / no cash cost)

    Returns a dict with probabilities and EVs.
    """
    # infer true probabilities if not provided
    p_win = true_prob_win if true_prob_win is not None else 1.0 / (win_odds + house_edge)
    p_place = true_prob_place if true_prob_place is not None else 1.0 / (place_odds + house_edge)

    # probabilities of each payout-relevant event
    p_place_not_win = (1.0 - p_win) * p_place
    p_lose_all = (1.0 - p_win) * (1.0 - p_place)

    # payouts
    payout_win = win_odds * wager                 # stake returned on win
    payout_betback = bonus_val * wager            # bet back as bonus cash
    payout_lose = 0.0

    # EV of payout
    ev_payout = p_win * payout_win + p_place_not_win * payout_betback + p_lose_all * payout_lose

    return {
        "true_prob_win": p_win,
        "true_prob_place_conditional": p_place,
        "p_win": p_win,
        "p_place_not_win": p_place_not_win,
        "p_lose_all": p_lose_all,
        "payout_win": payout_win,
        "payout_betback": payout_betback,
        "ev_payout": ev_payout,
        "ev_net_cash": ev_payout - wager,
        "ev_net_if_bonus_stake": ev_payout,
    }
