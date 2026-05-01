#!/usr/bin/env python3
"""
Position Monitor Service — persistent daemon that checks positions every 5 min
during market hours and sends alerts via inbox for critical conditions.

Alerts on:
1. Short calls losing > 100% (CRITICAL)
2. Short calls losing > 50% (WARNING)
3. Positions with < 7 DTE (EXPIRING)
4. Large P&L swings (> 15% change since last check)
5. New fills or position closures

Sends alerts via jarvis_utils.inbox so Jarvis sees them next turn.
"""

import json
import os
import sys
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add trading_bot to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/opt/jarvis-utils/lib")

# Spread detection lives in position_monitor.py (single source of truth).
from position_monitor import detect_spread_membership  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logger = logging.getLogger("position_monitor_svc")

STATE_FILE = os.path.expanduser("~/trading_bot/monitor_state.json")
JOURNAL_FILE = os.path.expanduser("~/memory/trading_journal.md")
CHECK_INTERVAL = 300  # 5 minutes
HEARTBEAT_INTERVAL_SEC = 600  # Edge 108b — emit cycle heartbeat every 10 min
MARKET_CHECK_INTERVAL = 60  # Check market status every 60s when waiting

# Alert thresholds
CRITICAL_LOSS_PCT = -100.0  # Short call losing > 100%
WARNING_LOSS_PCT = -50.0    # Short call losing > 50%
EXPIRY_WARNING_DTE = 7      # Days to expiry warning
PNL_CHANGE_THRESHOLD = 15.0 # % change since last check


ALPACA_USER_ID = 'a4dc8459-608d-49f5-943e-e5e105ed5207'


def get_alpaca_creds():
    from jarvis_utils.secrets import get
    api_key = get('Alpaca', 'api_key_id', user=ALPACA_USER_ID)
    secret = get('Alpaca', 'secret_key', user=ALPACA_USER_ID)
    return api_key, secret


# === Edge 123: auto-refresh Alpaca credentials on 401 ===
# Previously used plain requests.get which caches stale Alpaca creds
# across server-side key rotations. Migrated Apr 22 2026 to the canonical
# _AutoRefreshSession pattern from alpaca_client.py. See
# ~/memory/edge123_audit_apr22.md for the audit.
_session = None
_session_api_key = None
_session_api_secret = None


def _refresh_alpaca_credentials(session):
    """Callback invoked by _AutoRefreshSession on 401. Pulls fresh creds
    from the portal and rewrites the session headers in place."""
    global _session_api_key, _session_api_secret
    from jarvis_utils.secrets import get
    new_key = get('Alpaca', 'api_key_id', user=ALPACA_USER_ID)
    new_secret = get('Alpaca', 'secret_key', user=ALPACA_USER_ID)
    if not new_key or not new_secret:
        raise EnvironmentError("position_monitor_service: portal returned empty Alpaca creds")
    _session_api_key = new_key
    _session_api_secret = new_secret
    session.headers['APCA-API-KEY-ID'] = new_key
    session.headers['APCA-API-SECRET-KEY'] = new_secret
    logger.info("position_monitor: Alpaca credentials refreshed (key prefix %s...)", new_key[:6])


def _get_alpaca_session():
    """Return the module-level auto-refreshing session (lazy-created)."""
    global _session, _session_api_key, _session_api_secret
    if _session is None:
        from alpaca_client import _AutoRefreshSession
        key, secret = get_alpaca_creds()
        _session_api_key = key
        _session_api_secret = secret
        _session = _AutoRefreshSession(_refresh_alpaca_credentials)
        _session.headers.update({
            'APCA-API-KEY-ID': key or '',
            'APCA-API-SECRET-KEY': secret or '',
        })
    return _session


def is_market_open(session=None):
    """Check if market is open via Alpaca clock.

    Signature change: now accepts a session (auto-refreshing) instead of
    a headers dict. Pre-Edge-123 callers passed headers — keep backward
    compat by creating a session lazily if not passed.
    """
    try:
        sess = session if session is not None else _get_alpaca_session()
        r = sess.get("https://paper-api.alpaca.markets/v2/clock", timeout=10)
        if r.status_code == 200:
            clock = r.json()
            return clock.get("is_open", False)
    except Exception as e:
        logger.warning(f"Clock check failed: {e}")
    return False


UNSENT_ALERTS_LOG = os.path.expanduser("~/logs/position_monitor_unsent_alerts.log")


def send_alert(message, source="position-monitor", max_retries=3):
    """Send alert via inbox with retry-and-fallback.

    Bug #8 fix (Apr 21 2026): previously a single `send()` call — any transient
    orchestrator HTTP timeout silently dropped the alert. Now retries with
    exponential backoff (1s, 2s, 4s), and on total failure appends the alert
    to ~/logs/position_monitor_unsent_alerts.log so nothing is lost.
    Worst-case total wall-time per alert: ~7s.
    """
    try:
        from jarvis_utils.inbox import send
    except Exception as e:
        logger.error(f"Failed to import inbox module: {e}")
        _persist_unsent(message, source, reason=f"import-failed: {e}")
        return

    last_exc = None
    for attempt in range(max_retries):
        try:
            send(message, source=source)
            if attempt > 0:
                logger.info(f"Alert sent on retry {attempt}: {message[:80]}...")
            else:
                logger.info(f"Alert sent: {message[:80]}...")
            return
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                backoff = 2 ** attempt  # 1, 2, 4
                logger.warning(
                    f"Alert send failed (attempt {attempt + 1}/{max_retries}, "
                    f"retrying in {backoff}s): {e}"
                )
                time.sleep(backoff)
            else:
                logger.error(
                    f"Alert send failed after {max_retries} attempts: {e}"
                )

    _persist_unsent(message, source, reason=f"retries-exhausted: {last_exc}")


def _persist_unsent(message, source, reason):
    """Append a dropped alert to disk so it can be recovered / re-sent later."""
    try:
        os.makedirs(os.path.dirname(UNSENT_ALERTS_LOG), exist_ok=True)
        with open(UNSENT_ALERTS_LOG, "a") as f:
            ts = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps({
                "ts": ts, "source": source, "reason": reason,
                "message": message,
            }) + "\n")
        logger.warning(f"Unsent alert persisted to {UNSENT_ALERTS_LOG}")
    except Exception as e:
        # Last-resort: log the full alert so at least it's in the service log.
        logger.error(f"Could not persist unsent alert ({e}); message was: {message}")


def parse_option_symbol(sym):
    """Parse OCC option symbol."""
    import re
    m = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', sym)
    if not m:
        return None
    ticker = m.group(1)
    date_str = m.group(2)
    put_call = m.group(3)
    strike = int(m.group(4)) / 1000.0
    exp_date = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    return ticker, exp_date, put_call, strike


def days_to_expiry(exp_date_str):
    exp = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    return (exp - today).days


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"positions": {}, "last_check": None, "alerts_sent": {}}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def check_and_alert():
    """Main check cycle — pull positions, compare, alert."""
    session = _get_alpaca_session()
    if not _session_api_key or not _session_api_secret:
        logger.error("No Alpaca credentials")
        return

    base = 'https://paper-api.alpaca.markets/v2'

    # Get account
    try:
        acct_r = session.get(f'{base}/account', timeout=10)
        acct = acct_r.json()
    except Exception as e:
        logger.error(f"Account fetch failed: {e}")
        return

    # Get positions
    try:
        pos_r = session.get(f'{base}/positions', timeout=10)
        positions = pos_r.json()
    except Exception as e:
        logger.error(f"Positions fetch failed: {e}")
        return

    prev_state = load_state()
    prev_positions = prev_state.get("positions", {})
    alerts_sent = prev_state.get("alerts_sent", {})
    alerts = []

    current_positions = {}
    # Pass 1: build current_positions dict (needed for spread detection
    # before alert generation in pass 2).
    for p in positions:
        sym = p['symbol']
        qty = float(p['qty'])  # Edge 117 — preserve fractional crypto qty (was int(float(...)))
        pnl_pct = float(p['unrealized_plpc']) * 100
        pnl = float(p['unrealized_pl'])
        current_price = float(p['current_price'])
        entry_price = float(p['avg_entry_price'])

        current_positions[sym] = {
            "qty": qty, "pnl_pct": pnl_pct, "pnl": pnl,
            "current_price": current_price, "entry_price": entry_price,
        }

    # Spread membership: legs of paired hedged structures (same ticker,
    # same call/put, opposite qty signs). Used to suppress per-leg
    # P&L MOVE alerts (which are noise — losing on one leg means the
    # other leg is gaining) and emit a single net-spread alert instead.
    spread_map = detect_spread_membership(current_positions)
    spread_pnl_now: dict[str, float] = {}
    spread_pnl_prev: dict[str, float] = {}
    for sym, sid in spread_map.items():
        spread_pnl_now[sid] = spread_pnl_now.get(sid, 0.0) + current_positions[sym]["pnl"]
        if sym in prev_positions:
            spread_pnl_prev[sid] = spread_pnl_prev.get(sid, 0.0) + prev_positions[sym].get("pnl", 0.0)

    # Pass 2: alert generation.
    for p in positions:
        sym = p['symbol']
        qty = float(p['qty'])
        pnl_pct = float(p['unrealized_plpc']) * 100
        pnl = float(p['unrealized_pl'])

        # Short call checks
        if qty < 0 and len(sym) > 10:
            parsed = parse_option_symbol(sym)
            if parsed:
                ticker, exp_date, put_call, strike = parsed
                dte = days_to_expiry(exp_date)

                # Critical loss alert (deduplicated — only alert once per threshold crossing)
                if put_call == 'C':
                    alert_key_crit = f"{sym}_critical"
                    alert_key_warn = f"{sym}_warning"
                    if pnl_pct < CRITICAL_LOSS_PCT and alert_key_crit not in alerts_sent:
                        msg = (f"🚨 CRITICAL: {sym} short call at {pnl_pct:+.1f}% "
                               f"(${pnl:+,.0f}). {ticker} ${strike} exp {exp_date} ({dte}d). "
                               f"Consider closing or rolling.")
                        alerts.append(msg)
                        alerts_sent[alert_key_crit] = datetime.now(timezone.utc).isoformat()
                        # Also mark warning as sent so it doesn't fire separately
                        alerts_sent[alert_key_warn] = datetime.now(timezone.utc).isoformat()
                    elif pnl_pct < WARNING_LOSS_PCT and alert_key_warn not in alerts_sent and alert_key_crit not in alerts_sent:
                        msg = (f"⚠️ WARNING: {sym} short call at {pnl_pct:+.1f}% "
                               f"(${pnl:+,.0f}). {ticker} ${strike} exp {exp_date} ({dte}d).")
                        alerts.append(msg)
                        alerts_sent[alert_key_warn] = datetime.now(timezone.utc).isoformat()

                # Expiry warning
                if dte <= EXPIRY_WARNING_DTE:
                    alert_key_exp = f"{sym}_expiring_{dte}"
                    if alert_key_exp not in alerts_sent:
                        msg = (f"⏰ EXPIRING: {sym} expires in {dte} days. "
                               f"P&L: {pnl_pct:+.1f}% (${pnl:+,.0f})")
                        alerts.append(msg)
                        alerts_sent[alert_key_exp] = datetime.now(timezone.utc).isoformat()

        # Large P&L swing check.
        # Suppress per-leg alerts for spread members — those legs hedge
        # each other and per-leg P&L swings are noise. Net spread alert
        # is emitted below.
        if sym in prev_positions and sym not in spread_map:
            prev_pnl_pct = prev_positions[sym].get("pnl_pct", 0)
            change = abs(pnl_pct - prev_pnl_pct)
            if change > PNL_CHANGE_THRESHOLD:
                direction = "improved" if pnl_pct > prev_pnl_pct else "worsened"
                msg = (f"📊 {sym} {direction}: {prev_pnl_pct:+.1f}% → {pnl_pct:+.1f}% "
                       f"(Δ{change:.1f}%)")
                alerts.append(msg)

    # Spread-level P&L MOVE alerts: emit one alert per spread when the
    # net P&L moved by more than $25 (absolute, since spread sizes vary
    # so % is unstable). Replaces the per-leg noise that was misleading
    # for hedged structures (Apr 22 2026 fix — COPX 100C "worsened" -50%
    # when the long leg gained more, net spread was +$70).
    SPREAD_NET_DOLLAR_THRESHOLD = 25.0
    for sid, pnl_now in spread_pnl_now.items():
        pnl_prev = spread_pnl_prev.get(sid)
        if pnl_prev is None:
            continue
        delta = pnl_now - pnl_prev
        if abs(delta) > SPREAD_NET_DOLLAR_THRESHOLD:
            direction = "improved" if delta > 0 else "worsened"
            legs = sorted(s for s, m in spread_map.items() if m == sid)
            legs_str = ",".join(legs)
            alerts.append(
                f"📊 SPREAD {sid} {direction}: net=${pnl_prev:+.0f} → "
                f"${pnl_now:+.0f} (Δ${delta:+.0f}) [{legs_str}]"
            )

    # Closed positions
    for sym in prev_positions:
        if sym not in current_positions:
            msg = f"🔒 CLOSED: {sym} (was {prev_positions[sym].get('pnl_pct', 0):+.1f}%)"
            alerts.append(msg)

    # New positions
    for sym in current_positions:
        if sym not in prev_positions:
            msg = f"🆕 NEW: {sym} qty={current_positions[sym]['qty']}"
            alerts.append(msg)

    # Clean up old alert dedup keys (older than 24h)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    alerts_sent = {k: v for k, v in alerts_sent.items() if v > cutoff}

    # Save state
    new_state = {
        "positions": current_positions,
        "last_check": datetime.now(timezone.utc).isoformat(),
        "equity": float(acct.get('equity', 0)),
        "cash": float(acct.get('cash', 0)),
        "alerts_sent": alerts_sent,
    }
    save_state(new_state)

    # Update journal
    update_journal(acct, positions)

    # Send consolidated alert if any
    if alerts:
        header = f"Position Monitor — {len(alerts)} alert(s)"
        full_msg = header + "\n" + "\n".join(alerts)
        send_alert(full_msg)
        logger.info(f"Sent {len(alerts)} alerts")
    else:
        logger.info("No alerts this cycle")


def update_journal(acct, positions):
    """Update trading_journal.md."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M ET')
    equity = float(acct.get('equity', 0))
    cash = float(acct.get('cash', 0))
    pnl = equity - 100000

    lines = [f"# Trading Journal", f"Last updated: {now}", ""]
    lines.append("## Account Summary")
    lines.append(f"- Equity: ${equity:,.2f}")
    lines.append(f"- Cash: ${cash:,.2f}")
    lines.append(f"- P&L from $100K: ${pnl:,.2f} ({pnl/1000:.1f}%)")
    lines.append("")

    shares = []
    options = []
    for p in positions:
        sym = p['symbol']
        qty = int(float(p['qty']))
        entry = float(p['avg_entry_price'])
        current = float(p['current_price'])
        upnl = float(p['unrealized_pl'])
        pnl_pct = float(p['unrealized_plpc']) * 100
        rec = (sym, qty, entry, current, upnl, pnl_pct)
        if len(sym) > 10:
            options.append(rec)
        else:
            shares.append(rec)

    lines.append("## Open Positions")
    if shares:
        lines.append("\n### Shares")
        lines.append("| Ticker | Qty | Entry | Current | P&L | P&L % |")
        lines.append("|--------|-----|-------|---------|-----|-------|")
        for sym, qty, entry, cur, upnl, pct in sorted(shares):
            lines.append(f"| {sym} | {qty} | ${entry:.2f} | ${cur:.2f} | ${upnl:+,.2f} | {pct:+.1f}% |")

    if options:
        lines.append("\n### Options")
        lines.append("| Symbol | Qty | Entry | Current | P&L | P&L % | DTE |")
        lines.append("|--------|-----|-------|---------|-----|-------|-----|")
        for sym, qty, entry, cur, upnl, pct in sorted(options):
            parsed = parse_option_symbol(sym)
            dte = days_to_expiry(parsed[1]) if parsed else "?"
            side = "SHORT" if qty < 0 else "LONG"
            lines.append(f"| {sym} | {qty} ({side}) | ${entry:.2f} | ${cur:.2f} | ${upnl:+,.2f} | {pct:+.1f}% | {dte} |")

    lines.append("")
    with open(JOURNAL_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    logger.info("Position Monitor Service starting")

    # Validate credentials on startup (startup-validation safety)
    api_key, secret = get_alpaca_creds()
    if not api_key or not secret:
        logger.error("No Alpaca credentials — exiting cleanly")
        sys.exit(0)

    # Verify API access using the auto-refreshing session
    session = _get_alpaca_session()
    try:
        r = session.get("https://paper-api.alpaca.markets/v2/account", timeout=10)
        r.raise_for_status()
        logger.info("Alpaca API connection verified")
    except Exception as e:
        logger.error(f"Alpaca API check failed: {e} — exiting")
        sys.exit(0)

    send_alert("Position Monitor Service started.", source="position-monitor")

    last_heartbeat = 0.0  # Edge 108b — wall-clock of last HEARTBEAT line
    while True:
        try:
            market_open = is_market_open(session)
            if market_open:
                logger.info("Market open — running position check")
                check_and_alert()
            # else: silent fall-through; the heartbeat below handles closed-hours visibility

            # Edge 108b — per-cycle heartbeat at INFO level. Without this,
            # the service is silent during closed hours (DEBUG suppressed)
            # and indistinguishable from a wedged process.
            now_ts = time.time()
            if now_ts - last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
                last_heartbeat = now_ts
                logger.info(
                    "HEARTBEAT market_open=%s next_check_in=%ds",
                    market_open, CHECK_INTERVAL,
                )

            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
