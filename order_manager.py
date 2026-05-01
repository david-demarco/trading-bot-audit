"""
order_manager.py - Active order management for the covered call scalper.

Replaces the fire-and-forget ExecutionLayer workflow with a full lifecycle:

    POST limit order -> MONITOR (poll status every N seconds) ->
        IF filled     -> record position, done
        IF not filled -> evaluate market, cancel, adjust price, re-post
        IF max attempts reached -> cancel and abort

Smart price adjustment:
    - Selling calls: start above mid, step toward mid if unfilled, never below mid.
    - Buying back calls: start below mid, step toward mid if unfilled, never above mid.

Rate-limiting:
    - Sliding-window counter keeps us under 180 req/min (Alpaca max = 200).
    - Minimum 3 seconds between cancel/replace on the same order.
    - Max 10 cancel/replace cycles per order.

Works in both dry-run (simulated fills) and live mode (Alpaca REST).
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from slvr_cc_scalper import ExecutionLayer

from slvr_cc_config import (
    ALPACA_BASE_URL,
    ORDER_CHECK_INTERVAL,
    ORDER_MAX_WAIT,
    ORDER_MAX_ATTEMPTS,
    ORDER_PRICE_STEP,
    ORDER_MIN_INTERVAL,
    RATE_LIMIT_WINDOW,
    RATE_LIMIT_MAX_REQUESTS,
    ORDER_TIF,
)

logger = logging.getLogger("cc_scalper.order_mgr")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ManagedOrder:
    """Tracks every aspect of an order through its full lifecycle."""
    order_id: str
    contract_symbol: str
    side: str                       # "sell" or "buy"
    original_price: float
    current_price: float
    contracts: int
    posted_at: float                # time.time() when first posted
    last_check: float               # time.time() of last status poll
    attempts: int                   # cancel/replace count so far
    status: str                     # pending, filled, partial, cancelled, expired, failed
    fill_price: Optional[float] = None
    filled_qty: int = 0
    # Internal tracking for price adjustment logic
    last_bid: float = 0.0
    last_ask: float = 0.0
    last_mid: float = 0.0
    last_adjust_time: float = 0.0   # time.time() of last cancel/replace
    cancel_pending: bool = False     # True while waiting for cancel confirmation
    is_gtc: bool = False             # True for GTC orders (skip aggressive cancel/replace)
    # Callback context -- lets the scalper know when an order completes
    _on_fill: Optional[Callable] = field(default=None, repr=False)
    _on_abort: Optional[Callable] = field(default=None, repr=False)

    @property
    def elapsed(self) -> float:
        """Seconds since the order was first posted."""
        return time.time() - self.posted_at

    @property
    def side_label(self) -> str:
        return "SELL" if self.side == "sell" else "BUY"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Sliding-window rate limiter for Alpaca API requests.

    Tracks timestamps of recent requests in a deque and blocks if the
    count within the window would exceed the limit.
    """

    def __init__(self, max_requests: int = RATE_LIMIT_MAX_REQUESTS,
                 window_seconds: float = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: Deque[float] = deque()

    def _prune(self) -> None:
        """Remove timestamps older than the window."""
        cutoff = time.time() - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def can_request(self, count: int = 1) -> bool:
        """Return True if *count* more requests fit within the window."""
        self._prune()
        return len(self._timestamps) + count <= self.max_requests

    def record(self, count: int = 1) -> None:
        """Record that *count* requests were just made."""
        now = time.time()
        for _ in range(count):
            self._timestamps.append(now)

    def wait_if_needed(self, count: int = 1) -> None:
        """Block until *count* requests can be made within the window.

        If we are at the limit, sleeps until the oldest request falls
        out of the window.
        """
        while not self.can_request(count):
            self._prune()
            if self._timestamps:
                wait = self._timestamps[0] + self.window_seconds - time.time() + 0.05
                if wait > 0:
                    logger.debug("Rate-limit: waiting %.1fs before next request", wait)
                    time.sleep(wait)
            else:
                break

    @property
    def requests_in_window(self) -> int:
        self._prune()
        return len(self._timestamps)


# ---------------------------------------------------------------------------
# Dry-run fill simulator
# ---------------------------------------------------------------------------

class _DryRunSimulator:
    """Simulates realistic option-fill behavior for dry-run mode.

    - First attempt: ~50% chance of fill within 3-10 seconds.
    - After adjustment: higher fill probability (70-90%).
    - Partial fills are possible.
    """

    def __init__(self):
        # order_id -> {created_at, will_fill_at, fill_pct, filled}
        self._orders: Dict[str, Dict[str, Any]] = {}

    def create_order(self, order_id: str, side: str, price: float,
                     contracts: int) -> None:
        """Register a new simulated order."""
        delay = random.uniform(3.0, 10.0)
        fill_chance = random.random()  # 0-1
        self._orders[order_id] = {
            "created_at": time.time(),
            "will_fill_at": time.time() + delay,
            "will_fill": fill_chance < 0.50,   # 50% on first attempt
            "fill_price": round(price + random.uniform(-0.02, 0.02), 2),
            "contracts": contracts,
            "filled_qty": 0,
            "status": "new",
            "side": side,
        }

    def create_adjusted_order(self, order_id: str, side: str, price: float,
                              contracts: int, attempt: int) -> None:
        """Register a re-posted order after adjustment (higher fill chance)."""
        delay = random.uniform(1.5, 5.0)
        # Fill probability increases with each attempt
        fill_chance = random.random()
        threshold = min(0.70 + attempt * 0.05, 0.95)
        self._orders[order_id] = {
            "created_at": time.time(),
            "will_fill_at": time.time() + delay,
            "will_fill": fill_chance < threshold,
            "fill_price": round(price + random.uniform(-0.01, 0.01), 2),
            "contracts": contracts,
            "filled_qty": 0,
            "status": "new",
            "side": side,
        }

    def check_status(self, order_id: str) -> Dict[str, Any]:
        """Return simulated order status mimicking Alpaca response fields."""
        info = self._orders.get(order_id)
        if info is None:
            return {"status": "canceled", "filled_qty": "0",
                    "filled_avg_price": None}

        now = time.time()
        if info["status"] == "canceled":
            return {"status": "canceled", "filled_qty": str(info["filled_qty"]),
                    "filled_avg_price": str(info["fill_price"]) if info["filled_qty"] > 0 else None}

        if now >= info["will_fill_at"] and info["will_fill"]:
            info["status"] = "filled"
            info["filled_qty"] = info["contracts"]
            return {
                "status": "filled",
                "filled_qty": str(info["contracts"]),
                "filled_avg_price": str(info["fill_price"]),
            }

        return {"status": "new", "filled_qty": "0", "filled_avg_price": None}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a simulated order. Returns True on success."""
        info = self._orders.get(order_id)
        if info is None:
            return False
        if info["status"] in ("filled",):
            return False  # can't cancel a fill
        info["status"] = "canceled"
        return True


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """Active order management loop for the covered call scalper.

    Instantiated by CCScalper; holds a reference to the ExecutionLayer for
    placing and cancelling orders via Alpaca (or dry-run simulation).

    Public API:
        submit_sell(contract_symbol, contracts, limit_price, mid_price, bid, ask,
                    on_fill=None, on_abort=None) -> ManagedOrder
        submit_buy_back(contract_symbol, contracts, limit_price, mid_price, bid, ask,
                        on_fill=None, on_abort=None) -> ManagedOrder
        manage_orders() -> List[ManagedOrder]   # call each cycle
        get_pending_orders() -> List[ManagedOrder]
        get_filled_orders() -> List[ManagedOrder]
    """

    def __init__(self, executor: "ExecutionLayer", dry_run: bool = True):
        self.executor = executor
        self.dry_run = dry_run
        self.rate_limiter = SlidingWindowRateLimiter()
        self._orders: Dict[str, ManagedOrder] = {}      # order_id -> ManagedOrder
        self._completed: List[ManagedOrder] = []         # filled / cancelled / failed
        self._sim = _DryRunSimulator() if dry_run else None

        logger.info(
            "OrderManager initialized  dry_run=%s  check_interval=%ds  "
            "max_wait=%ds  max_attempts=%d  price_step=$%.2f  "
            "rate_limit=%d/%ds",
            dry_run, ORDER_CHECK_INTERVAL, ORDER_MAX_WAIT,
            ORDER_MAX_ATTEMPTS, ORDER_PRICE_STEP,
            RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW,
        )

    # ------------------------------------------------------------------
    # Submit helpers
    # ------------------------------------------------------------------

    def submit_sell(
        self,
        contract_symbol: str,
        contracts: int,
        limit_price: float,
        mid_price: float = 0.0,
        bid: float = 0.0,
        ask: float = 0.0,
        on_fill: Optional[Callable] = None,
        on_abort: Optional[Callable] = None,
    ) -> Optional[ManagedOrder]:
        """Post a limit SELL order and begin managing it.

        Returns the ManagedOrder on success, None if the order could not
        be posted (network error, rate-limit, etc.).
        """
        return self._submit(
            side="sell",
            contract_symbol=contract_symbol,
            contracts=contracts,
            limit_price=limit_price,
            mid_price=mid_price,
            bid=bid,
            ask=ask,
            on_fill=on_fill,
            on_abort=on_abort,
        )

    def submit_buy_back(
        self,
        contract_symbol: str,
        contracts: int,
        limit_price: float,
        mid_price: float = 0.0,
        bid: float = 0.0,
        ask: float = 0.0,
        on_fill: Optional[Callable] = None,
        on_abort: Optional[Callable] = None,
        tif: Optional[str] = None,
    ) -> Optional[ManagedOrder]:
        """Post a limit BUY order and begin managing it.

        Args:
            tif: Time-in-force override. Use "gtc" for PMCC profit-taking
                 buy-backs to avoid DAY order cancel/replace spam.

        Returns the ManagedOrder on success, None if the order could not
        be posted.
        """
        return self._submit(
            side="buy",
            contract_symbol=contract_symbol,
            contracts=contracts,
            limit_price=limit_price,
            mid_price=mid_price,
            bid=bid,
            ask=ask,
            on_fill=on_fill,
            on_abort=on_abort,
            tif=tif,
        )

    def _submit(
        self,
        side: str,
        contract_symbol: str,
        contracts: int,
        limit_price: float,
        mid_price: float,
        bid: float,
        ask: float,
        on_fill: Optional[Callable],
        on_abort: Optional[Callable],
        tif: Optional[str] = None,
    ) -> Optional[ManagedOrder]:
        """Internal: post the order through the ExecutionLayer and create a
        ManagedOrder to track it."""

        # === Edge 104(b) same-contract debounce ===
        # Refuse to submit a 2nd order for the same (contract_symbol, side)
        # while a prior one is still in-flight. Eliminates the AG
        # cancel-vs-fill race that produced 3 unintended fills in one
        # buy-back cycle. Same-contract OPPOSITE side is allowed --
        # OrderDeduplicator handles wash-trade collisions separately.
        for existing_id, existing_mo in self._orders.items():
            if (existing_mo.contract_symbol == contract_symbol
                    and existing_mo.side == side
                    and existing_mo.status in ("pending", "partial")):
                logger.warning(
                    "DEBOUNCE BLOCK (Edge 104b): %s %s already in-flight "
                    "(order_id=%s, age=%.1fs, status=%s); refusing duplicate.",
                    side.upper(), contract_symbol, existing_id,
                    time.time() - existing_mo.posted_at, existing_mo.status,
                )
                return None
        # === end Edge 104(b) debounce ===

        # Rate-limit guard
        self.rate_limiter.wait_if_needed(count=1)

        # Place the order
        if side == "sell":
            order_id = self.executor.sell_call(contract_symbol, contracts, limit_price)
        else:
            order_id = self.executor.buy_back_call(contract_symbol, contracts, limit_price, tif=tif)

        if order_id is None:
            logger.error(
                "ORDER POST FAILED: %s %dx %s @ $%.2f -- executor returned None",
                side.upper(), contracts, contract_symbol, limit_price,
            )
            return None

        self.rate_limiter.record(count=1)

        # Register with dry-run simulator
        if self._sim is not None:
            self._sim.create_order(order_id, side, limit_price, contracts)

        now = time.time()
        mo = ManagedOrder(
            order_id=order_id,
            contract_symbol=contract_symbol,
            side=side,
            original_price=limit_price,
            current_price=limit_price,
            contracts=contracts,
            posted_at=now,
            last_check=now,
            attempts=0,
            status="pending",
            last_bid=bid,
            last_ask=ask,
            last_mid=mid_price if mid_price > 0 else (bid + ask) / 2.0 if bid > 0 and ask > 0 else limit_price,
            last_adjust_time=0.0,
            is_gtc=(tif == "gtc"),
            _on_fill=on_fill,
            _on_abort=on_abort,
        )

        self._orders[order_id] = mo

        logger.info(
            "ORDER POSTED: %s %dx %s @ $%.2f (mid=$%.2f, bid=$%.2f, ask=$%.2f) "
            "(order_id=%s)",
            mo.side_label, contracts, contract_symbol, limit_price,
            mo.last_mid, bid, ask, order_id,
        )

        return mo

    # ------------------------------------------------------------------
    # Order management loop -- called every cycle
    # ------------------------------------------------------------------

    def manage_orders(self) -> List[ManagedOrder]:
        """Check all pending orders, adjust prices if needed.

        Returns a list of orders that reached a terminal state (filled,
        cancelled, failed) during this call.
        """
        newly_completed: List[ManagedOrder] = []
        now = time.time()

        # Work on a snapshot of current order IDs to avoid mutation during iteration
        order_ids = list(self._orders.keys())

        for oid in order_ids:
            mo = self._orders.get(oid)
            if mo is None:
                continue

            # Respect check interval
            if now - mo.last_check < ORDER_CHECK_INTERVAL:
                continue

            # --- Poll order status ---
            status_data = self._check_order_status(mo)
            if status_data is None:
                # Network error -- skip this cycle, try again later
                mo.last_check = now
                continue

            alpaca_status = status_data.get("status", "")
            filled_qty = int(status_data.get("filled_qty", 0) or 0)
            filled_avg = status_data.get("filled_avg_price")
            filled_avg_price = float(filled_avg) if filled_avg else None

            mo.last_check = now

            elapsed = mo.elapsed

            logger.info(
                "ORDER CHECK: %s %dx %s -- status=%s, %.0fs elapsed, "
                "attempt %d/%d, price=$%.2f",
                mo.side_label, mo.contracts, mo.contract_symbol,
                alpaca_status, elapsed, mo.attempts, ORDER_MAX_ATTEMPTS,
                mo.current_price,
            )

            # --- Handle terminal states ---
            if alpaca_status == "filled":
                mo.status = "filled"
                mo.fill_price = filled_avg_price
                mo.filled_qty = filled_qty if filled_qty > 0 else mo.contracts
                self._complete_order(mo, newly_completed)
                logger.info(
                    "ORDER FILLED: %s %dx %s @ $%.2f avg (%d attempts, %.0fs)",
                    mo.side_label, mo.filled_qty, mo.contract_symbol,
                    mo.fill_price or mo.current_price,
                    mo.attempts, elapsed,
                )
                continue

            if alpaca_status in ("canceled", "cancelled", "expired", "done_for_day"):
                # If we did not initiate the cancel, treat as external cancel
                if not mo.cancel_pending:
                    mo.status = "cancelled"
                    mo.filled_qty = filled_qty
                    if filled_qty > 0 and filled_avg_price:
                        mo.fill_price = filled_avg_price
                        mo.status = "partial"
                        logger.info(
                            "ORDER PARTIAL FILL (then cancelled): %s %d/%dx %s @ $%.2f",
                            mo.side_label, filled_qty, mo.contracts,
                            mo.contract_symbol, filled_avg_price,
                        )
                    else:
                        logger.info(
                            "ORDER CANCELLED (external): %s %dx %s after %.0fs",
                            mo.side_label, mo.contracts, mo.contract_symbol, elapsed,
                        )
                    self._complete_order(mo, newly_completed)
                    continue
                else:
                    # We initiated the cancel for a price adjustment -- handle below
                    mo.cancel_pending = False
                    # Fall through to the adjustment logic

            if alpaca_status in ("pending_cancel", "pending_replace"):
                # Still waiting for cancel to confirm -- do nothing this cycle
                continue

            # --- Handle partially filled ---
            if alpaca_status == "partially_filled" and filled_qty > 0:
                mo.filled_qty = filled_qty
                if filled_avg_price:
                    mo.fill_price = filled_avg_price
                # Do not cancel partially filled orders prematurely; let them
                # work unless max wait exceeded.
                # (Fall through to the adjustment check.)

            # --- Decide whether to adjust price ---

            # If we just got cancel confirmation, post replacement order
            if alpaca_status in ("canceled", "cancelled") and not mo.cancel_pending:
                self._repost_order(mo, newly_completed)
                continue

            # Still working -- check if we should cancel and adjust
            if alpaca_status in ("new", "accepted", "partially_filled"):
                # GTC orders: only cancel/replace if market moved significantly
                # (>10% from current limit price).  This prevents the DAY order
                # cancel/replace spam that caused 50+ canceled orders on Mar 13.
                if mo.is_gtc:
                    mid = mo.last_mid
                    if mid > 0 and mo.current_price > 0:
                        price_drift = abs(mid - mo.current_price) / mo.current_price
                        if price_drift <= 0.10:
                            # Market hasn't moved enough -- let the GTC order sit
                            continue
                        else:
                            logger.info(
                                "GTC ORDER PRICE DRIFT: %s %dx %s | limit=$%.2f, "
                                "mid=$%.2f (%.1f%% drift) -- canceling to re-price",
                                mo.side_label, mo.contracts, mo.contract_symbol,
                                mo.current_price, mid, price_drift * 100,
                            )
                    else:
                        continue  # no market data yet, let GTC sit

                should_adjust = self._should_adjust(mo, now)
                if should_adjust:
                    if mo.attempts >= ORDER_MAX_ATTEMPTS:
                        logger.info(
                            "ORDER TIMEOUT: %s %dx %s -- %d attempts exhausted, "
                            "cancelling",
                            mo.side_label, mo.contracts, mo.contract_symbol,
                            mo.attempts,
                        )
                        self._cancel_and_abort(mo, newly_completed)
                        continue

                    # Initiate cancel -- we will repost on next check when
                    # Alpaca confirms the cancel
                    self._initiate_cancel(mo, newly_completed)

        return newly_completed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_order_status(self, mo: ManagedOrder) -> Optional[Dict[str, Any]]:
        """Poll Alpaca for the current status of an order.

        Returns the status dict, or None on error.
        """
        if self._sim is not None:
            return self._sim.check_status(mo.order_id)

        # Live mode
        self.rate_limiter.wait_if_needed(count=1)
        try:
            session = self.executor._get_session()
            resp = session.get(
                f"{ALPACA_BASE_URL}/v2/orders/{mo.order_id}",
                timeout=10,
            )
            self.rate_limiter.record(count=1)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(
                "ORDER STATUS CHECK FAILED: %s (order_id=%s): %s",
                mo.contract_symbol, mo.order_id, e,
            )
            return None

    def _should_adjust(self, mo: ManagedOrder, now: float) -> bool:
        """Decide whether to cancel and re-post at a better price.

        Rules:
        - Must wait at least ORDER_MAX_WAIT seconds before first adjustment.
        - Must wait at least ORDER_MIN_INTERVAL seconds between adjustments.
        - After those thresholds, always adjust (the price step logic decides
          how much to move).
        """
        # First adjustment: wait ORDER_MAX_WAIT from initial post
        if mo.attempts == 0:
            if mo.elapsed < ORDER_MAX_WAIT:
                return False
        else:
            # Subsequent: wait ORDER_MIN_INTERVAL from last adjustment
            if mo.last_adjust_time > 0 and (now - mo.last_adjust_time) < ORDER_MIN_INTERVAL:
                return False

        return True

    def _compute_new_price(self, mo: ManagedOrder) -> float:
        """Compute the adjusted limit price based on side and market conditions.

        Selling calls (we are the seller):
            - We want to sell high. Start above mid.
            - If market moved UP (good for us), hold or raise price.
            - If market moved DOWN (bad for us), step toward mid.
            - Floor = mid-price (never sell below mid).

        Buying back calls (we are the buyer):
            - We want to buy low. Start below mid.
            - If market moved DOWN (good for us), hold or lower price.
            - If market moved UP (bad for us), step toward mid.
            - Ceiling = mid-price (never buy above mid).
        """
        mid = mo.last_mid
        current = mo.current_price

        if mo.side == "sell":
            # Step down toward mid, but never below mid
            new_price = current - ORDER_PRICE_STEP
            new_price = max(new_price, mid)
            # Ensure we move at least $0.01 (avoid infinite loop)
            if abs(new_price - current) < 0.005:
                # Already at or very near mid -- hold price
                new_price = mid
        else:
            # side == "buy": step up toward mid, but never above mid
            new_price = current + ORDER_PRICE_STEP
            new_price = min(new_price, mid)
            if abs(new_price - current) < 0.005:
                new_price = mid

        return round(new_price, 2)

    def _initiate_cancel(self, mo: ManagedOrder, newly_completed: List[ManagedOrder]) -> None:
        """Cancel the current order so we can re-post at adjusted price."""
        if self._sim is not None:
            ok = self._sim.cancel_order(mo.order_id)
            if ok:
                mo.cancel_pending = False  # instant in simulation
                self._repost_order(mo, newly_completed)
            else:
                logger.error(
                    "ORDER CANCEL FAILED (dry-run): %s %dx %s",
                    mo.side_label, mo.contracts, mo.contract_symbol,
                )
            return

        # Live cancel
        self.rate_limiter.wait_if_needed(count=1)
        try:
            session = self.executor._get_session()
            resp = session.delete(
                f"{ALPACA_BASE_URL}/v2/orders/{mo.order_id}",
                timeout=10,
            )
            self.rate_limiter.record(count=1)

            if resp.status_code in (200, 204):
                mo.cancel_pending = True
                logger.info(
                    "ORDER CANCEL SENT: %s %dx %s (order_id=%s)",
                    mo.side_label, mo.contracts, mo.contract_symbol, mo.order_id,
                )
            elif resp.status_code == 422:
                # Order already filled or cancelled -- re-check on next cycle
                logger.warning(
                    "ORDER CANCEL REJECTED (422): %s %dx %s -- may already be "
                    "filled/cancelled, will re-check",
                    mo.side_label, mo.contracts, mo.contract_symbol,
                )
            else:
                resp.raise_for_status()

        except Exception as e:
            logger.error(
                "ORDER CANCEL FAILED: %s %dx %s (order_id=%s): %s -- "
                "will NOT re-post to avoid duplicate",
                mo.side_label, mo.contracts, mo.contract_symbol, mo.order_id, e,
            )
            # Do NOT repost -- if cancel fails, we might have the order
            # still working.  Let the next manage_orders() cycle re-check.

    def _repost_order(self, mo: ManagedOrder, newly_completed: List[ManagedOrder]) -> None:
        """Post a new order at an adjusted price after the old one was cancelled."""
        old_price = mo.current_price
        new_price = self._compute_new_price(mo)

        # If price has not changed (already at mid), try one more time at mid
        # and if still the same, abort
        if new_price == old_price and mo.attempts >= 2:
            logger.info(
                "ORDER STALLED: %s %dx %s -- price stuck at $%.2f (mid=$%.2f), "
                "cannot improve further, aborting",
                mo.side_label, mo.contracts, mo.contract_symbol,
                old_price, mo.last_mid,
            )
            mo.status = "failed"
            self._complete_order(mo, newly_completed)
            return

        mo.attempts += 1
        mo.last_adjust_time = time.time()

        logger.info(
            "ORDER ADJUST: %s %dx %s -- $%.2f -> $%.2f (mid=$%.2f, attempt %d/%d)",
            mo.side_label, mo.contracts, mo.contract_symbol,
            old_price, new_price, mo.last_mid,
            mo.attempts, ORDER_MAX_ATTEMPTS,
        )

        # Rate-limit guard
        self.rate_limiter.wait_if_needed(count=1)

        # Place new order
        remaining = mo.contracts - mo.filled_qty
        if remaining <= 0:
            # Everything was filled during partial fill before cancel
            mo.status = "filled"
            self._complete_order(mo, newly_completed)
            return

        if mo.side == "sell":
            new_order_id = self.executor.sell_call(
                mo.contract_symbol, remaining, new_price,
            )
        else:
            new_order_id = self.executor.buy_back_call(
                mo.contract_symbol, remaining, new_price,
            )

        if new_order_id is None:
            logger.error(
                "ORDER REPOST FAILED: %s %dx %s @ $%.2f -- aborting",
                mo.side_label, remaining, mo.contract_symbol, new_price,
            )
            mo.status = "failed"
            self._complete_order(mo, newly_completed)
            return

        self.rate_limiter.record(count=1)

        # Register with simulator
        if self._sim is not None:
            self._sim.create_adjusted_order(
                new_order_id, mo.side, new_price, remaining, mo.attempts,
            )

        # Update ManagedOrder in-place (swap order_id, keep history)
        old_oid = mo.order_id
        del self._orders[old_oid]

        mo.order_id = new_order_id
        mo.current_price = new_price
        mo.contracts = remaining
        mo.status = "pending"
        mo.cancel_pending = False
        mo.last_check = time.time()

        self._orders[new_order_id] = mo

        logger.info(
            "ORDER REPOSTED: %s %dx %s @ $%.2f (new order_id=%s, "
            "old order_id=%s)",
            mo.side_label, remaining, mo.contract_symbol, new_price,
            new_order_id, old_oid,
        )

    def _cancel_and_abort(self, mo: ManagedOrder, newly_completed: List[ManagedOrder]) -> None:
        """Cancel the order and give up (max attempts reached)."""
        if self._sim is not None:
            self._sim.cancel_order(mo.order_id)
        else:
            self.rate_limiter.wait_if_needed(count=1)
            try:
                session = self.executor._get_session()
                resp = session.delete(
                    f"{ALPACA_BASE_URL}/v2/orders/{mo.order_id}",
                    timeout=10,
                )
                self.rate_limiter.record(count=1)
                if resp.status_code not in (200, 204, 422):
                    resp.raise_for_status()
            except Exception as e:
                logger.error(
                    "ORDER ABORT CANCEL FAILED: %s (order_id=%s): %s",
                    mo.contract_symbol, mo.order_id, e,
                )

        mo.status = "failed"
        self._complete_order(mo, newly_completed)

        logger.info(
            "ORDER ABORTED: %s %dx %s -- %d attempts, %.0fs elapsed, "
            "last price=$%.2f",
            mo.side_label, mo.contracts, mo.contract_symbol,
            mo.attempts, mo.elapsed, mo.current_price,
        )

    def _complete_order(self, mo: ManagedOrder, newly_completed: List[ManagedOrder]) -> None:
        """Move an order from active tracking to the completed list."""
        self._orders.pop(mo.order_id, None)
        self._completed.append(mo)
        newly_completed.append(mo)

        # Invoke callbacks
        if mo.status == "filled" and mo._on_fill is not None:
            try:
                mo._on_fill(mo)
            except Exception as e:
                logger.error("on_fill callback failed for %s: %s", mo.order_id, e)
        elif mo.status in ("failed", "cancelled") and mo._on_abort is not None:
            try:
                mo._on_abort(mo)
            except Exception as e:
                logger.error("on_abort callback failed for %s: %s", mo.order_id, e)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_pending_orders(self) -> List[ManagedOrder]:
        """Return all orders still being managed (not yet terminal)."""
        return list(self._orders.values())

    def get_filled_orders(self) -> List[ManagedOrder]:
        """Return all completed orders that were filled."""
        return [o for o in self._completed if o.status == "filled"]

    def get_all_completed(self) -> List[ManagedOrder]:
        """Return all completed orders (filled, cancelled, failed)."""
        return list(self._completed)

    def has_pending_for(self, contract_symbol: str) -> bool:
        """Return True if there is already a pending order for this contract."""
        return any(
            mo.contract_symbol == contract_symbol
            for mo in self._orders.values()
        )

    def get_order_status(self, order_id: str):
        """Fetch order status from Alpaca. Returns order dict or None."""
        try:
            r = self.api.get(f"/v2/orders/{order_id}")
            if hasattr(r, 'status_code'):
                return r.json() if r.status_code == 200 else None
            # alpaca-py returns object directly
            return r if r else None
        except Exception as e:
            logger.warning("OrderManager: get_order_status(%s) failed: %s", order_id[:8], e)
            return None

    def has_pending_sell_for_ticker(self, ticker: str) -> bool:
        """Return True if there is a pending SELL order for any contract on this ticker.

        Unlike has_pending_for() which matches exact contract symbols, this
        checks the ticker prefix of all pending sell orders.  Used by the
        cross-engine dedup layer to prevent multiple engines selling calls
        on the same underlying.
        """
        ticker_upper = ticker.upper()
        for mo in self._orders.values():
            if mo.side != "sell":
                continue
            # Extract ticker prefix from OCC symbol (alphabetic chars before first digit)
            prefix = ""
            for ch in mo.contract_symbol:
                if ch.isalpha():
                    prefix += ch
                else:
                    break
            if prefix.upper() == ticker_upper:
                return True
        return False

    def pending_count(self) -> int:
        """Number of orders currently being managed."""
        return len(self._orders)

    def update_market_data(self, contract_symbol: str, bid: float, ask: float) -> None:
        """Update bid/ask/mid for all pending orders on a given contract.

        The scalper should call this when it refreshes option chain data so
        the price adjustment logic uses current market conditions.
        """
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        for mo in self._orders.values():
            if mo.contract_symbol == contract_symbol:
                mo.last_bid = bid
                mo.last_ask = ask
                if mid > 0:
                    mo.last_mid = mid

    def status_summary(self) -> str:
        """Return a human-readable summary of order manager state."""
        pending = self.get_pending_orders()
        lines = [
            f"OrderManager: {len(pending)} pending, "
            f"{len(self._completed)} completed, "
            f"rate={self.rate_limiter.requests_in_window}/{RATE_LIMIT_MAX_REQUESTS} "
            f"in last {RATE_LIMIT_WINDOW}s",
        ]
        for mo in pending:
            lines.append(
                f"  {mo.side_label} {mo.contracts}x {mo.contract_symbol} "
                f"@ ${mo.current_price:.2f} (orig=${mo.original_price:.2f}, "
                f"mid=${mo.last_mid:.2f}) -- {mo.elapsed:.0f}s elapsed, "
                f"attempt {mo.attempts}/{ORDER_MAX_ATTEMPTS}"
            )
        return "\n".join(lines)
