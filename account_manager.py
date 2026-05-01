#!/usr/bin/env python3
"""
account_manager.py - Multi-account abstraction layer for the combined trading system.

Manages multiple Alpaca trading accounts (master + sub-accounts) so the bot
can eventually manage outside capital alongside the owner's capital.

Currently uses a single Alpaca paper trading API key for all accounts.
When Alpaca Broker API integration is added later, each account will get
its own API connection.

Provides:
  - AccountConfig dataclass for per-account settings
  - AccountManager for loading configs, executing trades, scaling positions
  - CLI for listing accounts, adding new ones, generating reports

Data persists to ~/trading_bot/accounts.json and ~/trading_bot/multi_account_data/

Usage:
    python3 account_manager.py --list                # List all accounts
    python3 account_manager.py --add                 # Add a new account (interactive)
    python3 account_manager.py --report [account_id] # Generate performance report
    python3 account_manager.py --sync                # Sync account positions
    python3 account_manager.py --status              # Show account summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

TRADING_BOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = TRADING_BOT_DIR / "multi_account_data"
ACCOUNTS_FILE = TRADING_BOT_DIR / "accounts.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(TRADING_BOT_DIR))
sys.path.insert(0, "/opt/jarvis-utils/lib")

logger = logging.getLogger("account_manager")

# Alpaca paper trading endpoint
PAPER_BASE = "https://paper-api.alpaca.markets/v2"


# =============================================================================
# CREDENTIALS
# =============================================================================

def _load_alpaca_credentials() -> Tuple[str, str]:
    """
    Load Alpaca API credentials using the same chain as the rest of the system.

    Priority:
      1. Environment variables (APCA_API_KEY_ID, APCA_API_SECRET_KEY)
      2. ~/trading_bot/.env file
      3. jarvis-utils secrets (Settings Portal)
    """
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    api_secret = os.environ.get("APCA_API_SECRET_KEY", "")

    if api_key and api_secret:
        return api_key, api_secret

    # Try .env file
    env_file = Path.home() / "trading_bot" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key == "APCA_API_KEY_ID":
                api_key = val
            elif key == "APCA_API_SECRET_KEY":
                api_secret = val

    if api_key and api_secret:
        return api_key, api_secret

    # Try jarvis-utils secrets
    try:
        from jarvis_utils.secrets import get as get_secret
        api_key = get_secret(
            "Alpaca", "api_key_id",
            user="a4dc8459-608d-49f5-943e-e5e105ed5207",
        ) or ""
        api_secret = get_secret(
            "Alpaca", "secret_key",
            user="a4dc8459-608d-49f5-943e-e5e105ed5207",
        ) or ""
    except Exception as e:
        logger.debug("Could not load secrets from jarvis-utils: %s", e)

    return api_key, api_secret


# =============================================================================
# ACCOUNT CONFIG
# =============================================================================

@dataclass
class AccountConfig:
    """Configuration for a single managed account."""

    account_id: str               # Unique identifier (e.g., "master", "sub_001")
    owner_name: str               # Human-readable name
    capital_allocation: float     # Initial capital allocated ($)
    fee_rate: float               # Annual management fee (0.02 = 2%)
    is_master: bool               # True for the primary/owner account
    active: bool                  # Whether to trade this account
    api_key_group: str = "Alpaca" # Credential group in secrets store
    created_at: str = ""          # ISO date when account was created
    notes: str = ""               # Free-form notes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AccountConfig":
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @property
    def daily_fee_rate(self) -> float:
        """Daily prorated fee rate: annual_rate / 365."""
        return self.fee_rate / 365.0


# =============================================================================
# FEE SETTINGS
# =============================================================================

@dataclass
class FeeSettings:
    """Global fee calculation settings."""
    billing_cycle: str = "quarterly"     # "monthly" or "quarterly"
    fee_calculation: str = "daily_accrual"
    high_water_mark: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeeSettings":
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# =============================================================================
# ACCOUNT MANAGER
# =============================================================================

class AccountManager:
    """
    Manages multiple Alpaca trading accounts.

    Currently all accounts share a single paper trading API key.
    When Alpaca Broker API is integrated, each sub-account will have
    independent API connections.
    """

    def __init__(self, config_path: str = str(ACCOUNTS_FILE)):
        self.config_path = Path(config_path)
        self.accounts: List[AccountConfig] = []
        self.fee_settings: FeeSettings = FeeSettings()

        # Alpaca API credentials (single key for now)
        self._api_key: str = ""
        self._api_secret: str = ""
        self._session: Optional[requests.Session] = None

        # Per-account portfolio values (cached after sync)
        self._portfolio_values: Dict[str, float] = {}

        # Per-account positions (cached after sync)
        self._positions: Dict[str, List[Dict[str, Any]]] = {}

        self._load_config()
        self._init_api()

    # ----------------------------------------------------------------
    # Configuration Loading
    # ----------------------------------------------------------------

    def _load_config(self) -> None:
        """Load account configurations from JSON file."""
        if not self.config_path.exists():
            logger.warning(
                "No accounts config found at %s, creating default", self.config_path
            )
            self._create_default_config()
            return

        try:
            with open(self.config_path) as f:
                data = json.load(f)

            self.accounts = [
                AccountConfig.from_dict(a) for a in data.get("accounts", [])
            ]
            self.fee_settings = FeeSettings.from_dict(
                data.get("fee_settings", {})
            )

            logger.info(
                "Loaded %d account(s) from %s", len(self.accounts), self.config_path
            )
        except Exception as e:
            logger.error("Failed to load accounts config: %s", e)
            self._create_default_config()

    def _create_default_config(self) -> None:
        """Create a default accounts.json with just the master account."""
        master = AccountConfig(
            account_id="master",
            owner_name="Kedalion Capital",
            capital_allocation=100_000.0,
            fee_rate=0.0,
            is_master=True,
            active=True,
            api_key_group="Alpaca",
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        self.accounts = [master]
        self.fee_settings = FeeSettings()
        self._save_config()

    def _save_config(self) -> None:
        """Save current account configurations to JSON file."""
        data = {
            "accounts": [a.to_dict() for a in self.accounts],
            "fee_settings": self.fee_settings.to_dict(),
        }
        tmp = self.config_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(self.config_path)
        logger.info("Account config saved to %s", self.config_path)

    # ----------------------------------------------------------------
    # API Initialization
    # ----------------------------------------------------------------

    def _init_api(self) -> None:
        """Initialize the Alpaca API session."""
        self._api_key, self._api_secret = _load_alpaca_credentials()

        if not self._api_key or not self._api_secret:
            logger.warning(
                "Alpaca credentials not found. API operations will fail."
            )
            return

        # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
        from alpaca_client import _AutoRefreshSession
        self._session = _AutoRefreshSession(self._refresh_session_credentials)
        self._session.headers.update({
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
        })

    def _refresh_session_credentials(self, session) -> None:
        """Edge 123 port (Apr 22 2026): re-pull creds on 401."""
        from jarvis_utils.secrets import get
        new_key = get("Alpaca", "api_key_id",
                      user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        new_secret = get("Alpaca", "secret_key",
                         user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        if not new_key or not new_secret:
            raise EnvironmentError("account_manager cred refresh: empty creds")
        self._api_key = new_key
        self._api_secret = new_secret
        session.headers["APCA-API-KEY-ID"] = new_key
        session.headers["APCA-API-SECRET-KEY"] = new_secret

    def _ensure_session(self) -> requests.Session:
        """Return the API session, raising if not initialized."""
        if self._session is None:
            raise RuntimeError(
                "Alpaca API session not initialized. Check credentials."
            )
        return self._session

    # ----------------------------------------------------------------
    # Account Queries
    # ----------------------------------------------------------------

    def get_active_accounts(self) -> List[AccountConfig]:
        """Return all active accounts."""
        return [a for a in self.accounts if a.active]

    def get_account(self, account_id: str) -> AccountConfig:
        """Get a specific account config by ID."""
        for a in self.accounts:
            if a.account_id == account_id:
                return a
        raise KeyError(f"Account not found: {account_id}")

    def get_master_account(self) -> AccountConfig:
        """Return the master (owner) account."""
        for a in self.accounts:
            if a.is_master:
                return a
        raise KeyError("No master account configured")

    def get_sub_accounts(self) -> List[AccountConfig]:
        """Return all non-master accounts."""
        return [a for a in self.accounts if not a.is_master]

    def add_account(self, config: AccountConfig) -> None:
        """Add a new account to the configuration."""
        # Validate no duplicate ID
        existing_ids = {a.account_id for a in self.accounts}
        if config.account_id in existing_ids:
            raise ValueError(f"Account ID already exists: {config.account_id}")

        # Ensure only one master
        if config.is_master:
            for a in self.accounts:
                if a.is_master:
                    raise ValueError(
                        f"Master account already exists: {a.account_id}. "
                        "Cannot have multiple master accounts."
                    )

        if not config.created_at:
            config.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self.accounts.append(config)
        self._save_config()
        logger.info("Added account: %s (%s)", config.account_id, config.owner_name)

    def deactivate_account(self, account_id: str) -> None:
        """Deactivate an account (stop trading but keep config)."""
        acct = self.get_account(account_id)
        if acct.is_master:
            raise ValueError("Cannot deactivate the master account")
        acct.active = False
        self._save_config()
        logger.info("Deactivated account: %s", account_id)

    # ----------------------------------------------------------------
    # Portfolio & Position Queries
    # ----------------------------------------------------------------

    def get_account_value(self, account_id: str) -> float:
        """
        Get current portfolio value for an account.

        For the master account, queries Alpaca directly.
        For sub-accounts, uses capital_allocation as the base and adjusts
        by the proportional P&L from the master account.
        (Full per-account tracking requires Alpaca Broker API.)
        """
        if account_id in self._portfolio_values:
            return self._portfolio_values[account_id]

        acct = self.get_account(account_id)

        if acct.is_master:
            try:
                session = self._ensure_session()
                resp = session.get(
                    f"{PAPER_BASE}/account", timeout=10
                )
                resp.raise_for_status()
                value = float(resp.json().get("equity", acct.capital_allocation))
                self._portfolio_values[account_id] = value
                return value
            except Exception as e:
                logger.error("Failed to get master account value: %s", e)
                return acct.capital_allocation
        else:
            # Sub-account value = capital_allocation * (master_value / master_allocation)
            # This proportional model works until Broker API is integrated
            master = self.get_master_account()
            master_value = self.get_account_value(master.account_id)
            if master.capital_allocation > 0:
                ratio = master_value / master.capital_allocation
                value = acct.capital_allocation * ratio
            else:
                value = acct.capital_allocation
            self._portfolio_values[account_id] = value
            return value

    def get_positions(self, account_id: str) -> List[Dict[str, Any]]:
        """
        Get positions for a specific account.

        For the master account, queries Alpaca directly.
        For sub-accounts, scales the master's positions proportionally.
        """
        if account_id in self._positions:
            return self._positions[account_id]

        acct = self.get_account(account_id)

        if acct.is_master:
            try:
                session = self._ensure_session()
                resp = session.get(f"{PAPER_BASE}/positions", timeout=10)
                resp.raise_for_status()
                positions = resp.json()
                self._positions[account_id] = positions
                return positions
            except Exception as e:
                logger.error("Failed to get master positions: %s", e)
                return []
        else:
            # Scale master positions proportionally
            master = self.get_master_account()
            master_positions = self.get_positions(master.account_id)
            scale = self._get_scale_factor(account_id)

            scaled = []
            for pos in master_positions:
                sp = dict(pos)
                orig_qty = float(sp.get("qty", 0))
                sp["qty"] = str(max(1, int(orig_qty * scale)))
                sp["qty_available"] = sp["qty"]
                sp["_account_id"] = account_id
                sp["_scale_factor"] = scale
                scaled.append(sp)

            self._positions[account_id] = scaled
            return scaled

    # ----------------------------------------------------------------
    # Trade Execution
    # ----------------------------------------------------------------

    def scale_quantity(self, base_qty: int, account_id: str) -> int:
        """
        Scale a trade quantity based on account's capital relative to master.

        If master has $100k and sub-account has $50k, scale = 0.5,
        so a 100-share master trade becomes 50 shares.

        Returns at least 1 share if the account is active.
        """
        acct = self.get_account(account_id)

        if acct.is_master:
            return base_qty

        scale = self._get_scale_factor(account_id)
        scaled = int(base_qty * scale)
        return max(1, scaled) if scaled > 0 or base_qty > 0 else 0

    def _get_scale_factor(self, account_id: str) -> float:
        """Get the capital ratio between an account and the master."""
        acct = self.get_account(account_id)
        master = self.get_master_account()

        if master.capital_allocation <= 0:
            return 1.0

        return acct.capital_allocation / master.capital_allocation

    def execute_trade(
        self,
        account_id: str,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        dry_run: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a trade on a specific account.

        For the master account, places orders directly via Alpaca.
        For sub-accounts, scales quantity and places the order.

        NOTE: Currently all accounts share one API key, so orders go to the
        same Alpaca paper account. When Broker API is integrated, each
        sub-account will have its own connection.

        Args:
            account_id: Target account
            symbol: Ticker symbol
            qty: Share quantity (will be scaled for sub-accounts)
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit"
            time_in_force: "day", "gtc", "ioc"
            limit_price: For limit/stop_limit orders
            stop_price: For stop/stop_limit orders
            dry_run: If True, log but do not place order

        Returns:
            Order dict from Alpaca, or None on failure/dry_run
        """
        acct = self.get_account(account_id)

        if not acct.active:
            logger.warning("Account %s is not active, skipping trade", account_id)
            return None

        # Scale quantity for sub-accounts
        scaled_qty = self.scale_quantity(qty, account_id)

        logger.info(
            "[%s] %s TRADE: %s %s %d shares (base=%d, scale=%.3f) | type=%s",
            "DRY RUN" if dry_run else "LIVE",
            account_id, side.upper(), symbol, scaled_qty, qty,
            self._get_scale_factor(account_id),
            order_type,
        )

        if dry_run:
            return {
                "id": f"DRY_{account_id}_{symbol}_{int(time.time())}",
                "symbol": symbol,
                "qty": str(scaled_qty),
                "side": side,
                "type": order_type,
                "status": "dry_run",
                "_account_id": account_id,
            }

        try:
            session = self._ensure_session()
            order_data: Dict[str, Any] = {
                "symbol": symbol,
                "qty": str(scaled_qty),
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
            }
            if limit_price is not None:
                order_data["limit_price"] = str(limit_price)
            if stop_price is not None:
                order_data["stop_price"] = str(stop_price)

            resp = session.post(
                f"{PAPER_BASE}/orders",
                json=order_data,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            result["_account_id"] = account_id
            result["_scaled_qty"] = scaled_qty
            result["_base_qty"] = qty

            logger.info(
                "[%s] Order placed: %s %s %d shares -> order_id=%s",
                account_id, side.upper(), symbol, scaled_qty,
                result.get("id", "unknown"),
            )
            return result

        except Exception as e:
            logger.error(
                "[%s] Trade execution failed: %s %s %d shares: %s",
                account_id, side, symbol, scaled_qty, e,
            )
            return None

    def execute_trade_all_accounts(
        self,
        symbol: str,
        qty: int,
        side: str,
        dry_run: bool = True,
        **kwargs,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Execute the same trade across all active accounts (scaled per account).

        Args:
            symbol: Ticker symbol
            qty: Base quantity (master account size)
            side: "buy" or "sell"
            dry_run: If True, log but do not place orders
            **kwargs: Additional order params (order_type, limit_price, etc.)

        Returns:
            Dict mapping account_id -> order result
        """
        results = {}
        for acct in self.get_active_accounts():
            results[acct.account_id] = self.execute_trade(
                account_id=acct.account_id,
                symbol=symbol,
                qty=qty,
                side=side,
                dry_run=dry_run,
                **kwargs,
            )
        return results

    # ----------------------------------------------------------------
    # Account Sync
    # ----------------------------------------------------------------

    def sync_accounts(self) -> Dict[str, Any]:
        """
        Sync all accounts: refresh portfolio values and positions,
        verify sub-accounts are proportionally aligned with master.

        Returns a sync report dict.
        """
        # Clear caches
        self._portfolio_values.clear()
        self._positions.clear()

        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accounts": {},
        }

        master = self.get_master_account()
        master_value = self.get_account_value(master.account_id)
        master_positions = self.get_positions(master.account_id)

        report["accounts"][master.account_id] = {
            "owner": master.owner_name,
            "portfolio_value": master_value,
            "capital_allocation": master.capital_allocation,
            "return_pct": (
                (master_value - master.capital_allocation) / master.capital_allocation * 100
                if master.capital_allocation > 0 else 0.0
            ),
            "positions_count": len(master_positions),
            "is_master": True,
        }

        for acct in self.get_sub_accounts():
            if not acct.active:
                continue

            value = self.get_account_value(acct.account_id)
            positions = self.get_positions(acct.account_id)
            scale = self._get_scale_factor(acct.account_id)

            report["accounts"][acct.account_id] = {
                "owner": acct.owner_name,
                "portfolio_value": value,
                "capital_allocation": acct.capital_allocation,
                "scale_factor": scale,
                "return_pct": (
                    (value - acct.capital_allocation) / acct.capital_allocation * 100
                    if acct.capital_allocation > 0 else 0.0
                ),
                "positions_count": len(positions),
                "is_master": False,
            }

        # Save sync report
        sync_path = DATA_DIR / "last_sync.json"
        with open(sync_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            "Account sync complete: %d accounts, master=$%.2f",
            len(report["accounts"]), master_value,
        )
        return report

    # ----------------------------------------------------------------
    # Display
    # ----------------------------------------------------------------

    def print_accounts_table(self) -> None:
        """Print a formatted table of all accounts."""
        print("\n" + "=" * 90)
        print("MANAGED ACCOUNTS")
        print("=" * 90)
        print(
            f"{'ID':<15} {'Owner':<25} {'Capital':>12} {'Fee Rate':>10} "
            f"{'Active':>8} {'Master':>8}"
        )
        print("-" * 90)

        for acct in self.accounts:
            print(
                f"{acct.account_id:<15} {acct.owner_name:<25} "
                f"${acct.capital_allocation:>10,.2f} "
                f"{acct.fee_rate*100:>8.1f}% "
                f"{'Yes' if acct.active else 'No':>8} "
                f"{'Yes' if acct.is_master else 'No':>8}"
            )

        print("-" * 90)
        total_capital = sum(a.capital_allocation for a in self.accounts)
        active_count = len(self.get_active_accounts())
        print(f"Total: {len(self.accounts)} accounts ({active_count} active), "
              f"${total_capital:,.2f} allocated")
        print("=" * 90)

    def print_status(self) -> None:
        """Print detailed status including portfolio values."""
        report = self.sync_accounts()

        print("\n" + "=" * 90)
        print("MULTI-ACCOUNT STATUS")
        print(f"Synced at: {report['timestamp']}")
        print("=" * 90)

        for acct_id, info in report["accounts"].items():
            master_flag = " [MASTER]" if info.get("is_master") else ""
            print(
                f"\n  {acct_id}{master_flag}"
                f"\n    Owner:          {info['owner']}"
                f"\n    Allocation:     ${info['capital_allocation']:,.2f}"
                f"\n    Current Value:  ${info['portfolio_value']:,.2f}"
                f"\n    Return:         {info['return_pct']:+.2f}%"
                f"\n    Positions:      {info['positions_count']}"
            )
            if "scale_factor" in info:
                print(f"    Scale Factor:   {info['scale_factor']:.3f}")

        print("\n" + "=" * 90)


# =============================================================================
# CLI
# =============================================================================

def _interactive_add_account(manager: AccountManager) -> None:
    """Interactive prompt to add a new account."""
    print("\n--- Add New Managed Account ---\n")

    account_id = input("Account ID (e.g., sub_001): ").strip()
    if not account_id:
        print("Error: Account ID is required.")
        return

    owner_name = input("Owner name: ").strip()
    if not owner_name:
        print("Error: Owner name is required.")
        return

    try:
        capital = float(input("Capital allocation ($): ").strip())
    except ValueError:
        print("Error: Invalid capital amount.")
        return

    try:
        fee_pct = float(input("Annual fee rate (%, e.g., 2.0 for 2%): ").strip())
        fee_rate = fee_pct / 100.0
    except ValueError:
        print("Error: Invalid fee rate.")
        return

    notes = input("Notes (optional): ").strip()

    config = AccountConfig(
        account_id=account_id,
        owner_name=owner_name,
        capital_allocation=capital,
        fee_rate=fee_rate,
        is_master=False,
        active=True,
        api_key_group="Alpaca",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        notes=notes,
    )

    print(f"\nNew account to add:")
    print(f"  ID:         {config.account_id}")
    print(f"  Owner:      {config.owner_name}")
    print(f"  Capital:    ${config.capital_allocation:,.2f}")
    print(f"  Fee Rate:   {config.fee_rate*100:.1f}%")
    print(f"  Daily Fee:  ${config.capital_allocation * config.daily_fee_rate:.2f}/day")

    confirm = input("\nConfirm? (y/n): ").strip().lower()
    if confirm == "y":
        try:
            manager.add_account(config)
            print(f"Account '{account_id}' added successfully.")
        except ValueError as e:
            print(f"Error: {e}")
    else:
        print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Account Manager for the Combined Trading System"
    )
    parser.add_argument("--list", action="store_true", help="List all accounts")
    parser.add_argument("--add", action="store_true", help="Add a new account (interactive)")
    parser.add_argument(
        "--report", nargs="?", const="all", default=None,
        help="Generate performance report (optionally for a specific account_id)"
    )
    parser.add_argument("--sync", action="store_true", help="Sync account positions")
    parser.add_argument("--status", action="store_true", help="Show account status with values")
    parser.add_argument(
        "--config", default=str(ACCOUNTS_FILE),
        help="Path to accounts.json config file"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    manager = AccountManager(config_path=args.config)

    if args.list:
        manager.print_accounts_table()
    elif args.add:
        _interactive_add_account(manager)
    elif args.report is not None:
        # Delegate to performance_reporter
        try:
            from performance_reporter import PerformanceReporter
            reporter = PerformanceReporter()
            if args.report == "all":
                print(reporter.generate_all_reports(manager))
            else:
                print(reporter.generate_account_report(args.report, manager))
        except ImportError:
            print("Performance reporter not available. Run: python3 performance_reporter.py")
    elif args.sync:
        report = manager.sync_accounts()
        print(json.dumps(report, indent=2))
    elif args.status:
        manager.print_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
