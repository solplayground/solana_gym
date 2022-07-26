from enum import IntEnum
from typing import NamedTuple, List, Any, NewType

from jupiter_gym.envs.constants import USDC_MINT, SOL_MINT


class ActionType(IntEnum):
    no_action = 1  # do nothing
    buy_sol = 2  # buy SOL with USDC
    buy_base = 3  # sell SOL for USDC
    convert_base_sol_base = 4  # Arbitrage from USDC->SOL  ,SOL->USDC ,in parallel
    convert_sol_base_sol = 5  # Arbitrage from SOL->USDC  , USDC->SOL  ,in parallel


class DoneConditionType(IntEnum):
    change_rate_limit = 1  # total amount changes +-20%
    max_step = 2  # max steps reached for one episode
    insufficient_fund = 3  # no enough fund left
    too_many_exceptions = 4  # too many errors
    others = 5


TransactionId = NewType('TransactionId', str)

MIN_INIT_USDC = 1  # USDC 1,
MIN_WAIT_PERIOD = 5  # seconds
MIN_INIT_SOL = 0.1  # 0.1 SOL
MIN_PROFIT_MI_CENTS = 5  # 10^6 = 1 USDC
MIN_PROFIT_LAMPORTS = 5000  # 0.00005 SOLs  10^9 = 1SOL
SWAP_RATIO = 0.10  # 10% of total assets
SLIPPAGE = 0.2


class JupiterConfiguration(NamedTuple):
    base_mint: str = USDC_MINT
    base_mint_name: str = 'USDC'
    sol_mint: str = SOL_MINT
    sol_mint_name: str = "SOL"
    min_init_base_mint: int = MIN_INIT_USDC
    min_init_sol_mint: int = MIN_INIT_SOL
    min_wait_period: int = MIN_WAIT_PERIOD
    min_profit_base_mint: int = MIN_PROFIT_MI_CENTS
    min_profit_sol_mint: int = MIN_PROFIT_LAMPORTS
    swap_ratio: float = SWAP_RATIO
    slippage: float = SLIPPAGE
    use_direct_route: bool = False


class TokenAccountInfo(NamedTuple):
    pubkey: str
    mint: str
    owner: str
    amount: int
    lamports: int
    is_native: bool
    decimals: int


class MarketInfo(NamedTuple):
    label: str


class JupiterRouteInfo(NamedTuple):
    in_amount: int
    out_amount: int
    out_amount_with_slippage: int
    amount: int
    market_infos: List[MarketInfo]
    route: Any


class JupiterQuoteInfo(NamedTuple):
    time_taken: float
    routes: List[JupiterRouteInfo]


class AccountBalanceInfo(NamedTuple):
    sol_amount: int
    base_amount: int
    wrapped_sol_amount: int
    sol_unit_price: float
    adjusted_sol_amount: float
    adjusted_base_amount: float
    timestamp: int


class StepInfo(NamedTuple):
    balances: AccountBalanceInfo
    total_trades: int
    success_trades: int
    failed_trades: int
