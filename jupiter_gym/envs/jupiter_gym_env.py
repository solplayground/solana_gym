import asyncio
import io
import json
import statistics
from datetime import datetime
from typing import List
from typing import Tuple, Optional, Union

import cv2
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
# Data manipulation packages
import pandas as pd
# OpenAI packages
from gym import Env
from gym import spaces
from gym.core import ActType, ObsType
from gym.utils.colorize import *
from solana.rpc.async_api import AsyncClient

from jupiter_gym.envs.constants import MAX_AMOUNT_VALUE, SOLANA_RPC_URL, LAMPORTS_PER_SOL, BASE_PER_USDC
from jupiter_gym.envs.image_viewer import ImageViewer
from jupiter_gym.envs.jupiter_aggregator import JupiterAggregator
from jupiter_gym.envs.types import ActionType, JupiterConfiguration, MIN_INIT_USDC, MIN_INIT_SOL, MIN_WAIT_PERIOD, \
    MIN_PROFIT_MI_CENTS, MIN_PROFIT_LAMPORTS, SWAP_RATIO, SLIPPAGE, AccountBalanceInfo, DoneConditionType
# noinspection PyBroadException
from jupiter_gym.envs.utils import create_directory_if_not_exist


class JupiterGymEnv(Env):
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}

    def __init__(self, **kwargs):
        self._loop = asyncio.new_event_loop()
        create_directory_if_not_exist("images")
        create_directory_if_not_exist("data")

        self.observation_space = spaces.Dict(
            {
                "sol_amount": spaces.Box(low=0,
                                         high=LAMPORTS_PER_SOL * 10000,
                                         shape=(1,),
                                         dtype=np.int64),
                "base_amount": spaces.Box(low=0,
                                          high=BASE_PER_USDC * 100000,
                                          shape=(1,),
                                          dtype=np.int64),
                "wrapped_sol_amount": spaces.Box(low=0,
                                                 high=LAMPORTS_PER_SOL * 10000,
                                                 shape=(1,),
                                                 dtype=np.int64),
                "sol_unit_price": spaces.Box(low=0,
                                             high=BASE_PER_USDC * 100000,
                                             shape=(1,),
                                             dtype=np.int64),
                "adjusted_sol_amount": spaces.Box(low=0,
                                                  high=10000.0,
                                                  shape=(1,),
                                                  dtype=np.float32),
                "timestamp": spaces.Box(low=10000000000,
                                        high=30000000000,
                                        shape=(1,),
                                        dtype=np.int64),
                "adjusted_base_amount": spaces.Box(low=0,
                                                   high=100000.0,
                                                   shape=(1,),
                                                   dtype=np.float32),
                "change_rate_base_amount": spaces.Box(low=0,
                                                      high=100000.0,
                                                      shape=(1,),
                                                      dtype=np.float32),
                "change_rate_sol_price": spaces.Box(low=0,
                                                    high=100000.0,
                                                    shape=(1,),
                                                    dtype=np.float32),

                "total_trades": spaces.Box(low=0,
                                           high=1000000,
                                           shape=(1,),
                                           dtype=int),
                "success_trades": spaces.Box(low=0,
                                             high=1000000,
                                             shape=(1,),
                                             dtype=int),
                "failed_trades": spaces.Box(low=0,
                                            high=1000000,
                                            shape=(1,),
                                            dtype=int),
                "exceptions": spaces.Box(low=0,
                                         high=1000000,
                                         shape=(1,),
                                         dtype=int),

            }
        )

        self.action_space = spaces.Dict(
            {
                "operation": spaces.Discrete(5, start=1),
                "amount": spaces.Box(low=np.float32(0),
                                     high=np.float32(MAX_AMOUNT_VALUE),
                                     shape=(1,),
                                     dtype=np.float32),
            }
        )

        self._keypair = kwargs.get('keypair', None)
        if self._keypair is None:
            raise ValueError('Invalid keypair')
        solana_rpc_url = kwargs.get('solana_rpc_url', SOLANA_RPC_URL)
        self._client = AsyncClient(solana_rpc_url)

        min_init_base_mint = kwargs.get('min_init_base_mint', MIN_INIT_USDC) * BASE_PER_USDC
        min_init_sol_mint = kwargs.get('min_init_sol_mint', MIN_INIT_SOL) * LAMPORTS_PER_SOL
        min_wait_period = kwargs.get('min_wait_period', MIN_WAIT_PERIOD)
        min_profit_base_mint = kwargs.get('min_profit_base_mint', MIN_PROFIT_MI_CENTS)
        min_profit_sol_mint = kwargs.get('min_profit_sol_mint', MIN_PROFIT_LAMPORTS)
        swap_ratio = kwargs.get('swap_ratio', SWAP_RATIO)
        slippage = kwargs.get('slippage', SLIPPAGE)
        resolution = kwargs.get('resolution', 5)
        use_direct_route = kwargs.get('use_direct_route', False)

        self._option = JupiterConfiguration(min_init_base_mint=min_init_base_mint,
                                            min_init_sol_mint=min_init_sol_mint,
                                            min_wait_period=min_wait_period,
                                            min_profit_base_mint=min_profit_base_mint,
                                            min_profit_sol_mint=min_profit_sol_mint,
                                            swap_ratio=swap_ratio,
                                            slippage=slippage,
                                            use_direct_route=use_direct_route
                                            )

        self._jupiter = JupiterAggregator(keypair=self._keypair, client=self._client, options=self._option)
        self._resolution = resolution  # 5 minutes
        self._historical_data: List[AccountBalanceInfo] = []

        self._save_figure = True
        self._periods = 600
        self._leading_periods = 100
        self._async_wrapper(self._set_init_balance())
        self._start_balances: AccountBalanceInfo = self._historical_data[0]

        # figures
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(18.5, 10.5)
        self.axis = None
        self.viewer = ImageViewer()
        # plot styles
        mc = mpf.make_marketcolors(up='limegreen', down='orangered',
                                   edge='inherit',
                                   wick={'up': 'limegreen', 'down': 'orangered'},
                                   volume='deepskyblue',
                                   ohlc='i')
        self.style = mpf.make_mpf_style(base_mpl_style='seaborn-whitegrid',
                                        marketcolors=mc)

        self.max_step_for_one_episode = kwargs.get('max_step_for_one_episode', 2400)
        self.change_rate_limit = kwargs.get('change_rate_limit', 20)  # -20%,20%
        self.disable_low_balance_checking = kwargs.get('disable_low_balance_checking', False)  # -20%,20%
        self.global_step_count = 0
        self.step_count = 0
        self.exception_count = 0
        self.episode = 0
        self.observation = None
        self.action = None
        self.info = {}
        self.reward = 0
        self.start_obs = None
        self.is_start = False
        self.current_trade = None
        self.change_rate_base_amount = 0
        self.change_rate_sol_price = 0
        self.recent_performance = -1
        self.done_condition_type = DoneConditionType.others
        day = datetime.now()
        self.date_prefix = f"data/{day.strftime('%Y-%m-%d_%H-%M-%S')}"

    async def _set_init_balance(self):
        balances = await self._jupiter.get_crypto_amounts()
        while balances.sol_unit_price <= 0:
            print(colorize(f'try to get current account balance, but getting {balances.sol_unit_price}, '
                           f'retry in 2 seconds',
                           color='yellow'))
            await asyncio.sleep(2)
            balances = await self._jupiter.get_crypto_amounts()

        total_periods = self._periods + self._leading_periods
        for i in range(total_periods):
            dummy_balance = AccountBalanceInfo(sol_amount=int(balances.sol_amount),
                                               base_amount=int(balances.base_amount),
                                               wrapped_sol_amount=int(balances.wrapped_sol_amount),
                                               timestamp=int(balances.timestamp) - 10 * (total_periods - i),
                                               sol_unit_price=float(balances.sol_unit_price),
                                               adjusted_sol_amount=float(balances.adjusted_sol_amount),
                                               adjusted_base_amount=float(balances.adjusted_base_amount)
                                               )
            self._historical_data.append(dummy_balance)

    def _get_current_obs(self):
        last2_trades: List[AccountBalanceInfo] = self._historical_data[-2:]
        current_trade: AccountBalanceInfo = last2_trades[1]
        last_trade: AccountBalanceInfo = last2_trades[0]
        self.current_trade = current_trade
        obs = {
            "sol_amount": round(float(current_trade.sol_amount) / float(LAMPORTS_PER_SOL), 3),
            "base_amount": round(float(current_trade.base_amount) / float(BASE_PER_USDC), 3),
            "wrapped_sol_amount": round(float(current_trade.wrapped_sol_amount) / float(LAMPORTS_PER_SOL), 3),
            "sol_unit_price": round(float(current_trade.sol_unit_price) / float(BASE_PER_USDC), 3),
            "adjusted_sol_amount": round(float(current_trade.adjusted_sol_amount), 3),
            "adjusted_base_amount": round(float(current_trade.adjusted_base_amount), 3),
            "timestamp": current_trade.timestamp,
            "total_trades": self._jupiter.total_trade,
            "success_trades": self._jupiter.success_trade,
            "failed_trades": self._jupiter.failed_trade,
            "change_rate_base_amount": self.change_rate_base_amount,
            "change_rate_sol_price": self.change_rate_sol_price,
            "exceptions": self.exception_count

        }
        self.observation = obs
        self.reward = current_trade.adjusted_base_amount - last_trade.adjusted_base_amount
        if self.is_start:
            self.start_obs = obs
            self.is_start = False
        return obs

    def _save_summary(self):
        directory_name = f'{self.date_prefix}/episode_{str(self.episode).zfill(4)}'
        create_directory_if_not_exist(directory_name)
        summary_file = open(f'{directory_name}/summary.json', 'w')
        summaries = {"last_obs": self._get_current_obs(),
                     "first_obs": self.start_obs,
                     "info": self.info}
        json.dump(summaries, summary_file, indent=2)
        summary_file.close()

    def _is_done(self) -> bool:
        done = False
        self.done_condition_type = DoneConditionType.others
        if self.exception_count > 100:
            done = True
            self.done_condition_type = DoneConditionType.too_many_exceptions
        elif abs(self.change_rate_base_amount) > self.change_rate_limit:
            done = True
            self.done_condition_type = DoneConditionType.change_rate_limit
        elif self.step_count > self.max_step_for_one_episode:
            done = True
            self.done_condition_type = DoneConditionType.max_step
        elif (not self.disable_low_balance_checking) and self.current_trade \
                and self.current_trade.adjusted_base_amount < 10:
            done = True
            self.done_condition_type = DoneConditionType.insufficient_fund

        return done

    def _get_info(self):
        history = self._historical_data[self._periods + self._leading_periods - 1:]

        self.info = {"history": history,
                     "start_balances": self._start_balances,
                     "tps": self.recent_performance}

    async def _get_balance(self):
        balances = await self._jupiter.get_crypto_amounts()
        tps = await self._jupiter.get_recent_performance()
        if tps > 0:
            self.recent_performance = tps
        if balances.sol_unit_price > 0:
            print(colorize(str(balances), color='blue'))
            if len(self._historical_data) > 10:
                last_10_trades: List[AccountBalanceInfo] = self._historical_data[-10:]
                last_10_base_amount = [x.adjusted_base_amount for x in last_10_trades]
                # try to remove outlier
                std = max(statistics.stdev(last_10_base_amount), 0.1)
                avg: float = sum(last_10_base_amount) / len(last_10_trades)

                left = avg - 5 * std
                right = avg + 5 * std

                if left < balances.adjusted_base_amount < right:
                    self._historical_data.append(balances)
                else:
                    print(colorize(f'Outlier:{balances.adjusted_base_amount} ,last 10 average :{avg}', color='yellow'))

    def _render_ansi(self):
        try:
            resampled, unit_resampled, display_title = self._get_display_title()
            print(colorize(display_title, color='blue'))
            print(colorize('*' * 80, color='blue'))
            print('Sol price')
            print(unit_resampled[-10:])
            print(colorize('*' * 80, color='blue'))
            print('Total assets')
            print(resampled[-10:])
            print(colorize('*' * 80, color='blue'))
        except:
            print(
                colorize(
                    f'Total trade={self._jupiter.total_trade},success trade={self._jupiter.success_trade},'
                    f'failed trade={self._jupiter.failed_trade}',
                    color='yellow'))

    def _close_fig(self):
        # try to close exist fig if possible
        # noinspection PyBroadException
        try:
            plt.close(self.fig)
        except Exception as e:
            print(colorize(str(e), color='yellow'))

    def _get_display_title(self):
        decimals = pd.Series([2, 2, 2, 2, 1], index=['Open', 'High', 'Low', 'Close', 'Volume', ])
        trades = self._historical_data[-self._periods - self._leading_periods:]

        if len(trades) > 0:
            close_price = [round(float(trade.adjusted_base_amount), 3) for trade in trades]
            high_price = close_price.copy()
            low_price = close_price.copy()
            open_price = close_price.copy()
            volume = [1 for _ in trades]
            bar_time_date = [datetime.fromtimestamp(trade.timestamp) for trade in trades]
            history_data = {
                "Date": bar_time_date,
                "Close": close_price,
                "High": high_price,
                "Low": low_price,
                "Open": open_price,
                "Volume": volume,
            }
            logic = {'Open': 'first',
                     'High': 'max',
                     'Low': 'min',
                     'Close': 'last',
                     'Volume': 'sum'}
            history_data_df = pd.DataFrame.from_dict(history_data)
            history_data_df.set_index('Date', inplace=True)
            history_data_df.sort_index(inplace=True)
            resampled = history_data_df.resample(f'{self._resolution}Min').apply(logic)
            resampled = resampled.dropna()
            resampled = resampled.round(decimals=decimals)[-self._periods:]

            last_trade = resampled.iloc[-1]

            unit_close_price = [round(float(trade.sol_unit_price) / float(BASE_PER_USDC), 3) for trade in trades]
            unit_high_price = unit_close_price.copy()
            unit_low_price = unit_close_price.copy()
            unit_open_price = unit_close_price.copy()
            unit_volume = [1 for _ in trades]
            unit_bar_time_date = [datetime.fromtimestamp(trade.timestamp) for trade in trades]
            unit_history_data = {
                "Date": unit_bar_time_date,
                "Close": unit_close_price,
                "High": unit_high_price,
                "Low": unit_low_price,
                "Open": unit_open_price,
                "Volume": unit_volume,
            }
            logic = {'Open': 'first',
                     'High': 'max',
                     'Low': 'min',
                     'Close': 'last',
                     'Volume': 'sum'}
            unit_history_data_df = pd.DataFrame.from_dict(unit_history_data)
            unit_history_data_df.set_index('Date', inplace=True)
            unit_history_data_df.sort_index(inplace=True)
            unit_resampled = unit_history_data_df.resample(f'{self._resolution}Min').apply(logic)
            unit_resampled = unit_resampled.dropna()
            unit_resampled = unit_resampled.round(decimals=decimals)[-self._periods:]

            last_unit_trade = unit_resampled.iloc[-1]
            start_base_amount = self._start_balances.adjusted_base_amount
            change_rate_base_amount = (last_trade["Close"] - start_base_amount) / start_base_amount
            change_rate_base_amount = round(change_rate_base_amount * 100, 2)

            start_sol_price = round(float(self._start_balances.sol_unit_price) / float(BASE_PER_USDC), 3)
            change_rate_sol_price = (last_unit_trade["Close"] - start_sol_price) / start_sol_price
            change_rate_sol_price = round(change_rate_sol_price * 100, 2)
            self.change_rate_base_amount = change_rate_base_amount
            self.change_rate_sol_price = change_rate_sol_price

            statistics_data = f'Total trades={self._jupiter.total_trade},' \
                              f'success trades={self._jupiter.success_trade},' \
                              f'failed trades={self._jupiter.failed_trade},' \
                              f'tps={self.recent_performance}'
            asset_value = f'Total Value={last_trade["Close"]} USDC {change_rate_base_amount}%,' \
                          f'SOL Price={last_unit_trade["Close"]} USDC {change_rate_sol_price}%'
            display_title = f'\n\nOpenAI Jupiter Gym Env Episode:{self.episode} Step:{self.step_count}' \
                            f'\n{asset_value}\n{statistics_data}'

            return resampled, unit_resampled, display_title

    def _draw_crypto(self):
        try:
            resampled, unit_resampled, display_title = self._get_display_title()
            self.fig, self.axes = mpf.plot(resampled,
                                           type='candle',
                                           mav=(2, 4, 6),
                                           returnfig=True,
                                           volume=True,
                                           title=display_title,
                                           ylabel='Total Value',
                                           ylabel_lower='Trade',
                                           style=self.style,
                                           figratio=(8, 4.5),
                                           )

            ax_c = self.axes[3].twinx()
            changes = unit_resampled.loc[:, "Close"].to_numpy()
            ax_c.plot(changes, color='orange', marker='o', markeredgecolor='red')
            ax_c.set_ylabel('SOL Price')
            plt.figtext(0.99, 0.01, 'By OpenAI Jupiter Gym Env', horizontalalignment='right',
                        color='lavender')
            plt.figtext(0.01, 0.01, 'www.solana-playground.com',
                        horizontalalignment='left',
                        color='lavender')
        except Exception as e:
            print(colorize(str(e), color='yellow'))

    def render(self, mode="human"):

        if mode == 'ansi':
            self._render_ansi()
        else:
            img = self._get_img_from_fig(self.fig)
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                self.viewer.imshow(img)
                return self.viewer.is_open

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        try:
            self._close_fig()
            self.ax.clear()
            self._async_wrapper(self._get_balance())
            operation: ActionType = action['operation']
            amount: int = action['amount']
            self.step_count += 1
            self.global_step_count += 1

            print(colorize(f'Action={ActionType(operation).name},Amount={amount}', color='blue'))
            if operation == ActionType.no_action:
                pass
            elif operation == ActionType.buy_sol:
                self._async_wrapper(
                    self._jupiter.convert_base_to_sol(amount=round(float(amount) * float(BASE_PER_USDC))))

            elif operation == ActionType.buy_base:
                self._async_wrapper(
                    self._jupiter.convert_sol_to_base(amount=round(float(amount) * float(LAMPORTS_PER_SOL))))

            elif operation == ActionType.convert_base_sol_base:
                self._async_wrapper(
                    self._jupiter.from_base_to_sol_then_sol_to_base(initial=max(self._option.min_init_base_mint,
                                                                                round(float(amount) * float(
                                                                                    BASE_PER_USDC))),
                                                                    minimum_profit=self._option.min_profit_base_mint,
                                                                    waiting_period=self._option.min_wait_period))

            elif operation == ActionType.convert_sol_base_sol:
                self._async_wrapper(
                    self._jupiter.from_sol_to_base_then_base_to_sol(initial=max(self._option.min_init_sol_mint,
                                                                                round(float(amount) * float(
                                                                                    LAMPORTS_PER_SOL))),
                                                                    minimum_profit=self._option.min_profit_sol_mint,
                                                                    waiting_period=self._option.min_wait_period))

            self._draw_crypto()
            self._get_info()
            obs = self._get_current_obs()
            done = self._is_done()
            if done:
                self.info['Done'] = self.done_condition_type
                self._save_summary()
            return obs, self.reward, done, self.info
        except Exception as e:
            print(colorize(str(e), color='yellow'))
            self.exception_count += 1
            done = self._is_done()
            if done:
                self.info['Done'] = self.done_condition_type
                self._save_summary()
            return self.observation, self.reward, done, self.info

    def _async_wrapper(self, f):
        self._loop.run_until_complete(f)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        self._close_fig()
        self._jupiter = JupiterAggregator(keypair=self._keypair, client=self._client, options=self._option)
        self._historical_data: List[AccountBalanceInfo] = []
        self._async_wrapper(self._set_init_balance())
        self._start_balances: AccountBalanceInfo = self._historical_data[0]
        self.step_count = 0
        self.episode += 1
        self.exception_count = 0
        self.observation = None
        self.is_start = True
        self.action = None
        self.info = {}
        self.reward = 0
        self.change_rate_base_amount = 0
        self.change_rate_sol_price = 0
        obs = self._get_current_obs()
        if return_info:

            self._get_info()
            return obs, self.info
        else:
            return obs

    def close(self):
        self._async_wrapper(self._client.close())

    def _get_img_from_fig(self, fig, dpi=160):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        if self._save_figure and self.global_step_count > 0:
            fig.savefig(f'images/fig_{str(self.global_step_count).zfill(6)}.png', dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
