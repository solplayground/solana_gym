import asyncio
import base64
import time
from typing import Optional, List

import aiohttp
from gym.utils.colorize import *
from solana.keypair import Keypair
from solana.rpc import types as solana_types
from solana.rpc.async_api import AsyncClient
from solana.rpc.core import RPCException, RPCNoResultException
from solana.rpc.types import TokenAccountOpts
from solana.transaction import Transaction
from spl.token.constants import TOKEN_PROGRAM_ID

from jupiter_gym.envs import LAMPORTS_PER_SOL
from jupiter_gym.envs.types import JupiterConfiguration, TokenAccountInfo, JupiterQuoteInfo, JupiterRouteInfo, \
    MarketInfo, \
    AccountBalanceInfo, TransactionId


# noinspection PyBroadException
class JupiterAggregator:

    def __init__(self, keypair: Keypair, client: AsyncClient, options: Optional[JupiterConfiguration]):
        self._options = options if options else JupiterConfiguration()
        self._keypair = keypair
        self._total_trade = 0
        self._failed_trade = 0
        self._success_trade = 0
        self._client = client

    @property
    def total_trade(self):
        return self._total_trade

    @property
    def success_trade(self):
        return self._success_trade

    @property
    def failed_trade(self):
        return self._failed_trade

    async def get_crypto_amounts(self) -> AccountBalanceInfo:
        try:
            token_accounts = await self.get_token_account_info()
            sol_unit_price = await self.get_sol_unit_price()
            coins = list(filter(lambda t: getattr(t, 'mint') == self._options.sol_mint, token_accounts))
            wrapped_sol_amount = sum([getattr(a, 'amount') for a in coins])
            sol_amount = await self.get_sol_balance()
            sol_amount += wrapped_sol_amount
            coins = list(filter(lambda t: getattr(t, 'mint') == self._options.base_mint, token_accounts))
            decimals = getattr(coins[0], 'decimals') if coins else 6
            base_amount = sum([getattr(a, 'amount') for a in coins])
            adjusted_sol_amount = float(sol_amount) / float(LAMPORTS_PER_SOL) + float(base_amount) / float(
                sol_unit_price)
            adjusted_base_amount = float(base_amount) / (10 ** decimals) + float(sol_amount) / float(
                LAMPORTS_PER_SOL) * float(sol_unit_price) / (10 ** decimals)
            return AccountBalanceInfo(sol_amount=sol_amount,
                                      base_amount=base_amount,
                                      wrapped_sol_amount=wrapped_sol_amount,
                                      timestamp=int(time.time()),
                                      sol_unit_price=sol_unit_price,
                                      adjusted_sol_amount=adjusted_sol_amount,
                                      adjusted_base_amount=adjusted_base_amount)


        except:
            return AccountBalanceInfo(sol_amount=0,
                                      base_amount=0,
                                      wrapped_sol_amount=0,
                                      timestamp=0,
                                      sol_unit_price=-1,
                                      adjusted_sol_amount=0,
                                      adjusted_base_amount=0)

    async def get_sol_unit_price(self) -> int:
        try:
            quote_info: Optional[JupiterQuoteInfo] = await self.get_sol_quote(amount=LAMPORTS_PER_SOL)
            return quote_info.routes[0].out_amount
        except:
            return -1

    async def get_sol_quote(self, amount: int) -> Optional[JupiterQuoteInfo]:
        """"
           Get quote from SOL to base mint (USDC)
        """
        return await JupiterAggregator.get_coin_quote(slippage=self._options.slippage,
                                                      output_mint=self._options.base_mint,
                                                      input_mint=self._options.sol_mint,
                                                      amount=amount, direct_route=self._options.use_direct_route)

    async def get_base_quote(self, amount: int) -> Optional[JupiterQuoteInfo]:
        """"
        Get quote from base mint (USDC) to SOL
        """
        return await JupiterAggregator.get_coin_quote(slippage=self._options.slippage,
                                                      output_mint=self._options.sol_mint,
                                                      input_mint=self._options.base_mint, amount=amount,
                                                      direct_route=self._options.use_direct_route)

    async def get_sol_balance(self) -> int:
        res = await self._client.get_balance(pubkey=self._keypair.public_key)
        return int(res['result']['value'])

    async def get_recent_performance(self) -> int:
        try:
            res = await self._client.get_recent_performance_samples(5)
            result = res['result']
            total_seconds = 0
            total_transactions = 0
            for r in result:
                total_seconds += r['samplePeriodSecs']
                total_transactions += r['numTransactions']
            tps = total_transactions / total_seconds
            return round(tps)
        except:
            return -1

    async def get_token_account_info(self) -> List[TokenAccountInfo]:

        res = await self._client.get_token_accounts_by_owner(owner=self._keypair.public_key,
                                                             opts=TokenAccountOpts(encoding='jsonParsed',
                                                                                   program_id=TOKEN_PROGRAM_ID))

        return [TokenAccountInfo(pubkey=a['pubkey'],
                                 mint=a['account']['data']['parsed']['info']['mint'],
                                 owner=a['account']['data']['parsed']['info']['owner'],
                                 amount=int(a['account']['data']['parsed']['info']['tokenAmount']['amount']),
                                 lamports=int(a['account']['lamports']),
                                 is_native=bool(a['account']['data']['parsed']['info']['isNative']),
                                 decimals=int(a['account']['data']['parsed']['info']['tokenAmount']['decimals']))
                for a
                in
                res['result']['value']] if res else []

    async def from_base_to_sol_then_sol_to_base(self, initial: int, minimum_profit, waiting_period: int):
        await self.convert_from_mint1_to_mint2_then_to_mint1(initial=initial, minimum_profit=minimum_profit,
                                                             waiting_period=waiting_period,
                                                             mint1=self._options.base_mint,
                                                             mint1_name=self._options.base_mint_name,
                                                             mint2=self._options.sol_mint,
                                                             mint2_name=self._options.sol_mint_name)

    async def from_sol_to_base_then_base_to_sol(self, initial: int, minimum_profit, waiting_period: int):
        await self.convert_from_mint1_to_mint2_then_to_mint1(initial=initial, minimum_profit=minimum_profit,
                                                             waiting_period=waiting_period,
                                                             mint1=self._options.sol_mint,
                                                             mint1_name=self._options.sol_mint_name,
                                                             mint2=self._options.base_mint,
                                                             mint2_name=self._options.base_mint_name)

    async def convert_from_mint1_to_mint2_then_to_mint1(self, initial: int, minimum_profit: int, mint1: str, mint2: str,
                                                        mint1_name: str, mint2_name: str, waiting_period: int):
        mint1_to_mint2_jupiter_route = await self.get_coin_quote(slippage=self._options.slippage, input_mint=mint1,
                                                                 output_mint=mint2, amount=initial,
                                                                 direct_route=self._options.use_direct_route)
        if mint1_to_mint2_jupiter_route:
            mint2_to_mint1_jupiter_route = await self.get_coin_quote(slippage=self._options.slippage, input_mint=mint2,
                                                                     output_mint=mint1,
                                                                     amount=mint1_to_mint2_jupiter_route.routes[
                                                                         0].out_amount,
                                                                     direct_route=self._options.use_direct_route)
            if mint2_to_mint1_jupiter_route:
                received_mint1_amount = mint2_to_mint1_jupiter_route.routes[0].out_amount
                if received_mint1_amount > initial + minimum_profit:
                    print(colorize(
                        f'initial={initial} get={received_mint1_amount} profit={received_mint1_amount - initial},'
                        f'{mint1_name}->{mint2_name}->{mint1_name}', color='blue'))
                    tasks = [asyncio.create_task(self._execute_jupiter_route(
                        jupiter_route=
                        mint1_to_mint2_jupiter_route.routes[0])),
                        asyncio.create_task(self._execute_jupiter_route(
                            jupiter_route=
                            mint2_to_mint1_jupiter_route.routes[0]))
                    ]
                    await asyncio.gather(*tasks)
                    await asyncio.sleep(waiting_period)
                else:
                    await asyncio.sleep(waiting_period)
                    print(
                        colorize(f'NO PROFIT initial={initial} get={received_mint1_amount} '
                                 f'profit={received_mint1_amount - initial},'
                                 f'{mint1_name}->{mint2_name}->{mint1_name}', color='yellow'))
            else:
                print(colorize(f'Unable to get route info: {mint1_name}->{mint2_name}->{mint1_name}', color='yellow'))

        else:
            print(colorize(f'Unable to get route info: {mint1_name}->{mint2_name}->{mint1_name}', color='yellow'))

    async def _execute_jupiter_route(self, jupiter_route: JupiterRouteInfo):
        market_infos = [f"{getattr(m, 'label')}" for m in jupiter_route.market_infos]
        print(colorize(f'Market Info:{str(market_infos)}', color='green'))
        transactions = await JupiterAggregator.get_jupiter_transactions(jupiter_route=jupiter_route,
                                                                        public_key=str(self._keypair.public_key))

        encoded_transactions: List[str] = []

        if 'setupTransaction' in transactions:
            setup_transaction = transactions['setupTransaction']
            encoded_transactions.append(setup_transaction)

        if 'swapTransaction' in transactions:
            swap_transaction = transactions['swapTransaction']
            encoded_transactions.append(swap_transaction)

        if 'cleanupTransaction' in transactions:
            cleanup_transaction = transactions['cleanupTransaction']
            encoded_transactions.append(cleanup_transaction)
        for encoded_transaction in encoded_transactions:
            buffer = base64.b64decode(encoded_transaction)
            transaction = Transaction.deserialize(buffer)
            transaction_id = await JupiterAggregator._send_transaction(client=self._client, transaction=transaction,
                                                                       signers=self._keypair)
            try:
                await JupiterAggregator._confirm_transaction(client=self._client, transaction_id=transaction_id)
                print(colorize(f'Success: https://solscan.io/tx/{transaction_id}', color='green'))
                self._success_trade += 1
            except Exception:
                print(colorize(f'Failed: https://solscan.io/tx/{transaction_id}', color='red'))
                self._failed_trade += 1
            finally:
                self._total_trade += 1

    async def convert_mint1_to_mint2(self, amount: int, mint1: str, mint2: str):
        jupiter_route = await JupiterAggregator.get_coin_quote(slippage=self._options.slippage, input_mint=mint1,
                                                               output_mint=mint2, amount=amount,
                                                               direct_route=self._options.use_direct_route)
        if jupiter_route:
            route = jupiter_route.routes[0]
            await self._execute_jupiter_route(jupiter_route=route)
        else:
            print(colorize(f'Unable to get route info', color='yellow'))

    async def convert_sol_to_base(self, amount: int):
        await self.convert_mint1_to_mint2(amount=amount, mint1=self._options.sol_mint, mint2=self._options.base_mint)

    async def convert_base_to_sol(self, amount: int):
        await self.convert_mint1_to_mint2(amount=amount, mint1=self._options.base_mint, mint2=self._options.sol_mint)

    @classmethod
    async def get_jupiter_transactions(cls, jupiter_route: JupiterRouteInfo, public_key: str):
        async with aiohttp.ClientSession() as session:
            data = {
                "route": jupiter_route.route,
                "userPublicKey": public_key,
                "wrapUnwrapSOL": False,

            }
            headers = {'content-type': 'application/json'}
            async with session.post(url='https://quote-api.jup.ag/v1/swap', json=data, headers=headers) as resp:
                json_object = await resp.json()

        return json_object

    @classmethod
    async def get_coin_quote(cls, slippage: float, input_mint: str, output_mint: str, amount: int,
                             direct_route=False) -> \
            Optional[JupiterQuoteInfo]:
        async with aiohttp.ClientSession() as session:
            only_direct_routes = 'true' if direct_route else 'false'
            url = f'https://quote-api.jup.ag/v1/quote?outputMint={output_mint}&inputMint={input_mint}&amount={amount}' \
                  f'&slippage={slippage}&onlyDirectRoutes={only_direct_routes}'
            try:
                async with session.get(url) as resp:
                    json_object = await resp.json()
                    return JupiterQuoteInfo(routes=[JupiterRouteInfo(
                        in_amount=int(j['inAmount']),
                        out_amount=int(j['outAmount']),
                        amount=int(j['amount']),
                        out_amount_with_slippage=int(j['outAmountWithSlippage']),
                        market_infos=[MarketInfo(label=m['label']) for m in j['marketInfos']],
                        route=j) for j in
                        json_object['data']], time_taken=json_object['timeTaken']) if json_object else None
            except:
                return None

    @classmethod
    async def send_and_confirm_transaction(cls, client, transaction, signers) -> solana_types.RPCResponse:
        resp = await client.send_transaction(transaction, signers)
        resp = JupiterAggregator._post_send(resp)
        return await client.confirm_transaction(resp["result"], 'confirmed')

    @classmethod
    async def _send_transaction(cls, client, transaction, signers) -> TransactionId:
        resp = await client.send_transaction(transaction, signers, opts=solana_types.TxOpts(skip_preflight=True))
        resp = JupiterAggregator._post_send(resp)
        return resp['result']

    @classmethod
    async def _confirm_transaction(cls, client, transaction_id: TransactionId) -> TransactionId:
        return await client.confirm_transaction(transaction_id, 'confirmed')

    @classmethod
    def _post_send(cls, resp):
        error = resp.get("error")
        if error:
            raise RPCException(error)
        if not resp.get("result"):
            raise RPCNoResultException("Failed to send transaction")
        return resp
