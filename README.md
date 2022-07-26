# OpenAI Solana Jupiter Trading Gym Env
OpenAI Gym Env for crypto trading based on [Jupiter aggregator](https://docs.jup.ag/)

This gym environment is **NOT** a simulation, it is based on real-world crypto trading on Solana blockchain.

If you requires a simulated environment, please try to use [OpenAI ASX GYM](https://github.com/guidebee/asx_gym), which is a OpenAI Gym based on Australia Stock Market.

So if you need to trading with your agent, you need a solana wallet (keypair) and some SOL (minimum 1 SOL is recommended).
or you can use the provided *NoActionAgent* to get a glimpse of the gym environment. 

The default Jupiter trading pair is SOL and USDC, since it's the most common trading pair on Solana.

The environment can easily switch to different pair if you prefer, but that requires some knowledge of solana blockchain.

# OpenAI Gym

[Gym](https://www.gymlibrary.ml/) is a standard API for reinforcement learning, and a diverse collection of reference environments.
This is a Solana Trading Environment uses OpenAI Gym interface, like Actions, Observations, etc.
Gym implements the classic “agent-environment loop”:

![Gym](https://www.gymlibrary.ml/_images/AE_loop.png)

# Solana Gym Actions

In the Solana Gym Environments, there are 5 actions agent can perform:

1. no_action = 1  # do nothing
2. buy_sol = 2  # buy SOL with USDC
3. buy_base = 3  # sell SOL for USDC
4. convert_base_sol_base = 4  # Arbitrage from USDC->SOL  ,SOL->USDC ,in parallel
5. convert_sol_base_sol = 5  # Arbitrage from SOL->USDC  , USDC->SOL  ,in parallel

# Solana Gym Observations

```python
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
```
- sol_unit_price  Sol prices at give timestamp
- sol_amount total SOL amount (native SOL + wrapped SOL)
- wrapped_sol_amount wrapped SOL amount
- base_amount  here base mean USDC (stable coin) amount
- adjusted_sol_amount , the total SOL value of all tokens (SOL + USDC) ,based on sol price at current timestamp
- adjusted_base_amount, the total USDC value of all tokens (SOL + USDC), based on sol price at current timestamp
- change_rate_base_amount, adjusted_base_amount value change rate (in percentage since episode starts)
- change_rate_sol_price, sol_unit_price change rate (in percentage since episode starts)
- total_trades, how many trades since current episode starts
- success_trades, successful trades since current episode starts
- failed_trades, failed trades since current episode starts
- exceptions, program exceptions counter

# Solana Gym Rewards

The default rewards is the changes of adjusted_base_amount (total asset value)


# Solana Gym Done condition

````python
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
````
Default done condition is total asset values +20% or -20%, or step reaches 2400 steps, or no enough fund, these values
are configable .

# Solana Gym Render Modes

render for an environment actually is optional. Here ASX Gym provides 3 render modes:

````python
 metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}
````

# OpenAI Wrappers

Use OpenAI [Wrappers](https://www.gymlibrary.ml/content/wrappers/), you can change the definitions of default implementations of Actions, Rewards, Done Conditions, Observations.

Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly. Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular. Wrappers can also be chained to combine their effects.




# Jupiter Aggregator

Jupiter is the key liquidity aggregator for Solana, offering the widest range of tokens and best route discovery between any token pair. 


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

```bash
  pipenv install
```

Then install this package via
The code as tested with Python 3.10. so better use 3.10+ or the latest python version. and uses pipenv to manage python packages.

````text
[packages]
aiohttp = "*"
mplfinance = "*"
opencv-python = "*"
matplotlib = "*"
pandas = "*"
pyglet = "*"
ipython = "*"
numpy = "*"
gym = "===0.24.0"
base58 = "*"
solana = "*"
python-dotenv = "*"

[dev-packages]

[requires]
python_version = "3.10"

````


```
pip install -e .
```
## Sample configuration

the environment requires a wallet keypair , you can use [phantom wallet](https://phantom.app/) to export the private key

also default solana rpc end point is https://solana-api.projectserum.com, if you have your own rpc node ,you can configure it as following

![Export private key](https://github.com/solplayground/solana_gym/blob/main/docs/export_private_key.png?raw=true)
in the .env file

```
SOLANA_WALLET_PRIVATE_KEY=2TCzhvyyBFpzXPVYXTJxSBWtRhJ...
SOLANA_RPC_URL=https://solana-api.projectserum.com

```


## Usage

```python
from logging import INFO

import base58
import gym
from dotenv import dotenv_values
from solana.keypair import Keypair

from agents.SimpleArbitrageAgent import SimpleArbitrageAgent
from jupiter_gym.envs import SOLANA_RPC_URL

config = dotenv_values(".env")
secret_passphrase = config['SOLANA_WALLET_PRIVATE_KEY']
solana_rpc_url = SOLANA_RPC_URL
if 'SOLANA_RPC_URL' in config:
    solana_rpc_url = config['SOLANA_RPC_URL']
keyPairArray = [x for x in base58.b58decode(secret_passphrase)]
keypair = Keypair.from_secret_key(bytes(keyPairArray))
print(f'Public key:{str(keypair.public_key.to_base58())}')
print(f'Solana rpc url:{solana_rpc_url}')


def main():
    gym.logger.set_level(INFO)
    min_init_base_mint = 5
    min_init_sol_mint = 0.2

    env = gym.make("JupiterGym-v0", disable_env_checker=True, keypair=keypair, min_init_base_mint=min_init_base_mint,
                   min_profit_base_mint=100,
                   min_init_sol_mint=min_init_sol_mint, min_profit_sol_mint=1000, min_wait_period=1,
                   disable_low_balance_checking=True,
                   max_step_for_one_episode=2400,
                   solana_rpc_url=solana_rpc_url
                   )

    agent = SimpleArbitrageAgent(env, convert_sol_amount=min_init_sol_mint, convert_base_amount=min_init_base_mint + 1)
    # agent = NoActionAgent(env)
    # agent = BuyAndHoldAgent(env, convert_sol_amount=min_init_sol_mint)
    env.reset()
    for _ in range(2400 * 100):
        env.render()
        observation, reward, done, info = env.step(agent.action())
        if done:
            env.reset()
        if observation is not None:
            print(observation)

    env.close()


if __name__ == "__main__":
    main()



```
<img src="https://github.com/solplayground/solana_gym/blob/main/docs/solana_gym.gif?raw=true" alt="Solana Gym" width="1000"/>

# Sample Agents

The agent directory provides a couple of simples agents for your reference. 

<img src="https://github.com/solplayground/solana_gym/blob/main/docs/bots.png?raw=true" alt="Solana Gym" width="1000"/>

You can implement your own smarter agent ,hope you can gain some profit with your trading bots.



# Episode output

 Solana Gym creates some data and images each step Agent interacts with the Environment. 

## images
 when in 'human' or 'rgb_array' render mode, Asx Gym save all images in this directory, you can use these images to create videos.

## data
 Store trading history in the data directory.

# Some costs

**NOTE:** I don't charge any fee and there's no hidden fee from me. But there's some cost you need to know.

1. Solana account cost, Solana blockchain charges storage fee. A token account is a storage, currently each account cost
 0.00203928 SOL (about $0.07343447 at the moment). [Account](https://docs.solana.com/developing/programming-model/accounts)

2. Jupiter Transaction fee .

   Here is the diagram of Jupiter routing 
   ![Jupiter](https://1295924016-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FJS2ScZOOuPoaicKVRake%2Fuploads%2Fyi4YW4IOhXi3iY37CPQF%2Fdiagram%20-%20multi-hop%20routes.png?alt=media&token=f2210ff1-79ee-4c5b-9f98-85cc77129b6b)

   For indirect route ,if you don't have the intermediate token (i.e SOL -> SERUM -> USDC) a securm token account need to be created first (only once), this cost 0.00203928 SOL.
   also for the liquidity pool (for example, if Jupiter uses [radium swap](https://raydium.io/swap/), Radium swap may charge some fee)

3. Solana Transaction fee.

   For each transaction (failed or success), solana charges 5000 lamport (0.000005SOL),almost nothing. but if you do millions of thousands,there will some cost need to be considered.

# DISCLAIMER
  
  This is based on real-world trading platform, it allows you to test your trading bot, **it doesn't promise profit and might incur losses.Use it at your own risk.**
  
  If you like this project and want to donate , please donate to SOL wallet address:
  
   ```
    GYMTaSY6vhTDTGw6qTs5W4kg8URG4zoAjzV9v5C9XTNs
   ```
    
