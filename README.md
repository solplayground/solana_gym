# OpenAI Solana Jupiter Trading Gym Env
OpenAI Gym Env for crypto trading based on [Jupiter aggregator](https://docs.jup.ag/)

This gym environment is **NOT** a simulation, it is based on real-world crypto trading on Solana blockchain.

So if you need to trading with your agent, you need a solana wallet (keypair) and some SOL (minimum 1 SOL is recommended).
or you can use the provided *NoActionAgent* to get a glimpse of the gym environment. 

The default Jupiter trading pair is SOL and USDC, since it's the most common trading pair on Solana.

The environment can easily switch to different pair if you prefer, but that requires some knowledge of solana blockchain.



# Jupiter Aggregator

Jupiter is the key liquidity aggregator for Solana, offering the widest range of tokens and best route discovery between any token pair. 


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

```bash
  pipenv install
```

Then install this package via

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

## Some cost

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
    
