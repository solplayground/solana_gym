from logging import INFO

import base58
import gym
from dotenv import dotenv_values
from solana.keypair import Keypair

from agents.NoActionAgent import NoActionAgent
# from agents.SimpleArbitrageAgent import SimpleArbitrageAgent
# from agents.SimpleArbitrageAgent import SimpleArbitrageAgent
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

    # agent = SimpleArbitrageAgent(env, convert_sol_amount=min_init_sol_mint, convert_base_amount=min_init_base_mint + 1)
    agent = NoActionAgent(env)
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
