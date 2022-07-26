from agents.DummyAgent import DummyAgent
from jupiter_gym.envs import ActionType


class BuyAndHoldAgent(DummyAgent):

    def __init__(self, env, convert_sol_amount=1):
        super().__init__(env)
        self.no_action = self.env.action_space.sample()
        self.no_action["operation"] = ActionType.no_action
        self.no_action["amount"] = 0
        self._convert_sol_amount = convert_sol_amount
        self._is_first = True

    def action(self):
        if self._is_first:
            env_action = self.env.action_space.sample()
            env_action["operation"] = ActionType.buy_base
            env_action["amount"] = self._convert_sol_amount
            self._is_first = False
            return env_action
        else:
            return self.no_action
