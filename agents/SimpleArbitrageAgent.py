from agents.DummyAgent import DummyAgent
from jupiter_gym.envs import ActionType


class SimpleArbitrageAgent(DummyAgent):

    def __init__(self, env, convert_sol_amount=1, convert_base_amount=1):
        super().__init__(env)
        self.env_action = self.env.action_space.sample()
        self._convert_sol_amount = convert_sol_amount
        self._convert_base_amount = convert_base_amount
        self._swap = False
        self._is_first = True

    def action(self):
        if self._is_first:
            self.env_action["operation"] = ActionType.no_action
            self.env_action["amount"] = 0
            self._is_first = False
        else:
            if self._swap:
                self.env_action["operation"] = ActionType.convert_sol_base_sol
                self.env_action["amount"] = self._convert_sol_amount
            else:
                self.env_action["operation"] = ActionType.convert_base_sol_base
                self.env_action["amount"] = self._convert_base_amount

            self._swap = not self._swap

        return self.env_action
