from agents.DummyAgent import DummyAgent
from jupiter_gym.envs import ActionType


class NoActionAgent(DummyAgent):

    def __init__(self, env):
        super().__init__(env)
        self.env_action = self.env.action_space.sample()
        self.env_action["operation"] = ActionType.no_action
        self.env_action["amount"] = 0

    def action(self):
        return self.env_action
