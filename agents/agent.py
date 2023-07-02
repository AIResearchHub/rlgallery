

from abc import abstractmethod


class Agent:

    @abstractmethod
    def get_action(self, obs):
        pass

    @abstractmethod
    def remember(self, obs, action, reward, next_obs, done):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @staticmethod
    def soft_update(target, source, tau):
        """
        Soft weight updates: target slowly track the weights of source with constant tau
        See DDPG paper page 4: https://arxiv.org/pdf/1509.02971.pdf

        Parameters:
            target (nn.Module): target model
            source (nn.Module): source model
            tau (float): soft update constant
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    @staticmethod
    def hard_update(target, source):
        """
        Copy weights from source to target

        Parameters:
            target (nn.Module): target model
            source (nn.Module): source model
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
