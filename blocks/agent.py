import torch, os, six, random
import numpy as np
from abc import abstractmethod, ABCMeta


@six.add_metaclass(ABCMeta)
class BaseAgent(object):
    """
    base class for all agents
    """

    def __init__(
        self,
        name,
        trainer,
        action_size,
        n_episodes,
        starting_episode=0,
        checkpoints_dir="./checkpoints",
        results_dir="./results",
        eps_start=1.0,
        eps_end=0.0,
        stop_decay=0.9,
        beta_start=0.4,
        beta_end=1,
    ):
        """Initialize agent class
        Params:
        =========
            name (str): name of the experiment.
            trainer (trainer.py instance): a trainer algorithm class (i.e. Q-learning or double-Q-learning).
            action_size (int): number of possible actions, assumes a discrete action space.
            n_episodes (int): number of training episodes.
            starting_episode (optional[int]): starting episode if we load from checkpoint.
            checkpoints_dir (optional[str]): path were to save checkpoints (default="./checkpoints")
            results_dir (optional[str]): path were to save results (default="./results")
            eps_start (optional[float]): starting value for epsilon exploration factor (default=1.0)
            eps_end (optional[float]): final value for epsilon exploration factor (default=0.)
            beta_start (optional[float]): initial value for beta bias correction factor for prioritized experience (default=0.4 from original paper)
            beta_end (optional[float]): final value for beta bias correction factor for prioritized experience (default=1 from original paper)
            stop_decay (optional[float]): ratio of training episodes before we stop decaying eps and beta (default=0.9)

        NOTE: the original prioritized experience paper uses linear annealiation of the beta factor, for simplicity we use the same exponential
              decay structure for both beta and epsilon, might change this in the future.
        """
        self.name = name
        self.trainer = trainer
        # setup checkpoints and results dirs for any logging/ input output
        self.checkpoints_dir = os.path.join(checkpoints_dir, name)
        self.results_dir = os.path.join(results_dir, name)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        # setup the action size
        self.action_size = action_size
        # starting epsilon value for exploration/exploitation trade off
        self.eps = eps_start
        # formulate a suitable decay factor for epsilon given the queried options (exponential decay).
        self.EPS_DECAY_FACTOR = (eps_end / eps_start) ** (
            1 / (int(stop_decay * n_episodes) - starting_episode)
        )
        # starting beta value for bias correction in prioritized experience replay
        self.beta = beta_start
        # formulate a suitable decay factor for beta given the queried options. (since usually beta_end>beta_start, this will actually be an increase factor)
        # annealiate beta to 1 (or beta_end) as we go further in the episode (original P.E.R paper reccommends this)
        self.BETA_DECAY_FACTOR = (beta_end / beta_start) ** (
            1 / (int(stop_decay * n_episodes) - starting_episode)
        )
        # place holders for episode steps and episode counts
        self.t_step, self.episode = 0, 0

    def random_action(self):
        """Return a random discrete action."""
        return random.choice(np.arange(self.action_size))

    def greedy_action(self, state, local_model):
        """Returns the discrete action with max Q-value.
        Params:
        ==========
            state (torch.tensor): current state, make sure state tensor is on the same device as local_network.
            local_model (PyTorch model): takes input the state and outputs action value.
        returns:
            int (discrete action that maximizes Q-values)
        """
        with torch.no_grad():
            Q = local_model(state)
        return torch.argmax(Q, dim=1).item()

    def act(self, state, local_model, eps=0.0):
        """Generate an action given some input.
        Params:
        ==========
            state (torch.tensor): current state, make sure state tensor is on the same device as local_network.
            local_model (PyTorch model): takes input the slice and outputs action values
            eps (float): epsilon parameter governing exploration/exploitation trade-off
        """
        if random.random() > eps:
            return self.greedy_action(state, local_model)
        else:
            return self.random_action()

    @abstractmethod
    def learn(self):
        """update policy/value function/network through some routine."""

    @abstractmethod
    def play_episode(self):
        """Make the agent play a full episode."""

    def train(self):
        raise NotImplementedError()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def hard_update(self, local_model, target_model, N):
        """hard update model parameters.
        θ_target = θ_local every N steps.
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            N (flintoat): number of steps after which hard update takes place
        """
        if self.t_step % N == 0:
            for target_param, local_param in zip(
                target_model.parameters(), local_model.parameters()
            ):
                target_param.data.copy_(local_param.data)
