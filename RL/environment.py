import os, six, random
import numpy as np
from abc import abstractmethod, ABCMeta
import concurrent.futures


@six.add_metaclass(ABCMeta)
class BaseEnvironment(object):
    """
    base class for all environments objects
    """

    def __init__(
        self,
        name,
        action_size,
        checkpoints_dir="./checkpoints",
        results_dir="./results",
    ):
        """Initialize environment object
        Params:
        =====
            name (str): name of the experiment.
            action_size (int): number of possible actions, assumes a discrete action space.
            checkpoints_dir (optional[str]): path were to save checkpoints (default="./checkpoints")
            results_dir (optional[str]): path were to save results (default="./results")
        """

        # setup checkpoints and results dirs for any logging/ input output
        self.name = name
        self.action_size = action_size
        self.checkpoints_dir = os.path.join(checkpoints_dir, name)
        self.results_dir = os.path.join(results_dir, name)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    @abstractmethod
    def step(self, action):
        """Perform an input action, observe the next state and reward.
        returns the experience tuple (state, action, reward, next_state, done).
        =====
            action (int): a discrete action to be taken.
        Returns
            tuple (state, action, reward, next_state, done)
        """

    def reset(self):
        """Reset the environment at the end of an episode. It will set a random initial state for the next episode
        and reset all the logs that the environment is keeping.
        """
        raise NotImplementedError()

    def get_reward(self):
        """Calculates the corresponding reward of stepping into a new state."""
        raise NotImplementedError()

    def is_terminal(self, state):
        """Checks wether a given state is terminal.
        Params:
        =====
            state (torch.tensor): current state of the environment (represented as an array/number).
        """
        raise NotImplementedError()

    def render(self, transition):
        """Renders the environment under a given transition tuple (s, a, r, s', done)
        Params:
        =====
            transition (tuple): experience tuple (state, action, reward, next_state, done)."""
        raise NotImplementedError()

    def random_walk(self, n_steps):
        """Starts a random walk to gather observations (s, a, r, s', done).
        Params:
        =====
            n_steps (int): number of steps for which we want the random walk to continue.
        Returns
            list[tuple (s, a, r, s', done)]: trajectory of experience tuples generated throughout the walk
        """
        trajectory = []
        # start the random walk
        self.reset()
        for step in range(n_steps):
            # random action
            action = random.choice(np.arange(self.action_size))
            # step the environment according to this random action
            transition = self.step(action)
            trajectory.append(transition)
        return trajectory

    @staticmethod
    def discreteActionToEffect(action):
        """Maps a discrete action to a specific effect that will influence the environment to change to the next state."""
        raise NotImplementedError()

    @staticmethod
    def effectToDiscreteAction(effect):
        """Reverse mapping of the above function, gets you the discrete action corresponding to a particula environment change."""
        raise NotImplementedError()
