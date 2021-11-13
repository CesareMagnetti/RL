import torch


class QLearning(object):
    """
    Q-learning training routine object.
    """

    def __init__(self, gamma, prioritized=False):
        """Initializes the trainer class
        Params:
        ==========
            gamma (float): discount factor in (0,1).
            prioritized (bool): flag for prioritized experience (default=False)
        """
        assert gamma > 0 and gamma <= 1, (
            "discount factor gamma should be in (0,1], got: %f" % gamma
        )
        self.gamma, self.prioritized = gamma, prioritized

    def __call__(self, batch, local, criterion, target=None):
        """Update local network parameters using given batch of experience tuples.
        Params
        ======
            batch (dict): contains all training inputs
            local (PyTorch model): local Q network (the one we update)
            criterion (torch.nn.Module): the loss we use to update the network parameters
            target (optional[PyTorch model]): target Q network (the one we use to bootstrap future Q values)
        Returns
        ======
            loss (torch.tensor): final TD error loss tensor to compute gradients from.
            deltas (optional[torch.tensor]): bias correction coefficients for prioritized experience.

        NOTE: in order to handle prioritized experience the criterion does not have to aggregate the output by default
        (i.e. you must pass a criterion with 'reduction=None'). We then manually aggregate the loss by averaging over the batch.
        """
        assert (
            criterion.reduction == None
        ), "please pass an input ``criterion`` with reduction=None. We reduce manually averaging over the batch."
        # 1. split batch
        states, actions, rewards, next_states, dones, weights = (
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["dones"],
            batch["weights"],
        )
        # 2. get value estimates Q for the current state and the next state (using target network if needed)
        Q = local(states)
        Qnext = target(next_states) if target else local(next_states)
        # 3. evaluate the loss (TDerror) and/or bias correction weights
        return self.forward(Q, Qnext, actions, rewards, dones, criterion, weights)

    def forward(self, Q, Qnext, actions, rewards, dones, criterion, weights=None):
        """single Q-learning step.
        Params
        =====
            Q (torch.tensor): action values for the current state.
            Qnext (torch.tensor): action values for the next state.
            actions (torch.tensor): actions taken in batch.
            rewards (torch.tensor): rewards collected in batch.
            dones (torch.tensor): flags if goal was reached.
            criterion (torch.nn.Module): he loss we use to update the network parameters (i.e. MSELoss).
            weights (optional[torch.tensor]): priority weights if we are using priorituzed experience replay buffer.
        Returns
        =====
            loss (torch.tensor): final TD error loss tensor to compute gradients from.
            deltas (optional[torch.tensor]): bias correction coefficients for prioritized experience.
        """
        # 1. gather the values of the action taken
        Qa = Q.gather(1, actions).squeeze()
        # 2. get the target value of the greedy action at the next state
        MaxQ = Qnext.max(1)[0].detach()
        # 3. backup the expected value of this action by bootstrapping on the greedy value of the next state
        Qhat = rewards.squeeze() + (1 - dones.int()) * self.gamma * MaxQ
        # 4. evalauate TD error as a fit function for the netwrok
        loss = criterion(Qa, Qhat)
        # 5. deltas to update priorities
        if weights:
            loss *= weights
            deltas = torch.abs(Qa - Qhat)
            return (loss.mean(), deltas)
        else:
            return loss.mean()


class DoubleQLearning(object):
    """
    Double Q-learning training routine object.
    """

    def __init__(self, gamma, prioritized=False):
        """Initializes the trainer class
        Params:
        ==========
            gamma (float): discount factor in (0,1).
            prioritized (bool): flag for prioritized experience (default=False)
        """
        assert gamma > 0 and gamma <= 1, (
            "discount factor gamma should be in (0,1], got: %f" % gamma
        )
        self.gamma, self.prioritized = gamma, prioritized

    def __call__(self, batch, local, criterion, target):
        """Update local network parameters using given batch of experience tuples.
        Params
        ======
            batch (dict): contains all training inputs
            local (PyTorch model): local Q network (the one we update)
            criterion (torch.nn.Module): the loss we use to update the network parameters
            target (PyTorch model): target Q network (the one we use to bootstrap future Q values)
        Returns
        ======
            loss (torch.tensor): final TD error loss tensor to compute gradients from.
            deltas (optional[torch.tensor]): bias correction coefficients for prioritized experience.

        NOTE: in order to handle prioritized experience the criterion does not have to aggregate the output by default
        (i.e. you must pass a criterion with 'reduction=None'). We then manually aggregate the loss by averaging over the batch.
        """
        assert (
            criterion.reduction == None
        ), "please pass an input ``criterion`` with reduction=None. We reduce manually averaging over the batch."
        # 1. split batch
        states, actions, rewards, next_states, dones, weights = (
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["dones"],
            batch["weights"],
        )
        # 2. get value estimates Q for the current state and the next state (using both local and target networks)
        Q = local(states)
        Qnext = local(next_states)
        Qnext_target = target(next_states)
        # 3. evaluate the loss (TDerror) and/or bias correction weights
        return self.forward(
            Q, Qnext, Qnext_target, actions, rewards, dones, criterion, weights
        )

    def forward(
        self, Q, Qnext, Qnext_target, actions, rewards, dones, criterion, weights=None
    ):
        """single Q-learning step.
        Params
        =====
            Q (torch.tensor): action values for the current state.
            Qnext (torch.tensor): action values for the next state evaluated using local network.
            Qnext (torch.tensor): action values for the next state evaluated using target network.
            actions (torch.tensor): actions taken in batch.
            rewards (torch.tensor): rewards collected in batch.
            dones (torch.tensor): flags if goal was reached.
            criterion (torch.nn.Module): he loss we use to update the network parameters (i.e. MSELoss).
            weights (optional[torch.tensor]): priority weights if we are using priorituzed experience replay buffer.
        Returns
        =====
            loss (torch.tensor): final TD error loss tensor to compute gradients from.
            deltas (optional[torch.tensor]): bias correction coefficients for prioritized experience.
        """
        # 1. gather Q values for the actions taken at the current state
        Qa = Q.gather(1, actions).squeeze()
        # 2. get the discrete action that maximizes the target value at the next state
        a_next = Qnext.max(1)[1].unsqueeze(-1)
        Qa_next = Qnext_target.gather(1, a_next).detach().squeeze()
        # 3. backup the expected value of this action by bootstrapping on the greedy value of the next state
        Qhat = rewards.squeeze() + (1 - dones.int()) * self.gamma * Qa_next
        # 4. evalauate TD error as a fit function for the netwrok
        loss = criterion(Qa, Qhat)
        # 5. deltas to update priorities
        if weights:
            loss *= weights
            deltas = torch.abs(Qa - Qhat)
            return (loss.mean(), deltas)
        else:
            return loss.mean()
