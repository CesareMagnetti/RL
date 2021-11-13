# Code

This repo contains common algorithms and building blocks for RL problems.

# Structure

**trainers**<br>
common training (i.e. Q-Learning, policy gradient etc.) routines implemented in their own classes. Each trainer will take as input one or more models and will apply a single forward pass returning the batch-loss (and/or priorities).
<hr/>
**buffers**<br>
common replay buffer implementations.
<hr/>
