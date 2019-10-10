we need to build a replay buffer class and a class for a target Q network (function of s, a)
we have to use batch normalization

The policy is deterministic, how to handle explore exploit?

Deterministic policy means outputs the actual action instead of a probability, will need a way to bound the actions to the environment

we have two actors and teo critic networks a target for each

According to theta_prime = tau * theta + (1 - tau) * theta_prime, with tau << 1
the target actor is just the evaluation actor plus some noise process
they used Ornstein Uhlenbeck, (will need to look that up) -> will need a class for noise
