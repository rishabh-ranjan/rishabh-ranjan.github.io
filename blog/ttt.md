# test-time training as a drop-in replacement for causal attention

Test-time training (TTT) layer is a recent alternative
to the causal attention layer with linear complexity.
In this blog post,
I try to motivate TTT as a drop-in replacement for causal attention.
This differs slightly from the presentation adopted in the original paper,
hence the post.
In particular,
this blog presents the TTT layer as having no outer loop parameters.
One immediate consequence is that causal attention in a pre-trained
transformer-based LLM can be swapped out with a TTT layer,
to achieve linear complexity,
without too much performance regression.
To me this is one of the coolest parts of the TTT model,
but I feel it does not come across as clearly in the original paper.


First, I will motivate TTT as a replacement for single-query attention,
then show how it removes the quadratic dependence on the sequence length
by considering causal attention.


## single-query attention

Single-query attention has the following type signature:

__Inputs:__
1. query, $q \in \mathbb{R}^{d_k}$
2. keys, $K = [k_1, \dots, k_T]$, $k_i \in \mathbb{R}^{d_k}$
3. values, $V = [v_1, \dots, v_T]$, $v_i \in \mathbb{R}^{d_v}$

__Output:__ $a(q, K, V) \in \mathbb{R}^{d_v}$.

The standard scaled-dot-product-attention (SDPA) is computed as follows:

1. $w_i = \dfrac{\exp\left(q^\top k_i\right)}{\sum_j \exp\left(q^\top k_j\right)}$

2. $a^{\text{SDPA}} = \sum_i w_i v_i$

(We ignore the $\sqrt{d_k}$ denominator added for numerical stability.)

An intuitive breakdown of the two steps is:
1. soft-select a key $k_i$ which is most similar to the query $q$
2. return the value $v_i$ corresponding to that key $k_i$


This is reminiscent of the nearest-neighbor classifier with
$\\{(k_1, v_1), \dots, (k_T, v_T)\\}$ as the train set
and $q$ as the test input.

But, if we are training classifiers, why not train a neural network?
The two steps would look like this:
1. Train a neural network $f_\theta$ to predict $v_i$ given input $k_i$
2. Return the prediction of the network on $q$

This motivates $a^{TTT}$ as a replacement for $a^{SDPA}$,
computed as follows:
1. $\theta = \arg \min_\theta \sum_i (f_\theta(k_i) - v_i)^2$
2. $a^{TTT} = f_\theta(q)$, where,



## causal attention

$a_{TTT}$ in the previous section does not exactly remove the dependence
on the size of the KV-cache, because it still requires training the neural network,
which is atleast linear in the KV-cache if training on all KV pairs.

However, in practice, it is possible to amortize this cost.
To see this, we must step back and consider the entire causal attention computation.
It has the following type signature:

Inputs:
1. list of queries, $Q = [q_1, \dots, q_T]$, $q_i \in \mathbb{R}^{d_k}$
2. list of keys, $K = [k_1, \dots, k_T]$, $k_i \in \mathbb{R}^{d_k}$
3. list of values, $V = [v_1, \dots, v_T]$, $v_i \in \mathbb{R}^{d_v}$

Output: list of output vectors, $A(Q, K, V) = [o_1, \dots, o_T]$, $o_i \in \mathbb{R}^{d_v}$

Standard SDPA attention is given by:
$$a_t = a^{SDPA}(q_t, [k_1, \dots, k_t], [v_1, \dots, v_t])$$

Note that to compute $a_t$ only $(k_i, v_i)$ for $i \le t$ is used,
which corresponds to causal attention.

For $a^{SDPA}$ no amortization is possible,
so the computation of $A(Q, K, V)$ is $O(T^2)$,
since the computation of $a(q_t, K[:t], V[:t])$ is $O(t)$.

For $A^{TTT}$, we have a parameter vector $\theta_t$ for every time step.
$$o_t = f_{\theta_t}(q_t)$$, where,
$$\theta_t = \arg \min_\theta \sum_{i=1}^t (f_\theta(k_i) - v_i)^2 $$

For $A_{TTT}$, the cost of training the neural network can be amortized over
the various values of $t$.
Intuitively, for every new token,
the neural network only needs to be updated on the training point
corresponding to its KV-pair.
For example, we can do one step of gradient descent:

$$\theta_t = \theta_{t-1} - \eta g_t(\theta_t)$$, where,

$$g_t(\theta) = \nabla_\theta (f_\theta(k_t) - v_t)^2$$ is the gradient,

and $\eta$ is the learning rate for $\theta_t$.

This keep the overall dependence on the size of the KV-cache only linear.


## epilogue

Of course, this simple explanation says nothing about
making the idea actually work,
or why it could do any better than all the other
sub-quadratic attention variants out there.
The original paper is profound in its motivations,
philosophy and techninal achievements.
Go read it!



<!--## acknowledgements-->
<!---->
<!--This blog was inspired by discussions with Krish Parikh and Marcel Roed.-->
<!--Thanks to them and xxx for feedback on the draft.-->
