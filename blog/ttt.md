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
transformer based LLM can be swapped out with a TTT layer,
to achieve linear complexity,
without too much performance regression.
To me this is one of the coolest parts of the TTT model,
but I feel it does not come across as clearly in the original paper.



## single-query attention

First we consider the attention module with a single query vector.
It has the following type signature:

Inputs:
1. query, $q \in \mathbb{R}^{d_k}$
2. list of keys, $K = [k_1, \dots, k_T]$, $k_i \in \mathbb{R}^{d_k}$
3. list of values, $V = [v_1, \dots, v_T]$, $v_i \in \mathbb{R}^{d_v}$

Output: $a(q, K, V) \in \mathbb{R}^{d_v}$.

The standard scaled-dot-product-attention is given by:

$a_{\text{SDPA}} = \sum_i w_i v_i$, where,
<!--$$w_i = \frac{\exp\left(q^\top k_i / \sqrt{d_k}\right)}{\sum_j \exp\left(q^\top k_j / \sqrt{d_k}\right)}$$-->
$$w_i = \frac{\exp\left(q^\top k_i\right)}{\sum_j \exp\left(q^\top k_j\right)}$$
ignoring the $\sqrt{d_k}$ denominator added for numerical stability.


Intuitively, $a_{SDPA}$ achieves the following:
1. soft-select a $k_i$ which is most similar to the query
2. return the $v_i$ corresponding to that $k_i$


This is reminiscent of the nearest-neighbor classifier with
$\\{(k_1, v_1), \dots, (k_T, v_T)\\}$ as the train set
and $q$ as the test input.

Whereas naive inference on a nearest neighbor classifier scales with the size of the training set,
parameteric classifiers such as neural networks
"compress" the training data into a fixed size parameter vector $\theta$,
and use that at inference, thereby removing the dependence on the training set size.

This motivates $a_{TTT}$ as a replacement for $a_{SDPA}$,
defined as follows:
$$a_{TTT} = f_\theta(q)$$, where,
$$\theta = \arg \min_\theta \sum_i (f_\theta(k_i) - v_i)^2 $$

Here $f_\theta$ is some neural network parameterized by $\theta$.
The second equation corresponds to training the neural network on the KV-cache
as the training set with the MSE loss,
while the first corresponds to making a prediction with that neural network
on the query vector $q$.


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
$$o_t = a_{SDPA}(q_t, [k_1, \dots, k_t], [v_1, \dots, v_t])$$

Note that to compute $o_t$ only $(k_i, v_i)$ for $i \le t$ is used,
which corresponds to causal attention.

For $a_{SDPA}$ no amortization is possible,
so the computation of $A(Q, K, V)$ is $O(T^2)$,
since the computation of $a(q_t, K[:t], V[:t])$ is $O(t)$.

For $A_{TTT}$, we have a parameter vector $\theta_t$ for every time step.
$$o_t = f_$\theta_t$(q_t)$$, where,
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
