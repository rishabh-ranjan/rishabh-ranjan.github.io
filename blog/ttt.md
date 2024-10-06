# test-time training as a drop-in replacement for causal self-attention

Test-time training (TTT) [1] is a recent approach to sequence modeling
which scales linearly with the sequence length.
[1] presents the TTT layer with both
outer-loop parameters, learnt at train-time,
and inner-loop parameters, learnt at test-time.
In this blog post,
I give an alternate formulation of the TTT layer
without outer-loop parameters.
In particular, I show that the TTT computation
is compatible with the parameter-free attention operator.
This suggests that TTT models can be initialized
with pre-trained transformer weights,
which is not particularly highlighted in the paper.
Further, I believe that some will find this presentation
of TTT easier to digest.

## single-query attention

Single-query attention has the type signature $o = a(q, K, V)$, where,
* $q \in \mathbb{R}^{d_k}$ is the query vector,
* $K = [k_1, \dots, k_T]$, $k_i \in \mathbb{R}^{d_k}$ is the list of key vectors,
* $V = [v_1, \dots, v_T]$, $v_i \in \mathbb{R}^{d_v}$ is the list of value vectors, and,
* $o \in \mathbb{R}^{d_v}$ is the output vector.


The standard scaled-dot-product-attention ($a^{\text{SDPA}}$) is computed as follows:

1. $w_i = \dfrac{\exp\left(q^\top k_i\right)}{\sum_j \exp\left(q^\top k_j\right)}$

2. $o = \sum_i w_i v_i$

(We ignore the $\sqrt{d_k}$ denominator added for numerical stability.)

An intuitive breakdown of the two steps is:
1. soft-select a key $k_i$ which is most similar to the query $q$
2. return the value $v_i$ corresponding to that key $k_i$


This is reminiscent of the nearest-neighbor classifier with
train set $\\{(k_1, v_1), \dots, (k_T, v_T)\\}$
and test input $q$.

But, if we are training classifiers, why not train a neural network?
The two steps would look like this:
1. Train a neural network $f_\theta$ to predict $v_i$ given input $k_i$
2. Return the prediction of the network on $q$

This motivates $a^{TTT}$ as a replacement for $a^{SDPA}$,
computed as follows:
1. $\theta = \arg \min_\theta \sum_i \Vert f_\theta(k_i) - v_i \Vert^2$
2. $o = f_\theta(q)$, where,



## causal self-attention

To see how TTT removes the quadratic dependence on sequence length $T$,
we must consider the entire causal self-attention computation.
It has the type signature $O = A(Q, K, V)$, where,
* $Q = [q_1, \dots, q_T]$, $q_i \in \mathbb{R}^{d_k}$ is the list of query vectors,
* $K = [k_1, \dots, k_T]$, $k_i \in \mathbb{R}^{d_k}$ is the list of key vectors,
* $V = [v_1, \dots, v_T]$, $v_i \in \mathbb{R}^{d_v}$ is the list of value vectors, and,
* $O = [o_1, \dots, o_T]$, $o_i \in \mathbb{R}^{d_v}$ is the list of output vectors.


This is related to single-query attention by:
$o_t = a(q_t, [k_1, \dots, k_t], [v_1, \dots, v_t])$

This is causal because $o_t$ depends on $(k_i, v_i)$ only for $i \le t$.

For $A^{\text{SDPA}}$, each $o_t$ is linear in the sequence length,
so the entire computation is quadratic in the sequence length.
No amortization is possible.

$A^{\text{TTT}}$ escapes this quadratic dependence
by only updating the learnt parameter vector from the previous timestep,
instead of training a fresh one from scratch.

For example, if we use one step of gradient descent
on the training point $(k_t, v_t)$ with learning rate $\eta$,
the entire $A^{\text{TTT}}$ computation can be achieved as below,
which is linear in the sequence length $T$:

For $t=1, \dots, T$:
1. $\theta_t = \theta_{t-1} - \eta \nabla_\theta (f_\theta(k_t) - v_t)^2 \big\|_{\theta_t}$
2. $o_t = f_\theta(q_t)$


## how the original paper presents it

Recall that in a transformer attention head,
$(q_t, k_t, v_t)$'s are obtained as projections from
the token vector $x_t$:
$q_t = W^Q x_t, k_t = W^K x_t, v_t = W^V x_t$.

The original paper includes these as "outer-loop" parameters
of the TTT layer. Then their inner-loop loss function

While their notation is suggestive,
the connection to query, key and value in attention
is not explicitly highlighted.
This obscures the possibility of reusing pre-trained
transformer weights in TTT-layer-based models.



## epilogue

This exposition is in contrast to the one used in the original
paper which treats $\theta_Q$, $\theta_K$, $\theta_V$ as "outer-loop"
parameters of the TTT layer,
and proposes

Of course, this simple explanation says nothing about
making the idea actually work,
or why it could do any better than all the other
sub-quadratic attention variants out there.
The original paper is profound in its motivations,
philosophy and techninal achievements.
Go read it!

If you are curious, my presentation offers an alternative to their
sections 2.3 and 2.4, where they include 



<!--## acknowledgements-->
<!---->
<!--This blog was inspired by discussions with Krish Parikh and Marcel Roed.-->
<!--Thanks to them and xxx for feedback on the draft.-->
