# test-time training as a drop-in replacement for causal self-attention

_Test-time training (TTT)_[^ttt] is a recent approach to sequence modeling
which scales linearly with the sequence length.
Sun et. al.[^ttt] present the TTT layer with both
_outer-loop parameters_, learnt at train-time,
and _inner-loop parameters_, learnt at test-time.
In this blog post,
I give an alternate formulation
without outer-loop parameters.
In particular, I show that the TTT computation
is compatible with the parameter-free attention operator.
This suggests that TTT models can be initialized
with pre-trained transformer weights,
which is not particularly highlighted in the paper.
Further, some might find this presentation
easier to digest than the original.

First, I motivate TTT as a substitute for
attention with a single query vector.
Then, I show how TTT achieves linear complexity
in the sequence length by considering
the full causal self-attention operator.

## single-query attention

Single-query attention has the type signature $o = a(q, K, V)$, where,
* $q \in \mathbb{R}^{d_k}$ is the _query_ vector,
* $K = [k_1, \dots, k_T]$, $k_i \in \mathbb{R}^{d_k}$ is the list of _key_ vectors,
* $V = [v_1, \dots, v_T]$, $v_i \in \mathbb{R}^{d_v}$ is the list of _value_ vectors, and,
* $o \in \mathbb{R}^{d_v}$ is the _output_ vector.


The standard _scaled-dot-product-attention_ ($a^{\text{SDPA}}$) is computed as follows:
1. $w = \text{softmax}\left(\left\\{\frac{q \cdot k_1}{\sqrt{d_k}}, \dots, \frac{q \cdot k_T}{\sqrt{d_k}}\right\\}\right)$
2. $o = \sum_{i=1}^t w_i v_i$

Below is an intuitive breakdown of the two steps:
1. Soft-select a key $k_i$ most _similar_ to the query $q$.
2. Return the associated value $v_i$.


This is reminiscent of applying a _nearest-neighbor classifier_ with
train set $\\{(k_1, v_1), \dots, (k_T, v_T)\\}$, and
test input $q$.
__So long as we have a classifier on the key-value pairs,
why not use a neural network?__
This would look like below:
1. Train a neural network $f_\theta$ to predict $v_i$ given input $k_i$, and,
2. Return the prediction of the network on $q$.


This motivates $a^{\text{TTT}}$, computed as follows:
1. $\theta = \arg \min_\theta \sum_{i=1}^T \Vert f_\theta(k_i) - v_i \Vert^2$
2. $o = f_\theta(q)$.

Step 1 corresponds to training the network with an $L_2$ loss.
If we have to train on all key-value pairs,
then $a^{\text{TTT}}$ is at least linear in the sequence length,
no better than $a^{\text{SDPA}}$.
To see how TTT can be more efficient,
we need to consider how single-query attention
is invoked inside the over-arching self-attention computation.


## causal self-attention

In _self-attention_, there is a query vector,
and an associated output vector,
for each key-value pair.
The type signature is $O = A(Q, K, V)$, where,
* $Q = [q_1, \dots, q_T]$, $q_i \in \mathbb{R}^{d_k}$ is the _list of_ query vectors,
* $K = [k_1, \dots, k_T]$, $k_i \in \mathbb{R}^{d_k}$ is the list of key vectors,
* $V = [v_1, \dots, v_T]$, $v_i \in \mathbb{R}^{d_v}$ is the list of value vectors, and,
* $O = [o_1, \dots, o_T]$, $o_i \in \mathbb{R}^{d_v}$ is the _list of_ output vectors.


The relation to single-query attention is given by:
$o_t = a(q_t, [k_1, \dots, k_t], [v_1, \dots, v_t])$
The dependence of $o_t$ on $(k_i, v_i)$ only for $i \le t$ makes this kind of self-attention _causal_.

$A^{\text{SDPA}}$ is quadratic in the sequence length,
as each $o_t$ is linear in $t$,
and no further optimization is possible.
$A^{\text{TTT}}$ escapes this quadratic dependence
by reusing the learnt neural network from the previous timestep,
instead of training a fresh one from scratch.
For example, if we use one step of gradient descent
on every pair $(k_t, v_t)$ with learning rate $\eta$,
$A^{\text{TTT}}$ can be computed as below:

For $t=1, \dots, T$:
1. $\theta_t = \theta_{t-1} - \eta \nabla_{\theta_t} \Vert f_{\theta_t}(k_t) - v_t \Vert^2$
2. $o_t = f_\theta(q_t)$

This is linear in the sequence length $T$.


## epilogue

What I have presented is only one way to
gain an introductory understanding of TTT.
This blog post has nothing to say about
making the idea work in practice,
or why it should do any better than all the other
sub-quadratic sequence models out there.
I believe that
__the original paper is profound in its philosophy and technical achievements.__
I highly recommend giving it a thorough read,
if you haven't already!
Sections 2.1-2.3 are most directly comparable
and they are excellent reads despite this post.



## acknowledgements

This post was inspired by discussions with Krish Parikh and Marcel Roed,
and indirectly from Yu Sun's many talks on the topic.

## references

[^ttt]: Learning to (Learn at Test Time): RNNs with Expressive Hidden States, <https://arxiv.org/abs/2407.04620>
