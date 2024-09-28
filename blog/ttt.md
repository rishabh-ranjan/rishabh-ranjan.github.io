# test-time training as a drop-in replacement for attention

Test-time training (TTT) layer is a recent alternative
to the attention layer with linear complexity.
In this blog post,
I try to motivate TTT as a drop-in replacement for attention.

Attention has the following type signature:
Inputs:
1. list of keys, $K = [k_1, \dots, k_T]$, $k_i \in \mathbb{R}^{d_k}$
2. list of values, $V = [v_1, \dots, v_T]$, $v_i \in \mathbb{R}^{d_v}$
3. query, $q \in \mathbb{R}^{d_k}$

Output, $a(K, V, q) \in \mathbb{R}^{d_v}$.

The standard scaled-dot-product-attention is given by:
$a_{\text{SDPA}} = \sum_i w_i v_i$, where,
$$w_i = \frac{\exp\left(\frac{q^\top k_i}{\sqrt{d_k}}\right)}{\sum_j \exp\left(\frac{q^\top k_j}{\sqrt{d_k}}\right)}$$

To clarify the discussion,
it helps to clarify what exactly we mean by attention.
And it's type signature.

We have a space of keys $\mathcal{K}$
($= \mathbb{R}^{d_k}$ in practice),
and a space of values $\mathcal{V}$
($= \mathbb{R}^{d_v}$ in practice).
A set of key-value pairs forms the KV-cache:
$\\{(k_1, v_1), \dots, (k_T, v_T)\\}$, with,
$k_i \in \mathcal{K}$,
$v_i \in \mathcal{V}$.

Next, we are given a query $q \in \mathcal{K}$

Attention is at the heart of the transformer architecture.
The quadratic cost of attention 


The attention operator is at the heart of the transformer architecture.
It takes as input
1. a query vector $q \in \mathbb{R}^d$, and,
2. a set of key-value vector pair $\\{\dots, (k_i, v_i), \dots\\}$,
$k_i, v_i \in \mathbb{R}^d$ (also called the KV-cache),

and returns a similarity weighted sum of values:
$$
a(q, \\{\dots, (k_i, v_i), \dots\\}) = \sum_i w_i v_i \quad (1)
$$
where the similarity function is
$$
w_i = \frac{\exp(q^\top k_i / \sqrt{d})}{\sum_j \exp(q^\top k_j / \sqrt{d})}
$$
for the most common variant, Scaled Dot Product Attention (SDPA).

The time complexity of SDPA is $O(Td)$ for T tokens
(one $(k, v)$ pair per token),
which leads to the $O(T^2d)$ term in the transformer's complexity
(there is a $q$ for every token).

Suppose we wanted to approximate SDPA in $O(d)$ time.
We could train a neural network $f_\theta \colon \mathbb{R}^d \to \mathbb{R}^d$ with
$\\{\dots, (k_i, v_i), \dots\\}$ as the training set.
Then evaluating $f_\theta(q)$ would be a replacement for Eqn. 1 (Why?).



