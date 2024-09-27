# approximate attention with test time training

The attention operator is at the heart of the transformer architecture.
It takes as input
1. a query vector $q \in \mathbb{R}^d$, and,
2. a set of key-value vector pair $\\{\dots, (k_i, v_i), \dots\\}$,
$k_i, v_i \in \mathbb{R}^d$ (also called the KV-cache),

and returns a similarity weighted sum of values:
$$
a(q, \\{\dots, (k_i, v_i), \dots\\}) = \sum_i \langle q, k_i \rangle v_i \quad (1)
$$
where the similarity function is
$$
\langle q, k_i \rangle = \frac{\exp(q^\top k_i / \sqrt{d})}{\sum_j \exp(q^\top k_j / \sqrt{d})}
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



