# approximating attention heads with test time training

At the heart of the transformer architecture
is the attention head computation, which,
given key, value matrix $K, V \in \R^{T \times d}$
(or key-value cache)
and query vector $q \in \R^d$,
returns the attention-score weighted sum of the values.

$$
h(q, K, V) = \sum softmax(q^T k) v
$$

This operation takes \(O(T^2d)\) time.
Can we approximate it in time linear in \(T\)?

Let's break it down:
$$
a = q^T k  # attention score
\alpha = \exp(q^T k) / \sum \exp(q^T k)  # attention weight
h = \sum \alpha v  # output
$$

This is similar to a non-parameteric classifier or kernel method.

What if we replace it with a parameteric classifier.

Let's say I train a neural network $f_\theta(x)$
on the training set $\{(k_1, v_1), (k_2, v_2), ..., (k_t, v_t)\}$.
Then I can approximate the head with
$$
h = f_\theta(q)
$$.

That's it. Read the paper for more.
