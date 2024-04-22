---
title: LLM Acceleration Techniques--How Attention Fusion Works in FlashAttention
tags: LLM ModelAcceleration FlashAttention
---

## Analyzing Memory Access Complexity for Standard Attention
The attention calculation formula is expressed as:

$$
O = softmax(\frac{QK^\top}{\sqrt{d}})V
$$

Here, $Q, K, V\in \mathbb{R}^{N\times d}$ represent the query, key, and value matrices. In this context, $N$ corresponds to the length of the input sequence, $d$ denotes the head dimension, and $O$ signifies the output of the attention mechanism.

When adhering strictly to the provided formula for code implementation, the process unfolds as follows:

1. Compute $S = \frac{QK^\top}{\sqrt{d_k}}$. This step involves accessing Q, K from High Bandwidth Memory (HBM), performing the computation for S, and subsequently storing the result back in HBM. The memory access complexity is $\Theta(Nd + N^2)$, where $Nd$ represents the size of the Q, K matrices, and $N^2$ is the size of the S matrix.

2. Calculate $P = \text{softmax}(S)$. This stage necessitates reading S from HBM, carrying out the softmax computation for P, and then storing the outcome back in HBM. The memory access complexity is $\Theta(N^2)$.

3. Compute $O = PV$. This step involves accessing P, V from HBM, performing the computation for O, and subsequently storing the result back in HBM. The memory access complexity is $\Theta(Nd + N^2)$.

In summary, the overall memory access complexity for the standard attention calculation is $\Theta(Nd + N^2)$.

## Softmax Decomposition

Let's explore a vector $x\in \mathbb{R}^B$ of dimension $B$. We define $\exp(x)$ as

$$
\exp(x) = [e^{x_1}, e^{x_2}... , e^{x_B}]
$$

The formula for computing the softmax is then expressed as

$$
softmax(x) = \frac{\exp(x)}{\sum_{j=1}^B e^{x_j}}
$$
 
Additionally, introduce $f(x) = \exp(x)$ and $l(x) = \sum_j\exp(x_j)$. Consequently, softmax can be reformulated as

$$
softmax(x) = \frac{f(x)}{l(x)}
$$
 
Now, let's examine two vectors $x^{(1)}, x^{(2)}$, and their concatenated vector $x = [x^{(1)}, x^{(2)}]$. For simplicity, let $f_i = f(x^{(i)}), l_i = l(x^{(i)})$. The softmax of $x$ can be articulated as


$$
softmax(x) = \frac{[f_1, f_2]}{l_1+ l_2} = \left[\frac{f_1}{l_1} \times \frac{l_1}{l_1 + l_2} , \frac{f_2}{l_1 + l_2}\right]
$$

In essence, if we don't possess the complete vector $x$ initially and only have $x^{(1)}$, we can compute $f_1$ and $l_1$. Upon acquiring $x^{(2)}$, we can then calculate $f_2$ and $l_2$, adjusting the previously computed $\frac{f_1}{l_1}$ to derive the final $softmax(x)$.


We can extend this concept to matrices. Let $X^{(1)}, X^{(2)}$ be matrices and their concatenated matrix be $X = [X^{(1)}, X^{(2)}]$. We denote $f_i = \exp(X^{(i)})$ and $l_i = rowsum(\exp(X^{(i)}))$. Then,

$$
softmax(X^{(i)}) = \frac{f_i}{l_i}
$$

Here, the division operation is defined as each row of the matrix divided by the corresponding element of the vector, yielding a matrix. For $X$, we have:

$$
softmax(X) = \left[\frac{f_1}{l_1 + l_2}, \frac{f_2}{l_1+l_2}\right] = \left[\frac{f_1}{l_1} \odot \frac{l_1}{l_1+ l_2}, \frac{f_2}{l_1+l_2}\right]
$$

In this context, the $(\odot)$ symbol represents element-wise multiplication of each row of the matrix with the corresponding element of the vector.

## Fusion of Attention Operators

The decomposition technique delineated above does not provide significant utility for a straightforward softmax computation. However, it facilitates the fusion of softmax with matrix multiplication, thereby diminishing the IO complexity.

![](/resources/2024-01-08-flash_attn/flash_attn-fuse_attn.png)

### Binary Block Scenario

Consider the Attention computation process depicted in the preceding figure. For the sake of simplicity, we partition the `Q`, `K`, `V` matrices into two blocks, each of size $B\times d$. Initially, consider the computation process of the block. Load $Q_1, K_1 ,V_1$ into the shared memory (assuming the matrix block is sufficiently small to fit into sm), then compute $S\_{11}$, $P'\_{11}$, and $O'\_1$ sequentially (note that $P'\_{11} \ne P\_{11}, O'\_1 \ne O\_1$ are not the final results, hence we use light yellow in the figure to represent them), and write $O'\_1$ back to HBM.

$$
\begin{aligned}
S_{11} &= \frac{Q_1 K_1}{\sqrt{d}}\\
P'_{11} &= softmax(S_{11}) \\
O'_1 &= P'_{11}V_{1} 
\end{aligned}
$$

Analogous to the previous section, we define the following:

$$
f_{11} = \exp(S_{11}), \quad l_{11} = rowsum(\exp(S_{11}))
$$

Therefore, $O'_1$ can be reformulated as:

$$
O'_1 = \frac{f_{11}}{l_{11}} V_1
$$

Subsequently, consider the overall Attention calculation:

$$
\begin{aligned}
S &= \frac{Q K}{\sqrt{d}} \\
P &= softmax(S)\\
O &= P V
\end{aligned}
$$

Where $O = [O_1, O_2]^\top$, with $O_1 = P_{11} V_1 + P_{12}V_2$, and $P_{11}, P_{12}$ are derived from:

$$
[P_{11}, P_{12}] = softmax([S_{11}, S_{12}])
$$

We then define:

$$
\begin{aligned}
f_{11} &= \exp(S_{11}) \\
f_{12} &= \exp(S_{12}) \\
l_{11} &= rowsum(\exp(S_{11})) \\
l_{12} &= rowsum(\exp(S_{12}))
\end{aligned}
$$

Based on the derivation in the previous section, we can obtain:

$$
[P_{11}, P_{12}] = \left[\frac{f_{11}}{l_{11}}\odot \frac{l_{11}}{l_{11} + l_{12}}, \frac{f_{12}}{l_{11} + l_{12}} \right]
$$

Substituting $P_1, P_2$ into the calculation formula of $O_1$ yields:

$$
\begin{aligned}
O_1 &= \frac{f_{11}}{l_{11}}\odot \frac{l_{11}}{l_{11} + l_{12}} V_1 + \frac{f_{12}}{l_{11} + l_{12}} V_2 \\
&= \frac{f_{11}}{l_{11}}V_1\odot \frac{l_{11}}{l_{11} + l_{12}} + \frac{f_{12}}{l_{11} + l_{12}} V_2
\end{aligned} 
$$

Given our previous derivation of $O'\_1 = \frac{f\_{11}}{l\_{11}}V\_1$, we can derive the relationship between $O\_1$ and $O'\_1$:

$$
O_1 = O'_1 \odot \frac{l_{11}}{l_{11} + l_{12}} + \frac{f_{12}}{l_{11} + l_{12}} V_2
$$

Similarly, a relationship exists between $O_2$ and $O'_2$:

$$
O_2 = O'_2 \odot \frac{l_{21}}{l_{21} + l_{22}} + \frac{f_{22}}{l_{21} + l_{22}} V_2
$$

### Multi-Block Scenario

![](/resources/2024-01-08-flash_attn/flash_attn-attention.png)

Subsequently, we extrapolate the binary block derivation to accommodate a multi-block scenario. As depicted in the figure above, `Q, K, V` are partitioned into blocks $Q_{1...T}, K_{1...T}, V_{1...T}$, each block of size $B\times d$.

To compute all the `O` blocks, we employ a nested loop to traverse all the blocks of `Q,K,V`. The outer loop iterates over the blocks of `K, V`, while the inner loop traverses the blocks of `Q, O`. In the initial iteration of the outer loop, the inner loop sequentially calculates $O_1, O_2, ... O_T$ (note that these results are not the final outcomes).

In the second iteration of the outer loop, we can adjust according to the formula discussed in the previous section, namely,

$$
O_j := O_j \odot \frac{l_{j1}}{l_{j1} + l_{j2}} + \frac{f_{j2}}{l_{j1} + l_{j2}} V_2
$$

In the third iteration of the outer loop, we continue to adjust:

$$
O_j := O_j \odot \frac{l_{j2}}{l_{j2} + l_{j3}} + \frac{f_{j3}}{l_{j2} + l_{j3}} V_3
$$

Therefore, we can deduce that for the $i$ th iteration of the outer loop, the iteration format is:

$$
O_j := O_j \odot \frac{l_{j,i-1}}{l_{j,i-1} + l_{ji}} + \frac{f_{ji}}{l_{j,i-1} + l_{ji}} V_i
$$

## Examination of Memory Access in Attention Fusion Computation

Based on the preceding analysis, the fusion computation of Attention necessitates loading segments of the `Q, K, V` data from HBM into SM in batches. Assuming its size is `M`, the memory access complexity of each computation loaded from the `Q, K, V` matrix is $\Theta(M)$.

Examining the loop relationship, the number of outer loop iterations is $\Theta(\frac{Nd}{M})$, and after loading $K_i, V_i$ once, all $Q_j, O_j$ need to be traversed. Consequently, the volume of memory accessed in each outer loop is on the order of $\Theta(Nd)$, resulting in a total memory access of $\Theta(N^2d^2M^{-1})$.

Consider a typical scenario: suppose `N = 1024, d = 64, M = 100kb`, then the memory access of standard Attention under float16 data precision is `(1024 x 1024 + 1024 x 64) x 2 / 1024 kb = 2176kb`, while under the condition of block computation, this value is `(1024 x 1024 x 64 x 64) / (100 x 1024) x 2 / 1024 kb = 81.92kb`. Although this is a rudimentary calculation based on asymptotic complexity, it is evident that utilizing block computation can significantly reduce the number of memory accesses, thereby enhancing the arithmetic intensity of Attention. Given that Attention is memory-intensive on most hardware, meaning its arithmetic intensity is consistently on the left side of the Roofline model, enhancing the arithmetic intensity can directly improve the hardware utilization.

## Conclusion

This article scrutinizes the operator fusion process of Flash Attention, with a focus on the fusion computation method of mulmat and softmax, and elucidates from the perspective of memory access complexity why Flash Attention is faster.