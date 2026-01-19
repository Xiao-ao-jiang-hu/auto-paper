<!-- Page 1 -->
# Learning to Accelerate Approximate Methods for Solving Integer Programming via Early Fixing

Longkang Li, Baoyuan Wu, Member, IEEE

**Abstract**—Integer programming (IP) is an important but challenging problem. Approximate methods have shown promising performance on solving the IP problem. However, we observed that a large fraction of variables solved by some iterative approximate methods fluctuate around their final converged discrete states in very long iterations. It implies that these approximate methods could be significantly accelerated by early fixing these fluctuated variables to their converged states, while not significantly harming the solution quality. To this end, we propose an innovative framework of learning to early fix variables along with the approximate method. Specifically, we formulate the early fixing process as a Markov decision process, and train it using imitation learning, where a policy network evaluates the posterior probability of each free variable concerning its discrete candidate states in each block of iterations. Extensive experiments on three typical IP applications are conducted, including constrained linear programming, MRF energy minimization and sparse adversarial attack, covering moderate and large-scale IP problems. The results demonstrate that our method could not only significantly accelerate the previous approximate method up to over 10 times in most cases, but also produce similar or even better solutions. The implementation of our method is publicly available at https://github.com/SCLBD/Accelerated-Lpbox-ADMM.

**Index Terms**—Integer programming, learning to accelerate, early fixing, imitation learning.

---

## 1 INTRODUCTION

INTEGER programming (IP) is an important and challenging problem in many fields, such as machine learning [1] and computer vision [2]. IP can be a versatile modeling tool for discrete or combinatorial optimization problems with a wide variety of applications, and thus has attracted considerable interests from the theory and algorithm design communities over the years [3]. There are rich literature and a wide range of developed methods and theories for solving IP. Generally, we could divide them into two categories: exact methods and approximate methods. Some exact methods are widely utilized, such as branch-and-bound [4], cutting plane [5] and branch-and-cut [6] methods. The branch-and-bound method [4] is an approach that partitions the feasible solution space into smaller subsets of solutions. The cutting-plane method [5] is any of a variety of optimization methods that iteratively refine a feasible set or objective function by means of linear inequalities, termed *cuts*. The branch-and-cut [6] method combines branch-and-bound and the cutting plane method. These exact methods are able to get the optimal solutions, however, they are usually suffering from time-consuming issues due to the repeated solving of relaxed linear problems. Thereafter, in the recent years, more and more research focuses on the approximate methods, where a feasible solution is obtained within the limited time. Linear relaxation [7] relaxes the binary constraints $x \in \{0, 1\}$ to the box constraints $x \in [0, 1]$. Spectral relaxation [8] relaxes the binary constraint to the $\ell_2$-ball, leading to a non-convex constraint. As regard to the SDP relaxation [9], the binary constraints are substituted with a positive semi-definite matrix constraint, i.e., $\mathbf{X} \succcurlyeq 0$.

Besides the above mentioned relaxation-based approximate methods for integer programming, recently there has been increasing attention to another type of approximate methods, which is iterative and based on the alternating direction method of multipliers (ADMM) [10]. ADMM is a powerful algorithm for distributed convex optimization, with an attempt to blend the benefits of dual decomposition [11] and augmented Lagrangian methods [12]. ADMM coordinates the solutions of small local subproblems to find a solution of the large global problem. Many variants of ADMM have been proposed with different purposes of better acceleration, convergence, or stability, and have been applied to solving different types of optimization tasks. Bethe ADMM [13] was proposed for tree decomposition based parallel MAP inference, which used an inexact ADMM augmented with a Bethe-divergence based proximal function. Bregman ADMM [14] was then proposed, which provided a unified framework for ADMM. Bregman ADMM then has a number of variants such as generalized ADMM and inexact ADMM. Linearized ADMM [15] [16] was also proposed for convex optimization. One state-of-the-art ADMM method for solving IP is $\ell_p$-Box ADMM [17], where the binary constraints are equivalently replaced by the intersection of a box constraint and a $\ell_p$-norm sphere constraint.

Towards those ADMM based approximate methods, we observed that regarding the approximate methods a large fraction of variables fluctuated around their final converged states in very long iterations. In Fig. 1, we solve a constrained linear programming instance with 500 variables by $\ell_p$-box ADMM, which converges after 7827 iterations. The left figure illustrates the objective changes with respect to the iterations. And we can see that the convergence reaches after a long fluctuation. In the right figure, We introduce "Flip number"

---

- Longkang Li and Baoyuan Wu are with the School of Data Science, The Chinese University of Hong Kong, Shenzhen, China and also with Secure Computing Lab of Big Data, Shenzhen Research Institute of Big Data, Shenzhen, China.
- Correspondence to Baoyuan Wu (wubaoyuan@cuhk.edu.cn).

<!-- Page 2 -->
Fig. 1. A constrained linear programming instance with 500 variables is solved by $\ell_p$-box ADMM, converging after 7827 iterations. Left: We record how the objective changes with respect to the iterations. Right: a flip histogram. We use "Flip number" to evaluate the iterating stableness of the variable. For one variable, if the values of two adjacent iterations go across 0.5, we call it a 'Flip'. When converged, each variable gets its Flip number. We build the percentage histogram of all these 500 variables according to their Flip number, where the minimum is 0, the maximum is 450 Flips, and the horizontal axis has 5 Flips as an interval. The results show that 59.6% of variables have [0,5) Flips, among which 34.0% have 0 Flip. We also present 4 different variable iterating processes, corresponding to 0, 20, 90, 450 Flip(s).

to evaluate the iterating stableness of the variable. For one variable, if the values of two adjacent iterations go across 0.5, we call it a 'Flip'. For example, the variable value at $t$-th iteration is 0.9 and that at $t+1$-th iteration is 0.3, thus it is a Flip. Each variable corresponds to one Flip number, the smaller the Flip number, the more stable the variable iterating. According to the Flip histogram, 59.6% of variables have [0,5) Flips, among which 34% have 0 Flip. Most variables have small Flip numbers. To that end, we believe that a large proportion of variables are fluctuating around their final converged states (0 or 1) within small ranges. Currently, one solution for algorithmic acceleration is to early stop the iterations [18] [19]. However, early stopping has two shortcomings: 1). There is trade-off between the objective efficiency and the runtime effectiveness, when stopping much earlier the objective accuracy may decrease more. 2). It always use the whole set of variables to consider whether and when to stop.

Inspired by this observation, we were thinking: why not take every single variable independently and then asynchronously fix them instead of early stopping all of the variables at one iteration? To the end, we propose an early fixing framework, which aims to accelerate the approximate method by early fixing these fluctuated variables to their converged states while not significantly harming the converged performance. Fig. 2 shows the comparison between early stopping and early fixing. And there are mainly three differences: Firstly, early stopping does only consider the depth of optimization, i.e., the number of iterations, while early fixing also thinks over the width of optimization, i.e., the dimension of variables. Secondly, early stopping regards the set of variables as a whole, while early fixing treats every single variable independently. And thirdly, decisions on whether to early stop are made in every single iteration, while those for early fixing are once every block of iterations, i.e., $\beta$ iterations.

Under our proposed early fixing framework, in each block of iterations, given the iterative values of the variables within the past $\beta$ iterations, a policy network will evaluate the posterior probability of each variable concerning all discrete candidate states. If the posterior probability with respect to one state exceeds the fixing threshold, then the action of fixing this variable to that discrete state will be conducted, and this variable will not be updated in later iterations; otherwise, no fixing action will be conducted and this variable will be further updated. Specifically, for each variable, the continuously iterative values within the past $\beta$ iterations are sequential according to the time series. Recently, the Transformer structure [20] has exhibited powerful performance in the sequential networks though the multi-headed attention (MHA) mechanism. Herein, we incorporate the attention layers in our policy network. When solving a problem with early fixing framework, one block of iterations only decides a proportion of variables to conduct early fixing, thus the process is episodic until termination. We can regard the solving process as a Markov decision process [21] and train it using imitation learning [22]. Since the input of policy network only requires the iterative values of variables, without any other constraint information, thus our early fixing framework can be versatile enough, available to all the IP problems of any orders or types, no matter linear or quadratic, constrained or unconstrained.

In this paper, in order to accelerate the ADMM based approximate methods for solving the IP problems, especially improving the scalability of the IP problems, we propose the early fixing framework combined with learning techniques. The main contributions of this paper are four-fold:

(i) To the best of our knowledge, we are the first to propose an early fixing framework to accelerate the approximate methods, where the variables are treated independently with one another and we can asynchronously fix them according to their continuously iterative values within the past series of iterations. Once fixed, the variables will not be further updated. And free variables will continue iterating and updating.

(ii) We formulate the whole early fixing process when accelerating solving an IP problem as a Markov decision process, and train it using behaviour cloning as a method of imitation learning. We also incorporate the weighted binary cross-entropy (WBCE) loss during the training.

(iii) We adopt the learning techniques with the attention layers in our policy network, to decide whether to fix the variable or not.

(iv) We apply our proposed early fixing framework to three different IP applications: constrained linear programming, MRF energy minimization and sparse adversarial attack. The former one is linear IP problem, while the latter two are quadratic IP problems.

<!-- Page 3 -->


<!-- Page 4 -->
TABLE 1  
Summary of notations

| Notation | Meaning | Notation | Meaning |
|----------|---------|----------|---------|
| $n$ | number of variables, $n > 0$. | $m$ | number of constraints, $m \geq 0$. |
| $t$ | iteration index, $t \in \{0, ..., T-1\}$. | $i$ | variable index, $i \in \{0, ..., n-1\}$. |
| $T$ | maximum iteration without early fixing, $T > 0$. | $T'$ | maximum iteration with early fixing, $T' > 0$. |
| $\delta$ | fixing threshold, deciding whether to fix, $\delta \in [0.5, 1]$. | $\beta$ | block size, denoting one block of iterations, $\beta > 1$. |
| $f(\cdot)$ | objective function, linear or quadratic. | $\mathcal{C}$ | set of constraints, $\mathcal{C} \in \mathbb{R}^m$. |
| $\mathbf{x}$ | set of variables, $\mathbf{x} \in \mathbb{R}^n$. | $\mathbf{A}$ | matrix in objective function, $\mathbf{A} \in \mathbb{R}^{n \times n}$. |
| $\mathbf{b}$ | vector in objective function, $\mathbf{b} \in \mathbb{R}^n$. | $\mathbf{C}$ | matrix in constraint set, $\mathbf{C} \in \mathbb{R}^{m \times n}$. |
| $\mathbf{d}$ | vector in constraint set, $\mathbf{d} \in \mathbb{R}^m$. | $\otimes$ | any relational symbol, $<, >, =, \geq$ or $\leq$. |
| $\pi(\cdot)$ | policy network for early fixing. | $\theta$ | weights of policy network. |
| $y, \mathbf{y}$ | $\mathbf{y}$ is the iterative values of variable $y$, $\mathbf{y} \in \mathbb{R}^{\beta \times 1}$ | $z$ | iteration embedding of one variable, $z \in \mathbb{R}^{\alpha \times d_h}$. |
| $\alpha$ | node number for iteration embedding, $\alpha > 1$. | $\hat{z}$ | iteration embedding with positional encoding, $\hat{z} \in \mathbb{R}^{\alpha \times 2d_h}$. |
| $d_h$ | node dimension, $d_h \geq 1$. | $d_n$ | hidden dimension, $d_n = 128$. |
| $k$ | position index, $k \in \{1, ..., \alpha\}$. | $j$ | dimension index for positional encoding, $2j \leq d_h$. |
| $\hat{z}_k$ | node input in attention layers, $\hat{z}_k \in \mathbb{R}^{2d_h}$ | $h_k, \hat{h}_k$ | node embeddings in attention layers, $h_k, \hat{h}_k \in \mathbb{R}^{d_n}$. |
| $\mathbb{L}$ | number of layers, $\mathbb{L} > 1$. | $\ell$ | layer index in attention layers, $\ell \in \{1, ..., \mathbb{L}\}$. |
| $H$ | number of heads in MHA sublayer, $H = 8$. | $d_{n'}$ | hidden dimensions in FF sublayer, $d_{n'} = 512$. |
| $\overline{z}$ | concatenated embedding, $\overline{z} \in \mathbb{R}^{(\alpha \cdot d_n) \times 1}$. | $p$ | the probability vector, each element $p_i \in [0, 1]$. |
| $u$ | number of free variables, $u > 0, u \leq n$. | $v$ | number of fixed variables, $v > 0, v \leq n$. |
| $r$ | number of rounds for conducting early fixing, $r > 1$. | $\mathcal{M}$ | the approximate method to be accelerated. |
| $\gamma$ | number of blocks, used in the network training, $\gamma > 1$. | $\mathcal{N}$ | number of training instances, $\mathcal{N} > 1$. |
| $e$ | instance index, $e \in \{0, ..., N-1\}$. | $q$ | one element of loss, a scalar. |
| $w$ | weight for training, a scalar. | $\mathcal{J}(\cdot)$ | the binary cross entropy loss for network training. |
| $\mathcal{I}$ | one instance as Formulation (1). | $N'$ | number of instances for inference, $N' > 1$. |
| $\xi$ | a constant for linear programming. | $\epsilon$ | perturbation for sparse adversarial attack. |
| $\zeta$ | the vector of perturbation magnitudes. | $\eta$ | the vector of perturbed positions. |

a great deal of attention. Khalil et al. [28] took the first step towards statistical learning of branching rules in BB. Alvarez et al. [29] and Gasse et al. [30] learn a branching rule offline on a collection of similar instances, and the branching policy is learned by the imitation of the strong branching expert. Graph Neural Network (GNN) approach for learning to branch has successfully reduced the runtime [30]. Gupta et al. [31] consider the availability of expensive GPU resources for training and inference, thus devise an alternate computationally inexpensive CPU-based model that retains the predictive power of the GNN architecture. Tang et al. [32] utilize reinforcement learning for learning to cut. Recently there is one set of approaches focusing on directly learning the mapping from an IP to its approximate optimal solution, instead of solving the IP by any exact or approximate solvers. Vinyals et al. [33] introduce the pointer network as a model that uses attention to output a permutation of the input, and train this model offline to solve the TSP problem. Bello et al. [34] introduce an Actor-Critic algorithm to train the pointer network without supervised solutions. Kool et al. [35] propose a model based on attention layers [20] to solve the routing problems. Andrychowicz et al. [36] and Li et al. [37] propose learning to optimize or learning to learn, casting an optimization problem as a learning problem. Nowak et al. [38] train a Graph Neural Network in a supervised manner to directly output a tour as an adjacency matrix, which is converted into a feasible solution by a beam search.

Most of the above mentioned papers use the networks to entirely substitute with the optimizer, different from that, in this paper we simply utilize the network to accelerate the optimization, just like an attachment module.

## Imitation learning

Imitation learning (IL) techniques aim to mimic the behaviour from an expert or teacher in a given task [39]. IL and reinforcement learning (RL) both work for the Markov decision processes (MDP). RL tends to have the agent learn from scratch through its exploration with a specified reward function, however, agent of IL does not receive task reward but learn by observing and mimicking [40]. Similar to traditional supervised learning (SL) where the samples represent pairs of features and ground-truth labels, IL has the samples demonstrating pairs of states and actions. One fundamental difference between SL and IL is that: SL follows the assumption that the training and test data are independent and identically distributed (IID), while those of IL are Non-IID where the current state is only correlated to the previous state. Broadly speaking, research in the IL can be split into two main categories: behavioral cloning (BC) [22], and inverse reinforcement learning (IRL) [41]. In this paper, we choose BC for the training.

## Early exiting and early stopping

The term "early exiting" originally comes from computer vision and image recognition, which is mainly aimed at improving the computation efficiency based on specific architectures during the inference phase [18] [19]. As networks continue to get deeper and larger, these costs become more prohibitive for real-time applications. To address the issue, the proposed architecture exits the network early via additional branch classifiers with high confidence [42]. Early stopping is a form of regularization used to avoid overfitting when training a learner with an iterative method, such as gradient descent [43], [44]. When a certain criterion is satisfied, early stopping will be conducted before the ultimate convergence. Kaya et al. [45] proposes to avoiding “over-thinking” by early stopping,

<!-- Page 5 -->
where the deep neural networks can reach correct predictions before the final layers to save the running time. In optimal control literature, optimal stopping [46] is a problem of choosing a time to take a given action based on sequentially observed random variables in order to maximize an expected payoff. Optimal stopping can be seen as a special case of early stopping. Becker et al. [47] and Chen et al. [48] use deep reinforcement learning to learn the optimal stopping policy. Besides, Chen et al. [48] provide a variational Bayes perspective to combine learning to predict with learning to stop. In this paper, we propose a novel early fixing framework for accelerating solving the generic IP problems. Whether early exiting or early stopping, the models focus on the depth of iterations, while our proposed early fixing also considers the width of variable dimensions.

**Fix-and-optimize** Fix-and-optimize [49] is a metaheuristic, firstly proposed by Gintner et al. for solving mixed integer linear programming (MIP), which iteratively decomposes a problem into smaller subproblems. In each iteration, a decomposition process is applied aiming at fixing most of the decision variables at their value in the current solution. Since the resulting subproblem is composed only by a small group of free variables to be optimized, each subproblem can be solved fairly quickly by a MIP solver, when compared with the full model. The solution obtained in each iteration becomes the current solution when it improves the objective value. In further iterations, a different group of variables is selected to be optimized. This process is repeated until a termination condition is satisfied. The fix-and-optimize algorithm has wide applications in lot sizing problem [50], timetabling problem [51], and etc. Different from the fix-and-optimize algorithm where the fixed variables in previous iteration will be released in the next iteration, our early fixing framework requires that the fixed variables will stay fixed and not appear in the following iterations.

**Variable fixing** The strategy of fixing variables [52] within optimization algorithms often proves useful for enhancing the performance of methods for solving constraint satisfaction and optimization problems. Such a strategy has come to be one of the basic strategies associated with Tabu search [53]. Two of the most important features in Tabu search are how to score the variables (variable scoring) and which variables should be fixed (variable fixing). Similar to fix-and-optimize algorithm, the fixed variables in Tabu search could possibly be freed, while those in our early fixing framework will keep fixed and not released in the next iterations.

## 3 BACKGROUND

### 3.1 Problem definition

Throughout this paper, we focus on the problem of generic IP problems, which can be generally formulated as a binary mathematical optimization problem as follows:

$$
\arg \max_{\mathbf{x}} f(\mathbf{x}), \quad s.t. \quad \mathbf{x} \in \mathcal{C}, \quad \mathbf{x} \in \{0,1\}^n,
\tag{1}
$$

where $\mathcal{C} \in \mathbb{R}^m$ is the set of constraints, and $\mathbf{x} \in \mathbb{R}^n$ is the set of binary variables. $n, m$ denotes the number of variables and constraints, respectively. In this paper, we mainly focus on the linear and quadratic IP problems.

<!-- Page 6 -->


<!-- Page 7 -->
which are sequential according to the time series. At this point, the dimension is $\beta \times 1$. Inspired by the sequential model [54], we use the sliding window to convert the $\boldsymbol{y} \in \mathbb{R}^{\beta \times 1}$ to $\boldsymbol{z} \in \mathbb{R}^{\alpha \times d_h}$, where $\boldsymbol{z}$ is the iteration embedding of $\boldsymbol{y}$, $\alpha$ denotes the node number, and $d_h$ is the node dimension.

**Positional encoding** There is no recurrence and no convolution in the attention models, so we need positional encoding. Before feeding $\boldsymbol{z}$ into the attention layers, we inject some information about the relative position of the tokens in the sequence of $\boldsymbol{z}$, so as to make the most use of the order of the sequence. We follow the setup in Vaswani et al. [20] and add the positional encodings to the iteration embeddings. Let $k$ be the node number, $d_h$ be the embedding dimension. Then the positional encodings have the same dimensions $d_h$ as the embeddings. We use sine and cosine functions of different frequencies:

$$
\begin{cases}
PE_{(k,2j)} &= \sin\left(k/10000^{2j/d_h}\right) \\
PE_{(k,2j+1)} &= \cos\left(k/10000^{2j/d_h}\right),
\end{cases}
\tag{2}
$$

where $k$ is the position, $k \in \{1, ..., \alpha\}$ and $j$ is the dimension, $2j \leq d_h$. The wavelengths form a geometric progression from $2\pi$ to $10000 \times 2\pi$. After adding the positional encoding, the iteration embedding turns from $\boldsymbol{z} \in \mathbb{R}^{\alpha \times d_h}$ to $\hat{\boldsymbol{z}} \in \mathbb{R}^{\alpha \times 2d_h}$.

**Attention layers** Then we apply the encoder part of Transformer-alike attention architecture [35] to our network to extract better iteration embeddings. First of all, to make it consistent with the dimensions, through a learned linear projection, one node input $\hat{\boldsymbol{z}}_k$ is projected to one node embedding $\boldsymbol{h}_k$, where $\hat{\boldsymbol{z}}_k \in \mathbb{R}^{2d_h}$ and $\boldsymbol{h}_k \in \mathbb{R}^{d_n}$, and the dimension $d_n=128$. Then all the node embeddings go through $\mathbb{L}$ attention layers. Each layer consists of two sublayers: a multi-head attention (MHA) layer that executes message passing between the nodes, and a node-wise fully connected feed-forward (FF) layer. Each sublayer adds a skip-connection [55] and a batch normalization (BN) [56]:

$$
\begin{cases}
\hat{\boldsymbol{h}}_k^{(\ell)} = \text{BN}^\ell \left( \hat{\boldsymbol{h}}_k^{(\ell-1)} + \text{MHA}_k^\ell \left( \hat{\boldsymbol{h}}_1^{(\ell-1)}, ..., \hat{\boldsymbol{h}}_\alpha^{(\ell-1)} \right) \right) \\
\boldsymbol{h}_k^{(\ell)} = \text{BN}^\ell \left( \hat{\boldsymbol{h}}_k^{(\ell)} + \text{FF}^\ell (\hat{\boldsymbol{h}}_k^{(\ell)}) \right).
\end{cases}
\tag{3}
$$

Any two layers do not share their parameters. Layer index $\ell \in \{1, .., \mathbb{L}\}$, node index $k \in \{1, ..., \alpha\}$. The MHA sublayer uses $H = 8$ heads, each with dimension $\frac{d_n}{H} = 16$. Moreover, the fully connected FF sublayer, which is applied to each node embedding separately and identically, consists of two linear transformations with a ReLU activation function in between: it first maps the node embedding from dimension $d_n$ to hidden dimension $d_{n'} = 512$, then transforms from $d_{n'}$ back to $d_n$. After $\mathbb{L}$ attentions layers, we get $\alpha$ node embeddings, each with dimension $d_n$. At this point, we do the concatenation and get embedded variable $\bar{\boldsymbol{z}}$ with dimension $(\alpha \cdot d_n) \times 1$.

**MLP layers** After the attention layers, the variable $\boldsymbol{y}$ obtains a new embedding $\bar{\boldsymbol{z}} \in \mathbb{R}^{(\alpha \cdot d_n) \times 1}$, which will be fed into another multi-layer perceptron (MLP). At this stage, we utilize three fully connected FF sublayers along with decreasing dimensions ($256 - 128 - 16$) with the ReLU activation functions between the hidden layers. Finally, in the last layer through a sigmoid function, we obtain a probability $p_y \in [0, 1]$, determining whether to fix the input variable $y$ or not.

If the probability is greater than the fixing threshold $\delta$, then the action of fixing this variable to 1 will be conducted. we can interpret this fixing threshold as a symmetric fixing confidence, *i.e.*, if the probability is less than $1 - \delta$, then the action of fixing this variable to 0 will be conducted. Otherwise, no fixing action will be conducted and this variable will be further updated. The early fixing process is given in the Algorithm 1.

## 4.3 Training: imitation learning

We train the policy network $\pi(\theta)$ as shown in Fig. 5. Specifically, our policy network is trained by behavioral cloning [22] as a method of expert-driven imitation learning, and here we use the approximate method to be accelerated $\mathcal{M}$ itself as the expert rule. Subsection 4.1 has explained the meaning for states and actions. Then mathematically, for those free variables, we assign $\boldsymbol{s}$ as the past $\beta$ iterative values, and assign $\boldsymbol{a}^*$ as the ultimately converged discrete solutions of the approximate method $\mathcal{M}$, *i.e.*, the expert solutions. We first run the expert on a collection of $N$ training instances, and pick the dataset of expert state-action pairs $\mathcal{D} = \{(\boldsymbol{s}_{e,r,i}, \boldsymbol{a}_{e,r,i}^*) |_{i=0}^{n-1} |_{r=0}^{\gamma-1} |_{e=0}^{N-1}\}$. And the policy is learned

<!-- Page 8 -->
by minimizing the weighted binary cross-entropy (WBCE) loss:

$$
\mathcal{J}(\theta) = -\frac{1}{N \cdot \gamma \cdot n} \sum_{e=0}^{N-1} \sum_{r=0}^{\gamma-1} \sum_{i=0}^{n-1} w_{e,r,i} q_{e,r,i},
\tag{4}
$$

$$
q_{e,r,i} = a^* \log \pi_\theta(a|s) + (1-a^*) \log(1-\pi_\theta(a|s)),
\tag{5}
$$

where for $a^*, a, s$, we hide the subscripts $_{e,r,i}$ for readability. $w_{e,r,i}$ is the weight, $w_{e,r,i} = \frac{1}{r+1}$. Mathematically, $s_{e,r,i} = \boldsymbol{x}_{(r-1)\beta:r\beta}^{(r,e)}$, $a_{e,r,i}^* = x_{T-1}^{(r,e)}$. We call one problem shown as in Formulation (1), as one instance. $N$ is the number of training instances.

## 4.4 Inference with early fixing

The general inference stage of early fixing framework is given in the Algorithm 1, and we also present the process as a MDP in Fig. 3. Our early fixing framework takes each variable independently, and decisions on whether to early fix are once every block of iterations. In each block of iterations, given the iterative values of the variables within the past $\beta$ iterations, the policy network will evaluate the posterior probability of each variable concerning each discrete candidate states (0 or 1). If the posterior probability with respect to one state exceeds a threshold, namely the fixing threshold $\delta$, then the action of fixing this variable to that discrete state will be conducted, and this variable will not be updated in later iterations; otherwise, no fixing action will be conducted and this variable will be further updated.

When a certain number of variables are fixed in previous iterations, then how to update the problem into a smaller-sized one will be discussed in this subsection. Our early fixing framework is available for both linear programming and quadratic programming, no matter constrained or unconstrained. We will give the mathematical assumptions and propositions based on a constrained quadratic programming problem:

$$
\arg\max_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x} + \mathbf{b}^\top \mathbf{x}, \quad \text{s.t. } \mathbf{C} \mathbf{x} \otimes \mathbf{d}, \quad \mathbf{x} \in \{0,1\}^n.
\tag{6}
$$

**Notations.** We denote the matrices and vectors in formulation (6) as: $\mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & \mathbf{A}_2 \\ \mathbf{A}_3 & \mathbf{A}_4 \end{bmatrix}$, $\mathbf{C} = \begin{bmatrix} \mathbf{C}_1 & \mathbf{C}_2 \end{bmatrix}$, $\mathbf{b} = \begin{bmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \end{bmatrix}$, $\mathbf{x} = \begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{bmatrix}$, where vectors $\mathbf{x}_1$ refers to the set of free variables and $\mathbf{x}_2$ refers to the set of fixed variables, and the same around for other vectors and matrices. Let $u, v$ be the number of free and fixed variables, then $\mathbf{b}_1, \mathbf{x}_1 \in \mathbb{R}^u$, $\mathbf{b}_2, \mathbf{x}_2 \in \mathbb{R}^v$, $\mathbf{A}_1 \in \mathbb{R}^{u \times u}$, $\mathbf{A}_2 \in \mathbb{R}^{u \times v}$, $\mathbf{A}_3 \in \mathbb{R}^{v \times u}$, $\mathbf{A}_4 \in \mathbb{R}^{v \times v}$, $\mathbf{C}_1 \in \mathbb{R}^{m \times u}$, $\mathbf{C}_2 \in \mathbb{R}^{m \times v}$, $\otimes$ denotes any relational symbol such as $<, >, =, \ge$ or $\le$.

**Proposition 1.** Problem reformulation: when doing the early fixing once every $\beta$ iterations, let $r$ be the rounds for conducting early fixing, then we can propose that:

- $\mathbf{x}^{(r+1)} = \mathbf{x}_1^{(r)}$
- $\mathbf{A}^{(r+1)} = \mathbf{A}_1^{(r)}$
- $\mathbf{b}^{(r+1)} = (\mathbf{A}_2^{(r)} + \mathbf{A}_3^{\top(r)}) \mathbf{x}_2^{(r)} + \mathbf{b}_1^{(r)}$
- $\mathbf{C}^{(r+1)} = \mathbf{C}_1^{(r)}$
- $\mathbf{d}^{(r+1)} = \mathbf{d}^{(r)} - \mathbf{C}_2^{(r)} \mathbf{x}_2^{(r)}$

---

### Algorithm 1 Inference: Early Fixing Framework

**Input:** accelerated approximate method $M$, instance $\mathcal{I}$, policy network $\pi_\theta$, total variable number $n$, block size $\beta$, fixing threshold $\delta \in [0.5, 1]$

**Output:** $\mathbf{x}^*$

1: $u \leftarrow n, v \leftarrow 0, t \leftarrow \beta$

2: **repeat**

3: $\quad$ **if** $t \% \beta == 0$ **then**

4: $\quad\quad$ $\mathbf{x}_{t-\beta:t} \leftarrow \mathcal{M}(\mathcal{I})$

5: $\quad\quad$ $\mathbf{s}_t \leftarrow \mathbf{x}_{t-\beta:t}$

6: $\quad\quad$ $\mathbf{p} \leftarrow \pi_\theta(\mathbf{a}_t | \mathbf{s}_t)$

7: $\quad\quad$ **for** $i = 1$ to $u$ **do**

8: $\quad\quad\quad$ **if** $p_i > \delta$ **then**

9: $\quad\quad\quad\quad$ Early fix variable $x_i$ to 1, $v \leftarrow v+1$

10: $\quad\quad\quad$ **end if**

11: $\quad\quad\quad$ **if** $p_i < 1-\delta$ **then**

12: $\quad\quad\quad\quad$ Early fix variable $x_i$ to 0, $v \leftarrow v+1$

13: $\quad\quad\quad$ **end if**

14: $\quad\quad$ **end for**

15: $\quad\quad$ Update the instance $\mathcal{I}$ as Subsection 4.4.

16: $\quad\quad$ $u \leftarrow u-v, v \leftarrow 0, t \leftarrow t+\beta$

17: $\quad$ **end if**

18: **until** (converged or $u \le 0$)

19: Record all the fixed/converged discrete solutions $\mathbf{x}^*$

20: **return** $\mathbf{x}^*$

---

### Proof 1. We eliminate the superscript $(r)$ for readability.

(i) For the objective function:

$$
\mathbf{x}^\top \mathbf{A} \mathbf{x} + \mathbf{b}^\top \mathbf{x}
\tag{7}
$$

$$
= \begin{bmatrix} \mathbf{x}_1^\top & \mathbf{x}_2^\top \end{bmatrix} \begin{bmatrix} \mathbf{A}_1 & \mathbf{A}_2 \\ \mathbf{A}_3 & \mathbf{A}_4 \end{bmatrix} \begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{bmatrix} + \begin{bmatrix} \mathbf{b}_1^\top & \mathbf{b}_2^\top \end{bmatrix} \begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{bmatrix}
\tag{8}
$$

$$
= \mathbf{x}_1^\top \mathbf{A}_1 \mathbf{x}_1 + \mathbf{x}_2^\top \mathbf{A}_3 \mathbf{x}_1 + \mathbf{x}_1^\top \mathbf{A}_2 \mathbf{x}_2 \\
+ \mathbf{x}_2^\top \mathbf{A}_4 \mathbf{x}_2 + \mathbf{b}_1^\top \mathbf{x}_1 + \mathbf{b}_2^\top \mathbf{x}_2
\tag{9}
$$

$$
= \mathbf{x}_1^\top \mathbf{A}_1 \mathbf{x}_1 + ((\mathbf{A}_2 + \mathbf{A}_3^\top) \mathbf{x}_2 + \mathbf{b}_1)^\top \mathbf{x}_1 \\
+ (\mathbf{x}_2^\top \mathbf{A}_4 \mathbf{x}_2 + \mathbf{b}_2^\top \mathbf{x}_2),
\tag{10}
$$

thus: $\mathbf{x}^{(r+1)} = \mathbf{x}_1^{(r)}$, $\mathbf{A}^{(r+1)} = \mathbf{A}_1^{(r)}$, $\mathbf{b}^{(r+1)} = (\mathbf{A}_2^{(r)} + \mathbf{A}_3^{\top(r)}) \mathbf{x}_2^{(r)} + \mathbf{b}_1^{(r)}$. Since previously fixed variables can be seen as the constants in the following iterations, then $(\mathbf{x}_2^\top \mathbf{A}_4 \mathbf{x}_2 + \mathbf{b}_2^\top \mathbf{x}_2)$ is constant.

(ii) For the constraints:

$$
\mathbf{C} \mathbf{x} \otimes \mathbf{d}
\tag{11}
$$

$$
\begin{bmatrix} \mathbf{C}_1 & \mathbf{C}_2 \end{bmatrix} \begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{bmatrix} \otimes \mathbf{d}
\tag{12}
$$

$$
\mathbf{C}_1 \mathbf{x}_1 \otimes \mathbf{d} - \mathbf{C}_2 \mathbf{x}_2
\tag{13}
$$

thus: $\mathbf{C}^{(r+1)} = \mathbf{C}_1^{(r)}$, $\mathbf{d}^{(r+1)} = \mathbf{d}^{(r)} - \mathbf{C}_2^{(r)} \mathbf{x}_2^{(r)}$.

Proof ends.

---

### Remark 1. As regard to the updating for $\mathbf{b}$, if $\mathbf{A}$ is a symmetric matrix, then $\mathbf{A}_2 = \mathbf{A}_3^\top$. At that point, the updating can be simplified as: $\mathbf{b}^{(r+1)} = 2 \mathbf{A}_2^{(r)} \mathbf{x}_2^{(r)} + \mathbf{b}_1^{(r)}$.

<!-- Page 9 -->
TABLE 2  
Performance evaluations for constrained linear programming on generated regular-sized Dataset I. The time limit is set to 1 hour.

| Size → | $n = 500$ |  |  | $n = 1000$ |  |  | $n = 1500$ |  |  | $n = 4000$ |  |  |
|--------|-----------|-----------|----------|------------|-----------|----------|------------|-----------|----------|------------|-----------|----------|
| Model | Obj.↑ | Obj. Gap↓ | Time↓ | Obj.↑ | Obj. Gap↓ | Time↓ | Obj.↑ | Obj. Gap↓ | Time↓ | Obj.↑ | Obj. Gap↓ | Time↓ |
| RPB [57] | 7464.8 | N/A | 2.79s | 14888 | N/A | 23.17s | 21231 | N/A | 169.16s | 59772 | N/A | 3600s |
| FiLM [31] | 7464.8 | N/A | 2.08s | 14888 | N/A | 19.78s | 21231 | N/A | 185.79s | 58809 | N/A | 3600s |
| GCNN [30] | 7464.8 | N/A | 1.78s | 14888 | N/A | 15.22s | 21231 | N/A | 144.74s | 60105 | N/A | 3600s |
| $\ell_p$-box ADMM [17] | **6953.7** | — | 1.17s | **14430** | — | 2.54s | **20930** | — | 4.04s | **56188** | — | 11.15s |
| $\ell_p$-box ADMM + LEF(w/o Att.) | 6804.9 | 0.69% | 0.18s | 13768 | 4.10% | 0.38s | 20415 | 2.40% | 0.43s | 53195 | 5.28% | 1.39s |
| $\ell_p$-box ADMM + LEF(with Att.) | 6883.5 | 1.96% | 0.27s | 13803 | 3.96% | 0.44s | 20500 | 1.85% | 0.56s | 54020 | 3.81% | 2.47s |

---

TABLE 3  
Performance evaluations for constrained linear programming on generated large-sized Dataset II. A negative objective gap means a better objective.

| Size → | $n = 1e4$ |  |  |  |  |  | $n = 5e4$ |  |  |  |  |  |
|--------|-----------|-----------|----------|-----------|-----------|----------|-----------|-----------|----------|-----------|-----------|----------|
| Model | Obj.↑ | Obj. Gap↓ | Time↓ | Speedup↑ | #Sol. Diff.↓ | Accuracy↑ | Obj.↑ | Obj. Gap↓ | Time↓ | Speedup↑ | #Sol. Diff.↓ | Accuracy↑ |
| $\ell_p$-box ADMM [17] | 9665.8 | N/A | 11.4s | N/A | N/A | N/A | 48372 | N/A | 111.5s | N/A | N/A | N/A |
| $\ell_p$-box ADMM + LEF(w/o Att.) | 9691.4 | -0.25% | 0.9s | 12.6× | 10.9 | 99.8910% | 48445 | -0.15% | 5.4s | 20.6× | 66.0 | 99.8680% |
| $\ell_p$-box ADMM + LEF(with Att.) | **9691.6** | -0.26% | 0.9s | 12.6× | 9.1 | 99.9090% | **48465** | -0.19% | 5.7s | 19.6× | 55.0 | 99.8900% |

| Size → | $n = 1e5$ |  |  |  |  |  | $n = 2e5$ |  |  |  |  |  |
|--------|-----------|-----------|----------|-----------|-----------|----------|-----------|-----------|----------|-----------|-----------|----------|
| Model | Obj.↑ | Obj. Gap↓ | Time↓ | Speedup↑ | #Sol. Diff.↓ | Accuracy↑ | Obj.↑ | Obj. Gap↓ | Time↓ | Speedup↑ | #Sol. Diff.↓ | Accuracy↑ |
| $\ell_p$-box ADMM [17] | 97579 | N/A | 327.3s | N/A | N/A | N/A | 195445 | N/A | 990.0s | N/A | N/A | N/A |
| $\ell_p$-box ADMM + LEF(w/o Att.) | 97631 | -0.05% | 13.8s | 23.7× | 129.8 | 99.8702% | 195710 | -0.14% | 39.4s | 25.1× | 247.1 | 99.8765% |
| $\ell_p$-box ADMM + LEF(with Att.) | **97682** | -0.11% | 14.4s | 22.7× | 126.8 | 99.8732% | **195758** | -0.16% | 41.3s | 24.0× | 220.2 | 99.8899% |

---

## 5 CONSTRAINED LINEAR PROGRAMMING

### 5.1 Setup

**Accelerated method and datasets** In this section, we will accelerate the approximate method: $\ell_p$-box ADMM [17] for solving constrained linear integer programming. For the datasets, we choose the combinatorial auction problems [30] with the following formulation:

$$
\arg\max_{\mathbf{x}} \mathbf{b}^\top \mathbf{x}, \quad s.t. \quad \mathbf{C}\mathbf{x} \leq \mathbf{d}, \quad \mathbf{x} \in \{0,1\}^n,
\tag{14}
$$

where $\mathbf{C} \in \mathbb{R}^{m \times n}$, $m,n$ refer to the number of constraints and variables. We follow the experimental setup of Gasse et al. [30], and generate two sets of instances in difference sizes.

$$
\mathbf{b}, \mathbf{C}, \mathbf{d} \leftarrow \mathbb{G}(\text{bid}=n, \text{item}=\xi \cdot m).
\tag{15}
$$

We generate the instances, namely, $\mathbf{b}, \mathbf{C}, \mathbf{d}$, according to the generator $\mathbb{G}$. The instance size is determined by two key parameters, bid and item. The bid number is equivalent to variable number $n$, while the item number is not equivalent but directly proportional to the constraint number $m$. $\xi$ is a constant. In order to ensure the feasibility and optimality of the instance, the constraint number $m$ could be different for different instances, given the same item number. We generate two sets of instances. For datasets I, there are regular-sized instances: (bid=500, item=100), (bid=1000, item=200), (bid=1500, item=300), (bid=4000, item=800). For datasets II, there are extra large scale instances: (bid=1e4, item=100), (bid=5e4, item=500), (bid=1e5, item=1000), (bid=2e5, item=2000).

**Training details** We train our model only on the smallest sized instances with $n=500$, and generalize to all other larger-sized instances. We train on $N=100$ instances, and do the inference on $N'=20$ instances for all the datasets. $\gamma, \beta, \delta, \mathbb{L}$ are set to 10, 100, 0.9, 2, respectively. We train for 10 epochs. The learning rate is set to $1e{-4}$. For our learning-based early fixing methods, training without attention for one epoch costs 77s, while training with attention costs 89s. We implement the functions of the accelerated methods in C++ and call the functions in Python via Cython interfaces. All the learning modules are implemented in Python.

**Evaluations** We evaluate all the different sizes, each with 20 instances. We set the time limit to 1 hour. For all datasets, we evaluate the objective, the objective gap and runtime, and we record the mean value of all instances. The objective gap is used to exhibit the gap between the $\ell_p$-box ADMM with learning-based early fixing (LEF) and the $\ell_p$-box ADMM without early fixing, given as: $\frac{obj_1 - obj_2}{obj_1}$, where $obj_1, obj_2$ refer to the objective obtained by $\ell_p$-box ADMM, $\ell_p$-box ADMM + LEF, respectively. A negative objective gap means achieving a better objective with early fixing. Specifically, for datasets II, we also evaluate the Speedup, the number of solution difference (#Sol. Diff.) and the accuracy. The speedup is the time speedup, simply dividing the runtime of $\ell_p$-box ADMM by that of $\ell_p$-box ADMM + LEF. #Sol. Diff is the number of solution difference where the variable solution by $\ell_p$-box ADMM is different from that by $\ell_p$-box ADMM + LEF, the maximum number should be the total variable number $n$. The accuracy is to evaluate the correct solution accuracy, given as: $\frac{n - n_d}{n} \times 100\%$, $n_d$ refers to (#Sol. Diff.). In the hyperparameter study, we also evaluate the number of infeasible constraints.

**Baselines** For datasets I, we compare against three exact methods based on the branch-and-bound algorithm. Then for Datasets II, since the size is too large for them to obtain a solution, we only compare with the approximate methods. Reliability pseudocost branching (RPB) [57] is a variant of hybrid branching which is used by default in SCIP, and we choose SCIP 6.0.1 [58] as the backend solver. Graph convolutional neural networks (GCNN) [30] are applied to learning branch-and-bound variable selection policies. Feature-wise Linear Modulation (FiLM) [31] [59] layers are used to construct the neural network for learning to branch which is purely CPU-based, but shows competitive performances against GPU-based neural networks.

<!-- Page 10 -->
Fig. 6. Convergence on different methods for constrained linear programming: how the objective changes with respect to the iterations during the inference stages for four different sized instances. LEF method leads to much faster convergence, and LEF with attention layers achieves better objective.

Fig. 7. Hyperparameter study on fixing threshold $\delta$ and ablation study on weighted loss: (a) How the number of infeasible constraints go with the fixing threshold. Different color refers to different fixing threshold. (b) How the objective gaps and runtime speedup of different sizes change when training with weighted loss or no weighted loss. No weighted loss means that all weights are equal to 1.

## 5.2 Experimental results

**Comparative study** As shown in Table 2 and Table 3, we firstly compare against three exact methods: RPB, FiLM and GCNN. GCNN generally outperforms RPB and FiLM with higher efficiency in runtime. The three exact methods have one obvious shortcoming in common: time-consuming, especially when the problem size increases, which is unacceptable in real life and large scale applications. Then, we compare our proposed LEF with the base method $\ell_p$-box ADMM. The results turn out that the LEF outperforms the base method regarding objective gaps, which reveals the effectiveness of our early fixing framework. In Fig. 6, we record how the objective changes with respect to the iterations during the inference stages for different sized instances. From the figure, we can see that our LEF method leads to much faster convergence. When zooming in $10\times$ in the left two figure, we obtain a general view about the LEF fluctuations. When zooming in $100\times$ in the right two figure, we can even clearly see that our LEF with attention layers achieves better objective.

**Ablation study** We also compare our LEF with or without attention layers. From the experiments, we can see that the extra attention layers will cost more runtime, while obviously achieving a better objective. The negative objective gaps refer to a better objective. Interestingly, for all the large-sized Datasets II, we could even achieve a better objective than the expert method, $\ell_p$-box ADMM itself. And with the problem size $n$ increases, the runtime speedup is also increasing. And with our attention layers, the accuracy of datasets II can be greater than 99.8%, which is impressive. These results exhibit the efficiency of attention layers within our early fixing framework. We also evaluate the efficiency of weighted loss in training, as shown in Fig. 7(b). We set the fixing threshold to 0.9, where all the constraints are feasible. From the figures, we can tell that the weighted loss generally achieves a smaller objective gap and a larger runtime speedup, compared to the no weighted loss.

**Hyperparameter study** We analyze one of the most important hyperparameters in our early fixing framework: the fixing threshold $\theta$, as shown in Fig. 7(a). We evaluate how the number of infeasible constraints go with the fixing threshold, for different problem sizes in dataset II. We set fixing threshold from 0.5 to 0.9. When $\theta$ is 0.5, all the variables are fixed after the first block of iterations. When $\theta$ is 1, no variable is early fixed at all. From the results, only when $\theta$ is greater than 0.9, the number of infeasible constraints is 0, which means that the solution with early fixing is feasible.

## 6 MRF ENERGY MINIMIZATION

### 6.1 Formulations

We consider the pairwise Markov Random Field (MRF) energy minimization problem based on a graph, which can be generally formulated as [63]:

$$
\begin{aligned}
\arg \min_{\mathbf{x}} & \ \mathbb{E}(\mathbf{x}) = \mathbf{x}^{\top} \mathbf{A} \mathbf{x} + \mathbf{b}^{\top} \mathbf{x}, \\
s.t. & \ \mathbf{C} \mathbf{x} = \mathbf{1}, \ \mathbf{x} \in \{0,1\}^{nK \times 1}.
\end{aligned}
\tag{16}
$$

where $\mathbf{x}$ is a concatenation of all indicator vectors for the states $\kappa \in \{1,...,K\}$ and all $n$ nodes. If $x_{\kappa}^{i} = 1$, then node $i$ is on the state $\kappa$; otherwise, $x_{\kappa}^{i} = 0$. Each node can only take on one state, therefore we ensure that $\sum_{\kappa=1}^{K} x_{\kappa}^{i} = 1$ for $\forall i, i \in \{1,...,n\}$. Thus we have the constraint set as in Formulation 16. As for the objective function, $\mathbf{A}$ is the un-normalized Laplacian of the graph, $\mathbf{A} = \mathbf{D} - \mathbf{W}$, $\mathbf{W}$ is the matrix of node-to-node similarities. When $K = 2$, the problem turns to a submodular minimization problem and it can be globally optimized using the min-cut algorithm [60] in polynomial time, however, it cannot be guaranteed when $K > 2$.

<!-- Page 11 -->
# 6.2 Experiments for Image segmentation

## Accelerated method and datasets
$\ell_p$-Box ADMM [17] has been proved to be efficient in image segmentation when $K = 2$. Thus in our experiment, we choose $\ell_p$-Box ADMM to be accelerated. We also regard the min-cut [60] as the ground-truth algorithm ($K=2$) for comparisons. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) dataset [64] has been widely used in computer vision tasks, such as object classification, object detection and object segmentation. We thus choose VOC2012 for our experiments, where 2913 images are available for segmentation. We then randomly select 100, 20, 20 images for the training, validation and testing, respectively. We resize the testing images to different sizes and do the testing, including $n = 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7$ and $5e7$.

## Training details
We train our model only on the smallest sized images with $n = 1e4$, and generalize to all other larger-sized instances. $\gamma, \beta, \delta, L$ are set to 5, 10, 0.9, 2, respectively. The learning rate is set to $1e{-4}$. We train for 20 epochs. For our learning-based early fixing methods, training without attention for one epoch costs 122s, while training with attention costs 129s. We implement the functions of the accelerated methods in C++ and call the functions in Python via Cython interfaces. All the learning modules are implemented in Python.

## Baselines
We compare our method against two generic IP solvers, namely linear relaxation [62] and spectral relaxation [61], and a state-of-the-art and widely used min-cut algorithm [60]. The linear relaxation is implemented using the built-in function `quadprog` in MATLAB. The closed-form solution based on eigen-decomposition is implemented for spectral relaxation. We also compare against the $\ell_p$-box ADMM in both MATLAB and C++. We then apply our learning-based early fixing framework to $\ell_p$-box ADMM, with or without attention layers. We implement the functions of the accelerated methods in C++ and call the functions in Python via Cython interfaces.

## Evaluations
We evaluate all the different sizes, each with 20 instances. We evaluate the energy and the runtime for MATLAB implementations, and for our early fixing we also evaluate the energy gap towards the accelerated method itself. We report the mean values for all the 20 instances of different sizes. The time limit is set to 1 hour.

## Comparison study
The experimental results are shown in Table 4. We compare all methods in terms of their final energy value and runtime in the case of binary submodular MRF ($K=2$). From the results, we can see that min-cut achieves the lowest energy, regarded as the ground-truth methods for the $K=2$ image segmentation task. Among all the methods, spectral relaxation achieves the worst performance, though it is fast in computation. Linear relaxation runs fast in small sized instances, while running slow for large scale instances. Linear relaxation obtains smaller energy than spectral relaxation, while it is much worse than $\ell_p$-box ADMM. The C++ implementation of $\ell_p$-box ADMM runs much faster than MATLAB version with significant speedup. Compared to the accelerated $\ell_p$-box ADMM, our early fixing framework generally achieves from $3\times$ to $5\times$ speedup in runtime. Especially for the smallest sized instances with $n = 1e4$, the runtime speedup is even greater than $10\times$. In Fig. 9, we record how the energy changes with respect to the iterations during the inference stages for one image in different sizes. From the figure, we can see that our LEF method leads to much faster convergence, and LEF with attention layers achieves lower energy.

## Ablation study
We also show the ablation study of our learning-based early fixing. We compare the results with or without attention layers. Equipping with the attention layers will cost a little bit more runtime, however, the energy will decrease a bit more. For LEF without attention layers, the average energy gap of all different sizes is 5.7%, while the average energy gap with attention layers is only 2.8%. The gap is overall decreased by 2.9%.

## Performance exhibition
We exhibit some segmented images of different shapes by the optimal min-cut algorithm, $\ell_p$-box ADMM as well as our learning-based early fixing with or

---

**TABLE 4**  
Performance evaluations for image segmentation on the PASCAL Visual Object Classes Challenge 2012 datasets (VOC2012).

| Size → | Model | Lang. | $n = 1e4$ |  | $n = 5e4$ |  | $n = 1e5$ |  | $n = 5e5$ |  |
|--------|-------|-------|-----------|----------|-----------|----------|-----------|----------|-----------|----------|
|        |       |       | Energy↓   | Time↓    | Energy↓   | Time↓    | Energy↓   | Time↓    | Energy↓   | Time↓    |
|        | min-cut [60] | M. | 8778      | 0.1s     | 41049     | 0.2s     | 79737     | 0.4s     | 378439    | 1.8s     |
|        | spectral relaxation [61] | M. | 10753     | 0.1s     | 51192     | 0.2s     | 105528    | 0.3s     | 448737    | 1.1s     |
|        | linear relaxation [62] | M. | 9157      | 1.0s     | 42275     | 5.5s     | 81631     | 12.1s    | 472253    | 121.0s   |
|        | $\ell_p$-box ADMM [17] | M. | 8864      | 1.8s     | 41181     | 7.1s     | 79897     | 17.8s    | 379860    | 98.9s    |
|        |       |       | Energy↓   | Gap↓     | Energy↓   | Gap↓     | Energy↓   | Gap↓     | Energy↓   | Gap↓     |
|        | $\ell_p$-box ADMM | C.+P. | 8864      | N/A      | 41181     | N/A      | 79897     | N/A      | 379860    | N/A      |
|        | $\ell_p$-box ADMM + LEF(w/o Att.) | C.+P. | 9341      | 6.7%     | 43168     | 6.4%     | 83612     | 6.5%     | 395511    | 6.1%     |
|        | $\ell_p$-box ADMM + LEF(with Att.) | C.+P. | 9124      | 3.5%     | 42334     | 3.6%     | 82009     | 3.5%     | 388722    | 3.4%     |

| Size → | Model | Lang. | $n = 1e6$ |  | $n = 5e6$ |  | $n = 1e7$ |  | $n = 5e7$ |  |
|--------|-------|-------|-----------|----------|-----------|----------|-----------|----------|-----------|----------|
|        |       |       | Energy↓   | Time↓    | Energy↓   | Time↓    | Energy↓   | Time↓    | Energy↓   | Time↓    |
|        | min-cut [60] | M. | 741741    | 3.7s     | 4036901   | 22.8s    | 8006436   | 46.3s    | 48931748  | 952.0s   |
|        | spectral relaxation [61] | M. | 1075301   | 2.4s     | 4248567   | 11.6s    | 8263794   | 25.6s    | 57834141  | 986.9s   |
|        | linear relaxation [62] | M. | 883686    | 482.9s   | 4405514   | 1152.5s  | N/A       | 3600s    | N/A       | 3600s    |
|        | $\ell_p$-box ADMM [17] | M. | 744442    | 170.3s   | 4056762   | 1170.2s  | 8037384   | 2183.1s  | N/A       | 3600s    |
|        |       |       | Energy↓   | Gap↓     | Energy↓   | Gap↓     | Energy↓   | Gap↓     | Energy↓   | Gap↓     |
|        | $\ell_p$-box ADMM [17] | C.+P. | 744442    | N/A      | 4056762   | N/A      | 8037384   | N/A      | 49075089  | N/A      |
|        | $\ell_p$-box ADMM + LEF(w/o Att.) | C.+P. | 773693    | 5.9%     | 4191890   | 4.9%     | 8289911   | 4.7%     | 49339894  | 4.5%     |
|        | $\ell_p$-box ADMM + LEF(with Att.) | C.+P. | 760334    | 3.1%     | 4123729   | 2.1%     | 8153911   | 1.9%     | 49286140  | 1.6%     |

<!-- Page 12 -->
Fig. 8. Performance exhibition of different methods for image segmentation on the PASCAL Visual Object Classes Challenge 2012 datasets (VOC2012). $n=1e5$. Min-cut can obtain optimal solutions when $K=2$. $\ell_p$-box ADMM is the method to be accelerated.

Fig. 9. Convergence on different methods for image segmentation: how the objective changes with respect to the iterations during the inference stages for one images in four different sizes. LEF method leads to much faster convergence, and LEF with attention layers achieves lower energy.

without attention layers. From Figure 8, we can see that $\ell_p$-box ADMM generally achieves a great segmentation performance. And with our early fixing framework, the segmentation efficiency is also excellent.

## 7 SPARSE ADVERSARIAL ATTACK

### 7.1 Background

We consider the sparse adversarial attack [24], which generates adversarial perturbations onto partial positions of the clean image, where the perturbed image is incorrectly predicted by the deep model. There are two challenges lying in the sparse adversarial attack. One is where to perturb and the other is how to determine the perturbation magnitude. Some works manually or heuristically determined the perturbed positions, and optimized the magnitude using an appropriate algorithm designed for the dense adversarial attack. However, Fan et al. [24] proposed to factorize the perturbation at each pixel to the product of two variables, including the perturbation magnitude and one binary selection factor (i.e., 0 or 1). One pixel is perturbed if its selection factor is 1, otherwise not perturbed. The perturbation $\epsilon$ can be factorized as:

$$
\epsilon = \zeta \odot \eta,
\tag{17}
$$

where $\zeta \in \mathbb{R}^n$ denotes the vector of perturbation magnitudes, and $\eta \in \{0,1\}^n$ denotes the vector of perturbed positions. $\odot$ represents the element-wise product. Then the sparse attack problem can be formulated as a mixed integer programming (MIP) by jointly optimizing the continuous perturbation magnitudes $\zeta$ and the binary selection factors $\eta$ of all pixels. Inspired by $\ell_p$-box ADMM [17], they proposed to reformulate the MIP problem to an equivalent continuous optimization problem. They update the $\zeta$ by gradient descent, and update the $\eta$ by ADMM. At this point, we are going to accelerate the $\eta$ updating parts with our early fixing framework.

### 7.2 Adversarial attack experiments

**Datasets and models** We follow the setup in SAPF (Sparse adversarial Attack via Perturbation Factorization) [24], and use the CIFAR-10 [65] and ImageNet [66] for the experiments. There are 50k training images and 10k validation images covering 10 classes for CIFAR-10. We randomly select 1000 images from the validation set for our experiments. Each image has 9 target classes except its ground-truth class. Thus there are totally 9000 adversarial examples for the adversarial attack method. ImageNet contains 1000 classes, with 1.28 million images for training and 50k images for validation. We randomly choose 100 images covering 100 different classes from the validation set. To reduce the time complexity, we randomly select 9 target classes for each image in ImageNet, resulting in 900 adversarial examples. As regard to the classification model, on CIFAR-10, we follow

<!-- Page 13 -->
TABLE 5  
Performance comparison of targeted sparse adversarial attack on CIFAR-10 and ImageNet. We consider the ASR and $\ell_p$-norm ($p = 0, 1, 2, \infty$) of the learned perturbation.

| Dataset   | Method               | Best case           | Average case        | Worse case          |
|-----------|----------------------|---------------------|---------------------|---------------------|
|           |                      | ASR(%) | $\ell_0$ | $\ell_1$ | $\ell_2$ | $\ell_\infty$ | ASR(%) | $\ell_0$ | $\ell_1$ | $\ell_2$ | $\ell_\infty$ | ASR(%) | $\ell_0$ | $\ell_1$ | $\ell_2$ | $\ell_\infty$ |
| CIFAR-10  | One-pixel [69]       | 15.0   | 3        | 1.57     | 0.96     | 0.68      | 5.5    | 3        | 2.19     | 1.29     | 0.82      | 0.7    | 3        | 2.66     | 1.54     | 0.92      |
|           | CornerSearch [70]    | 60.4   | 537      | 69.70    | 3.34     | 0.34      | 59.3   | 549      | 73.64    | 3.48     | 0.34      | 63.2   | 77.57    | 561      | 3.62     | 0.34      |
|           | PGD $\ell_0+\ell_\infty$ [70] | 99.4   | 555      | 18.11    | 0.97     | 0.12      | 98.6   | 555      | 23.17    | 1.17     | 0.12      | 99.3   | 555      | 26.82    | 1.35     | 0.13      |
|           | SparseFool [71]      | 100    | 255      | 11.87    | 0.67     | 0.05      | 99.9   | 555      | 25.81    | 1.04     | 0.05      | 99.8   | 852      | 39.67    | 1.34     | 0.05      |
|           | C&W-$\ell_0$ [67]    | 100    | 614      | 6.95     | 0.43     | 0.09      | 100    | 603      | 13.07    | 0.81     | 0.16      | 100    | 598      | 18.60    | 1.14     | 0.22      |
|           | StrAttack [72]       | 100    | 391      | 4.94     | 0.30     | 0.05      | 100    | 543      | 9.49     | 0.52     | 0.09      | 100    | 476      | 12.44    | 0.77     | 0.14      |
|           | SAPF [24]            | 100    | 387      | 4.61     | 0.25     | 0.04      | 100    | 603      | 8.51     | 0.44     | 0.06      | 100    | 471      | 10.39    | 0.60     | 0.10      |
|           | SAPF + LEF(w/o Att.) | 100    | 149      | 5.23     | 0.46     | 0.10      | 100    | 303      | 8.48     | 0.64     | 0.10      | 100    | 459      | 11.69    | 0.62     | 0.10      |
|           | SAPF + LEF(with Att.)| 100    | 276      | 4.43     | 0.25     | 0.04      | 100    | 510      | 8.37     | 0.44     | 0.06      | 100    | 506      | 10.24    | 0.55     | 0.09      |
| ImageNet  | One-pixel [69]       | 0      | 3        | 1.19     | 0.80     | 0.66      | 0      | 3        | 1.88     | 1.18     | 0.83      | 0      | 3        | 2.56     | 1.51     | 0.93      |
|           | CornerSearch [70]    | 4      | 58658    | 5962.46  | 28.06    | 0.44      | 1.3    | 58792    | 6018.31  | 28.29    | 0.44      | 2      | 58920    | 6076.07  | 28.53    | 0.44      |
|           | PGD $\ell_0+\ell_\infty$ [70] | 95     | 56922    | 798.89   | 4.21     | 0.06      | 95.6   | 56919    | 854.67   | 4.51     | 0.06      | 96     | 56920    | 925.27   | 4.90     | 0.06      |
|           | SparseFool [71]      | 97     | 34205    | 174.17   | 0.92     | 0.01      | 80.6   | 59940    | 305.18   | 1.22     | 0.01      | 46     | 82576    | 420.44   | 1.45     | 0.01      |
|           | C&W-$\ell_0$ [67]    | 100    | 73407    | 133.79   | 0.79     | 0.05      | 100    | 70885    | 199.20   | 1.12     | 0.06      | 100    | 69947    | 269.10   | 1.46     | 0.07      |
|           | StrAttack [72]       | 100    | 38354    | 77.28    | 0.69     | 0.06      | 100    | 58581    | 127.59   | 0.97     | 0.08      | 100    | 67348    | 171.25   | 1.28     | 0.10      |
|           | SAPF [24]            | 100    | 37275    | 70.25    | 0.59     | 0.04      | 100    | 56218    | 112.16   | 0.72     | 0.04      | 100    | 56843    | 150.55   | 1.12     | 0.04      |
|           | SAPF + LEF(w/o Att.) | 100    | 4146     | 54.21    | 1.19     | 0.10      | 100    | 4074     | 78.32    | 1.36     | 0.10      | 100    | 4570     | 111.09   | 1.75     | 0.10      |
|           | SAPF + LEF(with Att.)| 100    | 5311     | 47.44    | 0.95     | 0.08      | 100    | 5344     | 67.26    | 1.08     | 0.08      | 100    | 5582     | 88.16    | 1.28     | 0.08      |

---

TABLE 6  
Runtime comparison of targeted sparse adversarial attack on CIFAR-10 and ImageNet. We record the runtime for $\zeta$ updating, $\eta$ updating, and total. The fourth column is the time speedup for $\zeta$ updating which uses our early fixing to accelerate $\ell_p$-box ADMM.

| Dataset   | Method               | $\eta$ Updating | $\eta$ Speedup | $\zeta$ Updating | Total    |
|-----------|----------------------|-----------------|----------------|------------------|----------|
| CIFAR-10  | SAPF [24]            | 79.6s           | N/A            | 81.3s            | 160.8s   |
|           | SAPF + LEF(w/o Att.) | 14.1s           | 5.6$\times$    | 81.5s            | 95.6s    |
|           | SAPF + LEF(with Att.)| 14.4s           | 5.5$\times$    | 79.7s            | 94.1s    |
| ImageNet  | SAPF [24]            | 499.2s          | N/A            | 560.1s           | 1059.3s  |
|           | SAPF + LEF(w/o Att.) | 204.4s          | 2.4$\times$    | 561.5s           | 765.9s   |
|           | SAPF + LEF(with Att.)| 231.2s          | 2.2$\times$    | 565.3s           | 796.5s   |

---

the setting of C&W [67] and train a network that consists of four convolution layers, three fully-connected layers, and two max-pooling layers. The input size of the network is 32 x 32 x 3. On ImageNet, we use a pre-trained Inception-v3 network [68]. The input size of the network is 299 x 299 x 3. All other experimental hyper-parameters for SAPF are set to default [24].

**Training details** As regard to our early fixing framework, we randomly pick 20 images from CIFAR-10. And each image corresponds to one MIP problem. By default, solving one MIP problem for adversarial attack is with 6 search loops for $G$ updating. Thus we record these 20*6=120 instances for early fixing training. $\gamma, \beta, \delta, \mathbb{L}$ are set to 3, 50, 0.9, 2, respectively. The learning rate is set to $1e-4$. We train for 20 epochs. For our learning-based early fixing methods, training without attention for one epoch costs 4min 28s, while training with attention costs 4min 20s. All the implementations are based on Python. We use the pre-trained model on CIFAR-10 to test on both CIFAR-10 and ImageNet.

**Baselines and evaluations** We compare whether to use our learning-based early fixing (LEF) on SAPF or not. We also compare whether to use attention layers or not. Besides, we also record the results by other attack paradigms, including one-pixel [69], corner search [70], PGD $\ell_0+\ell_\infty$ [70], SparseFool [71], C&W-$\ell_0$ [67], StrAttack [72]. Those results of other attacks are from the SAPF paper [24].

As regard to the evaluations, the $\ell_p$-norm ($p = 0, 1, 2, \infty$) of perturbations and the attack success rate (ASR) are utilized to evaluate the attack performance of different methods. We

---

Fig. 10. Examples of perturbations generated by the SAPF method, the SAPF with our early fixing framework, equipped without or with attention layers. From the top to the bottom, the ground-truth class and target class label pairs are: (siamang, zucchini), (gorilla, Brabancon griffon), (hognose snake, red-backed sandpiper), (coyote, impala), (barn spider, greenhouse), (great grey owl, capuchin).

<!-- Page 14 -->
Fig. 11. Convergence on different methods for sparse adversarial attack: how the objective (loss) changes with respect to the iterations during the inference stages for one image in CIFAR-10 (left) and ImageNet (right).

follow the same setting in [24]. We keep increasing the upper bound of $\ell_p$-norm of perturbations until the attack is success. We compare different attack algorithms in terms of the $\ell_p$-norm of perturbations under 100% ASR, though some sparse attack methods fail to generate 100% ASR. Moreover, for each image, we evaluate three different cases, i.e., the average case: the average performance of all 9 target classes; the best case: the performance w.r.t. the target class that is the easiest to attack; and the worst case: the performance w.r.t. the target class that is the most difficult to attack.

**Results analysis** We exhibit the attack performances of best/average/worst cases with $\ell_p$-norm and ASR in Table 5. We present the runtime comparisons whether to use LEF or not on SAPF in Table 6. We show the examples of generated perturbations by different methods in Figure 10. From the tables, we see that SAPF method achieve 100% attack success rate under all three cases, in both CIFAR-10 and ImageNet datasets. And with our LEF, the ASR still remains 100% in different cases, no matter with attention layers or not.

The $\ell_\infty$-norm of the one-Pixel-Attack is the largest among all algorithms and it achieves the lowest attack success rate. It is difficult to perform targeted adversarial attacks by only perturbing one pixel (the $\ell_0 = 3$ relates to three channels), even on one database CIFAR-10. The CornerSearch and PGD $\ell_0+\ell_\infty$ also fails to generate 100% success attack rate. Comparing to all adversarial attack algorithms except one-Pixel-Attack, SAPF method achieves the best $\ell_1$-norm and $\ell_2$-norm under all three cases. On CIFAR-10, with our LEF with attention layers, it achieves the better $\ell_1$-norm and $\ell_2$-norm under all cases compared to SAPF. On ImageNet, it achieves the better $\ell_1$-norm under all cases.

More importantly, with the aim of algorithmic acceleration, we can see the obvious time speedup on the $\eta$ updating part, by Table 6. On CIFAR-10 dataset, the time speedup is more than $5\times$, while on ImageNet dataset, the time speedup is more than $2\times$. Those results demonstrate the efficiency and effectiveness of the proposed LEF method. In Fig. 11, we record the first loop out of the 6 search loops for $\eta$ Updating. The figure reveals that our LEF method leads to faster convergence than the SAPF itself, when they all have quite similar objective (loss).

## 8 CONCLUSIONS AND FUTURE WORK

We propose an early fixing framework, which aims to accelerate the approximate method by early fixing the fluctuated variables to their converged states, while not significantly harming the converged performance. To the best of our knowledge, we are the first to propose the framework of early fixing for solving integer programming. We construct the whole early fixing process as a Markov decision process, and incorporate the imitation learning paradigm for training with the weighted binary cross-entropy loss. Specifically, we adopt the powerful attention layers in the policy network. We conduct the extensive experiments for our proposed early fixing framework to three different IP applications: constrained linear programming, MRF energy minimization and sparse adversarial attack. The experimental results in different scenarios demonstrate the efficiency of our proposed learning-based early fixing framework. In the future work, we would like to apply the early fixing framework to more approximate methods, and discover more possibilities to improve the efficiency and effectiveness of these methods. Meanwhile, there is a plenty of room to mitigate the objective gap when using some other efficient policy networks.

## REFERENCES

[1] Elias B Khalil. Machine learning for integer programming. In *IJCAI*, pages 4004–4005, 2016.

[2] Xinchao Wang, Engin Türetken, François Fleuret, and Pascal Fua. Tracking interacting objects optimally using integer programming. In *European Conference on Computer Vision*, pages 17–32. Springer, 2014.

[3] Oktay Günlük, Jayant Kalagnanam, Minhan Li, Matt Menickelly, and Katya Scheinberg. Optimal decision trees for categorical data via integer programming. *Journal of Global Optimization*, pages 1–28, 2021.

[4] Eugene L Lawler and David E Wood. Branch-and-bound methods: A survey. *Operations research*, 14(4):699–719, 1966.

[5] James E Kelley, Jr. The cutting-plane method for solving convex programs. *Journal of the society for Industrial and Applied Mathematics*, 8(4):703–712, 1960.

[6] John E Mitchell. Branch and cut. *Wiley encyclopedia of operations research and management science*, 2010.

[7] Stephen Boyd and Lieven Vandenberghe. *Convex optimization*. Cambridge university press, 2004.

[8] Hongyuan Zha, Xiaofeng He, Chris Ding, Ming Gu, and Horst D Simon. Spectral relaxation for k-means clustering. In *Advances in neural information processing systems*, pages 1057–1064, 2001.

[9] Jean B Lasserre. An explicit exact sdp relaxation for nonlinear 0-1 programs. In *International Conference on Integer Programming and Combinatorial Optimization*, pages 293–303. Springer, 2001.

[10] Stephen Boyd, Neal Parikh, and Eric Chu. *Distributed optimization and statistical learning via the alternating direction method of multipliers*. Now Publishers Inc, 2011.

[11] Anders Rantzer. Dynamic dual decomposition for distributed control. In *2009 American Control Conference*, pages 884–888. IEEE, 2009.

[12] Nikolaos Chatzipanagiotis, Darinka Dentcheva, and Michael M Zavlanos. An augmented lagrangian method for distributed optimization. *Mathematical Programming*, 152(1):405–434, 2015.

[13] Qiang Fu, Huahua Wang, and Arindam Banerjee. Bethe-admm for tree decomposition based parallel map inference. *arXiv preprint arXiv:1309.6829*, 2013.

[14] Huahua Wang and Arindam Banerjee. Bregman alternating direction method of multipliers. *The Conference on Neural Information Processing Systems*, 2014.

[15] Xingyu Xie, Jianlong Wu, Guangcan Liu, Zhisheng Zhong, and Zhouchen Lin. Differentiable linearized admm. In *International Conference on Machine Learning*, pages 6902–6911. PMLR, 2019.

[16] Qinghua Liu, Xinyue Shen, and Yuantao Gu. Linearized admm for nonconvex nonsmooth optimization with convergence analysis. *IEEE Access*, 7:76131–76144, 2019.

[17] Baoyuan Wu and Bernard Ghanem. $\ell_p$-box admm: A versatile framework for integer programming. *IEEE transactions on pattern analysis and machine intelligence*, 41(7):1695–1708, 2019.

<!-- Page 15 -->
[18] Amir R Zamir, Te-Lin Wu, Lin Sun, William B Shen, Bertram E Shi, Jitendra Malik, and Silvio Savarese. Feedback networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 1308–1317, 2017.

[19] Gao Huang, Danlu Chen, Tianhong Li, Felix Wu, Laurens van der Maaten, and Kilian Q Weinberger. Multi-scale dense networks for resource efficient image classification. *arXiv preprint arXiv:1703.09844*, 2017.

[20] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *arXiv preprint arXiv:1706.03762*, 2017.

[21] Ronald A Howard. Dynamic programming and markov processes. 1960.

[22] Faraz Torabi, Garrett Warnell, and Peter Stone. Behavioral cloning from observation. *arXiv preprint arXiv:1805.01954*, 2018.

[23] Baoyuan Wu, Li Shen, Tong Zhang, and Bernard Ghanem. Map inference via $\ell$2-sphere linear program reformulation. *International Journal of Computer Vision*, 128(7):1913–1936, 2020.

[24] Yanbo Fan, Baoyuan Wu, Tuanhui Li, Yong Zhang, Mingyang Li, Zhifeng Li, and Yujun Yang. Sparse adversarial attack via perturbation factorization. In *European conference on computer vision*, pages 35–50. Springer, 2020.

[25] Tuanhui Li, Baoyuan Wu, Yujun Yang, Yanbo Fan, Yong Zhang, and Wei Liu. Compressing convolutional neural networks via factorized convolutional filters. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3977–3986, 2019.

[26] Xiaoqin Zhang, Mingyu Fan, Di Wang, Peng Zhou, and Dacheng Tao. Top-k feature selection framework using robust 0–1 integer programming. *IEEE Transactions on Neural Networks and Learning Systems*, 32(7):3005–3019, 2020.

[27] Yoshua Bengio, Andrea Lodi, and Antoine Prouvost. Machine learning for combinatorial optimization: a methodological tour d’horizon. *European Journal of Operational Research*, 2020.

[28] Elias Khalil, Pierre Le Bodic, Le Song, George Nemhauser, and Bistra Dilkina. Learning to branch in mixed integer programming. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 30, 2016.

[29] Alejandro Marcos Alvarez, Quentin Louveaux, and Louis Wehenkel. A machine learning-based approximation of strong branching. *INFORMS Journal on Computing*, 29(1):185–195, 2017.

[30] Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, and Andrea Lodi. Exact combinatorial optimization with graph convolutional neural networks. *arXiv preprint arXiv:1906.01629*, 2019.

[31] Prateek Gupta, Maxime Gasse, Elias B Khalil, M Pawan Kumar, Andrea Lodi, and Yoshua Bengio. Hybrid models for learning to branch. *arXiv preprint arXiv:2006.15212*, 2020.

[32] Yunhao Tang, Shipra Agrawal, and Yuri Faenza. Reinforcement learning for integer programming: Learning to cut. In *International Conference on Machine Learning*, pages 9367–9376. PMLR, 2020.

[33] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. *arXiv preprint arXiv:1506.03134*, 2015.

[34] Irwan Bello, Hieu Pham, Quoc V Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning. *arXiv preprint arXiv:1611.09940*, 2016.

[35] Wouter Kool, Herke Van Hoof, and Max Welling. Attention, learn to solve routing problems! *arXiv preprint arXiv:1803.08475*, 2018.

[36] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. *arXiv preprint arXiv:1606.04474*, 2016.

[37] Ke Li and Jitendra Malik. Learning to optimize. *arXiv preprint arXiv:1606.01885*, 2016.

[38] Alex Nowak, Soledad Villar, Afonso S Bandeira, and Joan Bruna. A note on learning algorithms for quadratic assignment with graph neural networks. *stat*, 1050:22, 2017.

[39] Ahmed Hussein, Mohamed Medhat Gaber, Eyad Elyan, and Chrisina Jayne. Imitation learning: A survey of learning methods. *ACM Computing Surveys (CSUR)*, 50(2):1–35, 2017.

[40] Faraz Torabi, Garrett Warnell, and Peter Stone. Recent advances in imitation learning from observation. *arXiv preprint arXiv:1905.13566*, 2019.

[41] Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In *Proceedings of the twenty-first international conference on Machine learning*, page 1, 2004.

[42] Surat Teerapittayanon, Bradley McDanel, and Hsiang-Tsung Kung. Branchynet: Fast inference via early exiting from deep neural networks. In *2016 23rd International Conference on Pattern Recognition (ICPR)*, pages 2464–2469. IEEE, 2016.

[43] Lutz Prechelt. Early stopping—but when? In *Neural Networks: Tricks of the trade*, pages 55–69. Springer, 1998.

[44] Yuan Yao, Lorenzo Rosasco, and Andrea Caponnetto. On early stopping in gradient descent learning. *Constructive Approximation*, 26(2):289–315, 2007.

[45] Yigitcan Kaya, Sanghyun Hong, and Tudor Dumitras. Shallow-deep networks: Understanding and mitigating network overthinking. In *International Conference on Machine Learning*, pages 3301–3310. PMLR, 2019.

[46] Albert N Shiryaev. *Optimal stopping rules*, volume 8. Springer Science & Business Media, 2007.

[47] Sebastian Becker, Patrick Cheridito, and Arnulf Jentzen. Deep optimal stopping. *Journal of Machine Learning Research*, 20:74, 2019.

[48] Xinshi Chen, Hanjun Dai, Yu Li, Xin Gao, and Le Song. Learning to stop while learning to predict. In *International Conference on Machine Learning*, pages 1520–1530. PMLR, 2020.

[49] Vitali Gintner, Natalia Kliewer, and Leena Suhl. Solving large multiple-depot multiple-vehicle-type bus scheduling problems in practice. *OR Spectrum*, 27(4):507–523, 2005.

[50] Stefan Helber and Florian Sahling. A fix-and-optimize approach for the multi-level capacitated lot sizing problem. *International Journal of Production Economics*, 123(2):247–256, 2010.

[51] Árton P Dorneles, Olinto CB de Araújo, and Luciana S Buriol. A fix-and-optimize heuristic for the high school timetabling problem. *Computers & Operations Research*, 52:29–38, 2014.

[52] Yang Wang, Zhipeng Lü, Fred Glover, and Jin-Kao Hao. Effective variable fixing and scoring strategies for binary quadratic programming. In *European Conference on Evolutionary Computation in Combinatorial Optimization*, pages 72–83. Springer, 2011.

[53] Michel Gendreau and Jean-Yves Potvin. Tabu search. In *Search methodologies*, pages 165–186. Springer, 2005.

[54] Jen-Tzung Chien and Chun-Wei Wang. Hierarchical and self-attended sequence autoencoder. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2021.

[55] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016.

[56] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *International conference on machine learning*, pages 448–456. PMLR, 2015.

[57] Tobias Achterberg and Timo Berthold. Hybrid branching. In *International Conference on AI and OR Techniques in Constraint Programming for Combinatorial Optimization Problems*, pages 309–311. Springer, 2009.

[58] Tobias Achterberg. Scip: solving constraint integer programs. *Mathematical Programming Computation*, 1(1):1–41, 2009.

[59] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 32, 2018.

[60] Yuri Boykov and Vladimir Kolmogorov. An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision. *IEEE transactions on pattern analysis and machine intelligence*, 26(9):1124–1137, 2004.

[61] Jianbo Shi and Jitendra Malik. Normalized cuts and image segmentation. *IEEE Transactions on pattern analysis and machine intelligence*, 22(8):888–905, 2000.

[62] George Dantzig. *Linear programming and extensions*. Princeton university press, 2016.

[63] Daphne Koller and Nir Friedman. *Probabilistic graphical models: principles and techniques*. MIT press, 2009.

[64] Mark Everingham, SM Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. *International journal of computer vision*, 111(1):98–136, 2015.

[65] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[66] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pages 248–255. Ieee, 2009.

<!-- Page 16 -->
[67] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In *2017 ieee symposium on security and privacy (sp)*, pages 39–57. IEEE, 2017.

[68] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 2818–2826, 2016.

[69] Jiawei Su, Danilo Vasconcellos Vargas, and Kouichi Sakurai. One pixel attack for fooling deep neural networks. *IEEE Transactions on Evolutionary Computation*, 23(5):828–841, 2019.

[70] Francesco Croce and Matthias Hein. Sparse and imperceptible adversarial attacks. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4724–4732, 2019.

[71] Apostolos Modas, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard. Sparsefool: a few pixels make a big difference. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 9087–9096, 2019.

[72] Kaidi Xu, Sijia Liu, Pu Zhao, Pin-Yu Chen, Huan Zhang, Quanfu Fan, Deniz Erdogmus, Yanzhi Wang, and Xue Lin. Structured adversarial attack: Towards general implementation and better interpretability. *arXiv preprint arXiv:1808.01664*, 2018.