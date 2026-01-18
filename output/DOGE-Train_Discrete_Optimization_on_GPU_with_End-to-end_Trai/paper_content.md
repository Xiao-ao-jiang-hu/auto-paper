<!-- Page 1 -->
# DOGE-Train: Discrete Optimization on GPU with End-to-end Training

**Ahmed Abbas**${}^1$  
**Paul Swoboda**${}^{1,2,3}$

${}^1$Max Planck Institute for Informatics, Saarland Informatics Campus  
${}^2$University of Mannheim, ${}^3$Heinrich-Heine University Düsseldorf

## Abstract

We present a fast, scalable, data-driven approach for solving relaxations of 0-1 integer linear programs. We use a combination of graph neural networks (GNN) and the Lagrange decomposition based algorithm (Abbas and Swoboda 2022b). We make the latter differentiable for end-to-end training and use GNNs to predict its algorithmic parameters. This allows to retain the algorithm’s theoretical properties including dual feasibility and guaranteed non-decrease in the lower bound while improving it via training. We overcome suboptimal fixed points of the basic solver by additional non-parametric GNN update steps maintaining dual feasibility. For training we use an unsupervised loss. We train on smaller problems and test on larger ones showing strong generalization performance with a GNN comprising only around $10k$ parameters. Our solver achieves significantly faster performance and better dual objectives than its non-learned version, achieving close to optimal objective values of LP relaxations of very large structured prediction problems and on selected combinatorial ones. In particular, we achieve better objective values than specialized approximate solvers for specific problem classes while retaining their efficiency. Our solver has better any-time performance over a large time period compared to a commercial solver.

solver for LP-relaxations of ILPs. LP solving is a key step taking most time in traditional ILP pipelines. State of the art LP solvers (Gurobi Optimization, LLC 2021; Cplex, IBM ILOG 2019; FICO 2022; MOSEK ApS 2022; Gamrath et al. 2020) are not amenable to ML since they are non-differentiable, sequential and have very complex implementations. This makes utilization of neural networks and GPUs for solver improvement difficult. For these reasons we build upon the massively parallel FastDOG (Abbas and Swoboda 2022b) solver and show that it can be made differentiable. This allows to train our problem agnostic solver for specific problem classes resulting in equal or better performance as compared to efficient hand-designed specialized solvers.

## Contributions

Our high-level contributions are conceptual and empirical: (i) We show that embedding good inductive biases coming from non-learned solvers (in our case highly parallel GPU-based block coordinate ascent (Abbas and Swoboda 2022b) and subgradients) into neural networks leads to greatly improved performance. In particular, we give evidence to the hypothesis that similar to vision (convolutions) and NLP (sequence models) the right inductive biases coming from solver primitives are a promising way to use the potential of ML for optimization. (ii) Our approach is more economical as compared to developing efficient problem specific heuristics, as is customary for large scale problems in structured prediction tasks for ML (Haller et al. 2020; Hutschenreiter et al. 2021). Instead of spending much time and effort in designing and implementing new algorithms, one can train our problem agnostic solver with a few problem instances coming from the problem class of interest and obtain a state of the art GPU-enabled solver for it${}^1$.

In detail, we propose to learn the Lagrange decomposition algorithm (Abbas and Swoboda 2022b) for solving LP relaxations of ILP problems and show its benefits. In particular,

- We generalize the dual optimization algorithm of (Abbas and Swoboda 2022b) to allow for a larger space of parameter updates.
- We make our dual optimization algorithm efficiently differentiable and embed it as a layer in a neural network. This enables us to predict parameters of the algorithm leading to faster convergence compared to manually designed rules.

---

${}^1$Code available at https://github.com/LPMP/BDD

Copyright © 2024, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

<!-- Page 2 -->
- We train a predictor for arbitrary non-parametric updates that allow to escape suboptimal fixed points encountered by parametric update steps of (Abbas and Swoboda 2022b).
- Our predictors for both of the above updates are trained in a fully unsupervised manner. Our loss optimizes for producing large improvements in the dual objective.
- We show the benefits of our learned approach on a wide range of problems. We have chosen structured prediction tasks including graph matching (Kainmueller et al. 2014) and cell tracking (Haller et al. 2020). From theoretical computer science we compare on the QAPLib (Burkard, Karisch, and Rendl 1997) dataset and on randomly generated independent set problems (Prouvost et al. 2020).

## 2 Related Work

### 2.1 Learning to solve combinatorial optimization

ML has been used to improve various aspects of solving combinatorial problems. For the standard branch-and-cut ILP solvers the works (Gasse et al. 2019; Gupta et al. 2020; Nair et al. 2020; Scavuzzo et al. 2022) learn variable selection for branching. The approaches (Ding et al. 2020; Nair et al. 2020) learn to fix a subset of integer variables in ILPs to their hopefully optimal values to improve finding high quality primal solutions. The works (Sonnerat et al. 2021; Wu et al. 2021) learn variable selection for the large neighborhood search heuristic for obtaining primal solutions to ILPs. Selecting good cuts through scoring them with neural networks was investigated in (Huang et al. 2022; Turner et al. 2022). While all these approaches result in runtime and solution quality improvements, only a few works tackle the important task of speeding up ILP relaxations by ML. Specifically, the work (Cappart et al. 2019) used graph neural network (GNN) to predict variable orderings of decision diagrams representing combinatorial optimization problems. The goal is to obtain an ordering such that a corresponding dual lower bound is maximal. To our knowledge it is the only work that accelerates ILP relaxation computation with ML. For constraint satisfaction problems (Selsam et al. 2018; Cameron et al. 2020; Tönshoff et al. 2021) train GNN while the latter train in an unsupervised manner. For inference in graphical models (Deng et al. 2022) learn parameters of belief propagation for faster convergence in a similar spirit to our work. However our method is applicable to a more general class of problems, allows escaping fixed-points, and is scalable to larger problems due to efficient implementation. For narrow subclasses of problems primal heuristics have been augmented through learning some of their decisions, e.g. for capacitated vehicle routing (Nazari et al. 2018), graph matching (Wang, Yan, and Yang 2021) and traveling salesman (Xin et al. 2021). For a more complete overview of ML for combinatorial optimization we refer to the detailed surveys (Bengio, Lodi, and Prouvost 2021; Cappart et al. 2023).

### 2.2 Unrolling algorithms for parameter learning

Algorithms containing differentiable iterative procedures are combined with neural networks for improving performance of such algorithms. Such approaches show more generalization power than pure neural networks based ones as shown in the survey (Monga, Li, and Eldar 2021). The work of (Gregor and LeCun 2010) embedded sparse coding algorithms in a neural network by unrolling. For solving inverse problems (Yang et al. 2020; Chen and Pock 2017) unroll through ADMM and non-linear diffusion resp. Lastly, neural networks were used to predict update directions for training other neural networks (e.g. in (Andrychowicz et al. 2016)).

## 3 Method

We first recapitulate the Lagrange decomposition approach to binary ILPs from (Lange and Swoboda 2021) and generalize the optimization scheme of (Abbas and Swoboda 2022b) for faster convergence. Then we will show how to backpropagate through the optimization scheme allowing to train a graph neural network for predicting its parameters.

### 3.1 Lagrange Decomposition

**Definition 1 (Binary Program (Lange and Swoboda 2021)).** Let a linear objective $c \in \mathbb{R}^n$ and $m$ variable subsets $\mathcal{I}_j \subset [n]$ of constraints with feasible set $\mathcal{X}_j \subset \{0,1\}^{\mathcal{I}_j}$ for $j \in [m]$ be given. The ensuing binary program is

$$
\min_{x \in \{0,1\}^n} \langle c, x \rangle \quad \text{s.t.} \quad x_{\mathcal{I}_j} \in \mathcal{X}_j \quad \forall j \in [m], \tag{BP}
$$

where $x_{\mathcal{I}_j}$ is the restriction to variables in $\mathcal{I}_j$.

Any binary ILP $\min_{x \in \{0,1\}^n} \langle c, x \rangle$ s.t. $Ax \leq b$ where $A \in \mathbb{R}^{m \times n}$ can be written as (BP) by associating each constraint $a_j^T x \leq b_j$ for $j \in [m]$ with its own subproblem $\mathcal{X}_j$. In order to obtain a problem formulation amenable for parallel optimization we consider its Lagrange dual which decomposes the full problem (BP) into a series of coupled subproblems.

**Definition 2 (Lagrangian dual problem (Lange and Swoboda 2021)).** Define the set of subproblems that constrain variable $i$ as $\mathcal{J}_i = \{ j \in [m] \mid i \in \mathcal{I}_j \}$. Let the energy for subproblem $j \in [m]$ w.r.t. Lagrange variables $\lambda_{\bullet j} = (\lambda_{ij})_{i \in \mathcal{I}_j} \in \mathbb{R}^{\mathcal{I}_j}$ be

$$
E^j(\lambda_{\bullet j}) = \min_{x \in \mathcal{X}_j} \langle \lambda_{\bullet j}, x \rangle. \tag{1}
$$

Then the Lagrangean dual problem is defined as

$$
\max_\lambda \sum_{j \in [m]} E^j(\lambda_{\bullet j}) \quad \text{s.t.} \quad \sum_{j \in \mathcal{J}_i} \lambda_{ij} = c_i \quad \forall i \in [n]. \tag{D}
$$

The problem (D) provides a lower bound to the NP-Hard optimization problem (BP) and is also useful for primal recovery (Abbas and Swoboda 2022b). Our goal is to learn a neural network for optimizing the dual (D) efficiently and to reach better objective values.

### 3.2 Optimization of Lagrangean dual

The work of Abbas and Swoboda (2022b) proposed a parallelization friendly iterative scheme for optimizing (D) with hand-designed parameters. We generalize their scheme in Algorithm 1, exposing a much larger set of parameters allowing more control over the optimization process. Since this large parameter space is difficult to be tuned manually, we will employ a GNN for predicting these parameters.

<!-- Page 3 -->
# ILP representation

![Figure 1: Our method for optimizing the Lagrangean dual (D). The dual problem is encoded on a bipartite graph containing features $f_\mathcal{I}$, $f_\mathcal{J}$ and $f_\mathcal{E}$ for primal variables, subproblems and dual variables resp. A graph neural network (GNN) predicts $\theta$, $\alpha$, $\omega$ for dual updates. In one dual update block (right), current set of Lagrange multipliers $\lambda$ are first updated by the non-parametric update using $\theta$. Afterwards parametric update is done via Alg. 1 using $\alpha, \omega$. The updated solver features $f$ and LSTM cell states $s_\mathcal{I}$ are sent to the GNN in next optimization round. See Sec. 3.6 for further details.]

In detail, Alg. 1 greedily assigns the Lagrange variables in $u$-many disjoint blocks $B_1, \ldots, B_u$ in such a way that each block contains at most one Lagrange variable from each subproblem and all variables within a block are updated in parallel (same as (Abbas and Swoboda 2022b)). The dual update scheme relies on computing min-marginal differences i.e., the difference of subproblem objectives when a certain variable is set to 1 minus its objective when the same variable is set to 0 (line 10). These min-marginal differences are averaged out across subproblems via updates to Lagrange variables (line 11). Our algorithm relies on two important set of parameters, damping factors and averaging weights. The damping factor $\omega_{ij}$ determine the fraction of min-marginal difference to subtract from variable $i$ in subproblem $j$. The averaging weights $\alpha_{ij}$ parameterize the fraction of total min-marginal difference $\sum_{ik} M_{ik}$ variable $i$ in subproblem $j$ receives.

**Remark.** The deferred min-marginal averaging algorithm of (Abbas and Swoboda 2022b) is a specialized form of our generalized Algorithm 1 if the parameters were set as $\omega_{ij} = 0.5$ and $\alpha_{ij} = 1/|\mathcal{J}_i|$ for all $i,j$.

We generalize the min-marginal update step by considering damping factors to be in $(0,1)$ and averaging weights to be arbitrary convex combinations. We show that this generalized update step still preserves the desirable property of guaranteed non-improvement in the dual objective.

**Proposition 1 (Dual Feasibility and Monotonicity of Generalized Min-Marginal Averaging).** For any $\alpha_{ij} \geq 0$ with $\sum_{j\in\mathcal{J}_i} \alpha_{ij} = 1$ and $\omega_{ij} \in [0,1]$ the min-marginal averaging step in line 11 in Algorithm 1 retains dual feasibility and is non-decreasing in the dual lower bound.

## 3.3 Backpropagation through dual optimization

We show below how to differentiate through Algorithm 1 with respect to its parameters $\alpha$ and $\omega$. This will ultimately allow us to learn these parameters such that faster convergence is achieved. To this end we describe backpropagation for a block update (lines 8-12) of Alg. 1. All other operations can be tackled by automatic differentiation. For a block $B$ in $\{B_1,\ldots,B_u\}$ we view the Lagrangean update as a mapping $\mathcal{H}: (\mathbb{R}^{|B|})^4 \to (\mathbb{R}^{|B|})^2$, $(\lambda^\text{in}, M^\text{in}, \alpha, \omega) \mapsto (\lambda^\text{out}, M^\text{out})$.

Given a loss function $\mathcal{L}: \mathbb{R}^N \to \mathbb{R}$ we denote $\partial\mathcal{L}/\partial x$ by $\dot{x}$. Algorithm 2 shows backpropagation through $\mathcal{H}$ to compute the gradients $\dot{\lambda}^\text{in}$, $\dot{M}^\text{in}$, $\dot{\alpha}$ and $\dot{\omega}$.

**Proposition 2.** Alg. 2 performs backprop. through $\mathcal{H}$.

**Efficient Implementation** Generally, the naive computation of min-marginal differences and its backpropagation are both expensive operations as they require solving two optimization problems for each dual variable. The works of Abbas and Swoboda (2022b); Lange and Swoboda (2021) represent each subproblem by a binary decision diagram (BDD) for fast computation of min-marginal differences. Their algorithm results in a computation graph involving only elementary arithmetic operations and taking minima over several variables. Using this computational graph we

---

### Algorithm 1: Generalized Min-Marginal Averaging

**Input:** Lagrange variables $\lambda_{ij}\ \forall i\in[n], j\in\mathcal{J}_i$,  
damping factors $\omega_{ij}\in(0,1)\ \forall i\in[n], j\in\mathcal{J}_i$,  
averaging weights $\alpha_{ij}\in(0,1)\ \forall i\in[n], j\in\mathcal{J}_i$,  
max. number of iterations $T$.

1. Initialize deferred min-marginal diff. $M = 0$
2. **for** $T$ iterations **do**
3. &nbsp;&nbsp;&nbsp; **for** block $B \in (B_1, \ldots B_u)$ **do**
4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\lambda, M \leftarrow \text{BlockUpdate}(B, \lambda, M, \alpha, \omega)$
5. &nbsp;&nbsp;&nbsp; **for** block $B \in (B_u, \ldots B_1)$ **do**
6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\lambda, M \leftarrow \text{BlockUpdate}(B, \lambda, M, \alpha, \omega)$
7. **return** $\lambda$, $M$

**Procedure BlockUpdate** $(B, \lambda^\text{in}, M^\text{in}, \alpha, \omega)$

8. &nbsp;&nbsp;&nbsp; **for** $ij \in B$ in parallel **do**
9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $M^\text{out}_{ij} = \omega_{ij} \big[ \min_{x\in\mathcal{X}_j} \langle \lambda^\text{in}_{\bullet j}, x \rangle - \min_{x\in\mathcal{X}_j: x_i=0} \langle \lambda^\text{in}_{\bullet j}, x \rangle \big]$
10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\lambda^\text{out}_{ij} = \lambda^\text{in}_{ij} - M^\text{out}_{ij} + \alpha_{ij} \sum_{k\in\mathcal{J}_i} M^\text{in}_{ik}$
11. **return** $\lambda^\text{out}$, $M^\text{out}$

<!-- Page 4 -->
**Algorithm 2: BlockUpdate backpropagation**

**Input:** Forward pass inputs: $B, \lambda^{\text{in}}, M^{\text{in}}, \alpha, \omega$,  
gradients of forward pass output: $\dot{\lambda}^{\text{out}}, \dot{M}^{\text{out}}$,  
gradients of parameters $\dot{\alpha}, \dot{\omega}$

1. for $ij \in B$ in parallel do
2.   $\dot{M}_{ij}^{\text{in}} = \sum_{k \in \mathcal{J}_i} \dot{\lambda}_{ik}^{\text{out}} \alpha_{ik}$
3.   $\dot{M}_{ij}^{\text{out}} = \dot{M}_{ij}^{\text{out}} - \dot{\lambda}_{ij}^{\text{out}}$
4.   $\dot{\alpha}_{ij} = \dot{\alpha}_{ij} + \dot{\lambda}_{ij} \sum_{k \in \mathcal{J}_i} M_{ik}^{\text{in}}$
5.   $\dot{\omega}_{ij} = \dot{\omega}_{ij} + \dot{M}_{ij}^{\text{out}} [M_{ij}^{\text{out}} / \omega_{ij}]$
6.   Compute minimizers for $\beta \in \{0,1\}$  
     $s^j(i,\beta) = \argmin_{x \in \mathcal{X}_j : x_i = \beta} \langle \lambda_{\bullet j}^{\text{in}}, x \rangle$
7.   $\dot{\lambda}_{pj}^{\text{in}} = \dot{\lambda}_{pj}^{\text{in}} + \dot{M}_{ij}^{\text{out}} \omega_{ij} [s_p^j(i,1) - s_p^j(i,0)], \forall p \in \mathcal{I}_j$
8. return $\dot{\lambda}^{\text{in}}, \dot{M}^{\text{in}}, \dot{\alpha}, \dot{\omega}$

can implement the abstract Algorithm 2 efficiently on GPU. For further performance gains we implement custom backpropagation routines in CUDA for more than an order of magnitude decrease in runtime and memory usage as shown in Table 4 of the Appendix.

## 3.4 Non-Parametric Update Steps

Although the min-marginal averaging scheme of Alg. 1 guarantees a non-decreasing lower bound, it can get stuck in suboptimal fixed points, see (Werner 2007) for a discussion for the special case of MAP inference in Markov Random Fields and (Werner, Prusa, and Dlask 2020) for a more general setting. To address this issue we allow arbitrary updates to Lagrange variables through a vector $\hat{\theta} \in \mathbb{R}^{|\lambda|}$ as

$$
\lambda_{ij} \leftarrow \lambda_{ij} + \hat{\theta}_{ij} - \frac{1}{|\mathcal{J}_i|} \sum_{k \in \mathcal{J}_i} \hat{\theta}_{ik}, \forall i \in [n], j \in \mathcal{J}_i,
\tag{2}
$$

where the last term ensures feasibility of updated Lagrange variables w.r.t. the dual problem (D).

## 3.5 Graph Neural Network

We train a graph neural network (GNN) to predict the parameters $\alpha, \omega \in \mathbb{R}^{|\lambda|}$ of Alg. 1 and also the non-parametric update $\theta \in \mathbb{R}^{|\lambda|}$ for (2). To this end we encode the dual problem (D) on a bipartite graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. Its nodes correspond to primal variables $\mathcal{I}$ and subproblems $\mathcal{J}$ i.e., $\mathcal{V} = \mathcal{I} \cup \mathcal{J}$ and edges $\mathcal{E} = \{ ij \mid i \in \mathcal{I}, j \in \mathcal{J}_i \}$ correspond to Lagrange multipliers. We need to predict values of $\alpha_{ij}, \omega_{ij}$ and $\theta_{ij}$ for each edge $ij$ in $\mathcal{E}$. We associate features $f = (f_\mathcal{I}, f_\mathcal{J}, f_\mathcal{E})$ with each entity (nodes, edges) of the bipartite graph. Lagrange multipliers $\lambda^{\text{in}}$ and deferred min-marginals $M^{\text{in}}$ encode the current state of Alg. 1 as a part of edge features. Additionally, we encode a number of quantities as features which can allow the GNN to make better updates. Specifically, a subgradient of the dual problem (D) is encoded in the edge features $f_\mathcal{E}$ and a history of previous dual objectives for each subproblem is encoded in the constraint features $f_\mathcal{J}$. This enables our GNN to effectively utilize more information in parameter prediction than conventional hand-designed updates rules can manage. For example (Abbas and Swoboda 2022b) can get stuck in suboptimal fixed points due to zero min-marginal differences (Werner, Prusa, and Dlask 2020). Since the GNN additionally has access to subgradient of the dual problem it can escape such fixed points. A complete list of features is provided in the Appendix.

### Graph convolution

We use the transformer based graph convolution scheme (Shi et al. 2021). We first compute embeddings of all subproblems $j$ in $\mathcal{J}$ by receiving messages from adjacent nodes and edges as

$$
\text{CONV}_\mathcal{J}(f_\mathcal{I}, f_\mathcal{J}, f_\mathcal{E}, \mathcal{E})_j = \mathbf{W}_\mathbf{s} f_j + \sum_{ij \in \mathcal{E}} a_{ij}(f_i, f_j, f_\mathcal{E}; \mathbf{W}_\mathbf{a}) \left[ \mathbf{W}_\mathbf{t} f_i + \mathbf{W}_\mathbf{e} f_{ij} \right],
\tag{3}
$$

where $\mathbf{W}_\mathbf{a}, \mathbf{W}_\mathbf{s}, \mathbf{W}_\mathbf{t}, \mathbf{W}_\mathbf{e}$ are trainable parameters and $a_{ij}(f_i, f_j, f_\mathcal{E}; \mathbf{W}_\mathbf{a})$ is the softmax attention weight between nodes $i$ and $j$ parameterized by $\mathbf{W}_\mathbf{a}$. Afterwards we perform message passing in the reverse direction to compute embeddings for variables $\mathcal{I}$. A similar strategy for message passing on bipartite graphs was followed in (Gasse et al. 2019).

### Recurrent connections

Our default GNN as mentioned above only uses hand-crafted features to maintain a history of previous optimization rounds. To learn a summary of the past updates we optionally allow recurrent connections through an LSTM with forget gate (Gers, Schmidhuber, and Cummins 1999). The LSTM is only applied on primal variable nodes $\mathcal{I}$ and maintains cell states $s_\mathcal{I}$ which can be updated and used for parameter prediction in subsequent optimization rounds.

### Prediction

The learned embeddings from GNN, LSTM outputs and solver features are consumed by a multi-layer perceptron $\Phi$ to predict the required variables for each edge $ij$ in $\mathcal{E}$. Afterwards we transform these outputs so that they satisfy Prop. 1. The exact sequence of operations performed by the graph neural network are shown in Alg. 3 where $[u_1, \dots, u_k]$ denotes concatenation of vectors $u_1, \dots, u_k$, LN denotes layer normalization (Ba, Kiros, and Hinton 2016) and $\text{LSTM}_\mathcal{I}$ stands for an LSTM cell operating on primal variables $\mathcal{I}$.

### Loss

Given the Lagrange variables $\lambda$ we directly use the dual objective (D) as an unsupervised loss to train the GNN. Thus, we maximize the loss $\mathcal{L}$ defined as

$$
\mathcal{L}(\lambda) = \sum_{j \in [m]} E^j(\lambda_{\bullet j}).
\tag{4}
$$

For a mini-batch of instances during training we take the mean of corresponding per-instance losses. For backpropagation, gradient of the loss $\mathcal{L}$ w.r.t. Lagrange variables of a subproblem $j$ is computed by finding a minimizing assignment for that subproblem, written as $\left( \frac{\partial \mathcal{L}}{\partial \lambda} \right)_{\bullet j} = \argmin_{x \in \mathcal{X}_j} \langle \lambda_{\bullet j}, x \rangle \in \{0,1\}^{\mathcal{I}_j}$. This gradient is then sent as input for backpropagation. For computing the minimizing assignment efficiently we use binary decision diagram representation of each subproblem as in (Abbas and Swoboda 2022b; Lange and Swoboda 2021).

<!-- Page 5 -->
**Algorithm 3: Parameter prediction by GNN**

**Input:** Primal variable features $f_\mathcal{I}$ and cell states $s_\mathcal{I}$,  
Subproblem features $f_\mathcal{J}$, Dual variable (edge)  
features $f_\mathcal{E}$, Set of edges $\mathcal{E}$.

// Compute subproblems embeddings  
1 $h_\mathcal{J} = \text{ReLU}\left(\text{LN}\left(\text{CONV}_\mathcal{J}\left(f_\mathcal{I}, f_\mathcal{J}, f_\mathcal{E}, \mathcal{E}\right)\right)\right)$

// Compute primal var embeddings  
2 $h_\mathcal{I} = \text{ReLU}\left(\text{LN}\left(\text{CONV}_\mathcal{I}\left(f_\mathcal{I}, [f_\mathcal{J}, h_\mathcal{J}], f_\mathcal{E}, \mathcal{E}\right)\right)\right)$

// Compute output and cell state  
3 $z_\mathcal{I}, s_\mathcal{I} = \text{LSTM}_\mathcal{I}(h_\mathcal{I}, s_\mathcal{I})$

// Prediction per edge  
4 $(\hat{\alpha}, \hat{\omega}, \hat{\theta}) = \Phi\left([f_\mathcal{I}, h_\mathcal{I}, z_\mathcal{I}], [f_\mathcal{J}, h_\mathcal{J}], f_\mathcal{E}, \mathcal{E}\right)$

// Ensure non-decreasing obj., Prop 1:  
5 $\alpha_{i\bullet} = \text{Softmax}(\hat{\alpha}_{i\bullet}), \forall i \in \mathcal{I}, \; \omega = \text{Sigmoid}(\hat{\omega})$

// Maintain dual feasibility  
6 $\theta_{i\bullet} = \hat{\theta}_{i\bullet} - \frac{1}{|\mathcal{J}_i|} \sum_{k \in \mathcal{J}_i} \hat{\theta}_{ik}, \forall i \in \mathcal{I}$

7 return $\alpha, \omega, \theta, s_\mathcal{I}$

---

### 3.6 Overall pipeline

We train our pipeline (Fig. 1) which contains multiple dual optimization rounds in a fashion similar to that of recurrent neural networks. One round of our dual optimization consists of message passing by GNN, a non-parametric update step and $T$ iterations of generalized min-marginal averaging. For computational efficiency we run our pipeline for at most $R$ dual optimization rounds during training. On each mini-batch we randomly sample a number of optimization rounds $r$ in $[R]$, run $r-1$ rounds without tracking gradients and backpropagate through the last round by computing the loss (4). For the pipeline with recurrent connections we backpropagate through last 3 rounds and apply the loss after each of these rounds. Since the task of dual optimization is relatively easier in early rounds as compared to later ones we use two neural networks. The early stage network is trained if the randomly sampled $r$ is in $[0, R/2]$ and the late stage network is chosen otherwise. During testing we switch to the later stage network when the relative improvement in the dual objective by the early stage network becomes less than $10^{-6}$. For computational efficiency during testing we query the GNN for parameter updates only after $T \gg 1$ iterations of Alg. 1.

---

## 4 Experiments

### 4.1 Evaluation

As main evaluation metric we report convergence plots of the relative dual gap $g(t) \in [0,1]$ at time $t$ by $g(t) = \min\left(\frac{d^* - d(t)}{d^* - d_{init}}, 1.0\right)$ where $d(t)$ is the dual objective at time $t$, $d^*$ is the optimal (or best known) objective value of the Lagrange relaxation (D) and $d_{init}$ is the objective before optimization as computed by (Abbas and Swoboda 2022b). Additionally we also report per dataset averages of best objective value ($E$), time taken ($t$) to obtain best objective and relative dual gap integral $g_I = \int g(t) dt$ (Berthold 2013). The latter metric $g_I$ allows to measure quality of the solution in conjunction with time take to obtain this solution.

---

### 4.2 Datasets

We evaluate our approach on a variety of datasets from different domains. For each dataset we train our pipeline on smaller instances and test on larger ones.

**Cell tracking (CT):** Instances of developing flying tissue from cell tracking challenge (Ulman et al. 2017) processed by (Haller et al. 2020) and obtained from (Swoboda et al. 2022). We use the largest and hardest 3 instances, train on the 2 smaller instances and test on the largest one.

**Graph matching (GM):** Instances of graph matching for matching nuclei in 3D microscopic images (Long et al. 2009) processed by (Kainmueller et al. 2014) and made publicly available through (Swoboda et al. 2022) as ILPs. We train on 10 instances and test on the remaining 20.

**Independent set (IS):** Random instances of independent set problem generated using (Prouvost et al. 2020). For training we use 240 instances with $10k$ nodes each and test on 60 instances with $50k$ nodes.

**QAPLib:** The benchmark dataset for quadratic assignment problems used in the combinatorial optimization community (Burkard, Karisch, and Rendl 1997). The benchmark contains problems arising from a variety of domains e.g., keyboard design, hospital layout, circuit synthesis, facility location etc. We train on 61 instances having up to $0.6M$ Lagrange variables and test on 35 instances having up to $10.6M$ Lagrange variables. Conversion to ILP is done via (2.7)-(2.16) of (Loiola et al. 2007)

For each dataset the size of problems (D) are reported in Table 3. Due to varying instance sizes we use a separate set of hyperparameters for each dataset given in Table 5 of the Appendix². For the CT dataset we only predict $\theta \in \mathbb{R}^{|\lambda|}$ for non-parametric update steps (2) and fix the parameters $\alpha, \omega$ in Alg. 1 to their default values from (Abbas and Swoboda 2022b). Learning these parameters gave slightly worse training loss at convergence possibly due to small training set size.

**Table 3: Statistics of datasets where the values are averaged within each train/test split. Number of edges in the GNN equal the number of Lagrange multipliers $\lambda$.**

| Dataset | # variables ($\times 10^6$) |          | # constraints ($\times 10^6$) |          | # edges ($\times 10^6$) |          |
|---------|-----------------------------|----------|-------------------------------|----------|-------------------------|----------|
|         | train                       | test     | train                         | test     | train                   | test     |
| CT      | 3.1                         | 10.1     | 0.6                           | 2.2      | 8.5                     | 27.5     |
| GM      | 1.5                         | 0.1      | 1.5                           | 0.1      | 3.3                     | 3.3      |
| IS      | 0.01                        | 0.05     | 0.04                          | 0.4      | 0.1                     | 1.1      |
| QAPLib  | 0.1                         | 2.5      | 0.02                          | 0.2      | 0.6                     | 10.6     |

---

### 4.3 Algorithms

**Gurobi:** The dual simplex algorithm from the commercial solver (Gurobi Optimization, LLC 2021).

**Spec.:** For graph matching and cell tracking datasets we also report results of state-of-the-art dataset specific solvers. For cell tracking the solver of (Haller et al. 2020) and for graph matching the best performing solver (fm-bca) from recent benchmark (Haller et al. 2022).

---

²Appendix in our full paper (Abbas and Swoboda 2022a).

<!-- Page 6 -->
Table 1: Results on tests instances where the values are averaged within a dataset. Numbers in bold highlight the best performance and underlines indicate the second best objective.

|                  | **Cell tracking**          |                          |                          | **Graph matching**         |                          |                          | **Independent set**        |                          |                          | **QAPLib**                 |                          |                          |
|------------------|----------------------------|--------------------------|--------------------------|----------------------------|--------------------------|--------------------------|----------------------------|--------------------------|--------------------------|----------------------------|--------------------------|--------------------------|
|                  | $g_I$                      | $E(\times 10^8)$         | $t[s]$                   | $g_I$                      | $E(\times 10^4)$         | $t[s]$                   | $g_I$                      | $E(\times 10^4)$         | $t[s]$                   | $g_I$                      | $E(\times 10^6)$         | $t[s]$                   |
| Gurobi           | 18                         | **-3.852**               | 809                      | 9                          | **-4.8433**              | 278                      | 14                         | **-2.4457**              | 52                       | 3472                       | 0.9                      | 2618                     |
| Spec.            | -                          | -3.866                   | 1673                     | -                          | -4.8443                  | 100                      | -                          | -                        | -                        | -                          | -                        | -                        |
| FastDOG          | 7                          | -3.863                   | 1005                     | 21                         | -4.8912                  | 61                       | 42                         | -2.4913                  | 9                        | 276                        | 5.7                      | 1680                     |
| DOGE             | 2.4                        | -3.854                   | 1015                     | 0.3                        | -4.8439                  | 17                       | 0.3                        | -2.4460                  | 8                        | 320                        | **12.1**                 | 720                      |
| DOGE-M           | **2.1**                    | _-3.854_                 | 730                      | **0.2**                    | _-4.8436_                | 21                       | **0.2**                    | _-2.4459_                | 5                        | **131**                    | **14.5**                 | 861                      |

Table 2: Ablation study results on the *Graph matching* dataset. w/o GNN: Use only the two predictors $\Phi$ without GNN for early and late stage optimization; same network: use one network (GNN, $\Phi$) for both early and late stage; only non-param., param.: predict only the non-parametric update (2) or the parametric update (Alg. 1); w/o $\alpha$, $\omega$: does not predict $\alpha$ or $\omega$ resp.

|                  | w/o learn. (FastDOG) | w/o GNN | same network | only non-param. | only param. | w/o $\alpha$ | w/o $\omega$ | DOGE   | DOGE-M |
|------------------|----------------------|---------|--------------|-----------------|-------------|--------------|--------------|--------|--------|
| $g_I$ ($\downarrow$) | 21                   | 0.42    | 0.95         | 2.3             | 0.7         | 0.36         | 0.35         | 0.33   | **0.19** |
| $E$ ($\uparrow$)     | -48912               | -48440  | -48444       | -48476          | -48444      | -48439       | -48439       | -48439 | **-48436** |
| $t[s]$ ($\downarrow$)| 61                   | 29      | 24           | 51              | 74          | 30           | 30           | 17     | 21     |

Figure 2: Convergence plots for $g(t)$ the relative dual gap to the optimum (or maximum suboptimal objective among all methods) of the relaxation (D). X-axis indicates wall clock time and both axes are logarithmic. The value of $g(t)$ is averaged over all test instances in each dataset.

<!-- Page 7 -->
FastDOG: The non-learned baseline (Abbas and Swoboda 2022b) with their hand designed parameters $\omega_{ij} = 0.5$ and $\alpha_{ij} = 1/|\mathcal{I}_i|$ as a specialization of Alg. 1.

DOGE: Our approach where we learn to predict parametric and non-parametric updates by using two graph neural networks for early and late-stage optimization. Size of the learned embeddings $h$ computed by the GNN in Alg. 3 is set to 16 for nodes and 8 for edges. For computing attention weights in (3) we use only one attention head for efficiency. The predictor $\Phi$ in Alg. 3 contains 4 linear layers with the ReLU activation. We train the networks using the Adam optimizer (Kingma and Ba 2014). To prevent gradient overflow we use gradient clipping on model parameters by an $l^2$ norm of 50. The number of trainable parameters is $8k$.

DOGE-M: Variant of our method where we additionally use recurrent connections using LSTM. The cell state vector $s_i$ for each primal variable node $i \in \mathcal{I}$ has a size of 16. The number of trainable parameters is $12k$.

Note the test instances require millions of solver parameters to be predicted (ref. Table 3) while our largest GNN has $12k$ parameters.

For training we use PyTorch and implement the Algorithms 1 and 2 in CUDA. CPU solvers use AMD EPYC 7702 CPU with 16 threads. GPU solvers use either an NVIDIA RTX 8000 (48GB) or a A100 (80GB) GPU depending on problem size.

## 4.4 Results

For each dataset we evaluate our methods on corresponding testing split. Convergence plots of relative dual gaps change (averaged over all test instances) are given in Figure 2. Other evaluation metrics are reported in Table 1. For further details we refer to the Appendix.

### Discussion

As compared to the non-learned baseline FastDOG we reach an order of magnitude more accurate relaxation solutions, almost closing the gap to optimum as computed by Gurobi. Even though given unlimited time Gurobi attains the optimum, we reach reasonably close values that are considered correct for practical purposes. For example the graph matching benchmark (Haller et al. 2022) considers a relative gap less than $10^{-3}$ as optimal (we achieve $10^{-4}$). Moreover our learned solvers reach much better objective values as compared to specialized solvers. Using LSTM in DOGE-M further improves the performance especially on the most difficult QAPLib dataset. On QAPLib Gurobi does not converge on instances with more than 40 nodes within the time limit of one hour. We show convergence plots for smaller instances in the Appendix. The difference to Gurobi is most pronounced w.r.t. anytime performance measured by $g_t$, as our solver reaches good solutions relatively early.

### Ablation study

We evaluate the importance of various components in our approach. Starting from (Abbas and Swoboda 2022b) as a baseline we first predict all parameters $\alpha, \omega, \theta$ through the two multi-layer perceptrons $\Phi$ for early and late stage optimization without using GNN. Next, we report results of using one network (instead of two) which is trained and tested for both early and later rounds of dual optimization. Lastly, we aim to seek the importance of learning parameters of Alg. 2 and the non-parametric update (2). To this end, we learn to predict only the non-parametric update and apply the loss directly on updated $\lambda$ without requiring backpropagation through Alg. 1. We also try learning a subset of parameters i.e., not predicting averaging weights $\alpha$ or damping factors $\omega$. Lastly, we report results of DOGE-M which uses recurrent connections. The results for graph matching dataset are in Table 2. Results on other datasets are provided in the Appendix.

Firstly, from our ablation study we observe that learning even one of the two types of updates i.e., non-parametric or parametric already gives better results than the non-learned solver FastDOG. This is because non-parametric update can help in escaping fixed-points when they occur and the parametric update can help Alg. 1 in avoiding such fixed-points. Combining both of these strategies further improves the results. Secondly, we observe that performing message passing with GNN gives improvement over only using the MLP $\Phi$. Thirdly, we find using separate networks for early and late stage optimization gives better performance than using the same network for all stages. Lastly, using recurrent connections through an LSTM gives the best performance.

### Limitations

Easy problem classes, including small cell tracking (Haller et al. 2020) and easy Markov Random Field (MRF) inference (Kappes et al. 2013) do not benefit from learning, since FastDOG already solves the problem in few iterations. Some problem classes have sequential bottlenecks due to long subproblems, including MRFs for protein folding (Jaimovich et al. 2006) and shape matching (Windheuser et al. 2011a,b), which makes training difficult due to slow dual optimization.

Although our method requires training for each problem class, the cost of training is manageable. Nonetheless devising a generalizable approach is an interesting research direction requiring at least: a large and diverse training set, powerful neural network and multi-GPU implementation.

## 5 Conclusion

We have proposed an self-supervised learning approach for solving relaxations to combinatorial optimization problems by backpropagating through and learning parameters for the dual LP solver (Abbas and Swoboda 2022b). We demonstrated its potential in obtaining close to optimal solutions much faster than with traditional methods. Although our solvers require training as compared to conventional solvers, this overhead is negligible as compared to the human effort required for developing efficient specialized solvers (which are also often outperformed by our approach).

Our work generalizes efficient approximate solver development: Instead of developing a specialized solver we propose to use a generically applicable one and train it to obtain fast and accurate optimization algorithm. Going one step further and training a universal model that generalizes across different problem classes remains a challenge for future work.

<!-- Page 8 -->
# Acknowledgements

We thank all anonymous reviewers for their feedback especially reviewer 4 for the detailed discussion and pointing out related work. We also thank Paul Roetzer for suggestions regarding writing.

# References

Abbas, A.; and Swoboda, P. 2022a. DOGE-Train: Discrete Optimization on GPU with End-to-end Training. (Current article with the Appendix). arXiv:2205.11638.

Abbas, A.; and Swoboda, P. 2022b. FastDOG: Fast Discrete Optimization on GPU. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.

Andrychowicz, M.; Denil, M.; Gomez, S.; Hoffman, M. W.; Pfau, D.; Schaul, T.; Shillingford, B.; and De Freitas, N. 2016. Learning to learn by gradient descent by gradient descent. *Advances in neural information processing systems*, 29.

Ba, J. L.; Kiros, J. R.; and Hinton, G. E. 2016. Layer normalization. *arXiv preprint arXiv:1607.06450*.

Bengio, Y.; Lodi, A.; and Prouvost, A. 2021. Machine learning for combinatorial optimization: a methodological tour d’horizon. *European Journal of Operational Research*, 290(2): 405–421.

Berthold, T. 2013. Measuring the Impact of Primal Heuristics. *Oper. Res. Lett.*, 41(6): 611–614.

Burkard, R. E.; Karisch, S. E.; and Rendl, F. 1997. QAPLIB—a quadratic assignment problem library. *Journal of Global optimization*, 10(4): 391–403.

Cameron, C.; Chen, R.; Hartford, J.; and Leyton-Brown, K. 2020. Predicting Propositional Satisfiability via End-to-End Learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(04): 3324–3331.

Cappart, Q.; Chételat, D.; Khalil, E. B.; Lodi, A.; Morris, C.; and Velickovic, P. 2023. Combinatorial optimization and reasoning with graph neural networks. *J. Mach. Learn. Res.*, 24: 130–1.

Cappart, Q.; Goutierre, E.; Bergman, D.; and Rousseau, L.-M. 2019. Improving optimization bounds using machine learning: Decision diagrams meet deep reinforcement learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, 1443–1451.

Chen, Y.; and Pock, T. 2017. Trainable Nonlinear Reaction Diffusion: A Flexible Framework for Fast and Effective Image Restoration. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(6): 1256–1272.

Cplex, IBM ILOG. 2019. CPLEX Optimization Studio 12.10.

Deng, Y.; Kong, S.; Liu, C.; and An, B. 2022. Deep Attentive Belief Propagation: Integrating Reasoning and Learning for Solving Constraint Optimization Problems. *Advances in Neural Information Processing Systems*, 35: 25436–25449.

Ding, J.-Y.; Zhang, C.; Shen, L.; Li, S.; Wang, B.; Xu, Y.; and Song, L. 2020. Accelerating primal solution findings for mixed integer programs based on solution prediction. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, 1452–1459.

Falcon, W.; and The PyTorch Lightning team. 2019. PyTorch Lightning.

Fey, M.; and Lenssen, J. E. 2019. Fast Graph Representation Learning with PyTorch Geometric. In *ICLR Workshop on Representation Learning on Graphs and Manifolds*.

FICO. 2022. FICO Xpress Optimization Suite.

Gamrath, G.; Anderson, D.; Bestuzheva, K.; Chen, W.-K.; Eifler, L.; Gasse, M.; Gemander, P.; Gleixner, A.; Gottwald, L.; Halbig, K.; Hendel, G.; Hojny, C.; Koch, T.; Bodic, P. L.; Maher, S. J.; Matter, F.; Miltenberger, M.; Mühlner, E.; Müller, B.; Pfetsch, M.; Schlösser, F.; Serrano, F.; Shinano, Y.; Tawfik, C.; Vigerske, S.; Wegscheider, F.; Weninger, D.; and Witzig, J. 2020. The SCIP Optimization Suite 7.0. Technical Report 20-10, ZIB, Takustr. 7, 14195 Berlin.

Gasse, M.; Chételat, D.; Ferroni, N.; Charlin, L.; and Lodi, A. 2019. Exact combinatorial optimization with graph convolutional neural networks. *arXiv preprint arXiv:1906.01629*.

Gers, F.; Schmidhuber, J.; and Cummins, F. 1999. Learning to forget: continual prediction with LSTM. In *1999 Ninth International Conference on Artificial Neural Networks ICANN 99*. (Conf. Publ. No. 470), volume 2, 850–855 vol.2.

Gregor, K.; and LeCun, Y. 2010. Learning Fast Approximations of Sparse Coding. In *Proceedings of the 27th International Conference on International Conference on Machine Learning*, ICML’10, 399–406. Madison, WI, USA: Omnipress. ISBN 9781605589077.

Gupta, P.; Gasse, M.; Khalil, E.; Mudigonda, P.; Lodi, A.; and Bengio, Y. 2020. Hybrid models for learning to branch. *Advances in neural information processing systems*, 33: 18087–18097.

Gurobi Optimization, LLC. 2021. Gurobi Optimizer Reference Manual.

Haller, S.; Feineis, L.; Hutschenreiter, L.; Bernard, F.; Rother, C.; Kainmüller, D.; Swoboda, P.; and Savchynskyy, B. 2022. A Comparative Study of Graph Matching Algorithms in Computer Vision. In *Proceedings of the European Conference on Computer Vision*.

Haller, S.; Prakash, M.; Hutschenreiter, L.; Pietzsch, T.; Rother, C.; Jug, F.; Swoboda, P.; and Savchynskyy, B. 2020. A Primal-Dual Solver for Large-Scale Tracking-by-Assignment. In *AISTATS*.

Hoberock, J.; and Bell, N. 2010. Thrust: A Parallel Template Library. Version 1.7.0.

Huang, Z.; Wang, K.; Liu, F.; Zhen, H.-L.; Zhang, W.; Yuan, M.; Hao, J.; Yu, Y.; and Wang, J. 2022. Learning to select cuts for efficient mixed-integer programming. *Pattern Recognition*, 123: 108353.

Hutschenreiter, L.; Haller, S.; Feineis, L.; Rother, C.; Kainmüller, D.; and Savchynskyy, B. 2021. Fusion moves for graph matching. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 6270–6279.

Jaimovich, A.; Elidan, G.; Margalit, H.; and Friedman, N. 2006. Towards an integrated protein–protein interaction network: A relational markov network approach. *Journal of Computational Biology*, 13(2): 145–164.

Jakob, W.; Rhinelander, J.; and Moldovan, D. 2017. pybind11 – Seamless operability between C++11 and Python. https://github.com/pybind/pybind11.

Kainmueller, D.; Jug, F.; Rother, C.; and Myers, G. 2014. Active graph matching for automatic joint segmentation and annotation of C. elegans. In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 81–88. Springer.

Kappes, J.; Andres, B.; Hamprecht, F.; Schnorr, C.; Nowozin, S.; Batra, D.; Kim, S.; Kausler, B.; Lellmann, J.; Komodakis, N.; et al. 2013. A comparative study of modern inference techniques for discrete energy minimization problems. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 1328–1335.

Kingma, D. P.; and Ba, J. 2014. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

Lange, J.-H.; and Swoboda, P. 2021. Efficient Message Passing for 0–1 ILPs with Binary Decision Diagrams. In *International Conference on Machine Learning*, 6000–6010. PMLR.

Loiola, E. M.; De Abreu, N. M. M.; Boaventura-Netto, P. O.; Hahn, P.; and Querido, T. 2007. A survey for the quadratic assignment problem. *European journal of operational research*, 176(2): 657–690.

<!-- Page 9 -->
Long, F.; Peng, H.; Liu, X.; Kim, S. K.; and Myers, E. 2009. A 3D digital atlas of C. elegans and its application to single-cell analyses. *Nature methods*, 6(9): 667–672.

Monga, V.; Li, Y.; and Eldar, Y. C. 2021. Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing. *IEEE Signal Processing Magazine*, 38(2): 18–44.

MOSEK ApS. 2022. 9.0.105.

Nair, V.; Bartunov, S.; Gimeno, F.; von Glehn, I.; Lichocki, P.; Lobov, I.; O’Donoghue, B.; Sonnerat, N.; Tjandraatmadja, C.; Wang, P.; et al. 2020. Solving mixed integer programs using neural networks. *arXiv preprint arXiv:2012.13349*.

Nazari, M.; Oroojlooy, A.; Snyder, L.; and Takác, M. 2018. Reinforcement learning for solving the vehicle routing problem. *Advances in neural information processing systems*, 31.

NVIDIA; Vingelmann, P.; and Fitzek, F. H. 2021. CUDA, release: 11.2.

Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; Desmaison, A.; Kopf, A.; Yang, E.; DeVito, Z.; Raison, M.; Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.; Bai, J.; and Chintala, S. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Wallach, H.; Larochelle, H.; Beygelzimer, A.; d’Alché-Buc, F.; Fox, E.; and Garnett, R., eds., *Advances in Neural Information Processing Systems 32*, 8024–8035. Curran Associates, Inc.

Paulus, M. B.; Zarpellon, G.; Krause, A.; Charlin, L.; and Maddison, C. 2022. Learning to cut by looking ahead: Cutting plane selection via imitation learning. In *INTERNATIONAL conference on machine learning*, 17584–17600. PMLR.

Prouvost, A.; Dumouchelle, J.; Scavuzzo, L.; Gasse, M.; Chételat, D.; and Lodi, A. 2020. Ecole: A Gym-like Library for Machine Learning in Combinatorial Optimization Solvers. In *Learning Meets Combinatorial Algorithms at NeurIPS2020*.

Qiu, R.; Sun, Z.; and Yang, Y. 2022. DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems. In Oh, A. H.; Agarwal, A.; Belgrave, D.; and Cho, K., eds., *Advances in Neural Information Processing Systems*.

Scavuzzo, L.; Chen, F.; Chételat, D.; Gasse, M.; Lodi, A.; Yorke-Smith, N.; and Aardal, K. 2022. Learning to branch with tree mdps. *Advances in Neural Information Processing Systems*, 35: 18514–18526.

Selsam, D.; Lamm, M.; Bünz, B.; Liang, P.; de Moura, L.; and Dill, D. L. 2018. Learning a SAT solver from single-bit supervision. *arXiv preprint arXiv:1802.03685*.

Shi, Y.; Huang, Z.; Feng, S.; Zhong, H.; Wang, W.; and Sun, Y. 2021. Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification. In Zhou, Z.-H., ed., *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21*, 1548–1554. International Joint Conferences on Artificial Intelligence Organization. Main Track.

Sonnerat, N.; Wang, P.; Ktena, I.; Bartunov, S.; and Nair, V. 2021. Learning a Large Neighborhood Search Algorithm for Mixed Integer Programs. *arXiv preprint arXiv:2107.10201*.

Sun, Z.; and Yang, Y. 2023. DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization. *arXiv preprint arXiv:2302.08224*.

Swoboda, P.; Hornakova, A.; Roetzer, P.; and Abbas, A. 2022. Structured Prediction Problem Archive. *arXiv preprint arXiv:2202.03574*.

Turner, M.; Koch, T.; Serrano, F.; and Winkler, M. 2022. Adaptive Cut Selection in Mixed-Integer Linear Programming. *arXiv preprint arXiv:2202.10962*.

Tönshoff, J.; Ritzert, M.; Wolf, H.; and Grohe, M. 2021. Graph Neural Networks for Maximum Constraint Satisfaction. *Frontiers in Artificial Intelligence*, 3.

Ulman, V.; Maška, M.; Magnusson, K. E.; Ronneberger, O.; Haubold, C.; Harder, N.; Matula, P.; Matula, P.; Svoboda, D.; Radojevic, M.; et al. 2017. An objective comparison of cell-tracking algorithms. *Nature methods*, 14(12): 1141–1152.

Wang, R.; Yan, J.; and Yang, X. 2021. Neural graph matching network: Learning lawler’s quadratic assignment problem with extension to hypergraph and multiple-graph matching. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(9): 5261–5279.

Werner, T. 2007. A linear programming approach to max-sum problem: A review. *IEEE transactions on pattern analysis and machine intelligence*, 29(7): 1165–1179.

Werner, T.; Prusa, D.; and Dlask, T. 2020. Relative Interior Rule in Block-Coordinate Descent. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

Windheuser, T.; Schlickwei, U.; Schmidt, F. R.; and Cremers, D. 2011a. Geometrically consistent elastic matching of 3d shapes: A linear programming solution. In *2011 International Conference on Computer Vision*, 2134–2141. IEEE.

Windheuser, T.; Schlickwei, U.; Schmidt, F. R.; and Cremers, D. 2011b. Large-scale integer linear programming for orientation preserving 3d shape matching. In *Computer Graphics Forum*, volume 30, 1471–1480. Wiley Online Library.

Wu, Y.; Song, W.; Cao, Z.; and Zhang, J. 2021. Learning Large Neighborhood Search Policy for Integer Programming. *Advances in Neural Information Processing Systems*, 34.

Xin, L.; Song, W.; Cao, Z.; and Zhang, J. 2021. NeuroLKH: Combining Deep Learning Model with Lin-Kernighan-Helsgaun Heuristic for Solving the Traveling Salesman Problem. *Advances in Neural Information Processing Systems*, 34.

Yang, Y.; Sun, J.; Li, H.; and Xu, Z. 2020. ADMM-CSNet: A Deep Learning Approach for Image Compressive Sensing. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(3): 521–538.

<!-- Page 10 -->
# Appendix

## A Proofs

### A.1 Proof of Proposition 1

The proof is an adaptation of the corresponding proof for $\omega_{ij} = 0.5$ and $\alpha_{ij} = \frac{1}{|\mathcal{J}_i|}$ given in (Abbas and Swoboda 2022b).

*Proof.*

**Feasibility of iterates.** We prove
$$
\sum_{j \in \mathcal{J}_i} \lambda_i^j + M_{ik} = c_i \tag{5}
$$
just after line 4 and 6 in Algorithm 1. We do an inductive proof over the number of iterates w.r.t iterations $t$.

$t = 0$: 
- After 4: Follows from $M = 0$ in line 1.
- After 6: Let $\lambda'$, $M'$, be the values that are used as input to line 6 and $\lambda$ and $M$ be the ones returned in line 6. It holds that
$$
\sum_{j \in \mathcal{J}_i} [\lambda_{ij} + M_{ij}] = \sum_{j \in \mathcal{J}_i} \left[ \lambda'_{ij} - M_{ij}(t) + \alpha_{ij} \sum_{k \in \mathcal{J}_i} (M'_{ik}) + M_{ij} \right] \tag{6}
$$
$$
= \sum_{j \in \mathcal{J}_i} \left[ \lambda'_{ij} + M'_{ij} \right] \tag{7}
$$
$$
= c_i \,. \tag{8}
$$
by the proved inequality on $\lambda'$, $M'$ and the assumption that $\sum_{k \in \mathcal{J}_i} \alpha_{ij} = 1$.

$t > 0$: Analogously to the second point for $t = 0$.

**Non-decreasing Lower Bound.** In order to prove that iterates have non-decreasing lower bound we will consider an equivalent lifted representation in which proving the non-decreasing lower bound will be easier.

**Lifted Representation.** Introduce $\lambda_{ij}^\beta$ for $\beta \in \{0,1\}$ and the subproblems
$$
E(\lambda_{\bullet j}^1, \lambda_{\bullet j}^0) = \min_{x \in \mathcal{X}_j} x^\top \lambda_{\bullet j}^1 + (1 - x)^\top \lambda_{\bullet j}^0 \tag{9}
$$
Then (D) is equivalent to
$$
\max_{\lambda^1, \lambda^0} \sum_{j \in \mathcal{J}} E(\lambda_{\bullet j}^1, \lambda_{\bullet j}^0) \text{ s.t. } \sum_{j \in \mathcal{J}_i} \lambda_{ij}^\beta = \beta \cdot c_i \tag{10}
$$
We have the transformation from original to lifted $\lambda$
$$
\lambda \mapsto (\lambda^1 \leftarrow \lambda, \lambda^0 \leftarrow 0) \tag{11}
$$
and from lifted to original $\lambda$ (except a constant term)
$$
(\lambda^1, \lambda^0) \mapsto \lambda^1 - \lambda^0 \,. \tag{12}
$$
It can be easily shown that the lower bounds are invariant under the above mappings and feasible $\lambda$ for (D) are mapped to feasible ones for (10) and vice versa.

The update rule line 11 in Algorithm 1 for the lifted representation can be written as
$$
\lambda_{ij}^\beta \leftarrow \lambda_{ij}^\beta - \max((2\beta - 1) M_{ij}^{out}, 0) + \alpha_{ij} \cdot \sum_{jk \in \mathcal{J}_i} \min((2\beta - 1) M_{ik}^{in}, 0) \tag{13}
$$
It can be easily shown that (13) and line 11 in Algorithm 1 are corresponding to each other under the transformation from lifted to original $\lambda$.

**Continuation of Non-decreasing Lower Bound Define**
$$
\lambda_{ij}'^\beta = \lambda_{ij} - \omega_{ij} \cdot \max((2\beta - 1)( \min_{x \in \mathcal{X}_j : x_j = \beta} \langle \lambda_{ij}^{in}, x \rangle - \min_{x \in \mathcal{X}_j : x_i = 1 - \beta} \langle \lambda_{ij}^{in}, x \rangle ), 0 ) \,. \tag{14}
$$
Then $E(\lambda'^{j,1}, \lambda'^{j,0}) = E(\lambda^{j,1}, \lambda^{j,0})$ are equal due to $\omega_{ij} \in [0,1]$. Define next
$$
\lambda_{ij}''^\beta = \lambda_{ij}' + \alpha_{ij} \sum_{k \in \mathcal{J}_i} \max((2\beta - 1) M_{ik}^{in}, 0) \,. \tag{15}
$$
Then $E(\lambda''^{j,1}, \lambda''^{j,0}) \geq E(\lambda'^{j,1}, \lambda'^{j,0})$ since $\lambda'' \geq \lambda'$ elementwise. This proves the claim.
$\square$

<!-- Page 11 -->
Figure 3: Computational graph of `BlockUpdate` in Alg. 1

## A.2 Proof of Proposition 2

**Proof.** The computational graph of `BlockUpdate` in Alg. 1 is shown in Figure 3. Assuming gradients $\partial \mathcal{L} / \partial M^{\text{out}}$ and $\partial \mathcal{L} / \partial \lambda^{\text{out}}$ are given. We first focus on lower part of Figure 3. By applying chain rule gradient of $M_{ij}^{\text{in}} \, \forall ij \in B$ is computed as

$$
\frac{\partial \mathcal{L}}{\partial M_{ij}^{\text{in}}} = \sum_{p \in \mathcal{I}} \sum_{k \in \mathcal{J}_p} \frac{\partial \mathcal{L}}{\partial \lambda_{pk}^{\text{out}}} \frac{\partial \lambda_{pk}^{\text{out}}}{\partial M_{ij}^{\text{in}}} = \sum_{k \in \mathcal{J}_i} \frac{\partial \mathcal{L}}{\partial \lambda_{ik}^{\text{out}}} \frac{\partial \lambda_{ik}^{\text{out}}}{\partial M_{ij}^{\text{in}}} = \sum_{k \in \mathcal{J}_i} \frac{\partial \mathcal{L}}{\partial \lambda_{ik}^{\text{out}}} \alpha_{ij}.
\tag{16}
$$

Similarly gradient for $\alpha_{ij} \, \forall ij \in B$ is

$$
\frac{\partial \mathcal{L}}{\partial \alpha_{ij}} = \sum_{p \in \mathcal{I}} \sum_{k \in \mathcal{J}_p} \frac{\partial \mathcal{L}}{\partial \lambda_{pk}^{\text{out}}} \frac{\partial \lambda_{pk}^{\text{out}}}{\partial \alpha_{ij}} = \frac{\partial \mathcal{L}}{\partial \lambda_{ij}^{\text{out}}} \frac{\partial \lambda_{ij}^{\text{out}}}{\partial \alpha_{ij}} = \frac{\partial \mathcal{L}}{\partial \lambda_{ij}^{\text{out}}} \sum_{k \in \mathcal{J}_i} M_{ik}^{\text{in}},
\tag{17}
$$

Since we allow running Alg. 1 for more than one iteration with same parameters $(\alpha, \omega)$, the above gradient (17) is accumulated to existing gradients of $\alpha$ to obtain the result given by Alg. 2.

For the upper part of Figure 3 we first backpropagate gradients of $\lambda^{\text{out}}$ to $M^{\text{out}}$ to account for subtraction ($-$) as

$$
\frac{\partial \mathcal{L}}{\partial M^{\text{out}}} = \frac{\partial \mathcal{L}}{\partial M^{\text{out}}} - \frac{\partial \mathcal{L}}{\partial \lambda^{\text{out}}}.
\tag{18}
$$

Then the gradient w.r.t. damping factors $\omega_{ij} \, \forall ij \in B$ is

$$
\frac{\partial \mathcal{L}}{\partial \omega_{ij}} = \frac{\partial \mathcal{L}}{\partial M_{ij}^{\text{out}}} \frac{\partial M_{ij}^{\text{out}}}{\partial \omega_{ij}} = \frac{\partial \mathcal{L}}{\partial M_{ij}^{\text{out}}} \left( m_{ij}^1 - m_{ij}^0 \right) = \frac{\partial \mathcal{L}}{\partial M_{ij}^{\text{out}}} \left( \frac{M_{ij}^{\text{out}}}{\omega_{ij}} \right),
\tag{19}
$$

which also needs to be accumulated to existing gradient as done for gradients of $\alpha$.

Lastly to backpropagate gradients to $\lambda^{\text{in}}$ we first calculate

$$
\frac{\partial \mathcal{L}}{\partial m_{ij}^0} = \frac{\partial \mathcal{L}}{\partial M_{ij}^{\text{out}}} \frac{\partial M_{ij}^{\text{out}}}{\partial m_{ij}^0} = -\frac{\partial \mathcal{L}}{\partial M_{ij}^{\text{out}}} \omega_{ij},
\tag{20a}
$$

$$
\frac{\partial \mathcal{L}}{\partial m_{ij}^1} = \frac{\partial \mathcal{L}}{\partial M_{ij}^{\text{out}}} \frac{\partial M_{ij}^{\text{out}}}{\partial m_{ij}^1} = \frac{\partial \mathcal{L}}{\partial M_{ij}^{\text{out}}} \omega_{ij}.
\tag{20b}
$$

Then (sub-)gradient of min-marginals $m_{ij}^0, m_{ij}^1 \, \forall ij \in B$ w.r.t. $\lambda^{\text{in}}$ are

$$
\frac{\partial m_{ij}^\beta}{\partial \lambda} = \frac{\partial m_{ij}^\beta}{\partial \lambda_{\bullet j}} = \operatorname{argmin}_{x \in \mathcal{X}_j : x_{ij} = \beta} \langle \lambda_{\bullet j}, x \rangle, \quad \forall \beta \in \{0,1\}.
\tag{21}
$$

<!-- Page 12 -->
Using the above relations (20), (21) and applying chain rule we obtain

$$
\frac{\partial \mathcal{L}}{\partial \lambda_{ij}^{\text{in}}} = \frac{\partial \mathcal{L}}{\partial \lambda_{ij}^{\text{out}}} + \sum_{\beta \in \{0,1\}} \sum_{p \in \mathcal{I}} \sum_{k \in \mathcal{J}_p} \frac{\partial \mathcal{L}}{\partial m_{pk}^{\beta}} \frac{\partial m_{pk}^{\beta}}{\lambda_{ij}^{\text{in}}}
\tag{22a}
$$

$$
= \frac{\partial \mathcal{L}}{\partial \lambda_{ij}^{\text{out}}} + \sum_{\beta \in \{0,1\}} \sum_{p \in \mathcal{I}_j} \frac{\partial \mathcal{L}}{\partial m_{pj}^{\beta}} \frac{\partial m_{pj}^{\beta}}{\lambda_{ij}^{\text{in}}}, \forall ij \in B.
\tag{22b}
$$

$\square$

## B Efficient min-marginal computation and backpropagation

Algorithms 1 and 2 in abstract terms require solving the subproblems each time a min-marginal value (or its gradient) is required. To make these procedures more efficient we represent each subproblem as binary decision diagrams (BDD) as done in (Abbas and Swoboda 2022b). We give a short overview below and refer to (Abbas and Swoboda 2022b) for more details.

**Binary decision diagrams (BDD).** A BDD is a directed acyclic graph with arc set $A$ starting at a root node $r$ and ending at two nodes $\top$ and $\bot$. For each variable $i$ the BDD contains one or more nodes in a set $\mathcal{P}_i$ where all $r \top$ paths pass through exactly one node in $\mathcal{P}_i$. All $r \top$ paths in the BDD correspond to feasible assignments of its corresponding subproblem. Lagrange variables of the subproblem can be used as weights in BDD arcs allowing also to calculate cost of these $r \top$ paths. This is done by creating two outgoing arcs for a node $v$ (except $\top$, $\bot$) in the BDD: a zero arc $vs^0(v)$ and a one arc $vs^1(v)$. If an $r \top$ path passes through zero arc $vs^0(v)$ it indicates that the corresponding variable has an assignment of 0 and 1 otherwise.

Therefore to compute the cost of assigning a 1 to variable $i$ one needs to check all $r \top$ paths which make use of the one arcs from all nodes in $\mathcal{P}_i$. In (Abbas and Swoboda 2022b) the authors compute min-marginals by maintaining shortest path distances. Each node $v$ in the BDD maintains the cost of shortest path from root node $r$ (denoted by $\text{SP}(r,v)$) and cost of shortest path to $\top$ node. These path costs are updated in BlockUpdate routine of Alg. 1. Min-marginals $m^0, m^1$ for a variable $i$ in subproblem $j$ can be computed efficiently as

$$
m_{ij}^{\beta} = \min_{\substack{vs^{\beta}(v) \in A \\ v \in \mathcal{P}_i}} \left[ \text{SP}(r,v) + \beta \cdot \lambda_{ij} + \text{SP}(s^{\beta}(v), \top) \right].
\tag{23}
$$

Backpropagation through min-marginals $m^0, m^1$ can then be done by finding the argmin in (23) instead of the min operation. Afterwards the gradients can be passed to Lagrange variables $\lambda$ and shortest path costs $\text{SP}(r,\cdot)$, $\text{SP}(\cdot,\top)$ which minimize (23). Since shortest path costs are also computed by min operations (see Alg. 3, 4 in (Abbas and Swoboda 2022b)), gradients of these path costs can subsequently be backpropagated to the Lagrange variables by the argmin operation.

**CUDA implementation:** Although above-mentioned operations help towards an efficient implementation via scatter, gather operations available in Pytorch (Paszke et al. 2019) and Pytorch Geometric (Fey and Lenssen 2019), GPU memory usage can still be high for large instances. This can be especially problematic for training since the GNN also needs a considerable amount of GPU memory. Therefore for further computational efficiency we implement both Algorithm 1 and its backpropagation in CUDA (NVIDIA, Vingelmann, and Fitzek 2021; Hoberock and Bell 2010) and expose via pybind (Jakob, Rhinelander, and Moldovan 2017). A comparison between our Pytorch implementation which relies on automatic differentiation with our CUDA implementation is given in Table 4. We observe that for all datasets containing large instances (i.e., all datasets except Independent set) GPU memory usage is drastically reduced through our CUDA implementation. Additionally, runtimes for both forward and backward pass are reduced.

Table 4: Runtime and peak GPU memory usage statistics of one generalized min-marginal averaging iteration in Algorithm 1 and its backpropagation via Algorithm 2. F. time: Runtime in milliseconds for forward pass i.e., one iteration of Alg. 1, B. time: Runtime for its backpropagation, mem. : Maximum GPU memory in GB used during both forward and backward pass. The values are averaged over all training instances within each dataset.

|                 | Cell tracking |               |               | Graph matching |               |               | Independent set |               |               | QAPLib          |               |               |
|-----------------|---------------|---------------|---------------|----------------|---------------|---------------|-----------------|---------------|---------------|-----------------|---------------|---------------|
|                 | F. time       | B. time       | mem.          | F. time        | B. time       | mem.          | F. time         | B. time       | mem.          | F. time         | B. time       | mem.          |
| PyTorch         | 305           | 1039          | 31            | 153            | 344           | 11            | 11              | 16            | 0.7           | 43.5            | 70            | 7.3           |
| Our CUDA        | 16            | 171           | 3.4           | 7              | 68            | 1.8           | 0.7             | 5.7           | 0.7           | 1.6             | 16            | 1.6           |

<!-- Page 13 -->
# C Neural network details

## C.1 Hyperparameters

The hyperparameters used in experiments are reported in Table 5. During training time we run the Alg. 1 for only a few iterations for computational efficiency since more iterations can make the backward pass much slower due to reverse mode autodiff. For test time we run Alg. 1 for more iterations since backward pass is not required. For $QAPLib$ dataset we need more training time than other datasets because training set is quite large and has more diversity as compared to other datasets. To manage training we additionally use PyTorch lightning (Falcon and The PyTorch Lightning team 2019).

**Table 5**: Statistics of hyperparameters of our approach for each dataset. $T$: Num. of iterations of Alg. 1 in each optimization round; $R$: max. number of training rounds; # itr. train: Num. of training iterations.

| Dataset | $T$        |           | $R$  | batch size | learn. rate | # itr. train | train time [hrs] |
|---------|------------|-----------|------|------------|-------------|--------------|------------------|
|         | train      | test      |      |            |             |              |                  |
| $CT$    | 1          | 100       | 400  | 1          | 1e-3        | 500          | 14               |
| $GM$    | 20         | 200       | 20   | 2          | 1e-3        | 400          | 4                |
| $IS$    | 20         | 50        | 20   | 8          | 1e-3        | 2500         | 10               |
| $QAPLib$| 5          | 20        | 500  | 4          | 1e-3        | 1600         | 48               |

## C.2 Hand-crafted features

The features used as input to the neural networks at every optimization round are provided in Table 6. The motivation of using these features is as follows:

1. For GNN to have complete ILP description we encode primal objective vector, constraint matrix, right-hand side vector, and constraint type. For more expressiveness we encode node degrees separately.
2. We provide GNN all of the information which FastDOG (Abbas and Swoboda 2022b) uses to perform dual updates. For this purpose we encode Lagrange variables $\lambda$ and deferred min-marginal differences $M$.
3. Features which can be computed easily and can potentially help the GNN. For example the optimal solution of each subproblem which actually corresponds to super-gradient of the dual objective (D) is given. In addition we provide gradients of smoothed dual objective (D). To this end we replace each subproblem (1) with its smoothed variant as

$$
E_\alpha^j(\lambda_{\bullet j}) = \alpha \cdot \log\left( \sum_{x \in \mathcal{X}_j} \exp\left( \frac{\langle \lambda_{\bullet j}, x \rangle}{\alpha} \right) \right),
\tag{24}
$$

with varying values of smoothing factor $\alpha$. For more details we refer to Sec. A.5 of (Lange and Swoboda 2021).
4. Moving average of previously computed features. This can help the GNN to be informed about the goodness of past updates, thus allowing for (implicit) change of step-sizes.

# D Detailed results

## D.1 Generalization study

To evaluate the generalization power we test our approach on different types of instances within a problem class. To this end we take the $QAPLib$ dataset which naturally contains instances of different types of problems e.g., keyboard layout design, facility location, circuit design etc. We train our solver on each sub-category and test on all categories. Note that for training we take our original training split and subdivide into different sub-categories. The results are available in Table 7.

We observe only a few cases where our approach generalizes to instances of different types e.g. training on $tai^*$ generalizes to $lipa$ but not the other way around. We hypothesize that it is due to each dataset having limited diversity causing overfitting.

## D.2 Ablation study

Extended results of ablation study in Section 4.4 on the datasets of *Independent set* and a smaller split (for efficiency reasons) of $QAPLib$ are given in Table 8. As before we observe that learning both parametric updates by our Alg. 1 and non-parametric updates (2) are essential for good performance. Moreover learning even one of these components allows us to surpass the hand-designed algorithm FastDOG (Abbas and Swoboda 2022b).

<!-- Page 14 -->
Table 6: Features used for learning. Exponentially averaged features are computed with a smoothing factor of 0.9. Features corresponding to the ILP remain fixed (i.e. node degrees, constraint type, $c$, $A$, $b$) whereas the remaining features are updated after every optimization round.

| Types             | Feature description                                                                 |
|-------------------|-------------------------------------------------------------------------------------|
| Primal variables $f_{\mathcal{I}}$ | Normalized cost vector $c/\|c\|_\infty$<br>Node degree ($|\mathcal{J}_i| \, \forall i \in \mathcal{I}$) |
| Subproblems $f_{\mathcal{J}}$      | Node degree ($|\mathcal{I}_j| \, \forall j \in \mathcal{J}$)<br>RHS vector $b$ in constraints $Ax \leq b$<br>Indicator for constraint type ($\leq$ or $=$)<br>Current objective value per subproblem $[E^1(\lambda_{\bullet 1}), \ldots, E^m(\lambda_{\bullet j})]$<br>Exp. moving avg. of first, second order change in obj. value<br>Change in objective value due to last non-parametric update (2) |
| Dual variables $f_{\mathcal{E}}$   | Current optimal assignment of each subproblem (i.e., subgradient of dual objective (D))<br>Gradients of the smoothed dual objective using (24) for all $\alpha$ in $\{1.0, 10.0, 100.0\}$.<br>Exp. moving avg. of optimal assignment<br>Coefficients of constraint matrix $A$<br>Current (normalized) Lagrange variables $\lambda/\|\lambda + M + \epsilon\|$<br>Current (normalized) deferred min-marginal differences $M/\|\lambda + M + \epsilon\|$ |

## D.3 Results on smaller instances of QAPLib

In Figure 4 we provide additional convergence plot calculated only on smaller instances of QAPLib dataset. These instances contain on average 1.6 million dual variables (instead of the overall test split with 11 million). We observe that on relatively smaller instances our solvers DOGE, DOGE-M are surpassed by the barrier method but not by the dual simplex method of (Gurobi Optimization, LLC 2021). However, on larger instances the barrier method could not perform any iteration within 1 hour timelimit.

![Figure 4: Convergence plots of smaller test instances of QAPLib ($\leq 40$ nodes).](https://i.imgur.com/placeholder.png)

*Legend:*
- `Gurobi`: blue dash-dot line
- `FastDOG`: pink dotted line
- `DOGE`: orange dash line
- `DOGE-M`: green dash-triangle line
- `Gurobi-barrier`: purple dot-dash line

<!-- Page 15 -->
Table 7: Results of training and testing on different problem types within the $QAPLib$ dataset. Values in the table depict relative dual gap (lower values are better). We divide our original training and testing sets into smaller splits where each split contains instance of only one problem type. Each row then indicates a training run on one training split and the columns indicate results on different test instances. The instances used in training splits are $bur^*$: $bur26a$-$d$, $lipa^*$: $lipa\{20a$-$b$, $30a$-$b\}$, $nug^*$: $nug\{12$, $14$, $15$, $16a$-$b$, $17$, $18$, $20$, $21$, $22$, $24$, $25\}$, $tai^*$: $tai\{10a$-$b$, $12a$-$b$, $15a$-$b$, $17a$, $20a$-$b$, $25a$-$b$, $30a$-$b\}$, $chr^*$: $chr\{12a$-$c$, $15a$-$c$, $18a$-$b$, $20a$-$c$, $22a\}$; $full$: Our full training split of $QAPLib$ as defined in Sec 4.2 (does not contain any test instance of this experiment).

| Train \ Test | bur26e-h | lipa40a-b | nug28.30 | tai35a-b | chr22b | had20 | kra32 | rou20 | scr20 | ste36c | tho40 |
|--------------|----------|-----------|----------|----------|--------|-------|-------|-------|-------|--------|-------|
| $bur^*$      | 0.0073   | 0.3220    | 0.3394   | 0.6180   | 0.0244 | 0.2129| 0.0601| 0.3760| 0.0116| 0.1313 | 0.4951|
| $lipa^*$     | 0.3371   | 0.0024    | 0.0218   | 0.2647   | 0.0530 | 0.0003| 0.0099| 0.0024| 0.0493| 0.1198 | 0.0629|
| $nug^*$      | 0.0056   | 0.0839    | 0.1903   | 0.3390   | 0.0161 | 0.0325| 0.0348| 0.1185| 0.0111| 0.0778 | 0.1913|
| $tai^*$      | 0.0697   | 0.0067    | 0.0191   | 0.0009   | 0.0803 | 0.0001| 0.0306| 0.0000| 0.0217| 0.2430 | 0.0964|
| $chr^*$      | 0.0914   | 0.0185    | 0.0349   | 0.3050   | 0.0212 | 0.0323| 0.0354| 0.0472| 0.0094| 0.1023 | 0.0584|
| $full$       | 0.0026   | 0.0056    | 0.0009   | 0.0096   | 0.0358 | 0.0000| 0.0126| 0.0000| 0.0022| 0.0345 | 0.0491|

Table 8: Ablation study results on $Independent$ set and $QAPLib$ datasets. $FastDOG$: Non-learned baseline from (Abbas and Swoboda 2022b); only non-parametric: predict only the non-parametric update (2), only parametric: predict only the parameters of Algorithm 1 without non-parametric update.

|                | Independent set             |                             |                             |                             | QAPLib                      |                             |                             |                             |
|----------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
|                | w/o learn. ($FastDOG$)      | only non-parametric         | only parametric             | DOGE-M                      | w/o learn. ($FastDOG$)      | only non-parametric         | only parametric             | DOGE-M                      |
| $g_I$ ($\downarrow$) | 14                          | 3.12                        | 1.76                        | **0.2**                     | 1310                        | 641                         | 564                         | **301**                     |
| $E$ ($\uparrow$)    | $-24913$                    | $-24913$                    | $-24472$                    | **$-24459$**                | $3.9e6$                     | $4.3e6$                     | $3.6e6$                     | **$6.8e6$**                 |

<!-- Page 16 -->
## D.4 Cell tracking

Table 9: Detailed results on *Cell tracking* dataset. Until termination criteria contain results where we stop our solvers early w.r.t. relative improvement. These results are averaged and reported in Table 1. Best until max. itr.: We run our solver for at most 50000 iterations and report best results (so $R = 500$, $T = 100$).

| instance     | method   | Until termination criteria         |                                      |              |             | Best until max. num itr.       |                                  |              |             |
|--------------|----------|------------------------------------|--------------------------------------|--------------|-------------|--------------------------------|----------------------------------|--------------|-------------|
|              |          | $E\ (\uparrow)$                   | $g(t)\ (\downarrow)$                 | $t\ (\downarrow)$ | # itr.      | $E\ (\uparrow)$                | $g(t)\ (\downarrow)$             | $t\ (\downarrow)$ | # itr.      |
| flywing-245  | Gurobi   | -                                  | -                                    | -            | -           | -385235600                     | 0                                | 809          | -           |
|              | DOGE     | -385424704                         | 0.00108                              | 1380         | 50000       | -385424704                     | 0.00108                          | 1380         | 50000       |
|              | DOGE-M   | -385428640                         | 0.00111                              | 730          | 28900       | -385428544                     | 0.00111                          | 760          | 30100       |

## D.5 Graph matching

Table 10: Detailed results on *Graph matching* dataset. Until termination criteria contain results where we stop our solvers early w.r.t. relative improvement. These results are averaged and reported in Table 1. Best until max. itr.: We run our solver for at most 10000 iterations and report best results (so $R = 50$, $T = 200$).

| instance              | method   | Until termination criteria         |                                      |              |             | Best until max. num itr.       |                                  |              |             |
|-----------------------|----------|------------------------------------|--------------------------------------|--------------|-------------|--------------------------------|----------------------------------|--------------|-------------|
|                       |          | $E\ (\uparrow)$                   | $g(t)\ (\downarrow)$                 | $t\ (\downarrow)$ | # itr.      | $E\ (\uparrow)$                | $g(t)\ (\downarrow)$             | $t\ (\downarrow)$ | # itr.      |
| worm10-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -42557                         | 0                                | 2356         | -           |
|                       | DOGE     | -42645                             | 0.00305                              | 35           | 5200        | -42631                         | 0.0026                           | 65           | 9600        |
|                       | DOGE-M   | -42611                             | 0.0188                               | 37.5         | 5600        | -42599                         | 0.00148                          | 60           | 8800        |
| worm11-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -48672                         | 0                                | 220          | -           |
|                       | DOGE     | -48677                             | 0.00015                              | 12.5         | 3000        | -48675                         | 0.0001                           | 25           | 5800        |
|                       | DOGE-M   | -48674                             | 0.00006                              | 27.5         | 6400        | -48674                         | 0.00005                          | 42.5         | 9800        |
| worm12-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -50411                         | 0                                | 68           | -           |
|                       | DOGE     | -50411                             | 0                                    | 22.5         | 4800        | -50411                         | 0                                | 22.5         | 4800        |
|                       | DOGE-M   | -50411                             | 0                                    | 25           | 5400        | -50411                         | 0                                | 25           | 5400        |
| worm13-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -45836                         | 0                                | 265          | -           |
|                       | DOGE     | -45837                             | 0.00003                              | 15           | 3800        | -45837                         | 0.00003                          | 15           | 3800        |
|                       | DOGE-M   | -45836                             | 0                                    | 17.5         | 4600        | -45836                         | 0                                | 27.5         | 7200        |
| worm14-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -47092                         | 0                                | 509          | -           |
|                       | DOGE     | -47108                             | 0.00058                              | 20           | 4400        | -47100                         | 0.00029                          | 27.5         | 6000        |
|                       | DOGE-M   | -47100                             | 0.00027                              | 42.5         | 9400        | -47100                         | 0.00027                          | 42.5         | 9400        |
| worm15-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -49551                         | 0                                | 63           | -           |
|                       | DOGE     | -49551                             | 0                                    | 12.5         | 3200        | -49551                         | 0                                | 12.5         | 3200        |
|                       | DOGE-M   | -49551                             | 0                                    | 12.5         | 3200        | -49551                         | 0                                | 12.5         | 3200        |
| worm16-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -48423                         | 0                                | 238          | -           |
|                       | DOGE     | -48428                             | 0.00019                              | 15           | 3800        | -48427                         | 0.00014                          | 30           | 7400        |
|                       | DOGE-M   | -48425                             | 0.00009                              | 15           | 3800        | -48424                         | 0.00004                          | 22.5         | 5600        |
| worm17-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -48082                         | 0                                | 118          | -           |
|                       | DOGE     | -48083                             | 0.00001                              | 17.5         | 4200        | -48082                         | 0                                | 37.5         | 8800        |
|                       | DOGE-M   | -48083                             | 0.00003                              | 12.5         | 2800        | -48082                         | 0                                | 20           | 4600        |
| worm18-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -48242                         | 0                                | 98           | -           |
|                       | DOGE     | -48242                             | 0.00001                              | 25           | 5200        | -48242                         | 0                                | 32.5         | 6800        |
|                       | DOGE-M   | -48242                             | 0                                    | 12.5         | 2600        | -48242                         | 0                                | 12.5         | 2600        |
| worm19-16-03-11-1745  | Gurobi   | -                                  | -                                    | -            | -           | -48804                         | 0                                | 195          | -           |
|                       | DOGE     | -48807                             | 0.00011                              | 15           | 3400        | -48806                         | 0.00008                          | 32.5         | 7200        |

*Continued on next page*

<!-- Page 17 -->
Table 10 Continued from previous page

| instance             | method   | Until termination criteria                 | Best until max. num itr.                |
|----------------------|----------|--------------------------------------------|-----------------------------------------|
|                      |          | $E(\uparrow)$ | $g(t)(\downarrow)$ | $t(\downarrow)$ | \# itr. | $E(\uparrow)$ | $g(t)(\downarrow)$ | $t(\downarrow)$ | \# itr. |
| DOGE-M               |          | -48806       | 0.00008              | 17.5            | 3800    | -48805       | 0.00004              | 42.5            | 9400    |
| worm20-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -49443       | 0                    | 216             | -       |
|                      | DOGE     | -49445       | 0.00009              | 15              | 3000    | -49443       | 0.00001              | 42.5            | 8800    |
|                      | DOGE-M   | -49444       | 0.00006              | 37.5            | 7800    | -49444       | 0.00006              | 37.5            | 7800    |
| worm21-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -49844       | 0                    | 67              | -       |
|                      | DOGE     | -49844       | 0                    | 20              | 4400    | -49844       | 0                    | 20              | 4400    |
|                      | DOGE-M   | -49844       | 0                    | 20              | 4400    | -49844       | 0                    | 20              | 4400    |
| worm22-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -48012       | 0                    | 277             | -       |
|                      | DOGE     | -48018       | 0.00022              | 17.5            | 4200    | -48013       | 0.00002              | 40              | 9600    |
|                      | DOGE-M   | -48014       | 0.00009              | 20              | 4800    | -48013       | 0.00003              | 32.5            | 7800    |
| worm23-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -49986       | 0                    | 51              | -       |
|                      | DOGE     | -49986       | 0                    | 10              | 2200    | -49986       | 0                    | 10              | 2200    |
|                      | DOGE-M   | -49986       | 0                    | 7.5             | 1600    | -49986       | 0                    | 7.5             | 1600    |
| worm24-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -49330       | 0                    | 79              | -       |
|                      | DOGE     | -49333       | 0.00012              | 22.5            | 4800    | -49333       | 0.00012              | 22.5            | 4800    |
|                      | DOGE-M   | -49330       | 0.00002              | 27.5            | 6000    | -49330       | 0.00001              | 37.5            | 8000    |
| worm25-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -47241       | 0                    | 205             | -       |
|                      | DOGE     | -47242       | 0.00002              | 17.5            | 4200    | -47241       | 0                    | 30              | 7200    |
|                      | DOGE-M   | -47242       | 0.00002              | 27.5            | 6600    | -47241       | 0                    | 37.5            | 9200    |
| worm26-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -46145       | 0                    | 595             | -       |
|                      | DOGE     | -46161       | 0.00055              | 17.5            | 4000    | -46158       | 0.00046              | 35              | 8000    |
|                      | DOGE-M   | -46150       | 0.00019              | 30              | 6800    | -46148       | 0.00011              | 42.5            | 9600    |
| worm27-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -50063       | 0                    | 60              | -       |
|                      | DOGE     | -50063       | 0                    | 12.5            | 2600    | -50063       | 0                    | 12.5            | 2600    |
|                      | DOGE-M   | -50063       | 0                    | 12.5            | 2600    | -50063       | 0                    | 12.5            | 2600    |
| worm28-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -49500       | 0                    | 59              | -       |
|                      | DOGE     | -49500       | 0.00002              | 15              | 3400    | -49500       | 0.00002              | 25              | 5600    |
|                      | DOGE-M   | -49500       | 0.00001              | 15              | 3200    | -49500       | 0                    | 27.5            | 6000    |
| worm29-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -50070       | 0                    | 46              | -       |
|                      | DOGE     | -50070       | 0                    | 15              | 3000    | -50070       | 0                    | 15              | 3000    |
|                      | DOGE-M   | -50070       | 0.00001              | 17.5            | 3400    | -50070       | 0                    | 27.5            | 5400    |
| worm30-16-03-11-1745 | Gurobi   | -            | -                    | -               | -       | -49784       | 0                    | 58              | -       |
|                      | DOGE     | -49784       | 0                    | 12.5            | 2800    | -49784       | 0                    | 12.5            | 2800    |
|                      | DOGE-M   | -49784       | 0                    | 15              | 3400    | -49784       | 0                    | 15              | 3400    |

## D.6 QAPLib

Table 11: Detailed results on QAPLib dataset. Until termination criteria contain results where we stop our solvers early w.r.t. relative improvement. These results are averaged and reported in Table 1. Best until max. itr.: We run our solver for at most 100000 iterations and report best results (so $R = 5000$, $T = 20$). *: Gurobi did not converge within 1 hour timelimit.

| instance | method   | Until termination criteria                 | Best until max. num itr.                |
|----------|----------|--------------------------------------------|-----------------------------------------|
|          |          | $E(\uparrow)$ | $g(t)(\downarrow)$ | $t(\downarrow)$ | \# itr. | $E(\uparrow)$ | $g(t)(\downarrow)$ | $t(\downarrow)$ | \# itr. |
| bur26g*  | Gurobi   | -            | -                    | -               | -       | 9886478        | 0.01997              | 3599            | -       |
|          | DOGE     | 10018869     | 0.00566              | 45              | 4820    | 10054780       | 0.00177              | 935             | 100000  |

Continued on next page

<!-- Page 18 -->
Table 11 Continued from previous page

| instance | method | Until termination criteria | | | | Best until max. num itr. | | | |
|----------|--------|-----------------------------|------------|---------|--------|-----------------------------|------------|---------|--------|
|          |        | $E$ (↑)                     | $g(t)$ (↓) | $t$ (↓) | # itr. | $E$ (↑)                     | $g(t)$ (↓) | $t$ (↓) | # itr. |
| DOGE-M   |        | 10010474                    | 0.00656    | 170     | 46340  | 10014676                    | 0.00611    | 235     | 64080  |
| bur26b*  | Gurobi | -                           | -          | -       | -      | 6060753                     | 0.1538     | 3600    | -      |
|          | DOGE   | 7005771                     | 0.008      | 50      | 5360   | 7034310                     | 0.0036     | 930     | 99740  |
|          | DOGE-M | 6997285                     | 0.00931    | 185     | 50540  | 7001718                     | 0.00863    | 250     | 68320  |
| had20*   | Gurobi | -                           | -          | -       | -      | 6402                        | 0.02938    | 3600    | -      |
|          | DOGE   | 6495                        | 0.01392    | 280     | 62680  | 6512                        | 0.0112     | 450     | 100000 |
|          | DOGE-M | 6487                        | 0.01532    | 225     | 99220  | 6487                        | 0.01522    | 230     | 100000 |
| kra32    | Gurobi | -                           | -          | -       | -      | 7703                        | 0          | 3333    | -      |
|          | DOGE   | 7481                        | 0.0317     | 115     | 12260  | 7545                        | 0.02259    | 940     | 100000 |
|          | DOGE-M | 7457                        | 0.03509    | 360     | 100000 | 7457                        | 0.03509    | 360     | 100000 |
| lipa40a* | Gurobi | -                           | -          | -       | -      | 4217                        | 0.91943    | 3597    | -      |
|          | DOGE   | 31506                       | 0.00109    | 255     | 5480   | 31538                       | 0          | 2465    | 52920  |
|          | DOGE-M | 31417                       | 0.00406    | 175     | 15300  | 31537                       | 0.00004    | 1145    | 100000 |
| lipa40b* | Gurobi | -                           | -          | -       | -      | 46637                       | 0.92619    | 3598    | -      |
|          | DOGE   | 439236                      | 0.08045    | 1140    | 24980  | 442771                      | 0.02284    | 1495    | 32760  |
|          | DOGE-M | 471432                      | 0.01109    | 485     | 43340  | 474399                      | 0.0047     | 1120    | 100000 |
| lipa50a* | Gurobi | -                           | -          | -       | -      | 3494                        | 0.98823    | 3598    | -      |
|          | DOGE   | 58664                       | 0.05782    | 905     | 38720  | 60010                       | 0.03512    | 2340    | 100000 |
|          | DOGE-M | 61497                       | 0.01005    | 145     | 5840   | 62093                       | 0          | 1665    | 66480  |
| lipa50b* | Gurobi | -                           | -          | -       | -      | 51648                       | 0.97176    | 3595    | -      |
|          | DOGE   | 1070018                     | 0.10392    | 2310    | 99900  | 1070103                     | 0.10385    | 2315    | 100000 |
|          | DOGE-M | 1173647                     | 0.1561     | 1190    | 47280  | 1191963                     | 0          | 2490    | 100000 |
| lipa60a* | Gurobi | -                           | -          | -       | -      | 3713                        | 1          | 3595    | -      |
|          | DOGE   | 105267                      | 0.01789    | 735     | 15800  | 106426                      | 0.00667    | 4660    | 100000 |
|          | DOGE-M | 105786                      | 0.01287    | 315     | 6420   | 107114                      | 0          | 4955    | 100000 |
| lipa60b* | Gurobi | -                           | -          | -       | -      | 66471                       | 0.98356    | 3596    | -      |
|          | DOGE   | 2093148                     | 0.15489    | 145     | 3180   | 2186837                     | 0.11658    | 4585    | 100000 |
|          | DOGE-M | 2328269                     | 0.05875    | 520     | 10720  | 2471953                     | 0          | 4860    | 100000 |
| lipa70a* | Gurobi | -                           | -          | -       | -      | 6598                        | 0.99197    | 3600    | -      |
|          | DOGE   | 165123                      | 0.02784    | 1140    | 13160  | 167966                      | 0.01054    | 8625    | 100000 |
|          | DOGE-M | 167322                      | 0.01446    | 565     | 6000   | 169700                      | 0          | 9495    | 100000 |
| lipa70b* | Gurobi | -                           | -          | -       | -      | 121986                      | 0.98152    | 3600    | -      |
|          | DOGE   | 4293967                     | 0.03577    | 1990    | 23520  | 4382764                     | 0.01564    | 4625    | 55120  |
|          | DOGE-M | 4230582                     | 0.05014    | 1355    | 14620  | 4451768                     | 0.03884    | 9310    | 100000 |
| nug27*   | Gurobi | -                           | -          | -       | -      | 2545                        | 0.10356    | 3599    | -      |
|          | DOGE   | 2693                        | 0.04496    | 425     | 50200  | 2713                        | 0.03731    | 850     | 100000 |
|          | DOGE-M | 2688                        | 0.04713    | 335     | 100000 | 2688                        | 0.04713    | 335     | 100000 |
| nug28*   | Gurobi | -                           | -          | -       | -      | 2446                        | 0.04303    | 3599    | -      |
|          | DOGE   | 2468                        | 0.03344    | 225     | 23300  | 2486                        | 0.02522    | 965     | 100000 |
|          | DOGE-M | 2456                        | 0.03891    | 360     | 99580  | 2456                        | 0.03884    | 365     | 100000 |
| nug30*   | Gurobi | -                           | -          | -       | -      | 1595                        | 0.70017    | 3600    | -      |
|          | DOGE   | 4481                        | 0.07076    | 760     | 48520  | 4535                        | 0.0588     | 1565    | 100000 |
|          | DOGE-M | 4630                        | 0.03817    | 510     | 99820  | 4630                        | 0.03815    | 515     | 100000 |
| rou20*   | Gurobi | -                           | -          | -       | -      | 586646                      | 0.09057    | 3599    | -      |
|          | DOGE   | 612538                      | 0.04922    | 445     | 99660  | 612560                      | 0.04918    | 450     | 100000 |

Continued on next page

<!-- Page 19 -->
Table 11 Continued from previous page

| instance | method   | Until termination criteria                 | Best until max. num itr.               |
|----------|----------|--------------------------------------------|----------------------------------------|
|          |          | $E$ (↑) | $g(t)$ (↓) | $t$ (↓) | # itr. | $E$ (↑) | $g(t)$ (↓) | $t$ (↓) | # itr. |
|----------|----------|---------|------------|---------|--------|---------|------------|---------|--------|
|          | DOGE-M   | 612818  | 0.04877    | 225     | 99760  | 612844  | 0.04873    | 230     | 100000 |
| scr20    | Gurobi   | -       | -          | -       | -      | 75474   | 0          | 43      | -      |
|          | DOGE     | 75401   | 0.00132    | 45      | 15620  | 75415   | 0.00107    | 140     | 48680  |
|          | DOGE-M   | 75404   | 0.00126    | 30      | 16620  | 75415   | 0.00106    | 65      | 36600  |
| sko42*   | Gurobi   | -       | -          | -       | -      | 1599    | 0.8898     | 3599    | -      |
|          | DOGE     | 9949    | 0.15128    | 1355    | 99980  | 9949    | 0.15125    | 1360    | 100000 |
|          | DOGE-M   | 11597   | 0.00557    | 1165    | 78220  | 11659   | 0          | 1480    | 100000 |
| sko49*   | Gurobi   | -       | -          | -       | -      | 1268    | 0.94759    | 3599    | -      |
|          | DOGE     | 15745   | 0.05392    | 2210    | 99800  | 15747   | 0.05383    | 2215    | 100000 |
|          | DOGE-M   | 16439   | 0.01109    | 1650    | 70580  | 16619   | 0          | 2340    | 100000 |
| sko56*   | Gurobi   | -       | -          | -       | -      | 1421    | 0.96053    | 3594    | -      |
|          | DOGE     | 22410   | 0.06144    | 125     | 3380   | 23430   | 0.01774    | 3695    | 100000 |
|          | DOGE-M   | 23254   | 0.02529    | 2020    | 52180  | 23845   | 0          | 3875    | 100000 |
| sko64*   | Gurobi   | -       | -          | -       | -      | 1053    | 0.98536    | 3586    | -      |
|          | DOGE     | 30410   | 0.0819     | 15      | 240    | 31798   | 0.03917    | 5915    | 99980  |
|          | DOGE-M   | 31517   | 0.04784    | 2065    | 33000  | 33071   | 0          | 6255    | 100000 |
| ste36c*  | Gurobi   | -       | -          | -       | -      | 2661689 | 0.65051    | 3599    | -      |
|          | DOGE     | 6759633 | 0.0412     | 820     | 26300  | 6967435 | 0.0103     | 3110    | 99660  |
|          | DOGE-M   | 6933566 | 0.01533    | 485     | 60140  | 6976984 | 0.00888    | 765     | 94840  |
| tai35a*  | Gurobi   | -       | -          | -       | -      | 180704  | 0.91506    | 3598    | -      |
|          | DOGE     | 1794138 | 0.09976    | 1285    | 46320  | 1802223 | 0.09568    | 1405    | 50640  |
|          | DOGE-M   | 1896341 | 0.04812    | 735     | 99900  | 1896382 | 0.0481     | 740     | 100000 |
| tai35b*  | Gurobi   | -       | -          | -       | -      | 5212148 | 0.9588     | 3601    | -      |
|          | DOGE     | 96397776| 0.07721    | 540     | 19280  | 99942776| 0.04294    | 2795    | 99880  |
|          | DOGE-M   | 98365992| 0.05818    | 750     | 100000 | 98365992| 0.05818    | 750     | 100000 |
| tai40a*  | Gurobi   | -       | -          | -       | -      | 121132  | 0.95377    | 3599    | -      |
|          | DOGE     | 2140900 | 0.07833    | 480     | 42920  | 2142402 | 0.07768    | 595     | 53280  |
|          | DOGE-M   | 2321628 | 0          | 1165    | 100000 | 2321628 | 0          | 1165    | 100000 |
| tai40b*  | Gurobi   | -       | -          | -       | -      | 5241527 | 0.97122    | 3599    | -      |
|          | DOGE     | 121210376| 0.09553   | 295     | 25960  | 121210376| 0.09553   | 295     | 25960  |
|          | DOGE-M   | 133862088| 0          | 1185    | 100000 | 133862088| 0          | 1185    | 100000 |
| tai50a*  | Gurobi   | -       | -          | -       | -      | 61532   | 0.98721    | 3592    | -      |
|          | DOGE     | 2988845 | 0.18113    | 535     | 23040  | 2994846 | 0.17948    | 1610    | 69440  |
|          | DOGE-M   | 3622177 | 0.00673    | 2125    | 87480  | 3646619 | 0          | 2435    | 100000 |
| tai50b*  | Gurobi   | -       | -          | -       | -      | 179580  | 1          | 3599    | -      |
|          | DOGE     | 83312440| 0.1107     | 1995    | 84120  | 85635848| 0.08563    | 2375    | 100000 |
|          | DOGE-M   | 93571496| 0          | 2480    | 100000 | 93571496| 0          | 2480    | 100000 |
| tai60a*  | Gurobi   | -       | -          | -       | -      | 87106   | 0.98697    | 3593    | -      |
|          | DOGE     | 3984766 | 0.26193    | 265     | 5620   | 4319060 | 0.19975    | 4625    | 100000 |
|          | DOGE-M   | 5216043 | 0.03289    | 1840    | 38040  | 5392868 | 0          | 4860    | 100000 |
| tai60b*  | Gurobi   | -       | -          | -       | -      | 125579  | 1          | 3590    | -      |
|          | DOGE     | 55250419| 0.60577    | 255     | 5520   | 88722336| 0.36445    | 4625    | 99960  |
|          | DOGE-M   | 106619936| 0.23542   | 915     | 18680  | 139274272| 0          | 4910    | 100000 |
| tai64c   | Gurobi   | -       | -          | -       | -      | 487500  | 0          | 3283    | -      |
|          | DOGE     | 482685  | 0.01197    | 5       | 800    | 487483  | 0.00004    | 275     | 43760  |

Continued on next page

<!-- Page 20 -->
Table 11 Continued from previous page

| instance | method   | Until termination criteria                 | Best until max. num itr.               |
|----------|----------|-------------------------------------------|-----------------------------------------|
|          |          | $E$ (↑)  $g(t)$ (↓)  $t$ (↓)  # itr.     | $E$ (↑)  $g(t)$ (↓)  $t$ (↓)  # itr.   |
|          | DOGE-M   | 486733   0.00191    25      760         | 486733   0.00191    25      760        |
| tho30*   | Gurobi   | -        -          -       -           | 33467    0.70179    3598    -          |
|          | DOGE     | 90078    0.11192    945     60560       | 91072    0.10157    1560    100000     |
|          | DOGE-M   | 95420    0.05626    510     100000      | 95420    0.05626    510     100000     |
| wil50*   | Gurobi   | -        -          -       -           | 3037     0.94051    3597    -          |
|          | DOGE     | 35943    0.08066    1655    24640       | 36941    0.05458    6715    100000     |
|          | DOGE-M   | 38775    0.00667    2140    85100       | 39030    0        2515    100000       |