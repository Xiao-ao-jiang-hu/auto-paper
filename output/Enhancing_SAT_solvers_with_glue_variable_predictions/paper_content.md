<!-- Page 1 -->
# Enhancing SAT solvers with glue variable predictions

Jesse Michael Han\*
Department of Mathematics  
University of Pittsburgh  
Pittsburgh, PA 15213  
jmh288@pitt.edu

## Abstract

Modern SAT solvers routinely operate at scales that make it impractical to query a neural network for every branching decision. NeuroCore, proposed by [32], offered a proof-of-concept that neural networks can still accelerate SAT solvers by only periodically refocusing a score-based branching heuristic. However, that work suffered from several limitations: their modified solvers require GPU acceleration, further ablations showed that they were not better than a random baseline on the SATCOMP 2018 benchmark, and their training target of unsat cores required an expensive data pipeline which only labels relatively easy unsatisfiable problems. We address all these limitations, using a simpler network architecture allowing CPU inference for even large industrial problems with millions of clauses, and training instead to predict *glue variables*—a target for which it is easier to generate labelled data, and which can also be formulated as a reinforcement learning task. We demonstrate the effectiveness of our approach by modifying the state-of-the-art SAT solver CADICAL, improving its performance on SATCOMP 2018 and SATRACE 2019 with supervised learning and its performance on a dataset of SHA-1 preimage attacks with reinforcement learning.

## 1 Introduction

Branching heuristics for search procedures in automated theorem provers are an attractive target for deep learning methods, and have been the focus of recent work ranging from higher-order theorem proving [6] and first-order theorem proving [24], to QBF solving [20, 41] and SAT solving [32, 18, 19, 42]. Branching heuristics for SAT solving are a particularly challenging target for applying deep learning, as modern SAT solvers are heavily optimized and routinely operate at scales (tens of thousands of decisions per second, problems with millions of variables and clauses) that make it impractical to query a neural network for every branching decision.

There are many design decisions whose trade-offs which must be carefully balanced in order to efficiently integrate machine learning into the branching heuristics of a modern SAT solver. These include whether or not to condition on the solver state and history, whether the model is trained on- or offline, how to integrate predictions into the solver, and the trade-off between model capacity and inference time. Perhaps most important is whether the model is conditioned on the global problem state. The CDCL algorithm, as typically implemented with lazy data structures [26], performs essentially *local* probing of the instance being solved: the solver only tracks the direct consequences of unit propagation of its assignment stack, and is almost never aware of the global state of the problem, i.e. how the problem actually simplifies under the current assignment. Doing so would require traversing every clause in the clause database, an operation which can take multiple seconds on large problems. Thus, any globally-informed heuristic already carries an enormous upfront

---

\*Work done while part of the Automated Reasoning Group at Amazon Web Services.

Preprint. Under review.

<!-- Page 2 -->
cost—the execution of the solver must be halted and all clause pointers dereferenced. This cost must be amortized by the quality of the heuristic’s decisions, incentivizing higher-capacity models such as neural networks.

To date, however, the most successful applications of machine learning to branching heuristics in SAT solving have appeared in the Maple family of solvers [22, 23], involving low-capacity, non-deep models for which inference is instantaneous, trained online during the execution of the solver, and queried for every branching decision, conditioned only on local information obtained from conflict analysis. Maple solvers have either won or placed highly in the annual international SAT competition [30] since 2016.

In contrast, much of the existing work in applying deep learning to SAT and QBF solving uses globally-conditioned graph neural networks [31] with much more expensive inference times to either solve trivially small problems end-to-end [34, 2, 41] or guide a search algorithm on every branching decision [20, 18, 19, 42] on problems with at most a few thousand clauses. It is unlikely that these methods can scale to large, real-world use-cases as represented in the SAT competitions. The most promising step in this direction is NeuroCore [32], which trained a neural network to predict unsat cores and avoids the performance overhead of querying for every branching decision by using the network’s predictions to only periodically refocus a score-based branching heuristic. However, their work had several limitations: (1) their modified solvers required GPUs for inference and were vastly more expensive to run than the CPU-only base solvers; (2) they only modified the SAT solvers Minisat and Glucose, which are no longer state of the art, and further ablations [33] showed that their modified solvers were no better than a random baseline on the SATCOMP 2018 benchmark; and (3) in order to produce enough labelled unsat cores for training, they relied on an expensive data pipeline that only labels relatively easy unsatisfiable problems.

In our present work, we address all these issues and show that we can realize the promise of using neural networks to accelerate state-of-the-art SAT solvers with no additional hardware. First, we use a simpler network architecture, allowing CPU inference for even large industrial problems with millions of clauses. Second, instead of unsat cores, we train to predict glue variables—those likely to occur in glue clauses, a type of conflict clause known to be extremely important to the reasoning of modern CDCL SAT solvers. Glue clauses arise frequently during search and do not require a solver to run to completion in order to generate training data. Finally, we target the state-of-the-art solver CADiCAL, and achieve improvements over the unmodified solver and a random baseline on SATCOMP 2018 and SATRACE 2019.$^{2}$ We also show that glue variable prediction can be formulated in terms of reinforcement learning; we use this to learn distribution-specific heuristics and improve solver performance on a dataset of problems encoding SHA-1 preimage attacks.

## 2 Background

A propositional logic formula is a Boolean expression (i.e. using the unary negation operator $\neg$ and the binary operators $\land$ and $\lor$) of the constants $0$ (false), $1$ (true), and variables. A literal is a variable or a negation of a variable. A clause is a disjunction of literals. A formula is in *conjunctive normal form* (CNF) if it is a conjunction of clauses; every formula in propositional logic is equivalent to a CNF formula [40]. The satisfiability problem (SAT) for propositional logic is to find an assignment (to $1$ or $0$) of all variables of a given formula (sometimes called *instance*) $\phi$ such that the formula is equivalent to $1$ (i.e. $\phi$ is satisfiable), or prove that no such assignment exists (i.e. $\phi$ is unsatisfiable). The formula $\phi$ will typically be in CNF, in which case satisfiability is equivalent to simultaneously being able to satisfy all clauses of $\phi$. SAT is the prototypical NP-complete problem [13].

The DPLL algorithm [17, 16] was introduced as a complete decision procedure for propositional logic. It heavily relies on *unit propagation*. Given a clause $C = (\ell_1 \lor \cdots \lor \ell_n)$, if all literals but $\ell_n$ are set to $0$, then $C$ is equivalent to the *unit clause* $(\ell_n)$, and the value of $\ell_n$ is forced to $1$. This may lead to simplifications elsewhere which produce more unit clauses; unit propagation refers to repeating this procedure of identifying unit clauses and propagating simplifications until fixpoint.

The DPLL algorithm was significantly extended by the *conflict-driven clause learning* (CDCL) algorithm [35], which is now the dominant paradigm in SAT solving. CDCL performs *conflict analysis* before backtracking: upon encountering a conflict, a CDCL solver analyzes the directed acyclic

---

$^{2}$https://www.github.com/jesse-michael-han/neuro-cadical/

<!-- Page 3 -->
graph of unit propagations leading to the conflict and derives a *conflict clause* which, when added to the formula, prunes the part of the search tree which led to the conflict. Each new conflict clause is justifiable from existing clauses by a sequence of resolution steps; if the formula is unsatisfiable, then eventually the empty clause will be derived from conflict analysis, resulting in a proof of unsatisfiability. Unsatisfiability tends to be harder to prove than satisfiability, because all candidate assignments must be ruled out; generally, exponential lower bounds on DPLL runtime arise from families of unsatisfiable formulas [15, 14, 7, 1].

**Glue levels and glue clauses** A key insight of the Glucose series of solvers [4] is that the quality of a conflict clause can be approximated by its literal block distance (LBD), or *glue level* [3], which counts the number of decision levels involved in the clause. A clause with low glue level requires fewer decisions to become unit, and will be disproportionately involved in unit propagation after being added to the clause database. Clauses with glue level $\leq 2$ are called *glue clauses*, and are so important that Glucose never removes them from its clause database while aggressively deleting high-LBD clauses; this led to massive performance gains over the existing state-of-the-art and is now standard practice.

**Score-based branching heuristics** The Variable State-Independent Decaying Sum (VSIDS) heuristic and its more popular variant Exponential-VSIDS (EVSIDS) have been the dominant branching heuristic in CDCL SAT solvers for over a decade [26]. EVSIDS greedily selects decision variables according to an *activity score* maintained for each variable, which is modified during conflict analysis: if a variable is involved in a conflict, its activity is *bumped* by a fixed increment, and after every conflict, regardless of participation, every variable’s activity score is multiplicatively *decayed* by a factor $0 < \rho < 1$. Thus, variables which participate frequently in conflicts will have higher activity scores, weighted towards more recent conflicts.

## 3 Network architecture

We describe our network architecture, which is similar to the graph neural network used in [20]. We encode a SAT problem with $N$ variables and $M$ clauses as the $M \times 2 \cdot N$ sparse bipartite adjacency matrix $G$ of its *clause-literal graph*, which has a node for every clause and literal, and an edge between a literal $\ell$ and a clause $C$ iff $\ell$ occurs in $C$. The adjacency matrix $G$ is the input to our network, which is parametrized by the following learnable parameters:

- An initial literal embedding $\mathbf{I}_{\text{init}}$
- An $n_C$-layer feedforward network $C_{\text{update}} : \mathbb{R}^{2 \cdot \delta_L} \to \mathbb{R}^{\delta_C}$
- An $n_L$-layer feedforward network $L_{\text{update}} : \mathbb{R}^{\delta_C} \to \mathbb{R}^{\delta_L}$
- An $n_P$-layer feedforward network $V_{\text{policy}} : \mathbb{R}^{2 \cdot \delta_L} \to \mathbb{R}^1$
- A layer normalization [5] $\text{LayerNorm} : \mathbb{R}^{\delta_L} \to \mathbb{R}^{\delta_L}$.

The network computes forward as follows. For every literal $\ell$, we initialize an embedding $\mathbf{l} = \mathbf{I}_{\text{init}}$. Let $\bar{\ell}$ denote the negation of $\ell$, and let $\bar{\mathbf{l}}$ denote the embedding of $\bar{\ell}$. Let $\mathbf{L}$ denote the $2 \cdot N_{\text{var}} \times \delta_L$ array of all literal embeddings, and let $\overline{\mathbf{L}}$ denote the operation of interchanging each row $\mathbf{l}$ of $\mathbf{L}$ with $\bar{\mathbf{l}}$. We compute the clause and literal embeddings as follows. In what follows, function application notation denotes row-wise application. For up to $\tau$ iterations, we perform the following updates:

$$
\begin{aligned}
\mathbf{C} &\leftarrow C_{\text{update}}(G \cdot \text{Concat}(\mathbf{L}, \overline{\mathbf{L}})) \\
\mathbf{C} &\leftarrow \frac{\mathbf{C} - \mathbb{E}[\mathbf{C}]}{\sqrt{\text{Var}(\mathbf{C}) + \varepsilon}} \\
\mathbf{L} &\leftarrow L_{\text{update}}(G^T \cdot \mathbf{C}) + 0.1 \cdot \mathbf{L} \\
\mathbf{L} &\leftarrow \text{LayerNorm}(\mathbf{L}).
\end{aligned}
$$

Finally, we obtain a probability distribution $\hat{\pi}$ over variables by applying

$$
\hat{\pi} \leftarrow \text{Softmax}\left(V_{\text{policy}}\left(\text{Concat}(\mathbf{L}, \overline{\mathbf{L}})\right)\right).
$$

<!-- Page 4 -->
Above, $\delta_C$ is the dimension of the clause embeddings, and $\delta_L$ is the dimension of the literal embeddings; the number of iterations $\tau$, the number of layers $n_L, n_C, n_P$ in the feedforward networks, and $\delta_L, \delta_C$ are hyperparameters. We use LeakyReLU nonlinearities for the hidden layers of the feedforward networks, and during training, we use a dropout fraction of 0.15 throughout. For reinforcement learning, we attach a value head $V_{\text{value}}$ which is identical to $V_{\text{policy}}$, and obtain a value estimate $\hat{v}$ with

$$
\hat{v} \leftarrow \text{Sigmoid}\left(\text{Mean}\left(V_{\text{value}}(\text{Concat}(\mathbf{L}, \overline{\mathbf{L}}))\right)\right).
$$

The main differences between our architecture and the one used in [20] are the normalization of clause and literal embeddings, the residual layer during the literal update step, the absence of clause and literal features extracted from the solver state, and our choice of hyperparameters, which are tuned for more expensive and less frequent queries instead of querying for every decision. Importantly, in contrast to NeuroSAT-style architectures [34, 18] which update $\mathbf{L}$ using $G^T \cdot \mathbf{C}$ and $\overline{\mathbf{L}}$, this architecture updates $\mathbf{C}$ using $G \cdot \mathbf{L}$ and $G \cdot \overline{\mathbf{L}}$. When the number of iterations increases beyond $\tau = 1$, this ensures that every clause embedding is updated partly according to the embeddings of its possible resolvents, i.e. the message-aggregation step for clause embeddings indirectly incorporates the structure of the resolution graph of the formula.

## 4 Data generation and training

### 4.1 Supervised learning of glue variable prediction

We modify the solver CADiCAL Section 5 to halt after 180 seconds and traverse the clause database, accumulating counters for the number of times each variable appears in a glue clause. These glue counts, along with the sparse clause-literal adjacency matrix of the original formula, form a single datapoint. We perform this procedure for all 750 main track problems in SATCOMP 2016 and SATCOMP 2017. To ensure uniformity in training data, if necessary, we split a problem into subproblems by randomly assigning variables until the resulting subproblems each have $\leq 150000$ clauses. We synthetically augment our dataset by periodically dumping the entire formula plus learned clauses every 100000 conflicts, then running the data-generation procedure again.

We generated a training set of approximately 50000 datapoints. During training, we softmax the glue counts to obtain a probability distribution $\pi$ and train to minimize the KL divergence between $\pi$ and the probability distribution $\hat{\pi}$ emitted by the network. We used the hyperparameters $\delta_L = 16, \delta_C = 64, \tau = 2, n_L = 2, n_c = 2, n_P = 3$, choosing relatively small values in anticipation of the large industrial problems in the evaluation set. We trained for 3 epochs with averaged stochastic gradient descent [29] with learning rate 1e-3, using RaySGD [25] and data-distributed Pytorch [28] on 32 GPUs in under an hour.

### 4.2 Reinforcement learning of glue level minimization

Motivated by recent work [20, 19, 42] showing that reinforcement learning techniques can learn effective distribution-specific branching heuristics from only dozens or hundreds of training problems, we frame glue variable prediction for a formula $\phi$ as an episodic reinforcement learning task on a finite Markov decision process (MDP), represented by the data $\mathcal{S}_\phi, \mathcal{A}, \mathcal{T}, \mathcal{R}$ defined as follows:

- The collection of possible states $\mathcal{S}_\phi$ comprises all subformulas of $\phi$, i.e CNFs obtained by simplifying and applying unit propagation to $\phi$ with respect to partial assignments. The environment enters a terminal state when all variables have been assigned. There are two terminal states, corresponding to whether the formula is satisfied or unsatisfied.
- $\mathcal{A}$ assigns to each $s \in \mathcal{S}_\phi$ the collection of valid actions; these are just the variables which have not yet been assigned.
- $\mathcal{T}: \Pi_{s \in \mathcal{S}_\phi} \mathcal{A}(s) \rightarrow \mathcal{S}_\phi$ is a stochastic transition function. Once a variable has been selected for assignment, the environment assigns it either 0 or 1 with uniform probability and simplifies the formula with unit propagation to obtain the next state.
- $\mathcal{R}$ is the reward function on states. For non-terminal states, we always assign the small negative reward $-1/n$, where $n$ is the number of variables in $\phi$. Upon reaching a terminal state, we assign 0 reward if the formula has been satisfied, and otherwise assign $1/g^2$, where $g$ is the glue level of the conflict clause learned from conflict analysis.

<!-- Page 5 -->
Note that in contrast to previous work in this vein [20, 19, 42], we use domain-specific knowledge to replace a sparse terminal reward (completely solving $\phi$) with a proxy reward (minimizing glue level of learned clauses) that allows us to ignore backtracking and treat every path through the DPLL search tree as a separate episode.

We convert CADICAL into a reinforcement learning environment which implements the dynamics outlined above. Upon receiving an action (a variable to assign), the environment sets it to a random polarity, performs unit propagation, and returns an observation in the form of a sparse clause-literal adjacency matrix, constructed as described in Section 5, along with the reward. Note that in keeping with the formal definition, we avoid non-stationarity in the environment by discarding any learned conflict clauses when resetting from a terminal state.

**The sha-1 dataset** We evaluate our reinforcement learning pipeline using a dataset of 250 formulas encoding SHA-1 preimage attacks, with an 80-20 train-test split. We generate the dataset using the tool CGEN [38, 36]. The sha-1 dataset is of similar difficulty to a collection of CGEN-generated problems submitted to SATRACE 2019 [37]. We generate a new six-character alphanumeric message string and randomly set the number of message variables between 70 and 90 for every instance in our dataset, leaving all other arguments to CGEN the same as the submitted benchmark.

**Training** Since all the problems in the sha-1 dataset are around the same size and relatively small ($\approx 3600$ variables and $\approx 15000$ clauses), we use the more expensive hyperparameters $\delta_L = 32$, $\delta_C = 64$, $\tau = 4$, $n_L = 3$, $n_c = 3$, $n_P = 4$. We train our network (Section 3) using synchronous multi-agent REINFORCE with a jointly-learned value function baseline. For each batch, each member of a pool of 128 workers equipped with the latest policy independently samples a formula $\phi$ from the training set and generates $b$ episodes, which are then processed by a GPU learner using the Adam optimizer with constant learning rate 1e-4. Besides standard optimizations like advantage normalization and gradient clipping, we use importance sampling to correct for policy lag across multiple gradient steps.

## 5 Solver modifications

As with NeuroCore [32], we avoid the performance overhead of querying our network for every branching decision by only *periodically refocusing* the EVSIDS branching heuristic with our trained networks (henceforth called NeuroGlue). We modify CADICAL [8, 9, 10], a state-of-the-art CDCL SAT solver which solved the most instances in SAT Race 2019 and won the unsatisfiable track in SATCOMP 2018. We use the version submitted to SAT Race 2019, which incorporates an EVSIDS branching heuristic during certain phases of search [27]. In order to more accurately measure the impact of periodic refocusing, which only updates EVSIDS scores, we run CADICAL in its `--sat` configuration, which specializes for satisfiable instances and exclusively relies on EVSIDS for branching. With this configuration, CADICAL is still state-of-the-art on satisfiable instances, winning the satisfiable track of SAT Race 2019 (and beating Glucose 4.1 on the main track regardless).

**Implementation of periodic refocusing** We implement periodic refocusing as an inprocessing routine in CADICAL which fires immediately before the next decision variable is selected, guarded by a scheduling heuristic (Section 5).

When refocusing, we construct a sparse clause-literal adjacency matrix $G$ by traversing all the original, then learned, clauses, simplifying with respect to the current assignment and compacting assigned variables. Like NeuroCore, we stop when the number of edges in the clause-literal graph exceeds a predetermined threshold of 10e6; if the original clauses do not meet this cutoff, then we do not query at all. $G$ is the input to our model, which is invoked via TorchScript [39] and runs in the same thread as the solver. The returned logits are multiplied by a temperature parameter $\tau = 4.0$, then softmaxed to produce a probability distribution $\hat{\pi}$ over variables. These probabilities are then rescaled by the number of variables and a fixed constant $\kappa = 1e4$ before replacing the existing EVSIDS scores.

**Refocusing schedule** Unlike [32], in which periodic refocusing is scheduled according to wall-clock intervals, we use a conflict schedule instead, performing the $N^{\text{th}}$ refocus after

$$
\min(50000 + 1000 \cdot (N - 1)^2, 250000)
$$

conflicts have occurred. Immediately after starting the solver, we also allot a fixed 15-second “warm-up” period during which no refocusing occurs.

<!-- Page 6 -->
Table 1: PAR-2 scores on SATCOMP 2018.

| Solver   | overall | sat    | unsat   |
|----------|---------|--------|---------|
| neuro    | 3194.81 | 969.68 | 1194.35 |
| vanilla  | 3249.99 | 1090.52| 1172.87 |
| random   | 3293.62 | 1033.93| 1452.72 |

Table 2: PAR-2 scores on SATRACE 2019.

| Solver   | overall | sat    | unsat   |
|----------|---------|--------|---------|
| neuro    | 4344.87 | 1039.68| 1817.05 |
| vanilla  | 4405.69 | 1134.12| 1908.87 |
| random   | 4419.33 | 1155.15| 1929.76 |

Glucose introduced a dynamic restart strategy [4] based on an exponential moving average (EMA) of glue levels: if the EMA is significantly worse than the global average of glue levels, a restart is triggered. CADiCAL uses a similar strategy to schedule restarts, replacing the global average with a slower-moving EMA. As a final optimization, we incorporate this statistic into the refocusing schedule: a refocus is triggered if and only if the conflict schedule has been satisfied and the fast glue level EMA is 10% higher than the slow glue level EMA. In practice, this occurs quite often, so the overall frequency of refocusing is unaffected.

## 6 Experiments

We evaluate three versions of CADiCAL: (1) *neuro-cadical*, which performs periodic refocusing using NeuroGlue, (2) *vanilla-cadical*, the unmodified baseline, and (3) *random-cadical*, a random baseline with identical logic to *neuro-cadical*, except that during refocusing, the logits obtained from NeuroGlue are replaced by logits uniformly and independently sampled from $[0, 1]$.

All evaluation runs were done in parallel on a cluster of ten r5d.24xlarge AWS EC2 instances with 48 cores and 768GB RAM each, and no hyperthreading.

### 6.1 SATCOMP 2018 and SATRACE 2019

In keeping with the rules of recent SAT competitions [30], each solver process runs with a 5000 second timeout, and our primary metric is the PAR-2 score, defined as the sum of runtimes for all solved instances plus $2 \cdot \text{timeout} \cdot (\#\ \text{of unsolved instances})$, so that a lower PAR-2 score is better. For conciseness, we divide all PAR-2 scores by the total number of instances. We also measure the *global learning rate* (GLR) i.e. ratio of conflicts to decisions, and *average glue level* for each problem, which have been empirically shown to be correlated with the quality of a branching heuristic [21].

As a measure against noise due to resource contention or unlucky random seeds for either *random-cadical* or randomized heuristics in the base solver, we perform 16 evaluation runs with distinct random seeds on SATCOMP 2018 and SATRACE 2019, averaging the results for each of the 400 instances in each benchmark. We consider solver $S$ to have solved $\phi$ if in any of the evaluation runs, $S$ solved $\phi$, and we only average the successful runtimes of $S$ on $\phi$. This resembles the construction of a virtual best solver [11], except we compare a solver against only itself across evaluation runs instead of against all other solvers, and take the average of successful runtimes instead of the minimum. We additionally calculate PAR-2 scores for satisfiable (resp. unsatisfiable) instances by restricting the score calculation to instances which were found to be satisfiable (resp. unsatisfiable) by any of the three solvers.

Our results are shown in Table 1 and Table 2. On SATCOMP 2018, *neuro-cadical* achieves a 1.67% better score than *vanilla-cadical* and a 2.98% better score than *random-cadical*. On SATRACE 2019, *neuro-cadical* achieves a 1.38% better score than *vanilla-cadical* and a 1.68% better score than *random-cadical*. To put this in perspective, the margin in PAR-2 scores between first and second place in the three most recent SAT competitions averages 1.15%. Figure 1 displays runtime and decision cactus plots on both datasets. In both cases, most of *neuro-cadical*'s lead is accumulated from more difficult problems towards the end of the plots, and it requires fewer decisions to solve more of these problems. Table 3 and Table 4 display the proportion of each benchmark that each solver attained better GLR and better average glue level. On both datasets, *neuro-cadical* has better GLR on more problems than both baselines.

### 6.2 SHA-1 preimage attacks

<!-- Page 7 -->
Figure 1: **Left**: Runtime cactus plots of all three variants on CADICAL on SATCOMP 2018 (top) and SATRACE 2019 (bottom). In both cases, most of neuro-cadical’s lead is accumulated from more difficult problems towards the end. **Right**: Decision cactus plots on SATCOMP 2018 with 175M decision limit, starting at the 250 problem cutoff (top) and on SATRACE 2019 with 250M decision limit, starting at the 220 problem cutoff (bottom). In both cases, the improved runtime of neuro-cadical is reflected in its superior decision efficiency over both baselines.

Table 3: Percent of problems on SATCOMP 2018 with better GLR/average glue level.

|          | higher GLR | lower avg glue |
|----------|------------|----------------|
| neuro    | 66.5%      | 54.9%          |
| vanilla  | 33.5%      | 45.1%          |
| neuro    | 56.2%      | 54.9%          |
| random   | 43.8%      | 45.1%          |
| vanilla  | 32.5%      | 53.0%          |
| random   | 67.5%      | 47.0%          |

Table 4: Percent of problems on SATRACE 2019 with better GLR/average glue level.

|          | higher GLR | lower avg glue |
|----------|------------|----------------|
| neuro    | 54.2%      | 48.8%          |
| vanilla  | 45.8%      | 51.2%          |
| neuro    | 52.5%      | 45.9%          |
| random   | 47.5%      | 54.1%          |
| vanilla  | 46.2%      | 50.4%          |
| random   | 53.8%      | 49.6%          |

<!-- Page 8 -->
In this section, neuro-cadical performs periodic refocusing with the version of NeuroGlue trained as a policy network via reinforcement learning (Section 4.2). On our test dataset of 50 SHA-1 preimage attack problems, we perform 7 evaluation runs with distinct random seeds with the same hardware and a 1000 second timeout, averaging the results as in Section 6.1. neuro-cadical achieves a PAR-2 score of 279.29, 1.23% better than that of vanilla-cadical (282.78) and 8.19% better than that of random-cadical (304.19). Figure 2 displays a decision cactus plot with 6M cutoff on the entire dataset. As in Section 6.1, neuro-cadical tends to be more decision efficient than both baselines. From Table 5, we see that, as in Section 6.1 neuro-cadical has better GLR on more problems than both baselines.

## 7 Discussion

### Related work on glue variables

We note that prioritizing the activity scores of variables related to glue clauses is not a novel idea, and dates back to Glucose, which additionally bumps the activity scores of variables in a learned clause which were propagated by a glue clause. The effectiveness of prioritizing glue variables in EVSIDS branching heuristics has already been demonstrated in [12], where *glue bumping* was shown to improve solver performance on SAT competition benchmarks. However, in contrast to our approach, glue bumping is an online heuristic that only increases the score of variables which have *already* frequently appeared in a glue clause for a single run of the solver, and does not attempt to *predict* glue variables from the formula itself, as we do.

### Future directions

Although in our present work, for the sake of simplicity, we have avoided exposing our policy network to any part of the solver state, the positive results of [20] indicate that even exposing very basic information can be beneficial for learning branching heuristics. We consider this to be a promising path to further improving solver performance.

### Conclusion

We have proposed training for *glue variable prediction* to guide SAT solvers through periodic refocusing, approaching the task in terms of both supervised and reinforcement learning. Along with a lightweight network architecture, we have demonstrated the effectiveness of both approaches by improving the performance of a state-of-the-art SAT solver on diverse benchmarks with no hardware acceleration, thus addressing the limitations of previous work in this vein [32] and showing that we can realize the promise of neural networks for accelerating high-performance SAT solvers in all contexts in which they are currently deployed. We are optimistic that refinements to our approach, possibly incorporating solver state and history, will push the state-of-the-art even further.

Table 5: Percent of problems on the sha-1 test set with better GLR/average glue level.

|          | higher GLR | lower avg glue |
|----------|------------|----------------|
| neuro    | 60.0%      | 50.0%          |
| vanilla  | 40.0%      | 50.0%          |
| neuro    | 56.0%      | 50.0%          |
| random   | 44.0%      | 50.0%          |
| vanilla  | 46.0%      | 48.0%          |
| random   | 54.0%      | 52.0%          |

Figure 2: Decision cactus plot on the sha-1 test dataset. For this evaluation, neuro-cadical was trained with reinforcement learning to minimize expected glue levels of conflict clauses. As before, neuro-cadical tends to be more decision efficient than either baseline.

<!-- Page 9 -->
# Acknowledgments and Disclosure of Funding

This work benefitted from conversations with John Harrison, Thomas Hales, and Daniel Selsam. We also thank Volodymyr Skladanivskyy for assistance with CGEN.

# References

[1] Michael Alekhnovich. Mutilated chessboard problem is exponentially hard for resolution. *Theor. Comput. Sci.*, 310(1-3):513–525, 2004.

[2] Saeed Amizadeh, Sergiy Matusevych, and Markus Weimer. Learning to solve circuit-sat: An unsupervised differentiable approach. In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net, 2019.

[3] Gilles Audemard and Laurent Simon. Predicting learnt clauses quality in modern SAT solvers. In Craig Boutilier, editor, *IJCAI 2009, Proceedings of the 21st International Joint Conference on Artificial Intelligence, Pasadena, California, USA, July 11-17, 2009*, pages 399–404, 2009.

[4] Gilles Audemard and Laurent Simon. On the glucose SAT solver. *Int. J. Artif. Intell. Tools*, 27(1):1840001:1–1840001:25, 2018.

[5] Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. *CoRR*, abs/1607.06450, 2016.

[6] Kshitij Bansal, Sarah M. Loos, Markus N. Rabe, Christian Szegedy, and Stewart Wilcox. Holist: An environment for machine learning of higher order logic theorem proving. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA*, volume 97 of *Proceedings of Machine Learning Research*, pages 454–463. PMLR, 2019.

[7] Paul Beame and Toniann Pitassi. An exponential separation between the parity principle and the pigeonhole principle. *Ann. Pure Appl. Logic*, 80(3):195–228, 1996.

[8] Armin Biere. CaDiCaL, Lingeling, Plingeling, Treengeling and YalSAT entering the SAT Competition 2017. *Proc. of SAT Competition 2017*, pages 14–15, 2017.

[9] Armin Biere. CaDiCaL, Lingeling, Plingeling, Treengeling and YalSAT entering the SAT Competition 2018. *Proc. of SAT Competition 2018*, pages 13–14, 2018.

[10] Armin Biere. CaDiCaL at the SAT Race 2019. *Proceedings of SAT Race (2019, Submitted)*, 2019.

[11] Martin Brain, James H. Davenport, and Alberto Griggio. Benchmarking solvers, sat-style. In Matthew England and Vijay Ganesh, editors, *Proceedings of the 2nd International Workshop on Satisfiability Checking and Symbolic Computation co-located with the 42nd International Symposium on Symbolic and Algebraic Computation (ISSAC 2017), Kaiserslautern, Germany, July 29, 2017*, volume 1974 of *CEUR Workshop Proceedings*. CEUR-WS.org, 2017.

[12] Md. Solimul Chowdhury, Martin Müller, and Jia-Huai You. Exploiting glue clauses to design effective CDCL branching heuristics. In Thomas Schiex and Simon de Givry, editors, *Principles and Practice of Constraint Programming - 25th International Conference, CP 2019, Stamford, CT, USA, September 30 - October 4, 2019, Proceedings*, volume 11802 of *Lecture Notes in Computer Science*, pages 126–143. Springer, 2019.

[13] Stephen A. Cook. The complexity of theorem-proving procedures. In Michael A. Harrison, Ranan B. Banerji, and Jeffrey D. Ullman, editors, *Proceedings of the 3rd Annual ACM Symposium on Theory of Computing, May 3-5, 1971, Shaker Heights, Ohio, USA*, pages 151–158. ACM, 1971.

[14] Stephen A Cook. A short proof of the pigeon hole principle using extended resolution. *Acm Sigact News*, 8(4):28–32, 1976.

[15] Stephen A Cook and Robert A Reckhow. The relative efficiency of propositional proof systems. *The Journal of Symbolic Logic*, 44(1):36–50, 1979.

[16] Martin Davis, George Logemann, and Donald W. Loveland. A machine program for theorem-proving. *Commun. ACM*, 5(7):394–397, 1962.

<!-- Page 10 -->
[17] Martin Davis and Hilary Putnam. A computing procedure for quantification theory. *J. ACM*, 7(3):201–215, 1960.

[18] Sebastian Jaszczur, Michał Łuszczek, and Henryk Michalewski. Neural heuristics for SAT solving. In *Representation Learning on Graphs and Manifolds Workshop at ICLR 2019*, 2019.

[19] Vitaly Kurin, Saad Godil, Shimon Whiteson, and Bryan Catanzaro. Improving SAT solver heuristics with graph networks and reinforcement learning. *arXiv preprint arXiv:1909.11830*, 2019.

[20] Gil Lederman, Markus N. Rabe, Sanjit Seshia, and Edward A. Lee. Learning heuristics for quantified boolean formulas through reinforcement learning. In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020.

[21] Jia Liang, Hari Govind V. K., Pascal Poupart, Krzysztof Czarnecki, and Vijay Ganesh. An empirical study of branching heuristics through the lens of global learning rate. In Jérôme Lang, editor, *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI 2018, July 13-19, 2018, Stockholm, Sweden*, pages 5319–5323. ijcai.org, 2018.

[22] Jia Hui Liang. *Machine Learning for SAT Solvers*. PhD thesis, University of Waterloo, Ontario, Canada, 2018.

[23] Jia Hui Liang, Vijay Ganesh, Pascal Poupart, and Krzysztof Czarnecki. Learning rate based branching heuristic for SAT solvers. In Nadia Creignou and Daniel Le Berre, editors, *Theory and Applications of Satisfiability Testing - SAT 2016 - 19th International Conference, Bordeaux, France, July 5-8, 2016, Proceedings*, volume 9710 of *Lecture Notes in Computer Science*, pages 123–140. Springer, 2016.

[24] Sarah M. Loos, Geoffrey Irving, Christian Szegedy, and Cezary Kaliszyk. Deep network guided proof search. In Thomas Eiter and David Sands, editors, *LPAR-21, 21st International Conference on Logic for Programming, Artificial Intelligence and Reasoning, Maun, Botswana, May 7-12, 2017*, volume 46 of *EPiC Series in Computing*, pages 85–105. EasyChair, 2017.

[25] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan, and Ion Stoica. Ray: A distributed framework for emerging AI applications. In Andrea C. Arpaci-Dusseau and Geoff Voelker, editors, *13th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2018, Carlsbad, CA, USA, October 8-10, 2018*, pages 561–577. USENIX Association, 2018.

[26] Matthew W. Moskewicz, Conor F. Madigan, Ying Zhao, Lintao Zhang, and Sharad Malik. Chaff: Engineering an efficient SAT solver. In *Proceedings of the 38th Design Automation Conference, DAC 2001, Las Vegas, NV, USA, June 18-22, 2001*, pages 530–535, 2001.

[27] Chanseok Oh. Between SAT and UNSAT: the fundamental difference in CDCL SAT. In Marijn Heule and Sean A. Weaver, editors, *Theory and Applications of Satisfiability Testing - SAT 2015 - 18th International Conference, Austin, TX, USA, September 24-27, 2015, Proceedings*, volume 9340 of *Lecture Notes in Computer Science*, pages 307–323. Springer, 2015.

[28] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada*, pages 8024–8035, 2019.

[29] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. *SIAM J. Control Optim.*, 30(4):838–855, July 1992.

[30] Satisfiability: Application and Theory (SAT) e.V. The International SAT competition web page. https://http://www.satcompetition.org/.

[31] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. *IEEE Trans. Neural Networks*, 20(1):61–80, 2009.

<!-- Page 11 -->
[32] Daniel Selsam and Nikolaj Bjørner. Guiding high-performance SAT solvers with unsat-core predictions. In *Theory and Applications of Satisfiability Testing - SAT 2019 - 22nd International Conference, SAT 2019, Lisbon, Portugal, July 9-12, 2019, Proceedings*, pages 336–353, 2019.

[33] Daniel Selsam and Nikolaj Bjørner. Neurocore: Guiding high-performance SAT solvers with unsat-core predictions. *CoRR*, abs/1903.04671, 2019.

[34] Daniel Selsam, Matthew Lamm, Benedikt Bünz, Percy Liang, Leonardo de Moura, and David L. Dill. Learning a SAT solver from single-bit supervision. In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*, 2019.

[35] João P. Marques Silva and Karem A. Sakallah. Conflict analysis in search algorithms for satisfiability. In *Eighth International Conference on Tools with Artificial Intelligence, ICTAI '96, Toulouse, France, November 16-19, 1996*, pages 467–469. IEEE Computer Society, 1996.

[36] Volodymyr Skladanivskyy. Cgen: a tool for encoding sha-1 and sha-256 hash functions into cnf in dimacs format.

[37] Volodymyr Skladanivskyy. Minimalistic round-reduced SHA-1 pre-image attack. *Proceedings of SAT Race (2019, Submitted)*, 2019.

[38] Volodymyr Skladanivskyy. Tailored compact CNF encodings for SHA-1. *J. Satisf. Boolean Model. Comput.*, 2020. Under review.

[39] The PyTorch team. Torch Script. https://pytorch.org/docs/stable/jit.html.

[40] Grigori S Tseitin. On the complexity of derivation in propositional calculus. In *Automation of reasoning*, pages 466–483. Springer, 1983.

[41] Zhanfu Yang, Fei Wang, Ziliang Chen, Guannan Wei, and Tiark Rompf. Graph neural reasoning for 2-quantified boolean formula solvers. *CoRR*, abs/1904.12084, 2019.

[42] Emre Yolcu and Barnabás Póczos. Learning local search heuristics for boolean satisfiability. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada*, pages 7990–8001, 2019.