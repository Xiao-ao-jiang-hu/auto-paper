<!-- Page 1 -->
# Learning Primal Heuristics for Mixed Integer Programs

Yunzhuang Shen  
School of Computing Technologies  
RMIT University  
Melbourne, Australia  
s3640365@student.rmit.edu.au

Yuan Sun  
School of Mathematics  
Monash University  
Melbourne, Australia  
yuan.sun@monash.edu

Andrew Eberhard  
School of Science  
RMIT University  
Melbourne, Australia  
andy.eberhard@rmit.edu.au

Xiaodong Li  
School of Computing Technologies  
RMIT University  
Melbourne, Australia  
xiaodong.li@rmit.edu.au

---

**Abstract**—This paper proposes a novel primal heuristic for Mixed Integer Programs, by employing machine learning techniques. Mixed Integer Programming is a general technique for formulating combinatorial optimization problems. Inside a solver, primal heuristics play a critical role in finding good feasible solutions that enable one to tighten the duality gap from the outset of the Branch-and-Bound algorithm (B&B), greatly improving its performance by pruning the B&B tree aggressively. In this paper, we investigate whether effective primal heuristics can be automatically learned via machine learning. We propose a new method to represent an optimization problem as a graph, and train a Graph Convolutional Network on solved problem instances with known optimal solutions. This in turn can predict the values of decision variables in the optimal solution for an unseen problem instance of a similar type. The prediction of variable solutions is then leveraged by a novel configuration of the B&B method, Probabilistic Branching with guided Depth-first Search (PB-DFS) approach, aiming to find (near-)optimal solutions quickly. The experimental results show that this new heuristic can find better primal solutions at a much earlier stage of the solving process, compared to other state-of-the-art primal heuristics.

**Index Terms**—Mixed Integer Programming, Primal Heuristics, Machine Learning

## I. INTRODUCTION

Combinatorial Optimization problems can be formulated as Mixed Integer Programs (MIPs). To solve general MIPs, sophisticated software uses Branch-and-Bound (B&B) framework, which recursively decomposes a problem and enumerates over the sub-problems. A bounding function determines whether a sub-problem can be safely discarded by comparing the objective value of the current best solution (incumbent) to a dual bound, generally obtained by solving a linear programming relaxation (i.e., relaxing integral constraints) of that sub-problem.

Inside solvers, primal heuristics play an important role in finding good primal solutions at an early stage [1]. A good primal solution strengthens the bounding functions which allows one to prune suboptimal branches more aggressively [2]. Moreover, finding good feasible solutions earlier greatly reduces the duality gap, which is important for user satisfaction [3]. Acknowledging the importance of primal heuristics, a modern open-source MIP solver SCIP [4], employs dozens of heuristics [2], including meta-heuristics [5], heuristics supported by mathematical theory [6], and heuristics mined by experts and verified by extensive experimental evidence [1]. Those heuristics are triggered to run with engineered timings during the B&B process.

In many situations, users are required to solve MIPs of a similar structure on a regular basis [7], so it is natural to seek Machine Learning (ML) solutions. In particular, a number of studies leverage ML techniques to speed up finding good primal solutions for MIP solvers. He et al. [8] train Support Vector Machine (SVM) to decide whether to explore or discard a certain sub-problem, aiming to devote more of the computational budget on ones that are likely to contain an optimal solution. Khalil et al. [9] train an SVM model to select which primal heuristic to run at a certain sub-problem. More related to our work here, Sun et al. [10] and Ding et al. [11] leverage ML to predict values of decision variables in the optimal solution, which are then used to fix a proportion of decision variables to reduce the size of the original problem, in the hope that the reduced space still contains the optimal solution of the original problem.

In this work, we propose a novel B&B algorithm guided by ML, aiming to search for high-quality primal solutions efficiently. Our approach works in two steps. Firstly, we train an ML model using a dataset formed by optimally-solved small-scale problem instances where the decision variables are labeled by the optimal solution values. Specifically, we employ the Graph Convolutional Network [12] (GCN), where an input graph associated with a problem instance to GCN is formed by representing each decision variable as a node and assigning an edge between two nodes if the corresponding decision variables appear in a constraint in the MIP formulation (see Section II-A). Then given an unseen problem instance, the trained GCN with the proposed graph representation method can efficiently predict for each decision variable its probability of belonging to an optimal solution in an unseen problem instance (e.g., whether a vertex is a part of the optimal

<!-- Page 2 -->
solution for the Maximum Independent Set Problem). The predicted probabilities are then used to guide a novel B&B configuration, called Probabilistic Branching technique with guided Depth-first Search (PB-DFS). PB-DFS enumerates over the search space starting from the region more likely to contain good primal solutions to the region of unpromising ones, indicated by GCN.

Although both the problem-reduction approaches [10], [11] and our proposed PB-DFS utilize solution prediction by ML, they are inherently different. The former can be viewed as a pre-processing step to prune the search space of the original problem, and the reduced problem is then solved by a B&B algorithm. In contrast, our PB-DFS algorithm configures the search order of the B&B method itself and directly operates on the original problem. In this sense, our PB-DFS is an exact method if given sufficient running time. However, as the PB-DFS algorithm does not take into account the size of the B&B search tree, it is not good at proving optimality. Therefore, we will limit the running time of PB-DFS and use it as a primal heuristic.

Our contributions can be summarized as follows:

1) We propose the Probabilistic Branching technique with guided Depth-first Search, a configuration of the B&B method specialized for boosting primal solutions empowered by ML techniques.

2) We propose a novel graph representation method that captures the relation between decision variables in a problem instance. The constructed graph can be input to GCN to make predictions efficiently.

3) Extensive experimental evaluation on NP-hard covering problems shows that 1) GCN with the proposed graph representation method is very competitive in terms of efficiency and effectiveness as compared to a tree-based model, a linear model, and a variant of Graph Neural Network [11]. 2) PB-DFS can find (near-)optimal solutions at a much earlier stage comparing to existing primal heuristics as well as the problem-reduction approaches using ML.

## II. BACKGROUND

### A. MIP and MIP solvers

MIP takes the form $\arg\min\{\boldsymbol{c}^T\boldsymbol{x} \mid \boldsymbol{x} \in \mathcal{F}\}$. For an MIP instance with $n$ decision variables, $\boldsymbol{c} \in \mathbb{R}^n$ is the objective coefficient vector. $\boldsymbol{x}$ denotes a vector of decision variables. We consider problems where the decision variables $\boldsymbol{x}$ are binary in this study, although our method can be easily extended to discrete domain [13]. $\mathcal{F}$ is the set of feasible solutions (search space), which is typically defined by integrality constraints and linear constraints $\boldsymbol{A}\boldsymbol{x} \leq \boldsymbol{b}$, where $\boldsymbol{A} \in \mathbb{R}^{m\times n}$ and $\boldsymbol{b} \in \mathbb{R}^m$ are the constraint matrix and constraint right-hand-side vector, respectively. $m$ is the number of constraints. The goal is to find an optimal solution in $\mathcal{F}$ that minimizes a linear objective function.

For solving MIPs, exact solvers employ B&B framework as their backbone, as outlined in Algorithm 1. There are two essential components in the B&B framework, *branching policy* and *node selection strategy*. A *node selection strategy* determines the next (sub-)problem (node) $Q$ to solve from the queue $\mathcal{L}$, which maintains a list of all unexplored nodes. B&B obtains a lower bound on the objective values of $Q$ by solving its Linear Programming (LP) relaxation $Q^{LP}$. If the LP solution (lower bound) is larger than the objective $\hat{c} = \boldsymbol{c}^T\hat{\boldsymbol{x}}$ of the incumbent $\hat{\boldsymbol{x}}$ (upper bound), then the sub-tree rooted at node $Q$ can be pruned safely. Otherwise, this sub-tree possibly contains better solutions and should be explored further. If the LP solution $\hat{\boldsymbol{x}}^{LP}$ is feasible in $Q$ and of better objective value, the incumbent is updated by $\hat{\boldsymbol{x}}^{LP}$. Otherwise, $Q$ is decomposed into smaller problems by fixing a candidate variable to a possible integral value. The resulting sub-problems are added to the node queue. *Branching policy* is an algorithm for choosing the branching variable from a set of candidate variables, which contains decision variables taking on fractional values in the solution $\hat{\boldsymbol{x}}^{LP}$. Modern solvers implement sophisticated algorithms for both components, aiming at finding good primal solutions quickly while maintaining a relatively small tree size. See [4] for a detailed review.

### B. Heuristics in MIP solvers

During computation, primal heuristics can be executed at any node, devoted to improve the incumbent, such that more sub-optimal nodes can be found earlier and thus pruned without further exploration. Berthold [2] classifies these primal heuristics into two categories: start heuristics and improvement heuristics. Start heuristics aim to find feasible solutions at an early solving stage, while improvement heuristics build upon a feasible solution (typically the incumbent) and seek better solutions. All

---

**Algorithm 1 Branch-and-Bound Algorithm**

Require: a problem instance: $\mathcal{I}$;

1: the node queue: $\mathcal{L} \leftarrow \{\mathcal{I}\}$;

2: the incumbent and its objective value: $\hat{x} \leftarrow \varnothing$, $\hat{c} \leftarrow \infty$;

3: **while** $\mathcal{L}$ is not empty **do**

4: &nbsp;&nbsp;&nbsp;&nbsp;choose $Q$ from $\mathcal{L}$; $\mathcal{L} \leftarrow \mathcal{L} \setminus Q$;

5: &nbsp;&nbsp;&nbsp;&nbsp;solve the linear relaxation $Q^{LP}$ of Q;

6: &nbsp;&nbsp;&nbsp;&nbsp;**if** $Q^{LP}$ is infeasible **then**

7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;go to Line 3 ;

8: &nbsp;&nbsp;&nbsp;&nbsp;**end if**

9: &nbsp;&nbsp;&nbsp;&nbsp;denote the LP solution $\hat{x}^{LP}$ ;

10: &nbsp;&nbsp;&nbsp;&nbsp;denote the LP objective $\hat{c}^{LP}$ ;

11: &nbsp;&nbsp;&nbsp;&nbsp;**if** $\hat{c}^{LP} \leq \hat{c}$ **then**

12: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $\hat{x}^{LP}$ is feasible in Q **then**

13: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\hat{x} \leftarrow \hat{x}^{LP}$; $\hat{c} \leftarrow \hat{c}^{LP}$;

14: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**else**

15: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;split $Q$ into subproblems $Q = Q_1 \cap ... \cap Q_n$ ;

16: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathcal{L} \leftarrow \mathcal{L} \cap \{Q_1..Q_n\}$;

17: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if**

18: &nbsp;&nbsp;&nbsp;&nbsp;**end if**

19: **end while**

20: **return** $\hat{x}$

<!-- Page 3 -->
## Algorithm 2 An MIP Instance to Linkage Graph

**Require:** the constraint matrix: $A \in \mathbb{R}^{m \times n}$;

1: the adjacency matrix: $\mathcal{G}^{adj} \leftarrow \mathbf{0}^{n \times n}$;

2: the row index of $A$: $i \leftarrow 0$;

3: the index set of variables: $\mathcal{C} \leftarrow \varnothing$;

4: **while** $i < m$ **do**

5: $\quad \mathcal{C} \leftarrow \{ j \mid A_{i,j} \neq 0 \}$;

6: $\quad$ **for** $k, l \in \mathcal{C},\ k \neq l$ **do**

7: $\quad\quad \mathcal{G}^{adj}_{k,l} \leftarrow 1,\ \mathcal{G}^{adj}_{l,k} \leftarrow 1$;

8: $\quad$ **end for**

9: $\quad i \leftarrow i + 1$;

10: **end while**

11: **return** $\mathcal{G}^{adj}$

heuristics run on external memory (e.g., a copy of a subproblem) and do not modify the structure of the B&B tree. For a more comprehensive description of primal heuristics, we refer readers to [2], [3].

---

### III. METHOD

In Section III-A, we describe how to train a machine learning model to predict the probability for each binary decision variable of its value in the optimal solution. In Section III-B, we illustrate the proposed PB-DFS that leverages the predicted probabilities to boost the search for high-quality primal solutions.

#### A. Solution Prediction

Given a combinatorial optimization problem, we construct a training dataset from multiple optimally-solved problem instances where each instance associates with an optimal solution. A training example corresponds to one decision variable $x_i$ from a solved problem instance. The label $y_i$ of $x_i$ is the solution value of $x_i$ in the optimal solution associated with a particular problem instance. The features $\boldsymbol{f}_i$ of $x_i$ are extracted from the MIP formulation, which describes the role of $x_i$ in that problem instance. We describe those features in Appendix A. Given the training data, an ML model can be trained by minimising the cross-entropy loss function [14] to separate the training examples with different class labels [10], [11].

We adapt the Graph Convolutional Network (GCN) [12] for this classification task, a type of Graph-based Neural Network (GNN), to take the relation between decision variables from a particular problem instance into account. To model the relation between decision variables, we propose a simple method to extract information from the constraint matrix of an MIP. Algorithm 2 outlines the procedure. Given an MIP we represent each decision variable as a node in a graph, and assign an edge between two nodes if the corresponding decision variables appear in a constraint. This graph representation method can capture the linkage of decision variables effectively, especially for graph-based problems. For example, the constructed graph for the Maximum Independent Set problem is exactly the same as the graph on which the problem is defined, and the constructed graph for the Traveling Salesman Problem is the line graph of the graph given by the problem definition.

Given the dataset containing training examples $(\boldsymbol{f}_i, y_i)$ grouped by problem instances and each problem instance associated with an adjacency matrix $\mathcal{G}^{adj}$ representing the relation between decision variables, we can then train GCN. Specifically for a problem instance, the adjacency matrix $\mathcal{G}^{adj}$ is precomputed to normalize graph Laplacian by

$$
L := I - D^{-\frac{1}{2}} \mathcal{G}^{adj} D^{-\frac{1}{2}},
\tag{1}
$$

where $I$ and $D$ are the identity matrix and diagonal matrix of $\mathcal{G}^{adj}$, respectively. The propagation rule is defined as

$$
H^{l+1} := \sigma(L H^l W^l + H^l),
\tag{2}
$$

where $l$ denotes the index of a layer. Inside the activation function $\sigma(\cdot)$, $W^l$ denotes the weight matrix. $H^l$ is a matrix that contains hidden feature representations for decision variables in that problem instance, initialized by the feature vectors of decision variables $H^0 = [\boldsymbol{f}_1, \cdots, \boldsymbol{f}_n]^T$. The second term is referred to as the residual connection, which preserves information from the previous layer. For hidden layers with arbitrary number of neurons, We adopt $ReLU(x) = \max(x, 0)$ as the activation function [15]. For the output layer $L$, there is only one neural to output a scalar value for each decision variable and sigmoid function is used as the activation function for prediction $H_L = [\hat{y}_1, \cdots, \hat{y}_n]$. We train GCN using Stochastic Gradient Descent to minimize the cross-entropy loss function between the predicted values $\hat{y}_i$ of decision variables and their optimal values $y_i$ defined as

$$
\min -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \times \log(\hat{y}_i) + (1 - y_i) \times \log(1 - \hat{y}_i) \right),
\tag{3}
$$

where $N$ is the total number decision variables from multiple training problem instances.

Given an unseen problem instance at test time, we can use the trained GCN to predict for each decision variable $x_i$ a value $\hat{y}_i$, which can be interpreted as the probability of a decision variable taking the value of 1 in the optimal solution $p_i = P(x_i = 1)$. We refer to the array of predicted values as the *probability vector*.

#### B. Probabilistic Branching with guided Depth-first Search

We can then apply the predicted *probability vector* to guide the search process of B&B method. The proposed branching strategy, Probabilistic Branching (PB) attempts to greedily select variable $x_i$ from candidate variables with the highest score $z_i$ to branch on. The score of variable $x_i$ can be computed by

$$
z_i \leftarrow \max(p_i, 1 - p_i),
\tag{4}
$$

where $p_i$ is the probability of $x_i$ being assigned to 1 predicted by the GCN model. This function assigns a higher score to variables whose $p_i$ is closer to either 0 or 1.

<!-- Page 4 -->
Figure 1: Probabilistic Branching with guided Depth-first Search on three decision variables. Given a node, PB branches on a variable our prediction is more confident. The variables that have already been branched are removed from the candidate set. The order of node selection by guided DFS is indicated by the arrow lines.

value tells how certain an ML model is about its prediction. We then can branch on the decision variable $x_i$ with the highest score,

$$
i \leftarrow \arg\max_i z_i; \quad s.t.\ i \in \mathcal{C}.
\tag{5}
$$

$\mathcal{C}$ is the index set of candidate variables that are not fixed at the current node. In this way, our PB method prefers to branch on the variables for which our prediction is more “confident” at the shallow level of the search tree while exploring the uncertain variables at the deep level of the search tree.

We propose to use a guided Depth-first Search (DFS) as the *node selection strategy* to select the next sub-problem to explore. After branching a node, we have a maximum of two child nodes to explore (because the decision variables are binary). Guided DFS selects the child node that results from fixing the decision variable to the nearest integer of $p_i$. When reaching a leaf node, guided DFS backtracks to the deepest node in the search tree among all unexplored nodes. Therefore, guided DFS always explores the node most likely to contain optimal solutions, instructed by the prediction of an ML model. We refer to this implementation of B&B method as PB-DFS.

Figure 1 illustrates PB-DFS applied to a problem with three decision variables. We note that as a configuration of the B&B framework, PB-DFS can be an exact search method if given enough time. However, it is specialized for aggressively seeking high-quality primal solutions while trading off the size of the B&B tree created during the computation. Hence, we implement and evaluate it as a primal heuristic, which partially solves an external copy of the (sub-)problem with a certain termination criterion.

## IV. EXPERIMENTS

In this section, we use numerical experiments to evaluate the efficacy of our proposed method. After describing the experiment setup, we first analyse different ML models in terms of both effectiveness and efficiency. Then, we evaluate the proposed PB-DFS equipped with different ML models against a set of primal heuristics. Further, we show the significance of PB-DFS with the proposed GCN model by comparing it to the full-fledged SCIP solver [11].

### A. Setup

#### a) Test Problems:

We select a set of representative NP-hard problems: Maximum Independent Set (MISP), Dominant Set Problem (DSP), Vertex Cover Problem (VCP), and an additional problem from Operational Research, Combinatorial Auction Problem (CAP). For each problem, we generate instances of three different scales, i.e., small, medium, and large. Small-scale problem instances and medium-scale problem instances are solved to optimality for training and evaluating ML models. Large-scale problem instances are used for evaluating different solution approaches. Details of problem formulations and instance generations are provided in Appendix B.

#### b) ML Models:

We refer to the GCN that takes the proposed graph representation of MIP as LG-GCN, where LG stands for Linkage Graph. We compare LG-GCN against three other machine learning (ML) models, Logistic Regression (LR), XGBoost [16], and a variant of Graph Neural Network (GNN) which represents the constraint matrix of MIP as a tripartite graph [11]. We address this GNN variant as TRIG-GCN, where TRIG stands for Tripartite Graph. We set the number of layers to 20 for LG-GCN and that of TRIG-GCN is set to 2 due to its high computational cost (explained later). For these two Graph Neural Networks, the dimension of the hidden vector of a decision variable is set to 32. For LR and XGBoost, the hyper-parameters are the default ones in the Scikit-learn [17] package. For all ML models, the feature vector of a decision variable has 57 dimensions, containing statistics extracted from the MIP formulation of a problem instance (listed in Appendix A). For each feature, we normalize its values to the range $[0,1]$ using min-max normalization with respect to the decision variables in a particular problem instance. Besides, LG-GCN and TRIG-GCN are trained with graphs of different structures with respect to each problem instance. For a problem, an ML model is trained using 500 optimally-solved small-scale instances.

#### c) Evaluation of ML Models:

We evaluate the classification performance of ML models on two test datasets, constructed from 50 small problem instances (different from the training instances) and medium-sized problem instances, respectively. We measure a model’s performance with the Average Precision (AP) [18], defined by accumulating the product of precision and the change in recall when moving the decision threshold on a set of ranked variables. AP is a more informative metric than the accuracy metric in our context, because it takes the ranking of decision variables into account. This allows AP to better measure the classification performance for imbalanced data that is common in the context of solution prediction for NP-hard problems. Further, AP can better measure the performance of the proposed PB-DFS, because exploring a node not containing the optimal solution in the upper level of the B&B tree is more harmful than exploring one in the lower level of the tree.

<!-- Page 5 -->
Table I: Comparison between ML models. AP column shows the mean statistic of Average Precision values over 50 problem instances. We conduct student’s $t$-test by comparing LG-GCN against other baselines. $p$-value less than 0.05 (a typical significance level) indicates that the LG-GCN is significantly better than other ML models.

| Problem Size | Model     | Independent Set |                 | Dominant Set |                | Vertex Cover |               | Combinatorial Auction |                       |
|--------------|-----------|-----------------|-----------------|--------------|----------------|--------------|---------------|------------------------|------------------------|
|              |           | AP              | $p$-value       | AP           | $p$-value      | AP           | $p$-value     | AP                     | $p$-value              |
| Small        | LG-GCN    | **96.53**       | -               | 87.25        | -              | **98.05**    | -             | **46.10**              | -                      |
|              | TRIG-GCN  | 88.62           | 8.7E-28         | 86.90        | 8.0E-02        | 92.42        | 2.8E-87       | 41.77                  | 4.7E-02                |
|              | XGBoost   | 74.11           | 3.9E-135        | 86.45        | 2.6E-01        | 84.82        | 1.7E-140      | 39.05                  | 1.4E-03                |
|              | LR        | 74.24           | 1.6E-128        | 86.59        | 3.5E-01        | 85.29        | 2.2E-137      | 41.31                  | 2.3E-02                |
| Medium       | LG-GCN    | **96.41**       | -               | 87.52        | -              | **98.24**    | -             | **47.07**              | -                      |
|              | TRIG-GCN  | 88.16           | 4.8E-31         | 87.18        | 5.1E-02        | 92.74        | 2.1E-51       | 42.67                  | 6.0E-04                |
|              | XGBoost   | 73.07           | 1.1E-125        | 86.80        | 2.0E-01        | 84.70        | 5.2E-67       | 40.85                  | 1.2E-06                |
|              | LR        | 73.99           | 2.0E-148        | 86.86        | 2.3E-01        | 85.33        | 3.3E-57       | 42.35                  | 1.8E-04                |

Figure 2: Increase of model prediction time when enlarging the size of problem instances: the $y$-axis is the prediction time in seconds in log-scale, and the $x$-axis is the size of problem instances in 3 scales.

$d)$ *Evaluation of Solution Methods*: We evaluate PB-DFS equipped with the best ML model against heuristic methods as well as problem-reduction approaches on large-scale problem instances. In the first part, PB-DFS is compared with primal heuristics that do not require a feasible solution on large problem instances. By examining the performance of the heuristics enabled by default in SCIP, four types of heuristics are selected as baselines: the Feasibility Pump [19] (FP), the Relaxation Enhanced Neighborhood Search [2] (RENS), a set of 15 diving heuristics, and a set of 8 rounding heuristics [1]. We allow PB-DFS to run only at the root node and terminate it upon the first-feasible solution is found. The compared heuristic methods run multiple times during the B&B process under a cutoff time of 50 seconds with default running frequency and termination criteria tuned by SCIP developers. Generating cutting planes is disabled to best measure the time of finding solutions by different heuristics. In the second part, we demonstrate the effectiveness of PB-DFS by comparing SCIP solver with only PB-DFS as the primal heuristic against full-fledged SCIP solver where all heuristics are enabled as well as a problem-reduction approach by Ding et al. [11]. The problem-reduction approach splits the root node of the search tree by a constraint generated from *probability vector*, and then solved by SCIP-DEF. To alleviate the effects of ML predictions, we use the *probability vector* generated by LG-GCN for both PB-DFS and ML-Split. The cutoff time is set to 1000 seconds. Note that for DSP, we employ an alternative score function $z_i \leftarrow p_i$. The corresponding DFS selects the child node that is the result of fixing the decision variable to one when those nodes are at the same depth. A comparison of alternative score functions is detailed in Appendix C.

$e)$ *Experimental Environment*: We conduct experiments on a cluster with 32 Intel 2.3 GHz CPUs and 128 GB RAM. PB-DFS is implemented using C-api provided by the state-of-the-art open-source solver, SCIP, version 6.0.1. The implementations of Logistic Regression and XGBoost are taken from Scikit-learn [17]. Both LG-GCN and TRIG-GCN are implemented using Tensorflow package [20] where offline training and online prediction are done by multiple CPUs in parallel. All solution approaches are evaluated on a single CPU core. Our code is available online¹.

B. *Results on Solution Prediction*

Table I presents the ML models’ performance on solution prediction. The small-scale test instances are of the same size as the training instances and the medium-scale ones are used for examining the generalization performance of tested ML models. We cannot measure the classification performance of ML models on large-scale instances because the optimal solutions for them are not available. By comparing the mean statistic of AP values, we observe that LG-GCN is very competitive among all problems indicated by mean statistics. We conduct student’s $t$-test

---

¹ Code is available at https://github.com/Joey-Shen/pb-dfs.

<!-- Page 6 -->
Table II: Comparison between PB-DFS and primal heuristics.

| Problem | Heuristic | Best Solution Objective | Best Solution Time | # Instances no feasible solution | # Calls | Heuristic Total Time |
|---------|-----------|--------------------------|--------------------|----------------------------------|---------|----------------------|
| VCP (Min.) | FP | 2137.3 | 1.6 | 19 | 1.0 | 1.0 |
|          | Roundings | 1784.7 | 35.2 | 0 | 252.0 | 8.0 |
|          | PB-DFS-LR | 1634.0 | 3.8 | 0 | 1.0 | 3.8 |
|          | PB-DFS-GCN | **1629.0 (1628.2)** | 5.7 (6.0) | 0 | 1.0 | 5.7 (21.6) |
| DSP (Min.) | FP | 325.1 | 4.0 | 2 | 1.0 | 0.4 |
|          | Roundings | 515.1 | 47.9 | 0 | 135.5 | 0.8 |
|          | RENS | 320.6 | 19.6 | 17 | 1.0 | 13.6 |
|          | PB-DFS-LR | **318.4** | 9.0 | 0 | 1.0 | 9.0 |
|          | PB-DFS-GCN | **318.7 (316.1)** | 10.6 (23.6) | 0 | 1.0 | 10.6 (21.6) |
| MISP (Max.) | FP | 845.8 | 1.3 | 2 | 1.0 | 0.9 |
|           | Roundings | 1225.6 | 38.0 | 0 | 681.0 | 18.0 |
|           | Divings | 1280.0 | 41.5 | 0 | 9.0 | 6.6 |
|           | PB-DFS-LR | 1365.8 | 3.9 | 0 | 1.0 | 3.9 |
|           | PB-DFS-GCN | **1371.0 (1371.6)** | 4.5 (5.6) | 0 | 1.0 | 4.5 (21.4) |
| CAP (Max.) | FP | - | - | 30 | 1.0 | 0.3 |
|           | Divings | 3633.2 | 26.1 | 0 | 30.2 | 6.1 |
|           | Roundings | 3274.4 | 20.4 | 0 | 990.5 | 0.25 |
|           | PB-DFS-LR | 3425.9 | 3.2 | 0 | 1.0 | 1.9 |
|           | PB-DFS-GCN | 3280.5 (3582.8) | 4.5 (12.9) | 0 | 1.0 | 4.5 (22.3) |

by comparing LG-GCN against other baselines where the AP values for a group of test problem instances of LG-GCN is compared with that of other ML models. In practice, $p$-value less than 0.05 (a typical significance level) indicates that the difference between the two samples is significant. Therefore, we confirm that the proposed LG-GCN can predict solutions with better quality as compared to other ML models on MISP, VCP, and CAP. The performances of ML models on DSP are comparable. Note that on CAP which is not originally formulated on graphs, LG-GCN’s AP value is significantly better than TRIG-GCN’s. This shows that the proposed graph construction method is more robust when extending to non-graph based problems as compared to TRIG-GCN. LR is slightly better than XGBoost overall. All models show their capability to generalize to larger problem instances.

In addition to prediction accuracy, the prediction time of an ML model is a part of the total solving time, which should also be considered for developing efficient solution methods. Figure 2 shows the increase of the prediction time when enlarging the problem size for the compared ML models. Comparing graph-based models, we observe that in practice the mean prediction time by LG-GCN is much less than the one by TRIG-GCN. The high computational cost of TRIG-GCN prevents us from building it with a large number of layers. The computation time of LG-GCN is close to those of linear models on VCP and MISP and shifts away on DSP and CAP. This is understandable because the complexity of LG-GCN is linear in the number of edges in a graph [12] and is polynomial with respect to the number of decision variables, as compared to linear growth for LR and XGBoost. However, for MISP and VCP where the constraint coefficient matrix is sparse (i.e., the fraction of zero values is high), in practice, the difference in the growth of computation time with increasing problem size is not as dramatic, but it may be significant when an MIP instance has a dense constraint matrix, e.g., Combinatorial Auction Problems.

## C. Results For Finding Primal Solutions

Table II shows the computational results of PB-DFS as compared to the most effective heuristic methods used in the SCIP solver. Recall that the FP and RENS are two standalone heuristics. *Roundings* refers to a set of 8 rounding heuristics and *Divings* covers 15 diving heuristics. PB-DFS-LR and PB-DFS-GCN stand for the PB-DFS method equipped with LG-GCN and LR model, respectively. Note that the PB-DFS method only runs at the root node and terminates upon finding the first feasible solution. Besides, we analyze an additional criterion using PB-DFS-GCN, terminating with a cutoff time 20 seconds (statistics are shown in brackets). For each problem, we run a heuristic 30 times on different instances. Since the SCIP solver assigns heuristics with different running frequencies based on the characteristic of a problem, we only show heuristics called at least once per instance. The column *# Instances no feasible solution* reports the number of instances that a heuristic does not find a feasible solution. Other columns show statistics with a geometric mean shifted by one averaged over instances where a heuristic finds at least one feasible solution. Note that we only consider the best solutions found by a heuristic. Solutions found by branching are not included. We also show the average number of calls of a heuristic and the total running time by a heuristic in the last columns for reference. For those PB-DFS heuristics, mean prediction time by models is added to the *Best Solution Time* and *Heuristic Total Time*. Note that the primal heuristics do not meet the running criteria by SCIP for a certain problem is excluded from the results.

Overall, the PB-DFS methods can find better primal solutions much earlier on VCP, DSP, and MISP. It is less competitive on CAP. This is understandable because the AP value of the prediction for CAP is low (Table I), indicating the prediction of an ML model is less effective. Comparing PB-DFS with different ML models, PB-DFS-GCN can find better solutions than PB-DFS-LR on VCP, MISP, and CAP, and they are comparable on DSP. This means that the AP values of *probability vectors* for different models can reflect the performance of PB-DFS heuristics to a certain extent. When comparing two termination criteria using the PB-DFS-GCN model, we observe that giving the proposed heuristic more time can lead to better solutions. These results show that the PB-DFS method can be a very strong primal heuristic when the ML prediction is of high-quality (i.e., high AP values).

We further demonstrate the effectiveness of PB-DFS by comparing the use of SCIP equipped with PB-DFS only, against both the full-fledged SCIP solver (SCIP-DEF) and a problem-reduction approach [11] using SCIP-DEF as the solver (ML-Split). Figure 3 presents the change of the primal bound during the solving process and the detailed solving statistics are shown in Table III. From Figure 3, we observe that PB-DFS finds (near-)optimal solutions at the very beginning and outperforms other methods on VCP and MISP. On DSP, early good solutions found by PB-DFS are still very useful such that it can help the solver without any primal heuristic outperform full-fledged solvers.

<!-- Page 7 -->
Figure 3: Change of the primal bound during computation on large-scale problem instances: the $x$-axis is the solving time in seconds, and the $y$-axis is the objective value of the incumbent.

Table III: PB-DFS compared with SCIP-DEF and ML-Split.

| Problem     | Method   | Best Solution Objective | Best Solution Time | Optimality Gap (%) | Heuristic Total Time |
|-------------|----------|-------------------------|--------------------|--------------------|----------------------|
| VCP (Min.)  | SCIP-DEF | 1636.4                  | 542.8              | 3.97               | 198.5                |
|             | ML-Split | 1630.1                  | 202.7              | 3.62               | 202.8                |
|             | PB-DFS   | **1629.0**              | **5.7**            | **3.46**           | **5.7**              |
| DSP (Min.)  | SCIP-DEF | 315.5                   | 390.8              | 3.0                | 114.2                |
|             | ML-Split | **315.3**               | 358.9              | **2.95**           | 110.1                |
|             | PB-DFS   | 316.1                   | 303.4              | 3.15               | 11.5                 |
| MISP (Max.) | SCIP-DEF | 1364.9                  | 559.9              | 4.67               | 174.5                |
|             | ML-Split | 1370.4                  | 281.6              | 4.34               | 200.4                |
|             | PB-DFS   | **1371.0**              | **4.5**            | **4.17**           | **4.5**              |
| CAP (Max.)  | SCIP-DEF | 3824.3                  | 535.0              | 11.62              | 29.2                 |
|             | ML-Split | 3821.2                  | 667.5              | 12.01              | 30.8                 |
|             | PB-DFS   | **3831.4**              | **514.6**          | **11.35**          | **5.5**              |

for a while. Further, PB-DFS is computationally cheap. Therefore, when incorporated into the SCIP solver, PB-DFS does not introduce a large computational overhead, and keeps more computational resources for the B&B process. This explains why the quality of the incumbent solution quickly catches up with other approaches on CAP. The detailed solving statistics in Table III are consistent with the above analysis, which confirms that the PB-DFS method is very competitive

## V. CONCLUSION

In this work, we propose a primal heuristic based on machine learning, Probabilistic Branching with guided Depth-First Search (PB-DFS). PB-DFS is a B&B configuration specialized for boosting the search for high-quality primal solutions, by leveraging a predicted solution to guide the search process of the B&B method. Results show that PB-DFS can find better primal solutions, and find them much faster on several NP-hard covering problems as compared to other general heuristics in a state-of-the-art open-source MIP solver, SCIP. Further, we demonstrate that PB-DFS can make better use of high-quality predicted solutions as compared to recent solution prediction approaches.

We would like to note several promising directions beyond the scope of this work. Firstly, we demonstrate that PB-DFS can be deployed as a primal heuristic that runs only at the root node during a solver’s B&B process. More sophisticated implementations, e.g. triggering PB-DFS to run on different nodes with engineered timings, can lead to further performance improvement. Secondly, PB-DFS relies on high-quality predicted solutions. We observe the drop in the performances of existing ML models when extending it to general MIP problems. We expect that improving ML models in the context of solution prediction for Mixed Integer Programs could be a fruitful avenue for future research.

## REFERENCES

[1] T. Achterberg, T. Berthold, and G. Hendel, “Rounding and propagation heuristics for mixed integer programming,” in *Operations research proceedings 2011*. Springer, 2012, pp. 71–76.

[2] T. Berthold, “Primal heuristics for mixed integer programs,” 2006.

[3] M. Fischetti and A. Lodi, “Heuristics in mixed integer programming,” *Wiley Encyclopedia of Operations Research and Management Science*, 2010.

[4] T. Achterberg, “Scip: solving constraint integer programs,” *Mathematical Programming Computation*, vol. 1, no. 1, pp. 1–41, 2009.

[5] E. Aarts, E. H. Aarts, and J. K. Lenstra, *Local search in combinatorial optimization*. Princeton University Press, 2003.

[6] T. Berthold, “Rens,” *Mathematical Programming Computation*, vol. 6, no. 1, pp. 33–54, 2014.

[7] M. Gasse, D. Chételat, N. Ferroni, L. Charlin, and A. Lodi, “Exact combinatorial optimization with graph convolutional neural networks,” in *Advances in Neural Information Processing Systems*, 2019, pp. 15 554–15 566.

[8] H. He, H. Daume III, and J. M. Eisner, “Learning to search in branch and bound algorithms,” in *Advances in neural information processing systems*, 2014, pp. 3293–3301.

[9] E. B. Khalil, B. Dilkina, G. L. Nemhauser, S. Ahmed, and Y. Shao, “Learning to run heuristics in tree search.” in *IJCAI*, 2017, pp. 659–666.

[10] Y. Sun, X. Li, and A. Ernst, “Using statistical measures and machine learning for graph reduction to solve maximum weight clique problems,” *IEEE transactions on pattern analysis and machine intelligence*, 2019.

[11] J.-Y. Ding, C. Zhang, L. Shen, S. Li, B. Wang, Y. Xu, and L. Song, “Accelerating primal solution findings for mixed integer programs based on solution prediction,” *arXiv preprint arXiv:1906.09575*, 2019.

[12] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” *arXiv preprint arXiv:1609.02907*, 2016.

[13] V. Nair, S. Bartunov, F. Gimeno, I. von Glehn, P. Lichocki, I. Lobov, B. O’Donoghue, N. Sonnerat, C. Tjandraatmadja, P. Wang *et al.*, “Solving mixed integer programs using neural networks,” *arXiv preprint arXiv:2012.13349*, 2020.

[14] C. M. Bishop, *Pattern recognition and machine learning*. springer, 2006.

[15] V. Nair and G. E. Hinton, “Rectified linear units improve restricted boltzmann machines,” in *ICML*, 2010.

<!-- Page 8 -->
[16] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*, 2016, pp. 785–794.

[17] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg *et al.*, “Scikit-learn: Machine learning in python,” *the Journal of machine Learning research*, vol. 12, pp. 2825–2830, 2011.

[18] M. Zhu, “Recall, precision and average precision,” *Department of Statistics and Actuarial Science, University of Waterloo, Waterloo*, vol. 2, p. 30, 2004.

[19] M. Fischetti, F. Glover, and A. Lodi, “The feasibility pump,” *Mathematical Programming*, vol. 104, no. 1, pp. 91–104, 2005.

[20] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard *et al.*, “Tensorflow: A system for large-scale machine learning,” in *12th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 16)*, 2016, pp. 265–283.

[21] K. Leyton-Brown, M. Pearson, and Y. Shoham, “Towards a universal test suite for combinatorial auction algorithms,” in *Proceedings of the 2nd ACM conference on Electronic commerce*, 2000, pp. 66–76.

## APPENDIX A

The features for decision variables are outlined as follows:

- original, positive and negative objective coefficients;
- number of non-zero, positive, and negative coefficients in constraints;
- variable LP solution of the original problem $\hat{x}$; $\hat{x} - \lfloor \hat{x} \rfloor$; $\lceil \hat{x} \rceil - \hat{x}$; a boolean indicator for whether $\hat{x}$ is fractional;
- variable’s upward and downward pseudo costs; the ratio between these pseudocosts; sum and product of these pseudo costs; variable’s reduced cost;
- global lower bound and upper bound;
- mean, standard deviation, minimum, and maximum degree for constraints in which the variable has a non-zero coefficient. The degree of a constraint is the number of non-zero coefficients of that constraint;
- the maximum and the minimum ratio between the left-hand-side and right-hand-side over constraints where the variable has a non-zero coefficient;
- statistics (sum, mean, standard deviation, maximum, minimum) for a variable’s positive and negative constraint coefficients respectively;
- coefficient statistics of all variables in the constraints (sum, mean, standard deviation, maximum, minimum) with respect to three weighting schemes: unit weight, dual cost, the inverse of the sum of the coefficients in the constraint.

## APPENDIX B

The MIP formulations for tested problems are as follows:

### a) Maximum Independent Set Problem (MISP):

In an undirected graph $\mathcal{G}(\mathcal{V},\mathcal{E})$, a subset of nodes $\mathcal{S} \subset \mathcal{V}$ is independent iff there is no edge between any pair of nodes in $\mathcal{S}$. A MISP is to find an independent set in $\mathcal{G}$ of maximum cardinality. The MIP formulation of the MISP is: $\max_{\mathbf{x}} \sum_{v \in \mathcal{V}} x_v$, subject to $x_u + x_v \leq 1, \forall (u,v) \in \mathcal{E}$ and $x_v \in {0,1}, \forall v \in \mathcal{V}$.

### b) Dominant Set Problem (DSP):

In an undirected graph $\mathcal{G}(\mathcal{V},\mathcal{E})$, a subset of nodes $\mathcal{S} \subset \mathcal{V}$ dominates the complementary subset $\mathcal{V} \setminus \mathcal{S}$ iff every node not in $\mathcal{S}$ is adjacent to at least one node in $\mathcal{S}$. The objective of a DSP is to find a dominant set in $\mathcal{G}$ of minimum cardinality. The MIP of the DSP is as follows: $\min_{\mathbf{x}} \sum_{v \in \mathcal{V}} x_v$, subject to $x_v + \sum_{u \in N(v)} x_u \geq 1, \forall v \in \mathcal{V}$ and $x_v \in {0,1}, \forall v \in \mathcal{V}$. $N(v)$ denotes the set of neighborhood nodes of $v$.

### c) Vertex Cover Problem (VCP):

In an undirected graph $\mathcal{G}(\mathcal{V},\mathcal{E})$, a subset of nodes $\mathcal{S} \subset \mathcal{V}$ is a cover of $\mathcal{G}$ iff for every edge $e \in \mathcal{E}$, there is at least one endpoint in $\mathcal{S}$. The objective of the VCP is to find a cover set in $\mathcal{G}$ of minimum cardinality. The MIP of the VCP is as follows: $\min_{\mathbf{x}} \sum_{v \in \mathcal{V}} x_v$, subject to $x_u + x_v \geq 1, \forall (u,v) \in \mathcal{E}$ and $x_v \in {0,1}, \forall v \in \mathcal{V}$.

### d) Combinatorial Auction Problem (CAP):

A seller faces to selectively accept offers from bidders. Each offer indexed by $i$ contains a bid $p_i$ for a set of items (bundle) $\mathcal{C}_i$ by a particular bidder. The seller has limited amount of goods and aims to allocate the goods in a way that maximizes the total revenue. We use $\mathcal{I}$ and $\mathcal{J}$ to denote the index set of offers and the index set of items. Formally, the MIP formulation of the problem can be expressed as $\max_{\mathbf{x}} \sum_{i \in \mathcal{I}} p_i x_i$, subject to $\sum_{k \in \{ i \mid j \in \mathcal{C}_i, \forall i \}} x_k \leq 1, \forall j \in \mathcal{J}$ and $x_i \in {0,1}, \forall i \in \mathcal{I}$.

For MISP, DSP, and VCP, we sample random graphs using Erdős-Rényi generator. The affinity is set to 4. The training data consists of examples from solved small graph instances between 500 to 1001 nodes. We form the small-scale and medium-scale testing datasets with solved graph instances of 1000 nodes and those of 2000 nodes. We evaluate heuristics on large-scale graph instances of 3000 nodes. The CAP instances are generated using an arbitrary relationship procedure. The CAP instances in small-scale are sampled with items in the range [100, 150] and bids in the range [500, 750]. Instances in medium-scale and large-scale are generated with 150 items for 750 bids and 200 items for 1000 bids, respectively. The detailed parameter settings are given by following [21].

## APPENDIX C

We consider two alternative score functions, $z_i \leftarrow p_i$ and $z_i \leftarrow 1 - p_i$, subject to $i \in \mathcal{C}$. When using $z_i \leftarrow p_i$ as the score function to select the decision variable with the maximum score to branch, our guided DFS always prefers the node that is the result of fixing the decision variable to 1. The behavior of the PB-DFS due to this score function can be interpreted as incrementally adding variables that more likely belong to an optimal solution until a feasible solution is obtained. When using the other score function

---

**Table IV: Effect of Score Functions**

| score function | Independent Set |                 | Dominant Set |              | Vertex Cover |             | Combinatorial Auction |                    |
|----------------|-----------------|-----------------|--------------|--------------|--------------|-------------|-----------------------|--------------------|
|                | objective       | time            | objective    | time         | objective    | time        | objective             | time               |
| $\mathbf{p}$   | 1371.0          | 2.8             | 318.7        | 9.9          | 1629.0       | 3.9         | 3226.6                | 2.3                |
| $\mathbf{1-p}$ | 1371.1          | 3.7             | 322.0        | 11.3         | 1629.1       | 3.3         | 3317.0                | 5.2                |
| $\max(\mathbf{p}, \mathbf{1-p})$ | 1371.0 | 3.8             | 321.9        | 11.6         | 1628.9       | 3.9         | 3316.0                | 5.2                |

<!-- Page 9 -->
$z_i \leftarrow 1 - p_i$, guided DFS always prefers the node that is the result of fixing the decision variable to 0, and the resulting behavior of the PB-DFS can be interpreted as continuously removing variables that less likely belong to an optimal solution until a feasible solution is obtained. In table IV, we observe that the score functions do not significantly affect the first-found solution.