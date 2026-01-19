<!-- Page 1 -->
# SATformer: Transformer-Based UNSAT Core Learning

Zhengyuan Shi$^1$, Min Li$^1$, Yi Liu$^1$, Sadaf Khan$^1$, Junhua Huang$^2$, Hui-Ling Zhen$^2$, Mingxuan Yuan$^2$ and Qiang Xu$^1$

$^1$ The Chinese University of Hong Kong, $^2$ Noah’s Ark Lab, Huawei

{zyshi21, mli, yliu22, skhan, qxu}@cse.cuhk.edu.hk

{huangjunhua15, zhenhuiling2, Yuan.Mingxuan}@huawei.com

---

**Abstract**—This paper introduces SATformer, a novel Transformer-based approach for the Boolean Satisfiability (SAT) problem. Rather than solving the problem directly, SATformer approaches the problem from the opposite direction by focusing on unsatisfiability. Specifically, it models clause interactions to identify any unsatisfiable sub-problems. Using a graph neural network, we convert clauses into clause embeddings and employ a hierarchical Transformer-based model to understand clause correlation. SATformer is trained through a multi-task learning approach, using the single-bit satisfiability result and the minimal unsatisfiable core (MUC) for UNSAT problems as clause supervision. As an end-to-end learning-based satisfiability classifier, the performance of SATformer surpasses that of NeuroSAT significantly. Furthermore, we integrate the clause predictions made by SATformer into modern heuristic-based SAT solvers and validate our approach with a logic equivalence checking task. Experimental results show that our SATformer can decrease the runtime of existing solvers by an average of 21.33%.

## I. INTRODUCTION

The Boolean Satisfiability (SAT) problem, fundamental to many fields, seeks to identify if there exists at least one assignment that makes a given Boolean formula *True*. Key applications in the electronic design automation (EDA) field include logic equivalence checking (LEC) [1], model checking [2], and automatic test pattern generation (ATPG) [3]. Being the first proven NP-complete problem, SAT has no complete solution achievable in polynomial time. Most solvers [4]–[10] use heuristic search techniques for large industrial SAT instances. They return a ‘satisfiable’ (SAT) result if a valid assignment is found, ‘unsatisfiable’ (UNSAT) if all search paths yield no valid assignment, or ‘unknown’ if the maximum runtime is exceeded. Despite impressive results from existing solvers on general benchmarks [10]–[12], limitations remain in their ability to address EDA problems effectively.

One notable issue is the inability of current solvers to self-direct heuristic selection based on satisfiability. Although modern solvers [7]–[10] offer satisfiability-specific modes (SAT or UNSAT), they rely on manual selection rather than self-prediction of satisfiability. Moreover, heuristic efficiency can vary markedly when solving SAT and UNSAT instances [13]. For instance, while the local search heuristic can expedite the resolution of SAT instances, it can slow down UNSAT instances, and vice versa for the dynamic restart heuristic. Inappropriate configuration of solver heuristics can potentially increase solving time by up to 50% [14], as solvers prove SAT and UNSAT in divergent manners. Thus, the prediction of SAT or UNSAT before solving is critical for heuristic adjustment and optimizing the efficiency of heuristic algorithms in solvers.

Another notable limitation of existing solvers is the absence of dedicated heuristics for UNSAT problems. Many EDA applications confront numerous UNSAT problems, which present a greater challenge than SAT problems. For instance, in logic equivalence checking and model checking, a majority of instances are unsatisfiable, necessitating significant runtime to confirm unsatisfiability [15], [16]. Similarly, in ATPG, proving that some faults are untestable, which corresponds to proving unsatisfiability, consumes a considerable amount of runtime [17], [18]. Hence, integrating an efficient UNSAT heuristic can substantially enhance the generalizability and efficiency of modern solvers within the EDA domain.

To address the above issues, we propose *SATformer*, a novel framework for Boolean satisfiability that not only predicts satisfiability but also accelerates the solving process for unsatisfiable problems. We reinterpret the nature of unsatisfiability as the presence of an unsatisfiable sub-problem and address this by modeling clause interactions. Specifically, the solver asserts unsatisfiability only when an inevitable conflict arises while seeking valid assignments. This conflict is embodied in a subset of clauses known as the Unsatisfiable Core (UNSAT Core), whereby the sub-problem formed by the UNSAT Core is also unsatisfiable. Consequently, by capturing clause interactions and identifying the UNSAT Core, we can determine the satisfiability of the given instance.

To model clause correlation, we formulate a deep learning model that represents clauses as embeddings using a graph neural network (GNN) and discerns clause connections through a novel hierarchical Transformer-based model. This hierarchical model consolidates clauses into clause groups incrementally, capturing relationships between these groups. Hence, our model learns interactions among multiple clauses, transcending the pairwise dependency typical of standard Transformers. Additionally, our model is dual-task trained: for satisfiability prediction with single-bit supervision, and for UNSAT Core prediction supervised by the minimal unsatisfiable core (MUC). Despite an UNSAT instance potentially containing multiple cores of varying size, the core with the minimal number of clauses exhibits less diversity, as highlighted in NeuroCore [19].

We further deploy SATformer as an initialization heuristic for SAT solvers. SATformer outputs a binary prediction indicating satisfiability and a ‘prediction score’ for each clause that estimates its likelihood of forming an UNSAT core. We view clauses with high prediction scores as key contributors to unsatisfiability, and propose that addressing sub-problems composed of such high-score clauses can hasten the discovery of UNSAT results. We leverage this clause prediction score to determine search priorities. Specifically, we calculate variable priority scores based on variable-clause connections and use these as the initial variable activity scores, e.g., the Variable State Independent Decaying Sum (VSIDS) [20]. As most SAT solvers provide a variable branching priority interface [4], [8], [9], [20], SATformer can be seamlessly integrated as a plug-in, negating extensive solver reconstruction. Unlike previous variable branching heuristics that solely concentrate on variable activity, our approach utilizes clause-level correlation. We validate the efficacy of SATformer by augmenting the performance of two modern SAT solvers: CaDiCaL [8] and Kissat [9].

This work makes the following contributions:

- We propose a fresh perspective on the unsatisfiability problem by modeling clause correlations and identifying the presence of unsatisfiable sub-problems.
- We introduce a hierarchical Transformer-based model to capture clause interactions, leveraging the minimal UNSAT core as

<!-- Page 2 -->
supervision, and training our model to distinguish clauses that are *likely* or *unlikely* to form an UNSAT core.

- We integrate our model as a learning-enhanced initialization heuristic into contemporary SAT solvers, yielding a notable acceleration in the solving process.

We organize the rest of the paper as follows. Section II reviews the related works about learning-aided SAT solvers and Transformer blocks. We detail our SATformer framework in Section III and demonstrate the effectiveness of our model with comprehensive experiments in Section V. Finally, Section VI concludes this paper.

## II. RELATED WORK

### A. Deep Learning for SAT Solving

In the area of combinatorial optimization, the increasing size of problem instances remains a critical challenge. For the SAT problem, deep learning provides an attractive solution to improve the solving efficiency, as surveyed in [21].

Generally, learning-based SAT solvers fall into two categories: standalone learning-based SAT solvers and learning-aided SAT solvers. On the one hand, standalone learning-based SAT solvers predict satisfiability and decode assignments with end-to-end deep learning models solely. For example, NeuroSAT [22] treats the SAT problem as a classification task and trains an end-to-end GNN-based framework to predict binary results (satisfiable or unsatisfiable). DG-DAGRNN [23] focuses on the circuit satisfiability problem, i.e., determining whether the single primary output of a given circuit can return logic '1'. They propose a differentiable method to predict a value that evaluates the satisfiability. By maximizing the value based on reinforcement learning, the model is more likely to find a satisfiable assignment. DeepSAT [24] formulates the SAT solution as a product of conditional Bernoulli distributions and obtains an assignment by maximizing the joint probability. However, the performance of these standalone learning-based solvers lags behind modern non-learning approaches by a large margin [19], [24]. Since all assertions produced by deep learning models are based on probabilistic statistic, while SAT problem is a strictly deterministic logic reasoning problem, it is naturally unsuitable to solve the SAT problem with a pure learning method.

On the other hand, combining deep learning model with modern SAT solvers has emerged as a promising research direction. As heuristics dominate mainstream modern SAT solvers, learning-aided solutions that replace manually-crafted heuristics with more effective and efficient heuristics produced by deep learning models can result in significant performance improvements [19], [25]–[27]. For example, NLocalSAT [25] achieves a runtime reduction of $27\% \sim 62\%$ on the stochastic local search (SLS) SAT solver by producing initialization assignments using a deep learning model. Graph-Q-SAT [26] replaces the searching heuristic with a reinforcement learning agent and reduces the number of iterations required to solve SAT problems. Moreover, NeuroCore [19] and NeuroComb [27] learn the distribution of UNSAT cores to guide SAT solving. Although these learning-aided solutions are powerful, they are generally designed and do not guarantee to speed up a specific kind of problem [10]–[12]. In industrial applications, engineers are often more focused on solving problems in a specific domain. Therefore, a task-specific solution that incorporates prior knowledge about the problems at hand is more practical than a general solution. In this paper, we focus on the unsatisfiable problem in the EDA area and propose a corresponding learning-aided solution.

### B. Transformer and Self-attention Mechanism

The Transformer [28] is well acknowledged as one of the most powerful architectures for modeling sequential data. The adoption of the Transformer has led to tremendous progress in the field of text comprehension [29], machine translation [30] and even computer vision [31], [32]. The self-attention mechanism in the Transformer block treats sequential input tokens as a fully connected graph and represents the connectivity between every pair of nodes. This enables Transformer-based models to capture the correlation between each pair of tokens.

The first attempt to involve the Transformer for solving MaxSAT problem (a variant of SAT problem) is introduced in [33]. The proposed model represents a given SAT instance as a bipartite graph and applies Transformers to aggregate the message over the nodes. In our work, we treat the clauses as a non-ordered sequence and apply the Transformer to capture the correlation among these clauses.

## III. SATFORMER

### A. Boolean Satisfiability Problem

A Boolean formula $\phi$ consists of a set of variables $\{x_j\}_{j=1}^{l} \in \{\text{True}, \text{False}\}$ and a set of Boolean operators $\{\text{AND} (\land), \text{OR}(\lor), \text{NOT}(\neg), ...\}$ over these variables. Solving the SAT problem involves determining whether there is at least one valid assignment of the variables so that the given formula $\phi$ evaluates to *true*.

Any Boolean formula can be converted into Conjunctive Normal Form (CNF) with an equivalent transformation in linear time [34]. Under CNF conventions, variables and their negations are referred to as *literals* (denoted as $x_j$ or $\neg x_j$). A disjunction of several literals forms a *clause* $C_i = (x_1 \lor x_2 \lor ...)$. In Eq. (1), we provide an example to expand CNF, wherein the instance has three variables $I = \{x_1, x_2, x_3\}$ and consists of three clauses $C_1 = \neg x_1 \lor x_2, C_2 = \neg x_2 \lor \neg x_3, C_3 = x_1 \lor x_3$.

$$
\phi := (\neg x_1 \lor x_2) \land (\neg x_2 \lor \neg x_3) \land (x_1 \lor x_3)
\tag{1}
$$

The Literal-Clause Graph (LCG) is a bipartite graph representation of CNF formulas, which consists of literal nodes and clause nodes. Two types of edges are defined in LCG: one connecting a literal node to a clause node, and another connecting a literal node to its corresponding negation. Fig. 1 is an example representing Eq. (1) in LCG.

![Fig. 1. An example of Literal-Clause Graph (LCG).](image_placeholder)

Besides, given an unsatisfiable CNF instance, if a sub-problem constructed by some of its clauses is also unsatisfiable, then such a sub-problem with specific clauses is referred to as an UNSAT core. Take the unsatisfiable instance $\phi_U$ shown in Eq. (2) as an example. We can extract an unsatisfiable sub-problem $\phi'_U = C_2 \land C_5 \land C_6$ as one of the UNSAT cores of the original instance $\phi_U$. The concept of UNSAT core can be further expanded by the searching-based solving process. As the clauses $C_2$ and $C_6$ each only include one

<!-- Page 3 -->
literal, to satisfy both clauses, the variables must be assigned as $x_1 = false$ and $x_2 = true$. Nevertheless, this assignment does not satisfy clause $C_5$ under any circumstances, regardless of the value assigned to the remaining variable $x_3$. Therefore, the solver proves there is an inevitable conflict among these clauses so that all searching branches for solving problem $\phi_U$ fail. It is worth noting that there is no UNSAT core in a satisfiable instance.

$$
\begin{aligned}
C_1 &= \neg x_1 \vee x_2 \vee \neg x_3 \\
C_2 &= x_2 \\
C_3 &= \neg x_1 \vee \neg x_2 \vee \neg x_3 \\
C_4 &= \neg x_1 \vee x_2 \vee x_3 \\
C_5 &= x_1 \vee \neg x_2 \\
C_6 &= \neg x_1 \\
C_7 &= x_1 \vee \neg x_2 \vee x_3 \\
C_8 &= x_1 \vee x_2 \vee x_3 \\
C_9 &= x_1 \vee \neg x_3 \\
\phi_U &:= C_1 \wedge C_2 \ldots \wedge C_9
\end{aligned}
\tag{2}
$$

Since UNSAT cores provide sufficient causality for unsatisfiability judgement, one can determine whether a CNF instance is unsatisfiable or not by extracting its UNSAT cores. In this work, we propose a novel approach to solve the SAT problem by first learning to identify the clauses that contribute to the unsatisfiability of the instance (i.e., clauses included in the UNSAT core) and then modeling the interactions among these clauses.

## B. Overview

Fig. 2 shows an overview of our proposed SATformer. Firstly, our model employs a GNN to obtain clause and literal representations from a CNF instance. After performing message-passing for several iterations based on the LCG, the GNN learns the literal embeddings with rich information from the connecting clauses and the clause embeddings containing information from the involved literals.

Secondly, after obtaining the clause embeddings, we design a classifier to predict which clauses are involved in the UNSAT core. Formally, the classifier is implemented by a Multilayer Perceptron (MLP) as shown in Eq. (3), where $H_i^0$ is the embedding of clause $C_i$ produced by the GNN, $y_i \in (0,1)$ is the predicted probability that $C_i$ belongs to the UNSAT core, and $n$ is the number of clauses. It should be noted that this structure may not be perfect for classifying clauses with high accuracy. However, similar to NeuroCore [19], which learns a heuristic that assigns higher priority to important variables, the proposed classifier can help to highlight the clauses that contribute more to unsatisfiability.

$$
y_i = MLP(H_i^0), i = 0,1,...,n
\tag{3}
$$

Thirdly, we consider that the unsatisfiability arises from the interactions among clauses in the UNSAT core. We take the clause embeddings from the GNN as inputs and apply the Transformer to capture such clause relationships. Benefiting from the self-attention mechanism inside the Transformer, the SATformer gives more attention to the clauses that are likely to be involved in the UNSAT core, which is proved in Section V-C. However, the standard Transformer only learns pairwise correlations, and there is almost no cause of unsatisfiability arising from a pair of clauses. To address this issue, we merge clauses into groups in a hierarchical manner and train a hierarchical Transformer-based model to learn these clause groups, which is elaborated in Section III-D.

## C. Training Objective

We adopt a multi-task learning (MTL) [35] strategy to improve the performance of SATformer. As shown in Fig. 2, we train our model to predict UNSAT core ($Task\_clause$) and satisfiability ($Task\_sat$) simultaneously.

Firstly, although the UNSAT core is not unique for a given instance, there are only a few possible combinations for minimal unsatisfiable cores (MUC), i.e., the UNSAT cores with the smallest size. We denote $n$ as the number of clauses and label a binary vector $Y_{clause}^* = \{y_1^*, y_2^*, ..., y_n^*\}$ as the ground truth for each instance, which represents whether the clause is included in MUC. Hence, the binary vector $Y_{clause}^*$ for a satisfiable instance consists of all zeroes while some bits are assigned ones for an unsatisfiable instance.

We denote the clause embedding matrix as $\mathbf{H}^0 = \{H_1^0, H_2^0, ..., H_n^0\}$ and obtain $Y_{clause} = \{y_1, y_2, ..., y_n\}$ to model a probability distribution over clauses using Eq. (4). We regard $Y_{clause}$ as the contribution of each clause to unsatisfiability, where a larger value indicates that the corresponding clause is more likely to form the MUC.

$$
Y_{clause} = softmax(MLP(\mathbf{H}^0))
\tag{4}
$$

The model is trained by minimizing the Kullback-Leibler (KL) divergence [36]:

$$
\mathcal{L}_{clause} = \sum_{i=1}^{n} y_i^* \cdot \log(\frac{y_i^*}{y_i})
\tag{5}
$$

Secondly, the SATformer predicts binary satisfiability as another task ($Task\_sat$). The model learns to produce a predicted result $Y_{sat} \in (0,1)$ with single-bit supervision $Y_{sat}^* \in \{0,1\}$. We apply Binary Cross-Entropy (BCE) loss to train the model.

$$
\mathcal{L}_{sat} = -(Y_{sat}^* \cdot \log Y_{sat} + (1 - Y_{sat}^*) \cdot \log(1 - (Y_{sat})))
\tag{6}
$$

Thus, the overall loss is the weighted sum of $\mathcal{L}_{clause}$ and $\mathcal{L}_{sat}$

$$
\mathcal{L} = \frac{p_{clause} \cdot \mathcal{L}_{clause} + p_{sat} \cdot \mathcal{L}_{sat}}{p_{clause} + p_{sat}}
\tag{7}
$$

Intuitively, $Task\_clause$ aims to distinguish which clauses contribute more to unsatisfiability. By doing so, our model learns to pay more attention to these clauses and predicts the final satisfiability as $Task\_sat$. Therefore, we can improve the prediction accuracy by leveraging useful information between these two related tasks.

## D. Hierarchical Design

We further propose a hierarchical Transformer-based model to capture the correlation among clauses. Formally, for the standard Transformer $\mathcal{T}$, we define the embeddings of the input tokens as $\mathbf{H} = \{H_1, H_2, ..., H_n\} \in \mathbb{R}^{n \times k}$. The self-attention matrix $\mathbf{A}$ and updated embeddings $\hat{\mathbf{H}} = (\hat{H}_1, \hat{H}_2, ..., \hat{H}_n)$ are denoted as:

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{W}^Q \mathbf{H}, \; \mathbf{K} = \mathbf{W}^K \mathbf{H}, \; \mathbf{V} = \mathbf{W}^V \mathbf{H} \\
\mathbf{A} &= softmax(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{k}}) \\
\hat{\mathbf{H}} &= \mathcal{T}(\mathbf{H}) = \mathbf{A} \mathbf{V}
\end{aligned}
\tag{8}
$$

where $\mathbf{W}^Q$, $\mathbf{W}^K$, $\mathbf{W}^V$ are three learnable projection matrices. $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ denote matrices packing sets of queries, keys, and values, respectively. And in our case, they have the same shape with $\mathbf{H}$.

In most cases, the unsatisfiability can not be determined by a pair of clauses. As demonstrated in the example instance in Eq. (2), the solver requires to consider at least 3 clauses ($C_2$, $C_5$ and

<!-- Page 4 -->
Fig. 2. An overview of the model architecture.

Fig. 3. The architecture of the hierarchical Transformer-based model.

$C_6)$ to assert unsatisfiable. However, capturing the dependencies among multiple tokens within one Transformer block is impractical. Although increasing the layer of Transformer blocks can mitigate this problem, it significantly increases the model complexity. In this paper, we propose a hierarchical Transformer-based model including many Attention Unit (AU), as shown in Fig. 3. Briefly, AU merges clauses within a fixed-size window into a clause group and then packages these groups into another group level by level. The resultant clause group embedding contains information from all clauses covered by the lower-level groups. At a higher level, the model takes clause group embeddings as inputs and learns the relationship between two clause groups.

More formally, the clause embeddings from the GNN model are denoted as $H_1^0, ..., H_n^0$, where $n$ is the number of clauses. We also denote the clause group embeddings produced by AU as $G_1^l, ..., G_{D^l}^l$, where $D^l$ is the number of windows in level $l$ and the maximum level is $L$. Given the fixed window size $w$, we calculate $D^l$ as Eq. (9). Before dividing clauses into groups, we randomly pad the length of the embedding matrix to $w \cdot D^l$. For example, if there are $n = 11$ clauses and the window size $w = 4$, we divide these 11 clauses with 1 more padding clause into $D^1 = 3$ windows at the first level.

$$
D^l = \left\lceil \frac{n}{\text{Pow}(w,\ l)} \right\rceil \tag{9}
$$

With a window of size $w$, the clause group embeddings are updated as:

$$
\begin{aligned}
\hat{\mathbf{H}}_d^l &= \mathcal{T}(\mathbf{H}_d^{l-1}) \\
\mathbf{G}_d^l &= \text{LN}(\hat{\mathbf{H}}_d^l)
\end{aligned} \tag{10}
$$

where $d$ is the index, matrix $\mathbf{H}_d^{l-1} = \{H_{(d-1)\cdot w}^{l-1}, ..., H_{d\cdot w-1}^{l-1}\}$ consists of embedding vectors of clauses in the window $d$ and the matrix $\hat{\mathbf{H}}_d^l$ after processed by Transformer block. We also combine these clause embeddings into one group embedding $\mathbf{G}_d^l$ for each window, which is implemented via a linear transformation LN. For the higher level, i.e., $l \geq 2$, the inputs for the Transformer should be clause group embeddings $\mathbf{G}_d^{l-1} = \{G_{(d-1)\cdot w}^{l-1}, ..., G_{d\cdot w-1}^{l-1}\}$ instead of $\mathbf{H}_d^{l-1}$.

We still take the instance in Eq. (2) as an example. When the window size is 3, these 9 clauses are divided into 3 clause groups, where $G_1^1$ contains the information from $\{C_1, C_2, C_3\}$, $G_2^1$ represents $\{C_4, C_5, C_6\}$, and so on. We cannot find any UNSAT cores by combining clauses pairwise, but at a high level, the clauses contained in $G_1^1$ and $G_2^1$ can construct an UNSAT core. Therefore, to determine whether the instance is satisfiable, the model should consider the pairwise relationships of groups at various grain sizes.

As shown in Eq. (11), similar to the Feature Pyramid Networks (FPN) [37], we also obtain level embedding $F^l$ based on the maximum pooling (MaxPooling) of all $\mathbf{H}^l$, $\hat{\mathbf{H}}^l$ and $\mathbf{G}^l$ as belows:

$$
F^l = 
\begin{cases}
\text{MaxPooling}(\mathbf{H}_0^l, ..., \mathbf{H}_{D^l}^l), & l = 0 \\
\text{MaxPooling}(\hat{\mathbf{H}}_0^l, ..., \hat{\mathbf{H}}_{D^l}^l), & l = 1 \\
\text{MaxPooling}(\hat{\mathbf{G}}_0^l, ..., \hat{\mathbf{G}}_{D^l}^l), & l = 2, ..., L
\end{cases} \tag{11}
$$

Finally, a MLP reads out the concatenation (Cat) of $F^l$ ($l = 1, 2, ..., L$) and the final clause group embedding $G^{L*}$. If there are

<!-- Page 5 -->
multiple clause groups in the final level, we need to apply maximum pooling to obtain the final $G^{L*}$.

$$
G^{L*} = \text{MaxPooling}(\mathbf{G}_0^L, ..., \mathbf{G}_{D^l}^L)
\tag{12}
$$

$$
Y_{sat} = \text{MLP}(\text{Cat}(F^0, ..., F^L, G^{L*}))
$$

With the hierarchical Transformer-based model, our proposed SATformer can capture the causes of unsatisfiability among multiple clauses rather than pairs of clauses.

## IV. COMBINATION WITH SAT SOLVERS

In this subsection, we employ the SATformer as an initialization heuristic and combine it with non-learning SAT solvers. To facilitate understanding, we briefly introduce the background of modern non-learning SAT solvers and related heuristics in Section IV-A. Following that, we demonstrate how to combine our SATformer with two modern solvers (CaDiCaL [8] and Kissat [9]) in Section IV-B.

### A. Background on SAT Solvers

The mainstream modern SAT solvers [4], [8]–[10] are based on the searching algorithm and reduce the search space by a heuristic named Conflict-Driven Clause Learning (CDCL) [38]. Specifically, CDCL-based SAT solvers first assign variables and perform Boolean constraint propagation until the current variable assignment is not valid, resulting in a *conflict*. Then, the solvers analyze the conflict to identify variable dependencies and parse them into new clauses, referred to as *learnt clauses*. Adding such learned clauses can introduce additional constraints into the solving procedure and significantly reduce the search space, especially for UNSAT problems [11].

In order to encounter conflicts as much as possible and speed up SAT solving, previous SAT solvers introduce a heuristic [20] to guide variable decision for more conflicts. This approach maintains a Variable State-Independent Decaying Sum (VSIDS) score for each variable, indicating how many conflicts the variable has been involved in. During the solving procedure, the solvers assign decision values to variables with high VSIDS scores as a priority. Due to the promising performance of this variable decision heuristic, many modern SAT solvers also use VSIDS scores as a variable decision heuristic [4], [8], [9]. However, the VSIDS scores are only obtained by an online algorithm during solving and lack an efficient initialization strategy.

### B. Hybrid Solver Design

We combine our SATformer with modern SAT solvers. Firstly, our SATformer predicts both binary satisfiability result and the clauses’ contribution to unsatisfiability. As the solving procedure for UNSAT instances is to cause more conflicts to reduce the search space, the contribution to unsatisfiability (see Eq. (4)) can be considered as the contribution to conflicts.

Secondly, we convert the above clause prediction to variable prediction. Specifically, based on the clause prediction, we calculate the score leading to conflicts for each variable using Eq. (13), where $A_{j,i}$ is the adjacency matrix of variable $x_j$ and clause $C_i$, and $M$ is the total number of variables.

$$
v_j = \sum_{C_i \in \phi} (A_{j,i} * y_i), j = 1, ..., M
\tag{13}
$$

Thirdly, we provide the above variable score $v_j$ as the initial VSIDS score to the modern SAT solvers. Since most solvers reserve the interface of VSIDS score, our SATformer can be embedded as a plug-in without modifying the SAT solvers. The details of the combination are shown in Algorithm 1.

---

**Algorithm 1 VSIDS Initialization with SATformer**

Problem instance $\phi$, with $M$ variables and $n$ clauses.  
Variable scores $v_j, j = 1, ..., M$, initialized with zeroes.  
SAT solver $\mathcal{S}$.

1: $Y_{sat}, Y_{clause} = \text{SATformer}(\phi)$  
2: if $Y_{sat} == \text{UNSAT}$ then  
3:   for $j = 1 \to M$ do  
4:     for $i = 1 \to n$ do  
5:       if Variable $x_j$ is connected with Clause $C_i$ then  
6:         $v_j = v_j + y_i$  
7:       end if  
8:     end for  
9:   end for  
10: end if  
11: Initialize solver $\mathcal{S}$ with instance $\phi$: $\mathcal{S}.\text{input}(\phi)$  
12: Update the initial VSIDS in $\mathcal{S}$ with $v$: $\mathcal{S}.\text{updateScore}(v)$  
13: Perform solving of $\mathcal{S}$: $\mathcal{S}.\text{run}()$  
14: Return results: $\mathcal{S}.\text{results}$

---

## V. EXPERIMENTS

In this section, we conduct three experiments in three parts, with detailed experimental settings provided in Section V-A. Firstly, we demonstrate the ability of SATformer to act as a standalone satisfiability classifier by comparing it with another end-to-end SAT solver, NeuroSAT [22] (see Section V-B). Secondly, we investigate the effectiveness of our model design through a series of ablation studies (see Section V-C, V-D and V-E). Thirdly, we integrate SATformer with modern SAT solvers: CaDiCaL [8] and Kissat [9], which are the state-of-the-art solvers to the best of our knowledge. As shown in Section V-F, the hybrid SAT solvers are employed to prove the unsatisfiability of hard logic equivalence checking (LEC) instances. Section V-G discusses our current limitations and proposes potential solutions for future work.

### A. Experimental setting

We present the details of the experimental setting, including the construction of datasets and the hyperparameters of the SATformer model. In the following experiments, we adopt the same settings described here for all models unless otherwise specified.

**Evaluation metric**: Our SATformer predicts the satisfiability of a given instance. We construct both satisfiable and unsatisfiable instances in the testing dataset and record the accuracy of binary classification.

**Dataset preparation**: We generate $10K$ satisfiable instances and $10K$ unsatisfiable instances. Following the same dataset generation scheme as [22], a pair of random $k$-SAT satisfiable and unsatisfiable instances is generated together with only one different edge connection. Here, SR($m$) indicates that the instance contains $m$ variables. In our training dataset, we set the problem size as SR(3-10). Furthermore, we enumerate all possible clause subsets of each instance and label the minimal UNSAT cores. We also generate default testing datasets, each containing 50 satisfiable and 50 unsatisfiable instances in SR(3-10), SR(20), SR(40) and SR(60), respectively.

**Implementation Details**: In the GNN structure, we adopt the same configurations as NeuroSAT [22], except for reducing the number of message-passing iterations from 26 to 10. In the Hierarchical Transformer structure, we directly use clause embeddings as input tokens, resulting in the same hidden state dimension of 128. The window size $w$ is set to 4, and the total number of levels in the

<!-- Page 6 -->
# THE MODEL COMPLEXITY OF NEUROSAT AND SATFORMER

**TABLE I**

|             | NeuroSAT | SATformer |
|-------------|----------|-----------|
| # Param.    | 429.31 K | 732.47 K  |
| # FLOPs     | 207.91 M | 152.93 M  |

---

# PERFORMANCE COMPARISON OF NEUROSAT AND SATFORMER

**TABLE II**

| $CV > 5$ | NeuroSAT | SR(3-10) | SR(20) | SR(40) | SR(60) |
|----------|----------|----------|--------|--------|--------|
|          |          | 87%      | 61%    | 58%    | 50%    |
| SATformer|          | 94%      | 77%    | 68%    | 61%    |
| Impr.    |          | 7%       | 16%    | 10%    | 11%    |

| $CV = 4$ | NeuroSAT | SR(3-10) | SR(20) | SR(40) | SR(60) |
|----------|----------|----------|--------|--------|--------|
|          |          | 83%      | 59%    | 58%    | 50%    |
| SATformer|          | 99%      | 98%    | 91%    | 86%    |
| Impr.    |          | 16%      | 39%    | 33%    | 36%    |

| $CV = 3$ | NeuroSAT | SR(3-10) | SR(20) | SR(40) | SR(60) |
|----------|----------|----------|--------|--------|--------|
|          |          | 89%      | 50%    | 57%    | 50%    |
| SATformer|          | 99%      | 98%    | 98%    | 98%    |
| Impr.    |          | 10%      | 48%    | 48%    | 48%    |

---

hierarchical structure is 4. Inside each Transformer block, we set the number of heads in the multi-head self-attention mechanism to 8. The MLPs used to produce the two predictions for *Task-clause* and *Task-sat* are both configured as 3 layers. We train the model for 80 epochs with a batch size 16 on 4 Nvidia V100 GPUs. We adopt the Adam optimizer [39] with a learning rate $10^{-4}$ and weight decay $10^{-10}$.

## Model Complexity:

The complexity of NeuroSAT and above SATformer are illustrated in Table I, where # FLOPs is the number of floating-point operations and # Param. is the total number of trainable parameters. Although our model requires more trainable parameters to build up Transformer structures, SATformer performs faster (lower # FLOPs) than NeuroSAT due to the fewer message-passing iterations.

## B. Performance Comparison with NeuroSAT

We compare the performance of SATformer and NeuroSAT [22] on satisfiability classification. We do not consider other standalone SAT solvers [23], [24], [40] because they cannot produce only a single binary result.

Both NeuroSAT and our SATformer require the input instances to be in CNF. While SR($m$) represents the problem scale, we use another metric, $CV = \frac{n}{m}$, to better quantify the problem difficulty, where $n$ is the number of clauses and $m$ is the number of variables. A higher $CV$ value indicates more clause constraints in the formula, making the instance more difficult to solve. The instances from the training dataset and the default testing dataset have $CV > 5$. These satisfiable instances only have 1 or 2 possible satisfying assignments in total. Besides, we also produce 50 satisfiable and 50 unsatisfiable simplified instances with $CV = 3$ and $CV = 4$ for each problem scale. To ensure fairness, we train NeuroSAT on the same training dataset in SR(3-10). The results of the binary classification are listed in Table II.

From Table II, we have a few observations. Firstly, our SATformer achieves higher performance than NeuroSAT across all datasets. For example, while NeuroSAT can only correctly classify 67% in SR(20) and 56% in SR(40) when $CV > 5$, our SATformer has an accuracy of 70% in SR(20) and 61% in SR(40). Secondly, the results of NeuroSAT have no significant difference when the problem difficulty is reduced, but our SATformer performs better on the simplified instances. For example, SATformer solves all instances with $CV = 3$ regardless of the problem scale. The reason for this is that SATformer relies on the correlation among clauses to solve SAT problems. With fewer clauses, SATformer harnesses such correlations more easily. In contrast, NeuroSAT only learns the instance-level features with single-bit supervision and does not obtain richer representations with the reduction of problem difficulty.

To conclude, our SATformer outperforms the NeuroSAT. Additionally, our SATformer shows a similar property to traditional SAT solvers, i.e., it can achieve higher performance on instances with lower difficulty.

## C. Effectiveness of Transformer Blocks

This section investigates the effectiveness of the Transformer Blocks in SATformer. On the one hand, we compare our SATformer with a derived model framework called SATformer-MLP, which replaces Transformer blocks with MLPs of the equivalent number of model parameters (# Param.). Table III shows the experimental results. Since the SATformer-MLP treats all clauses equally instead of selectively enhancing the correlation between some clauses, it shows an inferior performance compared to the original SATformer.

On the other hand, as the Transformer distinctively treats token pairs based on the self-attention mechanism, we explore the attention weights assigned to different token pairs. Based on the Task-clause UNSAT core prediction, we mostly divide the clauses into two categories: likely to form an UNSAT Core (denoted as C-clause) and Unlikely to form an UNSAT core (denoted as U-clause) cause unsatisfiable. Therefore, the pairwise connections are classified into 4 types: between C-clause and C-clause (CC), C-clause and U-clause (CU), U-clause and C-clause (UC), U-clause and U-clause (UU). We calculate the overall attention weights for these 4 types, respectively, and list the percentages in Table IV. A higher percentage indicates that the model pays more attention to the corresponding pairwise correlation. The Transformer assigns almost 70% of its attention weights to the CC connection. Moreover, along with the CU and UC connections, the model allocates 97.23% of its attention weights to capture the correlation related to C-clauses. Therefore, our Transformer-based model mainly focuses on the clauses that contribute more to unsatisfiability.

To summarize, our SATformer not only captures the relationship among clauses but also learns how to pay more attention to those clauses that are more likely to raise unsatisfiability.

## D. Effectiveness of Hierarchical Transformer

In this subsection, we conduct an ablation study and create some special cases to demonstrate the effectiveness of our Hierarchical Transformer.

---

# PERFORMANCE COMPARISON OF SATFORMER AND SATFORMER-MLP

**TABLE III**

|            | SR(3-10) | SR(20) | SR(40) | SR(60) |
|------------|----------|--------|--------|--------|
| SATformer  | 94%      | 77%    | 68%    | 61%    |
| SATformer-MLP | 90%   | 72%    | 60%    | 52%    |

---

# THE ATTENTION WEIGHT PERCENTAGES OF FOUR TYPE CONNECTIONS

**TABLE IV**

| CC     | CU     | UC     | UU     |
|--------|--------|--------|--------|
| 68.27% | 14.95% | 14.01% | 2.77%  |

<!-- Page 7 -->
TABLE V  
PERFORMANCE COMPARISON OF SATFormer AND SATFormer-NoHier

|          | SR(3-10) | SR(20) | SR(40) | SR(60) |
|----------|----------|--------|--------|--------|
| SATFormer | 94%      | 77%    | 68%    | 61%    |
| SATFormer-NoHier | 70%      | 57%    | 50%    | 50%    |

TABLE VI  
PERFORMANCE COMPARISON OF SATFormer AND SATFormer-NoCore

|          | SR(3-10) | SR(20) | SR(40) | SR(60) |
|----------|----------|--------|--------|--------|
| SATFormer | 94%      | 77%    | 68%    | 61%    |
| SATFormer-NoCore | 89%      | 67%    | 57%    | 51%    |

Fig. 4. Two special unsatisfiable cases in hierarchical structure

Firstly, we replace the hierarchical structure with plain Transformer blocks. This model takes clause embeddings as input tokens and updates them with 4 (same as $L=4$ in default SATFormer configuration) stacked Transformer blocks (denoted as *SATFormer-NoHier*). Then, the model produces an embedding vector by pooling all updated clause embeddings and predicts the final binary result. As shown in Table V, SATFormer-NoHier can handle fewer instances than SATFormer. For example, SATFormer-NoHier only achieves 70% accuracy in SR(3-10) and performs as a randomly guessing classifier in SR(40) and SR(60) instances. On the contrary, SATFormer is still effective in SR(40) and SR(60). The reason for this is that SATFormer-NoHier can only learn pairwise dependencies, while a pair of clauses may not have a decisive effect on the satisfiability. Although the model can learn multiple relationships by stacking Transformer blocks, 4 layers are still far from sufficient.

Secondly, to further demonstrate the ability of SATFormer to learn multiple relationships, we construct two equivalent unsatisfiable instances as Eq. (14) based on Eq. (2) with different clause orders. The three clauses $C_2$, $C_5$, and $C_6$ forming MUC are highlighted with underlines.

$$
\begin{aligned}
\phi_{U1} &:= C_1 \land C_2 \land C_3 \land C_4 \land C_5 \land C_6 \land C_7 \land C_8 \land C_9 \\
\phi_{U2} &:= C_1 \land C_2 \land C_3 \land C_4 \land C_5 \land C_9 \land C_7 \land C_8 \land C_6
\end{aligned}
\quad (14)
$$

As shown in Fig. 4, for the instance $\phi_{U1}$, the clauses (from $C_1$ to $C_6$) in the first two groups can construct an unsatisfiable sub-instance. Hence, by modeling the pairwise relationship between input token embeddings $G_1^1$ and $G_2^1$ in level $l=1$, the model can determine the satisfiability. However, there is no pairwise correlation leading to unsatisfiability in the first hierarchical level in $\phi_{U2}$. We train two additional SATFormer models with a window size of $w=3$, but with different hierarchical levels: one with $L=1$ and the other with $L=2$. According to the experimental results, the model with $L=1$ which relies on the group embeddings in level $l=1$ can only correctly predict $\phi_{U1}$ as unsatisfiable. However, if we add one more Transformer block in level $l=2$, the model (with $L=2$) can learn the group embedding $G_1^2$, which contains information from these 3 clause groups. In this case, the SATFormer model can correctly predict both instances as unsatisfiable.

In summary, our SATFormer can learn the relationships among multiple tokens due to the pairwise self-attention mechanism and hierarchical structure. Besides, to mitigate the effect of clause ordering, we shuffle the clauses every 10 training iterations.

## E. Effectiveness of Multi-task Learning Strategy

To investigate the effectiveness of the MTL strategy in our SATFormer, we train another model without UNSAT core supervision (denote as *SATFormer-NoCore*) and compare it with the original SATFormer. Table VI shows the experimental results. After removing the Task-clause UNSAT core prediction, the model’s performance on all four datasets decreases.

Generally, UNSAT core prediction retains rich information about the contribution of clauses to unsatisfiability, and it is significantly related to SAT solving. Therefore, the MTL strategy incorporating UNSAT core prediction benefits the model’s performance.

## F. Combination with SAT Solvers

In this subsection, we combine SATFormer with CaDiCaL [8] and Kissat [9] to demonstrate its effectiveness. The solvers integrated with SATFormer are denoted as w/ SATFormer, while the baseline solvers without SATFormer are denoted as w/o SATFormer. To ensure fairness, the runtime of the w/ SATFormer solver includes the model inference time. We test these solvers with 60 industrial LEC instances, comprising of 30 SAT instances and 30 UNSAT instances.

Fig. 5 shows the runtime reduction achieved by SATFormer on different solvers, where the red points represent SAT instances and the blue points represent UNSAT instances. To ensure better visualization, we ignore some cases with extremely large runtime increase or decrease. Both CaDiCaL and Kissat can achieve runtime reduction by integrating SATFormer in most cases. However, there are still a few cases (especially in satisfiable instances) indicating a performance degradation when combining the solvers with SATFormer.

To further elucidate the above phenomenon and evaluate the effectiveness of our SATFormer, we calculate the average runtime for UNSAT instances, SAT instances and all instances, respectively, as shown in Table VII. The table reveals several key observations. Firstly, compared to solvers w/o SATFormer, CaDiCaL and Kissat w/ SATFormer achieve a speedup of 21.64% and 7.60%, respectively, when solving UNSAT instances. Therefore, SATFormer can improve the efficiency of UNSAT solving. Secondly, SATFormer results in a runtime increase for both CaDiCaL and Kissat solvers. Taking CaDiCaL as an example, although w/ SATFormer requires 30.77% more runtime than w/o SATFormer, the initialization heuristic does not have a negative impact on the solver. The gap in solving time between w/ SATFormer (6.62s on average) and w/o SATFormer (6.63s on average) is quite small. Since the satisfiable instances in EDA are naturally easy to solve, the efficiency improvement brought by the heuristic is not significant. Thirdly, the inference time of SATFormer accounts for only a small fraction of the overall solving

<!-- Page 8 -->
TABLE VII  
AVERAGE RUNTIME COMPARISON BETWEEN THE SOLVERS W/O AND W/ SATFORMER

| Size   | Model(s) | CaDiCaL       |               |               | Kissat        |               |               |
|--------|----------|---------------|---------------|---------------|---------------|---------------|---------------|
|        |          | w/o (s)       | w/ (s)        | Overall (s)   | w/o (s)       | w/ (s)        | Overall (s)   |
| UNSAT  | 20,114   | 2.30          | 967.00        | 755.43        | 757.72        | 787.73        | 723.67        |
| SAT    | 20,102   | 2.05          | 6.63          | 6.62          | 8.67          | 3.79          | 3.78          |
| All    | 20,112   | 1.94          | 486.81        | 381.02        | 382.96        | 395.76        | 363.73        |
|        |          |               |               |               | Reduction     |               | Reduction     |
|        |          |               |               |               | 21.64%        |               | 7.84%         |
|        |          |               |               |               | -30.77%       |               | -53.77%       |
|        |          |               |               |               | 21.33%        |               | 7.60%         |

---

TABLE VIII  
SEARCHING PROCESS COMPARISON BETWEEN THE CADICAL W/O AND W/ SATFORMER

| Instances | Core Size | # Lemma  | Time (s) | Core Size | # Lemma  | Time (s) | # Lemma Reduction | Time Reduction |
|-----------|-----------|----------|----------|-----------|----------|----------|-------------------|----------------|
| C1        | 1343      | 601331   | 18.91    | 1341      | 557498   | 17.01    | 7.29%             | 10.06%         |
| C2        | 2755      | 935965   | 28.13    | 2753      | 663373   | 23.05    | 29.12%            | 18.07%         |
| C3        | 1836      | 677794   | 20.03    | 1834      | 782211   | 21.75    | -15.41%           | -8.59%         |
| C4        | 2744      | 717451   | 21.33    | 2739      | 852268   | 23.28    | -18.79%           | -9.12%         |

---

Fig. 5. The runtime reduction of SATformer on CaDiCaL [8] (upper) and Kissat [9] (below)

time, with 0.51% in CaDiCaL and 0.53% in Kissat, respectively. Moreover, SATformer is able to infer within polynomial time that is only proportional to the size of instances. Thus, the overhead of SATformer is within an acceptable range. In summary, our SATformer can significantly improve the efficiency of UNSAT solving without harming SAT solving.

## G. Failure Case Analysis and Discussion

In this subsection, we analyze the limitations of the current SATformer and propose potential solutions to further improve SATformer in the future. On the one hand, as shown in Table VII, SATformer cannot improve the efficiency of solving satisfiable problems. This is because the heuristic objective of finding a satisfying assignment is different from the objective of proving unsatisfiability [41], and the prediction of the UNSAT core is meaningless for satisfiable instances. To improve the generalization ability of SATformer, we can design a heuristic based on decoding assignments [22], especially for satisfiable instances. As a result, our next version of SATformer can predict satisfiability and accelerate the solving process for both SAT and UNSAT instances, allowing us to select the appropriate heuristic based on the satisfiability of the instance.

On the other hand, as shown in Fig. 5, we can find that integrating SATformer may reduce the efficiency of solving UNSAT instances in several cases. We investigate the reason for these failure cases with four representative instances. Table VIII presents the size of the UNSAT core (Core Size), solving time (Time) and the number of generated lemmas (# Lemma) using CaDiCaL solvers w/o and w/ SATformer. Each lemma is generated by conflict analysis in the search process. The number of lemmas has a positive correlation with the number of conflicts encountered during solving and the number of times the search backtracks. According to the results in Table VIII, we observe that for C1 and C2, SATformer guides the search towards a smaller UNSAT core and reduce the number of lemmas by 7.29% and 29.12%, respectively, leading to a runtime reduction of 10.06% and 18.07%, respectively. However, for C3 and C4, although SATformer tends to find a smaller UNSAT core, more backtracking is required during the search, which consumes more runtime. Since the search time is positively correlated with the size of the UNSAT core, rather than being absolutely proportional to the size of the UNSAT core [42], SATformer could derive a small but hard-to-prove UNSAT core. Consequently, to further improve the performance of SATformer in the future, the SATformer should guide the search path towards the easiest UNSAT core rather than the minimal UNSAT core.

## VI. CONCLUSION

This paper proposes SATformer, a novel Transformer-based framework for predicting satisfiability and accelerating the solving process for UNSAT problems. The focus of our work is on UNSAT core learning, which involves identifying the existence of an unsatisfiable core by modeling clause interactions. Specifically, SATformer maps clauses into embeddings through a graph neural network (GNN) and captures clause correlations using a hierarchical Transformer-based model. By leveraging both single-bit satisfiability and the unsatisfiable core as supervisions, SATformer can effectively predict the satisfiability of an instance and quantify each clause’s contribution to unsatisfiability. Our results demonstrate that SATformer outperforms NeuroSAT as an end-to-end learning-based satisfiability classifier. Furthermore, we integrate SATformer as an initialization heuristic into modern SAT solvers. Our experimental results show that SATformer can reduce the solving time by an average of 21.33% for the CaDiCaL solver and 7.60% for the Kissat solver on the logic equivalence checking task.

<!-- Page 9 -->
# ACKNOWLEDGMENTS

This work was supported in part by the General Research Fund of the Hong Kong Research Grants Council (RGC) under Grant No. 14212422 and in part by Research Matching Grant CSE-7-2022.

# REFERENCES

[1] E. I. Goldberg, M. R. Prasad, and R. K. Brayton, “Using sat for combinational equivalence checking,” in *Proceedings Design, Automation and Test in Europe. Conference and Exhibition 2001*. IEEE, 2001, pp. 114–121.

[2] K. L. McMillan, “Interpolation and sat-based model checking,” in *Computer Aided Verification: 15th International Conference, CAV 2003, Boulder, CO, USA, July 8-12, 2003. Proceedings 15*. Springer, 2003, pp. 1–13.

[3] K. Yang, K.-T. Cheng, and L.-C. Wang, “Trangen: A sat-based atpg for path-oriented transition faults,” in *ASP-DAC 2004: Asia and South Pacific Design Automation Conference 2004 (IEEE Cat. No. 04EX753)*. IEEE, 2004, pp. 92–97.

[4] N. Sorensson and N. Een, “Minisat v1. 13-a sat solver with conflict-clause minimization,” *SAT*, vol. 2005, no. 53, pp. 1–2, 2005.

[5] A. Biere, A. Fazekas, M. Fleury, and M. Heisinger, “CaDiCaL, Kissat, Paracooba, Plingeling and Treengeling entering the SAT Competition 2020,” in *Proc. of SAT Competition 2020 – Solver and Benchmark Descriptions*, ser. Department of Computer Science Report Series B, T. Balyo, N. Froleyks, M. Heule, M. Iser, M. Järvisalo, and M. Suda, Eds., vol. B-2020-1. University of Helsinki, 2020, pp. 51–53.

[6] B. Selman, H. A. Kautz, B. Cohen *et al.*, “Local search strategies for satisfiability testing,” *Cliques, coloring, and satisfiability*, vol. 26, pp. 521–532, 1993.

[7] M. Osana and A. Wijs, “Parafrost at the sat race 2021,” *SAT COMPETITION 2021*, vol. 32, 2021.

[8] S. D. QUEUE, “Cadical at the sat race 2019,” *SAT RACE 2019*, p. 8, 2019.

[9] A. Fleury and M. Heisinger, “Cadical, kissat, paracooba, plingeling and treengeling entering the sat competition 2020,” *SAT COMPETITION*, vol. 2020, p. 50, 2020.

[10] G. Audemard and L. Simon, “On the glucose sat solver,” *International Journal on Artificial Intelligence Tools*, vol. 27, no. 01, p. 1840001, 2018.

[11] F. Lu, L.-C. Wang, K.-T. Cheng, and R.-Y. Huang, “A circuit sat solver with signal correlation guided learning,” in *2003 Design, Automation and Test in Europe Conference and Exhibition*. IEEE, 2003, pp. 892–897.

[12] G. Audemard and L. Simon, “Glucose: a solver that predicts learnt clauses quality,” *SAT Competition*, pp. 7–8, 2009.

[13] J. Huang, H.-L. Zhen, H. Wang, M. Mao, Y. Yuan, and Y. Huang, “Neural fault analysis for sat-based atpg,” in *2022 IEEE International Test Conference (ITC)*. IEEE, 2022, pp. 36–45.

[14] M. Gagliolo and J. Schmidhuber, “Algorithm selection as a bandit problem with unbounded losses,” in *Learning and Intelligent Optimization: 4th International Conference, LION 4, Venice, Italy, January 18-22, 2010. Selected Papers 4*. Springer, 2010, pp. 82–96.

[15] E. Goldberg and Y. Novikov, “Verification of proofs of unsatisfiability for cnf formulas,” in *2003 Design, Automation and Test in Europe Conference and Exhibition*. IEEE, 2003, pp. 886–891.

[16] A. Mishchenko, S. Chatterjee, R. Brayton, and N. Een, “Improvements to combinational equivalence checking,” in *Proceedings of the 2006 IEEE/ACM international conference on Computer-aided design*, 2006, pp. 836–843.

[17] H.-C. Liang, C. L. Lee, and J. E. Chen, “Identifying untestable faults in sequential circuits,” *IEEE Design & Test of computers*, vol. 12, no. 03, pp. 14–23, 1995.

[18] K. Heragu, J. H. Patel, and V. D. Agrawal, “Fast identification of untestable delay faults using implications,” in *iccad*, 1997, pp. 642–647.

[19] D. Selsam and N. Bjørner, “Guiding high-performance sat solvers with unsat-core predictions,” in *International Conference on Theory and Applications of Satisfiability Testing*. Springer, 2019, pp. 336–353.

[20] M. W. Moskewicz, C. F. Madigan, Y. Zhao, L. Zhang, and S. Malik, “Chaff: Engineering an efficient sat solver,” in *Proceedings of the 38th annual Design Automation Conference*, 2001, pp. 530–535.

[21] W. Guo, J. Yan, H.-L. Zhen, X. Li, M. Yuan, and Y. Jin, “Machine learning methods in solving the boolean satisfiability problem,” *arXiv preprint arXiv:2203.04755*, 2022.

[22] D. Selsam, M. Lamm, B. Bünz, P. Liang, L. de Moura, and D. L. Dill, “Learning a sat solver from single-bit supervision,” *arXiv preprint arXiv:1802.03685*, 2018.

[23] S. Amizadeh, S. Matusevych, and M. Weimer, “Learning to solve circuit-sat: An unsupervised differentiable approach,” in *International Conference on Learning Representations*, 2018.

[24] M. Li, Z. Shi, Q. Lai, S. Khan, and Q. Xu, “Deepsat: An eda-driven learning framework for sat,” *arXiv preprint arXiv:2205.13745*, 2022.

[25] W. Zhang, Z. Sun, Q. Zhu, G. Li, S. Cai, Y. Xiong, and L. Zhang, “Nlocalsat: Boosting local search with solution prediction,” *arXiv preprint arXiv:2001.09398*, 2020.

[26] V. Kurin, S. Godil, S. Whiteson, and B. Catanzaro, “Can q-learning with graph networks learn a generalizable branching heuristic for a sat solver?” *Advances in Neural Information Processing Systems*, vol. 33, pp. 9608–9621, 2020.

[27] W. Wang, Y. Hu, M. Tiwari, S. Khurshid, K. McMillan, and R. Miikkulainen, “Neurocomb: Improving sat solving with graph neural networks,” *arXiv e-prints*, p. arXiv–2110, 2021.

[28] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” *Advances in neural information processing systems*, vol. 30, 2017.

[29] C. Meng, M. Chen, J. Mao, and J. Neville, “Readnet: A hierarchical transformer framework for web article readability analysis,” in *European Conference on Information Retrieval*. Springer, 2020, pp. 33–49.

[30] Q. Wang, B. Li, T. Xiao, J. Zhu, C. Li, D. F. Wong, and L. S. Chao, “Learning deep transformer models for machine translation,” *arXiv preprint arXiv:1906.01787*, 2019.

[31] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly *et al.*, “An image is worth 16x16 words: Transformers for image recognition at scale,” *arXiv preprint arXiv:2010.11929*, 2020.

[32] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021, pp. 10012–10022.

[33] F. Shi, C. Lee, M. K. Bashar, N. Shukla, S.-C. Zhu, and V. Narayanan, “Transformer-based machine learning for fast sat solvers and logic synthesis,” *arXiv preprint arXiv:2107.07116*, 2021.

[34] G. S. Tseitin, “On the complexity of derivation in propositional calculus,” in *Automation of reasoning*. Springer, 1983, pp. 466–483.

[35] R. Caruana, “Multitask learning,” *Machine learning*, vol. 28, no. 1, pp. 41–75, 1997.

[36] S. Kullback and R. A. Leibler, “On information and sufficiency,” *The annals of mathematical statistics*, vol. 22, no. 1, pp. 79–86, 1951.

[37] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature pyramid networks for object detection,” in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2017, pp. 2117–2125.

[38] J. Marques-Silva, I. Lynce, and S. Malik, “Conflict-driven clause learning sat solvers,” in *Handbook of satisfiability*. ios Press, 2021, pp. 133–182.

[39] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” *arXiv preprint arXiv:1412.6980*, 2014.

[40] E. Ozolins, K. Freivalds, A. Draguns, E. Gaile, R. Zakovskis, and S. Kozlovs, “Goal-aware neural sat solver,” *arXiv preprint arXiv:2106.07162*, 2021.

[41] C. Oh, “Between sat and unsat: the fundamental difference in cdcl sat,” in *Theory and Applications of Satisfiability Testing–SAT 2015: 18th International Conference, Austin, TX, USA, September 24-27, 2015, Proceedings 18*. Springer, 2015, pp. 307–323.

[42] A. Nadel, “Boosting minimal unsatisfiable core extraction,” in *Formal Methods in Computer Aided Design*. IEEE, 2010, pp. 221–229.