<!-- Page 1 -->
# Graph Neural Reasoning May Fail in Certifying Boolean Unsatisfiability

**Ziliang Chen**$^*$  
Department of Computer Science  
Sun Yat-sen University  
GuangZhou, Guangdong, China  
c.ziliang@yahoo.com

**Zhanfu Yang**$^*$  
Department of Computer Science  
Purdue University  
West Lafayette, IN, USA  
yang1676@purdue.edu

## Abstract

It is feasible and practically-valuable to bridge the characteristics between graph neural networks (GNNs) and logical reasoning. Despite considerable efforts and successes witnessed to solve Boolean satisfiability (SAT), it remains a mystery of GNN-based solvers for more complex *predicate logic* formulae. In this work, we conjecture with some evidences, that generally-defined GNNs present several limitations to certify the unsatisfiability (UNSAT) in Boolean formulae. It implies that GNNs may probably fail in learning the logical reasoning tasks if they contain proving UNSAT as the sub-problem included by most predicate logic formulae.

## 1 Introduction

Logical reasoning problems span from simple propositional logic to complex predicate logic and high-order logic, with known theoretical complexities from NP-completeness [3] to semi-decidable and undecidable [2]. Testing the ability and limitation of machine learning tools on logical reasoning problems leads to a fundamental understanding of the boundary of learnability and robust AI, helping to address interesting questions in decision procedures in logic, program analysis, and verification as defined in the programming language community.

There have been arrays of successes in learning propositional logic reasoning [1, 12], which focus on Boolean satisfiability (SAT) problems as defined below. A Boolean logic formula is an expression composed of Boolean constants ($\top$: true, $\bot$: false), Boolean variables ($x_i$), and propositional connectives such as $\land$, $\lor$, $\neg$ (for example $(x_1 \lor \neg x_2) \land (\neg x_1 \lor x_2)$). The SAT problem asks if a given Boolean formula can be satisfied (evaluated to $\top$) by assigning proper Boolean values to the literal variables. A crucial feature of the logical reasoning domain (as is visible in the SAT problem) is that the inputs are often structural, where logical connections between entities (variables in SAT problems) are the key information.

SAT and its variant problems are almost NP-complete or even more complicated in the complexity. The fact motivates the emergence of sub-optimal heuristic that trades off the solver performance to rapid reasoning. In terms of the fast inference process, deep learning models are favored as learnable heuristic solvers [1, 12, 16]. Among them Graph Neural Networks (GNNs) have grasped amount of attentions, since the message-passing process delivers the transparency to interpret the inference within GNNs, thus, revealing the black box behind neural logical reasoning in the failure instances.

However, it should be noticed that logical decision procedures is more complex that just reading the formulas correctly. It is unclear if GNN embeddings (from simple message-passing) contain all the information needed to reason about complex logical questions on top of the graph structures

---

$^*$indicates alphabetic order.

Preprint. Under review.

<!-- Page 2 -->
derived from the formulas, or whether the complex embedding schemes can be learned from back-propagation. Previous successes on SAT problems argued for the power of GNN, which can handle NP-complete problems [1, 12], whereas no evidences have been reported for solving semi-decidable predicate logic problems via GNN. The significant difficulty to prove the problems is the requirement of comprehensive reasoning over a search space, since a complete proof includes SAT and UNSAT (i.e., Boolean unsatisfiability).

Perhaps disappointingly, this work presents some theoretical evidences that support a pessimistic conjecture: GNNs do not simulate the complete solver for UNSAT. Specifically, we discover that the neural reasoning procedure learned by GNNs does simulate the algorithms that may allow a CNF formula changing over iterations. Those complete SAT-solvers, e.g., DPLL and CDCL, are almost common in the operation that adaptively alters the original Boolean formula that eases the reasoning process. So GNNs do not learn to simulate their behaviors. Instead, we prove that by appropriately defining a specific structure of GNN that a parametrized GNN may learn, the local search heuristic in WalkSAT can be simulated by GNN. Towards these results, we believe that GNN can not solve UNSAT in existing logical reasoning problems.

## 2 Embedding Logic Formulae by GNNs

**Preliminary: Graph Neural Networks (GNNs).** GNNs refer to the neural architectures devised to learn the embeddings of nodes and graphs via message-passing. Resembling the generic definition in [14], they consist of two successive operators to propagate the messages and evolve the embeddings over iterations:

$$
m_v^{(k)} = \text{Aggregate}^{(k)}\left(\left\{h_u^{(k-1)} : u \in \mathcal{N}(v)\right\}\right), \quad h_v^{(k)} = \text{Combine}^{(k)}\left(h_v^{(k-1)}, m_v^{(k)}\right)
\tag{1}
$$

where $h_v^{(k)}$ denotes the hidden state (embedding) of node $v$ in the $k^{th}$ iteration, and $\mathcal{N}(v)$ denotes the neighbors of node $v$. In each iteration, the $\text{Aggregate}^{(k)}(\cdot)$ aggregates hidden states from node $v$’s neighbors $\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}$ to produce the new message (i.e., $m_v^{(k)}$) for node $v$; $\text{Combine}^{(k)}(\cdot, \cdot)$ updates the embedding of $v$ in terms of its previous state and its current message. After a specific number of iterations (e.g., $K$ in our discussion), the embeddings should capture the global relational information of the nodes, which can be fed into other neural network modules for specific tasks.

Significant successes about GNNs have been witnessed in relational reasoning [6, 17, 20], where an instance could be departed into multiple objects then encoded by a series of features with their relation. It typically suits representation in Eq. 1. Whereas in logical reasoning, a Boolean formula is in Conjunctive Normal Form (CNF) that consists of literal and clause items. In term of the independence among literals in CNF (so do clauses), [12] embeds a formula into a *bipartite graph*, where the nodes denote the clauses and literals that are disjoint, respectively. In this principle, given a literal $v$ as a node, all the nodes of clauses that contains the literal are routinely treated as $v$’s neighbors, vice versa for the node of each clause. We assume $\Phi$ is a logic formula in CNF, i.e., a set of clauses, and $\Psi(v) \in \Phi$ denote one of clauses within the logic formula $\Phi$ that contains literal $v$. Derived from Eq. 1, GNNs for logical reasoning can be further specified by

$$
\begin{aligned}
m_v^{(k)} &= \text{Aggregate}_L^{(k)}\left(\left\{h_{\Psi(v)}^{(k-1)} : \Psi(v) \in \Phi\right\}\right), & h_v^{(k)} &= \text{Combine}_L^{(k)}\left(h_v^{(k-1)}, h_{\neg v}^{(k-1)}, m_v^{(k)}\right), & \text{s.t. } \forall v \in L \\
m_{\Psi(v)}^{(k)} &= \text{Aggregate}_C^{(k)}\left(\left\{h_u^{(k-1)} : u \in \Psi(v)\right\}\right), & h_{\Psi(v)}^{(k)} &= \text{Combine}_C^{(k)}\left(h_{\Psi(v)}^{(k-1)}, m_{\Psi(v)}^{(k)}\right), & \text{s.t. } \forall \Psi(v) \in \Phi
\end{aligned}
\tag{2}
$$

where $h_v^{(k)}$ and $h_{\Psi(v)}^{(k)}$ denote embeddings of the literal $v$ and the clause $\Psi(v)$ in the $k^{th}$ iteration ($h_{\neg v}^{(k)}$ denotes the embedding of the negation of $v$); $m_v^{(k)}$ and $m_{\Psi(v)}^{(k)}$ refer to their propagated messages. Since the value of a Boolean formula is determined by the value assignment of the literal variables, Eq. 2 solely requires the final-state literal embeddings $\{h_u^{(K)}, u \in L\}$ to predict the logical reasoning result. More specifically, we use $L$ and $C$ to denote a literal set and a clause set ($L$ and $C$ may be different for each CNF formula), then $\Psi(v)$ is a clause and $\Psi(v)$ denotes a clause including the literal $v \in L$.

Note that the graph embeddings for SAT [7] and 2QBF [7] are generally represented by Eq.2. Hence our further analysis is based on Eq.2.

<!-- Page 3 -->
# 3 Certifying UNSAT by GNNs may Fail

Although existing researches showed that GNN can learn a well-performed solver for satisfiability problems, GNN-based SAT solvers actually have terrible performances in predicting unsatisfiability with high confidence [12] in a SAT formula, if the formula does not have a small unsatisfiable core (minimal number of clauses that is enough to cause unsatisfiability). In fact, some previous work [1] even completely removed unsatisfiable formulas from the training dataset, since they slowed down the whole training process.

The difficulty in proving unsatisfiability is understandable, since constructing a proof of unsatisfiability demands a complete reasoning in the search space, which is more complex than constructing a proof of satisfiability that only requires a witness. Traditionally it relies on the recursive decision procedures that either traverse all possible assignments to construct the proof (DPLL [4]), or generate extra constraints from assignment trials that lead to conflicts, until some of the constraints contradict each other (CDCL [13]). The line of recursive algorithms include some operation branches that reconfigure the bipartite graph behind the CNF in each step while they search. In the terms of a graph that may iteratively change (e.g., DPLL), perhaps miserably, their recursive processes can not be simulated by GNNs.

**Observation 3.1.** Given a recursive algorithm that iteratively reconfigures the graph, GNNs in Eq.2 can not simulate this recursive process.

*Proof.* Associating the aggregate and combine functions in Eq. 2, we obtain the iterative update rule for the embedding of a literal $v$:

$$
\begin{aligned}
h_v^{(k)} &= \text{Combine}_L^{(k)}\left(h_v^{(k-1)}, h_{\neg v}^{(k-1)}, \text{Aggregate}_L^{(k)}\left(\{h_{\Psi(v)}^{(k-1)} : \Psi(v) \in \Phi\}\right)\right) \\
&= \text{Update}_L^{(k)}\left(h_v^{(k-1)}, h_{\neg v}^{(k-1)}, \{h_{\Psi(v)}^{(k-1)} : \Psi(v) \in \Phi\}\right), \quad \quad s.t.\ v \in L
\end{aligned}
\tag{3}
$$

Towards this principle, we observe that the embedding update of $v$ in the current stage relies on the last-stage embeddings of $v$ and its negation $\neg v$, and the embeddings of all the clauses that include $v$ in a CNF formula ($\Psi(v) \in \Phi$). The literal $v$, $\neg v$ and the clauses containing $v$ are consistent over iterations. Hence if the update function (Eq. 3) is consistent over the iterations in Eq.2, i.e., $\forall k \in \mathbb{N}_+$, $\text{Update}_L^{(k)} = \text{Update}_L$, where $\text{Update}_L$ means the update for literal embedding, GNNs derived from Eq. 3 receive a fixed graph generated by a CNF formula as input. However, if a recursive algorithm iteratively changes the graph that represents a CNF formula, it implies that there must be a clause that was changed (or eliminated) after this iteration, since clauses are permutation-invariant in a CNF formula. Accordingly there must be a literal embedding whose update process depends on a clause different from the previous iteration. It contradicts the literal embedding update function learned by Eq. 3 with $\forall k \in \mathbb{N}_+$, $\text{Update}_L^{(k)} = \text{Update}_L$. $\square$

Hence the message-passing in GNNs could not resemble the procedures in the complete SAT-solvers. In fact, GNNs are rather similar to learning a subfamily of incomplete SAT solvers (GSAT, WalkSAT [11]), which randomly assign variables and stochastically search for local witnesses.

**Observation 3.2.** GNNs in Eq. 2 may simulate the local search in WalkSAT.

*Proof.* Recall the iterative update routine of WalkSAT: starting by assigning a random value to each literal variable in a formula, it randomly chooses an unsatisfied clause in the formula and flips the value of a Boolean variable within that clause. Such process is repeated till the literal assignment satisfies all clauses in the formula. Here we construct the optimal aggregation and combine functions derived from Eq. 2, which are designed to simulate the procedure of WalkSAT. In this way, if the aggregation and combine functions in Eq. 2 approximate these optimal aggregation and combine functions, the GNN may simulate the local search in WalkSAT.

Given a universe of literals in logical reasoning, we first initiate the embeddings of them and their negation, thus, $\forall v \in L$, random value of $h_v^{(0)}$ and $h_{\neg v}^{(0)}$ are initiated. This assignment can be treated as the Boolean value that belong to different literals, which have been mapped from a binary vector into a real-value embedding space about the literals. We also randomly initiate the clause embeddings $h_{\Psi(v)}^{(0)}$ for reasoning each formula that contains the clause $\Psi(v)$. Here we define the optimal

<!-- Page 4 -->
aggregation and combine functions that encode literals and clauses respectively, which GNNs in Eq. 2 may learn if they attempt to simulate WalkSAT:

$$
m_v^{(k)} = \overline{\text{Aggregate}}_L\left(\{h_{\Psi(v)}^{(k-1)} : \Psi(v) \in \Phi\}\right),
$$

$$
= \begin{cases}
\boldsymbol{\epsilon}^{(k)}, & \prod_{\Psi(v)} \|h_{\Psi(v)}^{(k-1)}\| = 0 \\
\mathbf{0}, & \prod_{\Psi(v)} \|h_{\Psi(v)}^{(k-1)}\| \neq 0
\end{cases}
\quad s.t.\ \forall v \in L
\tag{4}
$$

where $\overline{\text{Aggregate}}_L(\cdot)$ denotes the optimal aggregation function to propagate literal messages and $m_v^{(k)}$ denotes the optimally propagated message of literal $v$ in the $k$ iteration; $\mathbf{0}$ is a zero-value vector; $\boldsymbol{\epsilon}^{(k)}$ denotes a bounded non-zero random vector generated in the $k$ iteration; $\|\cdot\|$ indicates a vector norm.

$$
h_v^{(k)} = \overline{\text{Combine}}_L\left(h_v^{(k-1)}, h_{\neg v}^{(k-1)}, m_v^{(k)}\right)
$$

$$
= \begin{cases}
h_{\neg v}^{(k-1)}, & v = \arg\max_{\forall u \in L} \{\|m_u^{(k)}\|\} \text{ and } \|m_v^{(k)}\| > 0 \\
h_v^{(k-1)}, & \text{otherwise}
\end{cases}
\quad s.t.\ \forall v \in L
\tag{5}
$$

where $\overline{\text{Combine}}_L(\cdot)$ denotes the optimal combine function that iteratively updates literal embeddings by the aid of the optimal message. Eq. 5 implies the local Boolean variable flipping in WalkSAT: if the norm of $m_v^{(k)}$ is the maximum among all the optimal literal messages, its literal embedding would be replaced by the embedding of its negation, otherwise, keep the identical value. The maximization ensures only one literal embedding that would be “flipped” per iteration, which simulates the local search behavior. Besides, the literal embedding selected for update would not be $\mathbf{0}$, which implies all the clauses containing this literal are satisfied (see the condition 2 in Eq. 4). Since all the satisfied clauses would not be selected in WalkSAT, this literal also would not be selected to update in this iteration. Finally, if a literal has been included by a clause that is unsatisfied, it would be randomly picked in some probability. The uncertainty is implied by the randomness of $\boldsymbol{\epsilon}^{(k)}$.

$$
m_{\Psi(v)}^{(k)} = \overline{\text{Aggregate}}_C\left(\{h_u^{(k-1)} : u \in \Psi(v)\}\right)
$$

$$
= \begin{cases}
h_{\Psi(v)}^{(0)}, & \text{Sigmoid}\left(\text{MLP}_2^*\left(\sum_{u \in \Psi(v)} \text{MLP}_1^*(h_u^{(k-1)})\right)\right) \geq 0.5 \\
\mathbf{0}, & \text{Sigmoid}\left(\text{MLP}_2^*\left(\sum_{u \in \Psi(v)} \text{MLP}_1^*(h_u^{(k-1)})\right)\right) < 0.5
\end{cases}
\quad s.t.\ \forall \Psi(v) \in \Phi
\tag{6}
$$

where $\overline{\text{Aggregate}}_C(\cdot)$ denotes the optimal aggregation function that conveys the clause embedding messages during reasoning. Note that $\text{MLP}_2\left(\sum_{u \in \Psi(v)} \text{MLP}_1(h_u^{(k-1)})\right)$ indicates Deep Sets [18], a neural network that encodes a literal embedding set $\{h_u^{(k-1)}\}_{u \in \Psi(v)}$ whose literals are included by a clause $\Psi(v)$. The reduced feature would be fed into the sigmoid clause predictor. We use $\text{MLP}_1^*$ and $\text{MLP}_2^*$ to denote the implicit optimal prediction to each clause: given the arbitrarily initiated literal embeddings that denote the Boolean value assignment of literals, the optimal Deep Sets can predict whether the literal-derived clause is satisfied ($\geq 0.5$) or not ($< 0.5$). Since the predictor is permutation-invariant to the input, Propositions 3.1 in [15] promises that it can be approximated arbitrarily closely by graph convolution, which exactly corresponds to the parameterized clause aggregation functions in Eq.2. On the other hand, Eq. 5 promises the literal embeddings staying in their initiated values over iterations, hence the optimal Deep Sets may always judge whether a clause (the set of literals as the input of Deep Sets) is satisfied or not.

$$
h_{\Psi(v)}^{(k)} = \overline{\text{Combine}}_C\left(h_{\Psi(v)}^{(k-1)}, m_{\Psi(v)}^{(k)}\right)
$$

$$
= \begin{cases}
h_{\Psi(v)}^{(k-1)}, & h_{\Psi(v)}^{(k-1)} = m_{\Psi(v)}^{(k)} \\
h_{\Psi(v)}^{(0)}, & \|h_{\Psi(v)}^{(k-1)}\| < \|m_{\Psi(v)}^{(k)}\| \\
\mathbf{0}, & \|h_{\Psi(v)}^{(k-1)}\| \geq \|m_{\Psi(v)}^{(k)}\|
\end{cases}
\quad s.t.\ \forall \Psi(v) \in \Phi
\tag{7}
$$

where $\overline{\text{Combine}}_C(\cdot)$ denotes the optimal clause combine function. Based on the propagated messages conveyed by Eq. 2, it determines how to iteratively update clause embeddings to simulate WalkSAT.

<!-- Page 5 -->
Here is the exact transcription of the provided page in Markdown format, adhering strictly to your instructions:

---

Here we elaborate how the four optimal functions above cooperate to simulate an iteration of Walk-SAT. Since GNNs use literal embeddings as the initial input, we first analyze Eq. 6 and takes a literal $v$ into our consideration. As we discussed, this function receives a set of literal embeddings that denotes a clause that contains $v$, and then, takes the optimal Deep Sets as an oracle to judge whether this clause is satisfied. The output, the optimal message about the clause, equals to the initiated embedding of the clause $h_{\Psi(v)}$ if it is satisfied, otherwise becomes 0. This process simulates the logical reasoning on a clause, which WalkSAT relies on to pick an unsatisfied clause and flip one of its variables (see Eq. 5). Based on $m_{\Psi(v)}^{(k)}$, the optimal clause combine function (Eq. 7) updates an arbitrary clause embedding that contains $v$. The first branch states that, if the current clause message $m_{\Psi(v)}^{(k)}$ is consistent with the previous clause embedding $h_{\Psi(v)}^{(k-1)}$, it implies the satisfiability of the clause $\Psi(v)$ is not changed in this iteration (the previously satisfied clause is still satisfied, vice and versa). In this case the clause embedding would not be updated. The second and third branches imply that when $m_{\Psi(v)}^{(k)}$ and $h_{\Psi(v)}^{(k-1)}$ are inconsistent, how to update the clause embedding $h_{\Psi(v)}^{(k)}$ to convey the current message about whether the clause $\Psi(v)$ is satisfied (return into the initial clause embeddings) or not (turn into 0). Therefore all updated embeddings about the clauses that contain $v$, as the neighbors of $v$, would be fed into the optimal aggregation function in Eq. 4. This function selects $v$ that only exists in satisfied clauses, i.e., $\prod_{\Psi(v)} ||h_{\Psi(v)}^{(k)}|| \neq 0$ (If there is an unsatisfied clauses, its embedding is 0 according to Eq. 7, and would lead to $\prod_{\Psi(v)} ||h_{\Psi(v)}^{(k)}|| = 0$), then the embedding of $v$ would become 0. The results by this operation are taken advantage by Eq. 5, which promises the literal that only exists in satisfied clauses would not be “flipped” (WalkSAT only chooses unsatisfied clause and select its variables to flip. If literals are not in any unsatisfied clauses, it would not be chosen). Towards the literal $v$ contained by one unsatisfied clause at least ($\prod_{\Psi(v)} ||h_{\Psi(v)}^{(k)}|| = 0$ since there exists a clause embedding equals to 0 according to Eq. 7), its literal message would be assigned by a random vector $\epsilon^{(k)}$. It implies the randomness when WalkSAT try to select one of literal in unsatisfied clauses to flip its value. The flipping process is simulated by Eq. 6 as we have discussed.

Here we futher verify if a CNF formula could be satisfied, literal embeddings generated by the optimal aggregation and combine functions that represent the Boolean assignment of literal to satisfy this CNF formula, would converge over iterations (It corresponds to the stop criteria in Walk-SAT). Specifically suppose that in the $k-1$ iteration, Eq. 5 has induced the literal embeddings so that all clauses with the literal in the formula have been satisfied. By Eq. 6 it is obvious that $\forall v \in L$, $m_{\Psi(v)}^{(k)} = h_{\Psi(v)}^{(0)}$. To this we have $h_{\Psi(v)}^{(k-1)} = m_{\Psi(v)}^{(k)}$ and $h_{\Psi(v)}^{(k)} = h_{\Psi(v)}^{(k-1)} = h_{\Psi(v)}^{(0)}$ since all clauses in the formula have already been satisfied before the current iteration. In this case, it holds $\prod_{\Psi(v)} ||h_{\Psi(v)}^{(k-1)}|| \neq 0$ and leads to $\forall v \in L$, $m_v^{(k)} = \mathbf{0}$ in this formula (Eq. 4). In term of this, Eq. 5 guarantees all the literal embeddings consistent with those in the previous iteration.

Concluding the analysis above, we know that the optimal aggregation and combine functions (Eq. 4 5 6 7) are cooperated to simulate the local search in WalkSAT.
$\square$

**Failure in 2QBF.** Notably the failure in proving UNSAT would not be a problem for GNNs applied to solve SAT, as predicting satisfiability with high confidence has already been good enough for a binary distinction. However, 2QBF problems imply solving UNSAT, which inevitably makes GNNs unavailable in proving the relevant formulae. It probably explains the mystery in [7] about why GNNs purely learned by data-driven supervised learning lead to the same performances as random speculation [16].

## 4 Further Discussion

In this manuscript, we provide some discussions about the GNNs that consider the SAT and 2QBF problem as static graph, we haven’t considered the shrinkage condition, which may apply dynamic GNN as [9], due to the difficulty about proving the dynamic graph as we need to prove all the dynamic updating methods are impossible or not. Ought to be regarded that, this manuscript *does not claim* GNN is provably unable to achieve UNSAT, which remains an open issue.

--- 

*(Note: Page number "5" at bottom center is ignored per instruction #3.)*

<!-- Page 6 -->
Belief propagation (BP) is a Bayesian message-passing method first proposed by [10], which is a useful approximation algorithm and has been applied to the SAT problems (specifically in 3-SAT [8]) and 2QBF problems [19]. BP can find the witnesses of unsatisfiability of 2QBF by adopting a bias estimation strategy. Each round of BP allows the user to select the most biased $\forall$-variable and assign the biased value to the variable. After all the $\forall$-variables are assigned, the formula is simplified by the assignment and sent to SAT solvers. The procedure returns the assignment as a witness of unsatisfiability if the simplified formula is unsatisfiable, or UNKNOWN otherwise. However, the fact that BP is used for each $\forall$-variable assignment leads to high overhead, similar to the RL approach given by [5]. It is interesting, however, to see that with the added overhead, BP can find witnesses of unsatisfiability, which is what one-shot GNN-based embeddings cannot achieve.

This manuscript revealed the previously unrecognized limitation of GNN in reasoning about unsatisfiability of SAT problems. This limitation is probably rooted in the simplicity of message-passing scheme, which is good enough for embedding graph features, but not for conducting complex reasoning on top of the graph structures.

## References

[1] Saeed Amizadeh, Sergiy Matusevych, and Markus Weimer. Learning to solve circuit-SAT: An unsupervised differentiable approach. In *International Conference on Learning Representations*, 2019.

[2] Alonzo Church. A note on the entscheidungsproblem. *J. Symb. Log.*, 1(1):40–41, 1936.

[3] Stephen A. Cook. The complexity of theorem-proving procedures. In *Proceedings of the Third Annual ACM Symposium on Theory of Computing*, STOC ’71, pages 151–158, New York, NY, USA, 1971. ACM.

[4] Martin Davis, George Logemann, and Donald W. Loveland. A machine program for theorem-proving. *Commun. ACM*, 5(7):394–397, 1962.

[5] Gil Lederman, Markus N. Rabe, and Sanjit A. Seshia. Learning heuristics for automated reasoning through deep reinforcement learning. *CoRR*, abs/1807.08058, 2018.

[6] Xiaodan Liang, Xiaohui Shen, Jiashi Feng, Liang Lin, and Shuicheng Yan. Semantic object parsing with graph LSTM. *CoRR*, abs/1603.07063, 2016.

[7] Florian Lonsing, Uwe Egly, and Martina Seidl. Q-resolution with generalized axioms. In Nadia Creignou and Daniel Le Berre, editors, *Theory and Applications of Satisfiability Testing – SAT 2016*, pages 435–452, Cham, 2016. Springer International Publishing.

[8] M. "Mézard", G. Parisi, and R. Zecchina. Analytic and algorithmic solution of random satisfiability problems. *Science*, 297(5582):812–815, 2002.

[9] Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, and Charles E. Leiserson. Evolvegcn: Evolving graph convolutional networks for dynamic graphs. *CoRR*, abs/1902.10191, 2019.

[10] Judea Pearl. Reverend bayes on inference engines: A distributed hierarchical approach. In *AAAI*, pages 133–136. AAAI Press, 1982.

[11] Bart Selman, Henry A. Kautz, and Bram Cohen. Local search strategies for satisfiability testing. In *Cliques, Coloring, and Satisfiability*, volume 26 of *DIMACS Series in Discrete Mathematics and Theoretical Computer Science*, pages 521–531. DIMACS/AMS, 1993.

[12] Daniel Selsam, Matthew Lamm, Benedikt Bünz, Percy Liang, Leonardo de Moura, and David L. Dill. Learning a SAT solver from single-bit supervision. In *ICLR (Poster)*. OpenReview.net, 2019.

[13] João P. Marques Silva, Inês Lynce, and Sharad Malik. Conflict-driven clause learning SAT solvers. In *Handbook of Satisfiability*, volume 185 of *Frontiers in Artificial Intelligence and Applications*, pages 131–153. IOS Press, 2009.

[14] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In *ICLR*. OpenReview.net, 2019.

[15] Keyulu Xu, Jingling Li, Mozhi Zhang, Simon S. Du, Ken-ichi Kawarabayashi, and Stefanie Jegelka. What can neural networks reason about? *CoRR*, abs/1905.13211, 2019.

<!-- Page 7 -->
[16] Zhanfu Yang, Fei Wang, Ziliang Chen, Guannan Wei, and Tiark Rompf. Graph neural reasoning for 2-quantified boolean formula solvers. *CoRR*, abs/1904.12084, 2019.

[17] Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Joshua B. Tenenbaum. Neural-symbolic VQA: disentangling reasoning from vision and language understanding. *CoRR*, abs/1810.02338, 2018.

[18] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan R Salakhutdinov, and Alexander J Smola. Deep sets. In *Advances in neural information processing systems*, pages 3391–3401, 2017.

[19] Pan Zhang, Abolfazl Ramezanpour, Lenka Zdeborová, and Riccardo Zecchina. Message passing for quantified boolean formulas. *CoRR*, abs/1202.2536, 2012.

[20] David Zheng, Vinson Luo, Jiajun Wu, and Joshua B. Tenenbaum. Unsupervised learning of latent physical properties using perception-prediction networks. *CoRR*, abs/1807.09244, 2018.