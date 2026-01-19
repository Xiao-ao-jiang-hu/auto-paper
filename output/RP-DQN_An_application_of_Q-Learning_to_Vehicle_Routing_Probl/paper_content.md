<!-- Page 1 -->
# RP-DQN: An application of Q-Learning to Vehicle Routing Problems

Ahmad Bdeir$^*$, Simon Boeder$^*$, Tim Dernedde$^{(\text{equal})}$, Kirill Tkachuk$^*$, Jonas K. Falkner, and Lars Schmidt-Thieme

University of Hildesheim, 31141 Hildesheim, Germany  
$\{$bdeir,boeders,dernedde,tkachuk$\}$@uni-hildesheim.de  
$\{$falkner,schmidt-thieme$\}$@ismll.uni-hildesheim.de

## Abstract

In this paper we present a new approach to tackle complex routing problems with an improved state representation that utilizes the model complexity better than previous methods. We enable this by training from temporal differences. Specifically Q-Learning is employed. We show that our approach achieves state-of-the-art performance for autoregressive policies that sequentially insert nodes to construct solutions on the CVRP. Additionally, we are the first to tackle the MDVRP with machine learning methods and demonstrate that this problem type greatly benefits from our approach over other ML methods.

**Keywords:** Reinforcement Learning · Deep Q-Learning · Combinatorial Optimization · Vehicle Routing Problem · CVRP · MDVRP

## 1 Introduction

Routing problems are very important in business and industry applications. Finding the best routes for delivery vehicles, finding the best pick up order of trading goods in a warehouse or the optimal machine paths in a manufacturing factory are just a few examples for such problems. Due to their importance, many of these problems have been thoroughly studied and the traditional operations research community has identified a wide variety of problem types including various constraints and developed many heuristics for solving these [1, 8, 6, 4, 2, 5, 9]. Recently, the Machine Learning (ML) community has proposed to learn heuristics with models instead of handcrafting them. The main advantage of ML methods is that once initially trained, they can solve new problem instances very quickly, while traditional heuristics solve every problem individually which in the general case takes up significantly more time.

As optimal labeled solutions are expensive or intractable to compute, usually Reinforcement Learning (RL) is employed to optimize these problems. Specifically, various forms of the REINFORCE [29] algorithm were applied [3, 21, 19, 11]. A recent method by Kool et al. [19] has proven to find good solutions on a variety of problem types including the capacitated vehicle routing problem (CVRP). Their

---

$^*$ Equal contribution

<!-- Page 2 -->
architecture consists of an encoder and decoder setup that constructs the tour sequentially. The encoder computes a representation of the graph and its nodes and is run once in the beginning of the tour construction. Then the decoder is applied on the embeddings repeatedly to add nodes to the solution until the tour is finished. This method is limited to static node features, since the encoder is applied only once. Using dynamic features, and hence a more rich representation, would require the encoder to be run at every step. This quickly becomes memory-expensive when using REINFORCE, as all intermediate representations have to be stored over the whole episode in order to execute backpropagation.

Motivated by this limitation, we propose a model that enables the use of dynamic features in the encoder. Encoder and decoder are then applied in every step. To make this memory-feasible, we optimize our model with a temporal-difference algorithm. When learning only on single transitions, the intermediate representations do not have to be stored for the whole episode. Specifically, we will use Q-Learning [20].

We show that the improved state representation leads to performance improvements on the CVRP. Additionally, we show that our approach is more sample efficient which can be an important factor when environments become more complex and expensive to compute. Finally, we show, when extending to more complicated problems like the Multiple Depot Vehicle Routing Problem (MDVRP), the improved state representation is of even greater importance. As to our knowledge we are the first to explore the MDVRP with RL-methods, thus we extended the model by Kool et al. [19] to solve the MDVRP and compare to our approach and several established OR-methods.

## 2 Related work

The operations research community has studied a plenitude of different routing problems over the last 70 years. Various different soft and hard constraints, objective functions and other problem properties like stochasticity have been considered and many optimal and heuristic solution approaches with and without formal guarantees were proposed. As a comprehensive overview in this context is not possible the following related work will focus on the ML literature. We refer the reader to a survey by Toth and Vigo [26].

Machine Learning has largely considered two approaches for solving routing problems. The first one autoregressively inserts one node at a time from a partial tour until a solution is complete. Vinyals et al. [28] proposed the Pointer-Network and used this approach to tackle the TSP. Their model was learnt from optimal examples. As these are expensive, Bello et al. [3], Nazari et al. [21] proposed approaches that used RL for optimization. Specifically, variants of REINFORCE [29] were employed and the problem catalogue was extended to the CVRP. Kool et al. [19] then proposed an encoder-decoder model based on the Transformer architecture and showed that good solutions can be learnt for various related problem types for instances with up to 100 nodes. Falkner and Schmidt-Thieme [11] proposed an extension to incorporate time windows.

<!-- Page 3 -->
While both also use variants of REINFORCE, Khalil et al. [17] built a graph model based on S2V optimized with Q-Learning. Instead of directly inserting the predicted next node, they choose its position with a best insertion heuristic. However their results on the TSP were only comparable with simpler constructive heuristics like farthest insertion [23]. The autoregressive approaches have considered different search algorithms to use during inference, like Sampling, Beam Search and Active Search.

The second paradigm that was considered with ML approaches is the use of improvement heuristics. They operate on initial solutions and improve these repeatedly. Usually local search is applied. Traditionally, Meta-heuristics like Large-Neighbourhood Search (LNS) or Guided-Local Search (GLS) are used in conjunction with local search operators like 2-opt to avoid local minima. Various Meta-heuristics and local operators are implemented in Perron and Furnon [22]. ML approaches here range from only learning the initial solution and then applying a traditional method [31] to learning to perform local search [30, 7].

Other approaches that do not fit these two categories include constructing a tour non-autoregressively by predicting a heatmap of promising edges with a graph convolutional neural network [16], augmenting traditional dynamic programming approaches for Routing with deep learning [18] and changing the action space from nodes to full tours and solving the sub-problem of the best route to insert [10].

## 3 Problem Definition

We consider two problem variants in this paper: the CVRP and the MDVRP.

**CVRP** The problem can be described by a graph $G(N, E)$, where $N = C \cup D$ is the set of nodes consisting of customers $C = \{1, ..., n\}$ and one depot $D = \{n+1\}$. Each customer $c \in C$ has a positive demand $d$. We assume a complete graph, thus the set of edges $E$ contains an edge $e_{i,j}$ representing the distance for every pair of nodes $(i, j) \in N$. We also assume a fleet of homogeneous vehicles $K$ having the same maximum Capacity $Q$ with no restriction on the fleet size. For a valid solution all customers have to be visited exactly once. All routes must start and end at the depot. For all routes the total sum of demands must satisfy the maximum vehicle capacity. As an objective we want to minimize the total sum of distances. This formulation is also used by other works like [19, 3]. Note however that other formulations with limited, heterogeneous vehicles or other objective functions are possible.

**MDVRP** The MDVRP generalizes this problem by allowing for multiple depots $D = \{n + 1, ..., n + m\}$. We require that every vehicle ends at the same depot that it has started but the model is allowed to decide at which depot a vehicle should start. Again note that other problem formulations are possible. Vehicles could be fixed and set to certain depots or allowed to return to any depot instead of the one they started for instance.

<!-- Page 4 -->
# 4 Method

We base our architecture on the model by Kool et al. [19]. First we present the original model and then show which adaptions our new approach makes as well as how solutions are constructed for the CVRP and MDVRP and how the model is optimized.

## 4.1 Original Attention-Model

The attention model [19] solves routing problems using an encoder-decoder architecture and learns a policy model $\pi$ that autoregressively inserts one node at a time until a solution is complete.

**Encoder** The encoder takes the set of nodes $N$ consisting of the customer nodes and the depot node and creates an embedding of each node. The node features consist of the coordinates and the demand. These features are considered static, meaning they don’t change during decoding. Thus the node embeddings are calculated once at the beginning of the episode and are reused at each step.

The encoder creates initial node embeddings $h_i$ by applying a node-wise linear projection, scaling up every node to the embedding dimension $h_{dim} = 128$. $h_i$ represents the $i$-th node embedding. To differentiate between the depot node and the customer nodes, weights are not shared and two projections are learned.

$$
h_i^{(0)} = W^{\text{node}} n_i + b^{\text{node}} \tag{1}
$$

$$
h_0^{(0)} = W^{\text{depot}} n_0 + b^{\text{depot}} \tag{2}
$$

These embeddings are then updated through $L$ attention blocks (AB):

$$
H^{(L)} = \text{AB}_L(\ldots(\text{AB}_1(H^{(0)}))) \tag{3}
$$

where $H$ is a concatenation of the initial embeddings, and $L = 3$ for VRPs. Every block AB consists of a multi-head self-attention layer (MHA) [27], a node-wise feed-forward layer (FF), along with batch normalization (BN) [15] and skip connections (skip).

$$
\text{AB}(H^{(l)}) = \text{BN}(\text{FF}^{\text{skip}}(\text{BN}(\text{MHA}^{\text{skip}}(H^{(l-1)})))) \tag{4}
$$

where the MHA layer uses 8 heads. The $\text{FF}^{\text{skip}}$ layer has one sublayer with ReLU activation and an intermediate dimension of size 512.

$$
\text{FF}^{\text{skip}}(h_i) = W_2 \max(0, W_1 h_i + b_1) + b_2 + h_i \tag{5}
$$

<!-- Page 5 -->
**Decoder** The decoder is run at every timestep $t$ and parametrizes a probability distribution over all the nodes. It takes a context $C^{(t)} = [h^{\text{graph}}; c^{(t)}; h^{\text{last}}]$ as input, where $h^{\text{graph}}$ is the mean of the node embeddings, $c^{(t)}$ is the current vehicle capacity and $h^{\text{last}}$ refers to the node embedding of the last position. This context is transformed with another MHA layer, where the context only attends to nodes that are feasible in this timestep. This is done with a mask $M$. Note that the context is the query and the node embeddings are the keys of this layer. In contrast the encoder computes the self attention between all pairs of nodes. The decoder instead only computes attention between context and node embeddings to arrive at a transformed context $\hat{C}^{(t)}$. This avoids the memory quadratic complexity of the MHA-layer.

$$
\hat{C}^{(t)} = \text{MHA}(C^{(t)}, H, M)
\tag{6}
$$

Now, compatibility scores $u_i$ between the transformed context and the nodes are calculated as in a Single-Head Attention (SHA) mechanism where the context is again the query:

$$
u_i = \frac{W^q \hat{C}^{(t)} W^k h_i}{\sqrt{d_k}}
\tag{7}
$$

This gives a single value for each node and infeasible nodes are masked with $-\infty$. To arrive at the final probabilities, a softmax is applied. For more details, we refer the reader to the original paper by Kool et al. [19].

## 4.2 RP-DQN

Most of the models complexity is in the encoder, however it only considers the static components of the problem. The dynamic information that can be used in the context is quite constrained. The original model has very limited access to information about which nodes have already been visited or are infeasible due to other constraints. The only time this information is passed is in equation 6 with an inner masking that prevents attention from infeasible nodes to the context. We feel that using dynamic node features in the encoder makes the model more

![Fig. 1. Architecture overview](image_placeholder.png)

*Fig. 1. Architecture overview*

<!-- Page 6 -->
expressive and utilizes its complexity better. However, in that case the encoder has to be executed in every step. As discussed already, to make this memory feasible, our model is optimized with Q-Learning. Thus, our model also has to output Q-values instead of probabilities and therefore we get rid of the softmax. The inner masking is also not needed anymore and we omit it completely.

Now, the four dynamic features $\mu_i^{(t)}, \rho_i^{(t)}, \tau_i^{(t)}, \phi_i^{(t)}$ represent boolean variables that indicate whether at timestep $t$ the node $i$ has already been inserted in the tour, cannot currently be inserted due to capacity constraints, represents the current position of the vehicle or in the multi-depot case represents the current depot.

$\mathcal{R}^{(t)}$ is the set of customer nodes that have been visited at time $t$. Then the set

$$
\mu_i^{(t)} = 
\begin{cases}
1 & \text{if } i \in \mathcal{R}^{(t)} \\
0 & \text{otherwise}
\end{cases}
\quad (8)
\qquad
\rho_i^{(t)} = 
\begin{cases}
1 & \text{if } d_i^{(t)} > c^{(t)} \\
0 & \text{otherwise}
\end{cases}
\quad (10)
$$

$$
\tau_i^{(t)} = 
\begin{cases}
1 & \text{if } i \in \mathcal{R}^{(t)} \land i \notin \mathcal{R}^{(t-1)} \\
0 & \text{otherwise}
\end{cases}
\quad (9)
\qquad
\phi_i^{(t)} = 
\begin{cases}
1 & \text{if } i \text{ is active depot} \\
0 & \text{otherwise}
\end{cases}
\quad (11)
$$

of node features for the CVRP includes $[x_i, y_i, d_i^{(t)}, \mu_i^{(t)}, \rho_i^{(t)}, \tau_i^{(t)}]$, where $x_i$ and $y_i$ are the coordinates of node $i$ and $d_i^{(t)}$ is the demand. $d_i^{(t)} = 0$ for nodes that have been served at timestep $t$. In the MDVRP case, $\phi_i^{(t)}$ is also added to the set of node features. The rest of the architecture stays the same. Although it potentially could be simplified, we opted to stay close to the original architecture. We show that performance improvements come mainly through our better state representation. Our model is called Routing Problem Deep Q-Network (RP-DQN). An architecture overview can be seen in Figure 1.

## MDVRP Decoding

For the MDVRP we expand the context by the current depot embedding. This is done for our model as well as for our extension of the model by Kool et al. [19] to the MDVRP. Next we describe the decoding procedure. In the first step the model has not selected any node yet. We mask all customers and force it to select a depot. This depot becomes the current depot and the model selects nodes until it chooses to come back to the current depot. We don’t allow for back to back depot selection and the model cannot come back to a different depot than it started from. After a route is completed the model can start another route by choosing the next depot and the procedure is repeated until all customers are inserted. The Q-values of infeasible actions are masked with $-\infty$.

## Q-Learning

To optimize our model we implemented Double DQN [20, 13] with $N$-step returns and a prioritized replay buffer [24]. We found $N = 1$ to

<!-- Page 7 -->
work best. For exploration, we experimented with Boltzman-Exploration [25] with decaying softmax temperature and $\epsilon$-Greedy with decaying $\epsilon$. In the end we used $\epsilon$-Greedy and linearly decayed the rate over half of the total training episodes. Although Boltzman-Exploration initially speeds up the convergence, the model also plateaus at slightly higher costs than with $\epsilon$-Greedy.

## 5 Experiments

### 5.1 Baselines

For the CVRP we differentiate baselines from four different categories. The first category includes specialized solvers. Gurobi [12] was used in Kool et al. [19] as an optimal solver for the CVRP. This becomes intractable for instances with more than 20 customers, thus the highly optimized LKH3 [14] was employed by Kool et al. [19] alternatively. It transforms constrained problems into TSPs via penalty functions and other means and then solves those heuristically. The second category consist of ML approaches that construct solutions sequentially and use greedy inference. We compare our model with the approaches by Kool et al. [19], Nazari et al. [21], Falkner and Schmidt-Thieme [11]. The third category also includes ML based approaches with sequential construction, but this time using sampling or beam search inference. We compare with the same models as in the previous category. Our model, Kool et al. [19] and Falkner and Schmidt-Thieme [11] use sampling while Nazari et al. [21] uses beam search. The last category encompasses improvement heuristics that use local search. We include it for completeness sake. They operate on an initial solution and improve it iteratively by selecting solutions in the neighborhood. OR-Tools [22] is a classical approach using meta-heuristics while Wu et al. [30] is a machine learning approach. For the MDVRP, no ML baselines are available. Thus, we adapt the model by Kool et al. [19] to support MDVRP and train it ourselves. We set up the decoding procedure and context as for our model. For a traditional baseline we stick to Google OR-Tools as it provides a flexible modeling framework. However OR-Tool does not provide an out of the box MDVRP solver, thus we use the CVRP framework and simulate the MDVRP by setting different starting and finishing points to the vehicles.

### 5.2 Data

For the training of the model we will generate data. We will create data with $|C| = 20, 50, 100$ and train a model for each of these. In the main experiments, the problem size stays the same for training and testing. Additionally, we conducted a generalization study that shows how the models perform on different problem sizes, which can be seen in Appendix B. Since the data can be generated unlimitedly, we will use every problem instance only once and generate new data after every episode. Note that the data can be generated with various different properties. For instance, the customers can have a special alignment in grids or

<!-- Page 8 -->
star-like structures, be sampled uniformly or according to some other distribution. In order to compare, our data generation follows Nazari et al. [21]. Their dataset is used by most of the ML literature. For testing we have used the exact test set that was provided by Nazari et al. [21] for each problem size. It consists of 10,000 instances that were generated with the same properties as the training data. It uses the euclidean distance. Demands $d \in [1,9]$ are sampled uniformly. Coordinates are sampled uniformly from the unit square. Vehicles have capacity 30, 40 and 50 depending on the problem size. For the MDVRP we generate the data in the same fashion and create our own test set.

## 5.3 CVRP Results

Table 1 shows that our approach outperforms all other models that construct the solution by sequentially selecting nodes. This applies for both greedy and sampling inference. Only the improvement method by Wu et al. [30] achieves better results. However, improvement heuristics operate on an initial solution and will try to improve this repeatedly. It has been shown that these approaches scale to better final performance when given a better initial starting solution [31]. Thus better approaches to find initial solutions like ours can be used to kick-start improvement methods like OR-Tools or the machine learning approach by Wu et al. [30]. We also notice that our percentage gain increases with the problem size, stressing the importance of dynamic features for larger, real world problems. Appendix A gives a more detailed comparison including the timings of some of the methods.

### Table 1. CVRP Results

| Method                      | Problem Size                  |
|-----------------------------|-------------------------------|
|                             | **20**        | **50**        | **100**       |
|                             | Mean Gap %    | Mean Gap %    | Mean Gap %    |
|-----------------------------|---------------|---------------|---------------|
| Specialized Solver          |               |               |               |
| &nbsp;&nbsp;Gurobi          | 6.1           | -             | -             |
| &nbsp;&nbsp;LKH3            | 6.14          | 0.66          | 10.38         | 0.00          | 15.65         | 0.00          |
| Sequential Policy (Greedy)  |               |               |               |
| &nbsp;&nbsp;RP-DQN          | **6.36**      | 4.26          | **10.92**     | 5.20          | **16.59**     | 6.01          |
| &nbsp;&nbsp;Kool            | 6.4           | 4.92          | 10.98         | 5.78          | 16.8          | 7.35          |
| &nbsp;&nbsp;Falkner         | 6.47          | 6.07          | 11.44         | 10.21         | -             | -             |
| &nbsp;&nbsp;Nazari          | 6.59          | 8.03          | 11.39         | 9.73          | 17.23         | 10.10         |
| Sequential Policy (Sampling/Beam search) |               |               |               |
| &nbsp;&nbsp;RP-DQN (1024s)  | **6.24**      | 2.30          | **10.59**     | 2.02          | **16.11**     | 2.94          |
| &nbsp;&nbsp;Kool (1280s)    | 6.25          | 2.46          | 10.62         | 2.31          | 16.23         | 3.71          |
| &nbsp;&nbsp;Falkner (1280s) | 6.26          | 2.62          | 10.84         | 4.43          | -             | -             |
| &nbsp;&nbsp;Nazari (10bs)   | 6.4           | 5.41          | 11.31         | 8.96          | 17.16         | 9.65          |
| Local Search                |               |               |               |
| &nbsp;&nbsp;Wu (5000 steps) | **6.12**      | 0.33          | **10.45**     | 0.67          | **16.03**     | 2.43          |
| &nbsp;&nbsp;OR-Tools        | 6.43          | 5.41          | 11.31         | 8.96          | 17.16         | 9.65          |

<!-- Page 9 -->
## 5.4 MDVRP Results

In Table 2 an even greater lift as for the CVRP can be seen. The model by Kool et al. [19] reaches only subpar performance when the problem size increases. We attribute the results to the more powerful state representation of our model. We assume that this has more impact on the MDVRP as it is a more complicated problem than the CVRP. We also want to highlight that training our model is much more memory efficient compared to the standard model by Kool et al. [19].

**Table 2. MDVRP Results**

| Method                 | Problem Size              |
|------------------------|---------------------------|
|                        | 20        | 50        | 100       |
|                        | Mean Gap % | Mean Gap % | Mean Gap % |
| RP-DQN - Greedy        | 5.48      | 2.62      | 8.04      | 4.15      | 11.99     | 4.08      |
| Kool - Greedy          | 5.68      | 6.37      | 8.84      | 14.51     | 13.17     | 14.32     |
| RP-DQN - Sampling 1024 | **5.34**  | 0.00      | **7.72**  | 0.00      | **11.52** | 0.00      |
| Kool - Sampling 1024   | 5.42      | 1.50      | 8.11      | 5.05      | 12.15     | 5.47      |
| OR-Tools               | 6.74      | 26.23     | 9.02      | 16.84     | 12.92     | 12.15     |

## 5.5 Learning Curves

In Figure 2 we exemplify the learning behavior of both models on the MDVRP with 50 customers run on the same hardware. Due to the use of a buffer, the sample efficiency is improved greatly. While this is less important for simple environments, it has significant impact for problems that are more expensive to compute. For routing, this can include environments with more hard and soft constraints, stochastic components and objective functions that trade-off multiple goals. Figure 2 shows that the learning of our model starts off slower, however we always converge significantly faster.

## 6 Conclusion

In this paper we present a new approach to tackle complex routing problems based on learning from temporal differences, specifically through Q-Learning, when optimizing autoregressive policies that sequentially construct solutions by inserting one node at a time for solving routing problems. We showed that this learning procedure allows the incorporation of dynamic node features, enabling more powerful models which lead to state-of-the-art performance on the CVRP for autoregressive policies. Additionally, the sample efficiency is greatly improved. Although our model still falls short of specialized solvers like LKH3 and improvement methods, it is useful to find very good initial solutions. Future

<!-- Page 10 -->
10

<div style="display: flex; justify-content: space-between;">
  <div>
    <img src="image_placeholder_1" alt="MDVRP50 – Wall Clock Time" />
  </div>
  <div>
    <img src="image_placeholder_2" alt="MDVRP50 – Samples" />
  </div>
</div>

**Fig. 2.** Learning Curves for the MDVRP with 50 customers comparing the wall clock time and sample efficiency.

work could include combining powerful initial solution finders like RP-DQN with improvement heuristics. We also demonstrated that the dynamic components become more important for the MDVRP, a problem type that was not explored with RL before. We assume that this holds for other more complicated problem types like CVRP with time windows. Future work could include extending our model to these problem types. Further, more work is needed to improve generalizability. This could include training on broader distributions and building automatic data generators from real life datasets.

**Acknowledgement** This work is co-funded via the research project L2O¹ funded by the German Federal Ministry of Education and Research (BMBF) under the grant agreement no. 01IS20013A and the European Regional Development Fund project TrAmP² under the grant agreement no. 85023841.

## A Runtime Comparison

Reporting the runtime is difficult as it can differ in order of magnitudes due to implementation differences (C++ vs Python) and hardware. Kool et al. [19] thus decided to report the runtime over the complete test set of 10,000 instances. Other literature like Wu et al. [30] has followed them. We feel that this adds another layer of obscurity over the runtime comparison as they then decided to parallelize over multiple instances. ML approaches were parallelized via batch computation. Most traditional methods are single threaded CPU applications. They were parallelized over instances by launching more threads if the CPU has them available. We feel that for practical applications if enough instances

---

¹ https://www.ismll.uni-hildesheim.de/projekte/l2o_en.html  
² https://www.ismll.uni-hildesheim.de/projekte/tramp.html

<!-- Page 11 -->
have to be solved for instance parallelization to be useful, the corresponding hardware can be bought. More important is the time it takes to solve a single instance or alternatively the time it takes to solve the 10,000 instances when not parallelizing over instances. In Table 3 we collect some timings reported by the literature and our own. Note that the timings are not directly comparable due to the discussed reasons. Also note that most methods have some way of trading performance and time off. Beam Searches can have bigger width, more solutions can be sampled and improvement methods can make more steps. Additionally, none of the methods will improve indefinitely but hit diminishing returns instead. Thus there are more and less reasonable spots to trade-off. Ultimately however, this is also application dependent. Reporting full trade-off curves for all of the methods is not possible.

**Table 3. CVRP Results and Timings to solve all 10,000 instances**

| Method             | Problem Size                                                                 |
|--------------------|--------------------------------------------------------------------------------|
|                    | **20**                 | **50**                 | **100**                |
|                    | Mean      Time         | Mean      Time         | Mean      Time         |
|--------------------|------------------------|------------------------|------------------------|
| Gurobi             | 6.1       -            | -       -              | -       -              |
| LKH3               | 6.14      (2h*)        | 10.38   (7h*)          | 15.65   (13h*)         |
| RP-DQN – Greedy    | 6.36      (3s†/10min††)| 10.92   (14s†/30min††) | 16.59   (50s†/78min††) |
| Kool – Greedy      | 6.4       (1s†)        | 10.98   (3s†)          | 16.8    (8s†)          |
| RP-DQN – 1024s     | 6.24      (52m**)      | 10.59   (5h**)         | 16.11   (15h**)        |
| Kool – 1280s       | 6.25      (6m**)       | 10.62   (28m**)        | 16.23   (2h**)         |
| Wu (5000 steps)    | 6.12      (2h†)        | 10.45   (4h†)          | 16.03   (5h†)          |
| OR-Tools           | 6.43      (2m‡)        | 11.31   (13m‡)         | 17.16   (46m‡)         |

*32 instances were solved in parallel on two CPUs [19].  
†Time for solving many instances in parallel through GPU batch computation.  
‡Time reported by Wu et al. [30]. Only one Thread.  
**Time for solving one instance at a time.  
††Time for solving one instance at a time on CPU

Our method is extremely quick in greedy and then takes more time the more samples are used. This behaviour is respectively the same for the Kool model although our model expectedly takes more time as the encoder is run at every step. 15 hours for the size 100 model with sampling seems high, however consider that one instance was solved at a time, thus a single instance still only takes less than 6 seconds. The only two methods that have better results are LKH3 and Wu et al. [30]’s improvement approach. For both the time was only reported with a high degree of instance parallelization, thus it should be expected that our model takes less time than them on a fair single instance comparison.

Additionally, 1024 samples is already at a point on the trade-off curves where there are negligible performance improvements as can be seen in figure 3.

<!-- Page 12 -->
12

![Trade-Off Curve](image-placeholder)  
**Fig. 3.** This figure shows how much additional samples improve the solution quality.

## B Generalization Study

This study tests the ability to generalize. All models were trained on one size and tested on instances between 20 and 200 nodes. Figure 4 shows the percentage gap to the best model from both inference settings.

![MDVRP Generalization Greedy Inference](image-placeholder)  
![MDVRP Generalization Sampling](image-placeholder)  
**Fig. 4.** This figure shows the generalization to different problem sizes.

<!-- Page 13 -->
# Bibliography

1. Balas, E.: The prize collecting traveling salesman problem. Networks **19**(6), 621–636 (1989)
2. Beasley, J.E.: Route first—cluster second methods for vehicle routing. Omega **11**(4), 403–408 (1983)
3. Bello, I., Pham, H., Le, Q.V., Norouzi, M., Bengio, S.: Neural combinatorial optimization with reinforcement learning. CoRR (2016), URL http://arxiv.org/abs/1611.09940
4. Bertsimas, D.J., Van Ryzin, G.: A stochastic and dynamic vehicle routing problem in the euclidean plane. Operations Research **39**(4), 601–615 (1991)
5. Braun, H.: On solving travelling salesman problems by genetic algorithms. In: International Conference on Parallel Problem Solving from Nature, pp. 129–133, Springer (1990)
6. Bräysy, O., Gendreau, M.: Vehicle routing problem with time windows, part I: Route construction and local search algorithms. Transportation science **39**(1), 104–118 (2005)
7. Chen, X., Tian, Y.: Learning to perform local rewriting for combinatorial optimization. In: Advances in Neural Information Processing Systems, vol. 32, Curran Associates, Inc. (2019)
8. Christofides, N.: Combinatorial optimization. A Wiley-Interscience Publication (1979)
9. Christofides, N., Mingozzi, A., Toth, P.: Exact algorithms for the vehicle routing problem, based on spanning tree and shortest path relaxations. Mathematical programming **20**(1), 255–282 (1981)
10. Delarue, A., Anderson, R., Tjandraatmadja, C.: Reinforcement learning with combinatorial actions: An application to vehicle routing. In: Advances in Neural Information Processing Systems, vol. 33, pp. 609–620, Curran Associates, Inc. (2020)
11. Falkner, J.K., Schmidt-Thieme, L.: Learning to solve vehicle routing problems with time windows through joint attention (2020), URL http://arxiv.org/abs/2006.09100
12. Gurobi Optimization, L.: Gurobi optimizer reference manual (2021)
13. Hasselt, H.v., Guez, A., Silver, D.: Deep reinforcement learning with double q-learning. In: Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, p. 2094–2100, AAAI’16, AAAI Press (2016)
14. Helsgaun, K.: An extension of the Lin-Kernighan-Helsgaun TSP solver for constrained traveling salesman and vehicle routing problems (2017), https://doi.org/10.13140/RG.2.2.25569.40807
15. Ioffe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: Proceedings of the 32nd International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 37, pp. 448–456, PMLR, Lille, France (2015)

<!-- Page 14 -->
16. Joshi, C.K., Laurent, T., Bresson, X.: An efficient graph convolutional network technique for the travelling salesman problem. CoRR (2019), URL http://arxiv.org/abs/1906.01227

17. Khalil, E., Dai, H., Zhang, Y., Dilkina, B., Song, L.: Learning combinatorial optimization algorithms over graphs. In: Advances in Neural Information Processing Systems, vol. 30, Curran Associates, Inc. (2017)

18. Kool, W., van Hoof, H., Gromicho, J., Welling, M.: Deep policy dynamic programming for vehicle routing problems (2021), URL http://arxiv.org/abs/2102.11756

19. Kool, W., van Hoof, H., Welling, M.: Attention, learn to solve routing problems! In: International Conference on Learning Representations (2019)

20. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., Hassabis, D.: Human-level control through deep reinforcement learning. Nature **518**(7540), 529–533 (2015), ISSN 00280836, https://doi.org/10.1038/nature14236

21. Nazari, M.R., Oroojlooy, A., Snyder, L., Takac, M.: Reinforcement learning for solving the vehicle routing problem. In: Advances in Neural Information Processing Systems, vol. 31, Curran Associates, Inc. (2018)

22. Perron, L., Furnon, V.: OR-Tools 7.2 (2019)

23. Rosenkrantz, D.J., Stearns, R.E., Lewis, P.M., Shukla, S.K.: An analysis of several heuristics for the traveling salesman problem, pp. 45–69. Springer Netherlands, Dordrecht (2009)

24. Schaul, T., Quan, J., Antonoglou, I., Silver, D.: Prioritized experience replay (2015), URL http://arxiv.org/abs/1511.05952

25. Sutton, R.S., Barto, A.G.: Reinforcement Learning: An Introduction. The MIT Press, second edn. (2018)

26. Toth, P., Vigo, D.: Vehicle Routing: Problems, Methods, and Applications, Second Edition. No. 18 in MOS-SIAM Series on Optimization, SIAM (2014), ISBN 9781611973587

27. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.u., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems, vol. 30, Curran Associates, Inc. (2017)

28. Vinyals, O., Fortunato, M., Jaitly, N.: Pointer networks. In: Advances in Neural Information Processing Systems, vol. 28, Curran Associates, Inc. (2015)

29. Williams, R.J.: Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning **8**(3-4), 229–256 (1992)

30. Wu, Y., Song, W., Cao, Z., Zhang, J., Lim, A.: Learning improvement heuristics for solving routing problems (2020), URL http://arxiv.org/abs/1912.05784

31. Zhao, J., Mao, M., Zhao, X., Zou, J.: A hybrid of deep reinforcement learning and local search for the vehicle routing problems. IEEE Transactions on Intelligent Transportation Systems pp. 1–11 (2020), https://doi.org/10.1109/TITS.2020.3003163