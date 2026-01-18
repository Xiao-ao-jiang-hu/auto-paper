<!-- Page 1 -->
Submitted to *Management Science* manuscript

# A Survey on Reinforcement Learning for Combinatorial Optimization

Yunhao Yang  
Department of Computer Science, University of Texas at Austin, Austin, TX 78705, yunhaoyang234@utexas.edu

Andrew Whinston  
Department of Information, Risk and Operations Management, University of Texas at Austin, Austin, TX 78705, abw@uts.cc.utexas.edu

This paper gives a detailed review of reinforcement learning in combinatorial optimization, introduces the history of combinatorial optimization starting in the 1960s, and compares it with the reinforcement learning algorithms in recent years. We explicitly look at a famous combinatorial problem known as the Traveling Salesperson Problem (TSP). We compare the approach of the modern reinforcement learning algorithms on TSP with an approach published in 1970. Then, we discuss the similarities between these algorithms and how the approach of reinforcement learning changes due to the evolution of machine learning techniques and computing power. We also mention the deep learning approach on the TSP, which is named Deep Reinforcement Learning. We argue that deep learning is a generic approach that can be integrated with traditional reinforcement learning algorithms and optimize the outcomes of the TSP.

*Key words*: reinforcement learning, combinatorial optimization, dynamic programming, machine learning, traveling salesperson

---

## 1. Introduction

Combinatorial optimization (discrete optimization), as opposed to continuous optimization is the focus of this paper. Discrete optimization is searching for an optimal solution in a finite or countably infinite set of potential solutions. Optimality is defined with respect to some criterion function, which is to be minimized or maximized. This paper discusses combinatorial optimization applied to the Quadratic Assignment Problem (QAP), and a special case which is the well-known Traveling Salesperson Problem (TSP).

The QAP was introduced by Koopmans and Beckman in 1957 (Koopmans, and Beckmann. 1957) in the context of locating "indivisible economic activities" (Anstreicher 2003). The QAP consists of two sets of interrelated objects, the solution of the problem is the optimal assignment among the objects. From an economic perspective, the objective of the QAP is to assign a set of facilities to a set of locations in such a way as to minimize the total assignment cost (Çela 1998).

<!-- Page 2 -->
The QAP is known to be an NP-Hard problem (Rainer 2013). There is no polynomial-time solution to the problem. However, there are many approximation algorithms for this problem in order to reduce the computational complexity. Recently, the idea of machine learning is broadly used in computation. Currently, many researchers try to apply reinforcement learning and neural networks for solving the QAP. It is an important way which we will discuss in the later part of the paper.

One of the most common example of the QAP is TSP. The TSP is also an NP-Complete problem in combinatorial optimization (Papadimitriou 1977), commonly studied in theoretical computer science and operations research. The approximation of the TSP is an important topic. Recently, some researchers are focusing on applying reinforcement learning algorithms on approximating the solution to the TSP. This paper is focusing on comparing the different reinforcement learning algorithms that generate approximate solutions to the TSP.

Reinforcement learning (RL) is an area of machine learning that develops approximate methods for solving dynamic optimization problems. The main concern of reinforcement learning is how software agents ought to take actions in an environment in order to maximize the concept of cumulative reward or minimize the cost/penalty. The environment is typically stated in the form of a Markov decision process (Van Otterlo, Wiering 2012). Because of the nature of reinforcement learning, it is one of the relatively efficient learning techniques.

In recent years, the concept of Deep RL was introduced and largely applied in the machine learning fields. Deep reinforcement learning is a combination of RL and deep learning (Francois-Lavet 2018). Deep RL utilizes a deep neural network structure to manipulate high-dimensional data. It typically generates outputs based on the probabilistic outcomes. In recent years, there are some Deep RL algorithm was also introduced in approximating the TSP, as well as other combinatorial optimization problems.

## 2. Motivation

With the evolution of the computing power, reinforcement learning techniques can be applied to various problems. Unlike supervised learning, reinforcement learning does not need labelled data to adjust the network base on the loss function. Instead, it focuses on finding balance between exploration and exploitation (Kaelbling, Littman, Moore 1996). A special case of the QAP- TSP is NP-hard, which is known to be challenging to find the optimal solution. While reinforcement learning is not developed to find the optimal solution but to approximate the optimal solution. Therefore, compare to approximating TSP arithmetically, reinforcement learning is a new direction that is worth to be explored. This paper will discuss the performance difference between the algorithm introduced in 1960s and the modern reinforcement learning algorithms, as well as how the limitations of computing powers would affect the performance of these algorithms.

<!-- Page 3 -->
# 3. Historical Timeline

Although Reinforcement Learning is a relatively new field in machine learning, there were some researchers in the past who introduced the idea of using Reinforcement Learning to solve the TSP. The idea of Reinforcement Learning can be traced back to mid 20th Century, which provided the theoretical support for modern Reinforcement Learning. For example, the "prototype" of Reinforcement Learning on Combinatorial Optimization was introduced in 1970.

## 3.1. Bellman Equation (1957)

Bellman Equation, also known as Dynamic Programming Equation, was introduced by Richard Bellman(Bellman 1957) and has been used in Dynamic Programming. It divides a dynamic optimization problem into a sequence of simpler sub-problems(Kirk 1970). The Bellman Equation is able to deal with the majority of the discrete-time problems that relate to Optimal Control Theory. However, the Bellman Equation is not feasible in solving the large scale NP-hard problems such as the TSP. Reinforcement Learning provides a way to approximate the Bellman Equation and solve the TSP.

The Bellman Equation is commonly used as the starting point of the Reinforcement Learning approach. To learn the optimal policy $\pi$ in Reinforcement Learning(Graves 2017), there are two types of value functions: the state value function $V(s)$, and the action value function $Q(s,a)$.

The state value function returns the value of a state $s$ according to the policy $\pi$ (a function or method to generate outputs).

$$
V^{\pi}(s) = \mathbb{E}_{\pi}[R_t | s_t = s]
\tag{1}
$$

Define $\wp_{ss'}^{a} = Pr[s_{t+1}=s' | s_t=s, a_t=a]$ and $\Re_{ss'}^{a} = \mathbb{E}[r_{t+1} | s_t=s, s_{t+1}=s', a_t=a]$. Where $\wp$ is the transition probability and $\Re$ is the expected or average reward when starting in state $s$, taking action $a$, and moving toward state $s'$. Then, derive the Bellman Equation, the state value function can be wrote as

$$
V^{\pi}(s) = \mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^{k}r_{t+k+1} | s_t=s] = \mathbb{E}_{\pi}[\gamma r_{t+1} + \gamma\sum_{k=0}^{\infty}\gamma^{k}r_{t+k+2} | s_t=s]
\tag{2}
$$

The equation above describes the expected return value if start from state $s$ and follow policy $\pi$. Then, inserting $\wp$ and $\Re$ that are defined above to this equation and use the fact that

$$
\mathbb{E}_{\pi}[r_{t+1} | s_t=s] = \sum_{a}\pi(s,a)\sum_{s'}\wp_{ss'}^{a}\Re_{ss'}^{a}
$$

$$
\mathbb{E}_{\pi}[\gamma\sum_{k=0}^{\infty}\gamma^{k}r_{t+k+2} | s_t=s] = \sum_{a}\pi(s,a)\sum_{s'}\wp_{ss'}^{a}\gamma\mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^{k}r_{t+k+2} | s_{t+1}=s']
\tag{3}
$$

Eventually, the state value function $V(s)$ can be rewrite as

$$
V^{\pi}(s) = \sum_{a}\pi(s,a)\sum_{s'}\wp_{ss'}^{a}(\Re_{ss'}^{a} + \gamma V^{\pi}(s'))
\tag{4}
$$

<!-- Page 4 -->
The action value function can also be derived using the Bellman Equation:

$$
\begin{aligned}
Q^{\pi}(s, a) &= \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^{k} r_{t+k+1} | s_t = s, a_t = a \right] \\
&= \mathbb{E}_{\pi} \left[ \gamma r_{t+1} + \gamma \sum_{k=0}^{\infty} \gamma^{k} r_{t+k+2} | s_t = s, a_t = a \right] \\
&= \sum_{s'} \wp_{ss'}^{a} \left( \Re_{ss'}^{a} + \gamma \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^{k} r_{t+k+2} | s_{t+1} = s' \right] \right) \\
&= \sum_{s'} \wp_{ss'}^{a} \left( \Re_{ss'}^{a} + \gamma \sum_{a'} \pi(s', a') Q^{\pi}(s', a') \right)
\end{aligned}
$$

The action value function can be rewrite as:

$$
Q^{\pi}(s, a) = E_{\pi}[R_t | s_t = s, a_t = a]
\quad \text{(5)}
$$

Therefore, the Bellman Equation enables expressing values of a specific state as values of other states. This makes the calculation of values between states become much simpler. This opens a lot of possibilities for iterative approaches for calculating the value for each state (Graves 2017). So, the Bellman Equation plays an important role in the inception of RL.

## 3.2. Graves and Whinston’s Algorithm (1970)- A Prototype of RL

Graves and Whinston’s Algorithm for solving the TSP was introduced in 1970 (Graves and Whinston 1970). This algorithm used the Bellman Equation and statistical properties of the criterion function. It could be considered as the prototype of Reinforcement Learning, which calculated the mean and variance to serve as a value function. Graves and Whinston’s Algorithm achieved one of the shortest distance on the TSP among the existing algorithms in that period.

### 3.2.1. Computational Scheme

The TSP is considered as an optimal permutation problem. The algorithm is attempting to discover the optimal mapping between the set S and R, where S consists of variables $(x_1, ..., x_n)$ and R consists of integers from $(1, ..., n)$.

The process is stated below:

1. Initialize by setting $k = 1$ and determine $S_1$ and $R_1$ according to a specific rule (eg. feasibility).
2. If $R_k$ is empty, go to step 8.
3. Take elements $i^* \in S_k$ and $j^* \in R_k$ arbitrarily. Let $i_k = i^*$, $j_k = j^*$, and $R_k = R_k \setminus \{j^*\}$.
4. If $k = n$, go to step 12.
5. Increment k by 1, $k = k + 1$.
6. Determine $S_k$ and $R_k$ according to some specified rule.
7. Repeat from step 2.

<!-- Page 5 -->
8. If $k = 1$, stop

9. Decrement $k$ by 1, $k = k - 1$.

10. If $R_k$ is empty, go back to step 8.

11. Take an arbitrary element $j^* \in R_k$, let $j_k = j^*$ to go with the current $i_k$ and set $R_k = R_k \setminus \{j^*\}$. Then go back to step 4

12. Record the current mapping, and go to step 10.

In this algorithm, as a criterion for selecting a path, is the value of the associated mean completion.

$$
\mathbb{E}[\phi(p), p \in C(i_1, ..., i_k)]
$$

For the $k^{th}$ selection, the algorithm will pick the path to the city which has the smallest mean among all unvisited cities. The mean can be calculated as follow:

$$
M = \frac{1}{n}(\sum_{i \in N a_i})(\sum_{j \in N b_j})
$$

And the variance can be written as:

$$
\frac{1}{n!}\sum_p[(\sum_j a_j b_{p(j)})(\sum_i a_i b_{p(i)})] \\
= \frac{(n-2)!}{n!}[(\sum_i a_i)^2 - \sum_t a_t^2][(\sum_k b_k)^2 - \sum_t b_t^2] + \frac{(n-1)!}{n!}(\sum_i a_i^2)(\sum_j b_j^2)
$$

Repeated examinations with the selection procedure indicates that while an optimal solution does not generally result from the first complete assignment, a very good solution is achieved. Then, the algorithm turns to the explicit computation of the mean and variance value of the completion of a k-partial map.

### 3.2.2. Implicit Enumeration

Define $\Psi(S)$ is a set that consists all the subsets of the set S. Let $L_\phi : \Psi(S) \to R$ and $L_\phi(E) \leq \phi(i), \forall i \in E$ where $E \subset S$. The functions give a lower bound for $\phi(i)$ over a subset E. Then, let $A$ denote value of the current best completed assignment, where $A = \infty$ if there is no complete assignment yet. If $L_\phi(C(i_1, ..., i_k)) \geq A$, there is no k-partial mapping from i to j consists a better solution than the current mapping. The completion class $C(i_1, ..., i_k)$ is *implicitly enumerated*.

To embed implicit enumeration into the algorithm, replace Step 11 that is presented in Section 3.2.1 by the following:

11. Take an arbitrary element $j^* \in R_k$, let $j_k = j^*$ to go with the current $i_k$ and set $R_k = R_k \setminus \{j^*\}$. Compute $L_\phi(C(i_1, ..., i_k))$. If $L_\phi(C(i_1, ..., i_k)) \geq A$ go to step 9, otherwise go back to step 4.

Define $G_\phi : P(S) \to R$ be $\alpha$-probabilistic lower bound set function for any given $\phi$ and $\alpha$. In terms of the overall algorithm, the $\alpha$-probabilistic lower bound functions $G_\phi$ are employed as the lower bound functions $L_\phi$ in Step 11 presented above. Which means, if

$$
G_\phi(C(i_1, ..., i_k)) \geq A
$$

<!-- Page 6 -->
then completion class $C(i_1, ..., i_k)$ is implicitly enumerated at the $\alpha$ confidence level. The computational experience presented in the paper indicates that the $\alpha$-probabilistic lower bound function gives much greater cutting power than the lower bound functions without substantial risk of overlooking the true minimum. It has been proved to achieve a complete confidence level enumeration with large practical problems.

### 3.2.3. Evaluation

The algorithm is illustrated from three classical TSP from Karg and Thompson (Karg, Thompson 1964). The three problems are 33-city, 42-city and 57-city, which are considered large scale problems. The outcomes of Grave and Whinston’s Algorithm and the optimal solution of the problems are presented in the table below. The optimal solution is retrieved from Karg and Thompson’s paper. The results show that the approximations are within 15% away from the optimal solution, moreover, the approximation will move closer to the optimal solution when the problem scale up.

|                           | 33-city | 42-city | 57-city |
|---------------------------|---------|---------|---------|
| Grave and Whinston’s Approximation | 12,406  | 707     | 13,159  |
| Optimal Solution          | 10,861  | 699     | 12,995  |

### 3.2.4. Analysis

The Grave and Whinston’s Algorithm reveals some characteristics of Reinforcement Learning. The algorithm utilizes the Bellman Equation for approximating the optimal solution of the partial TSP. Bellman equation is the basic block of solving reinforcement learning and is omnipresent in modern Reinforcement Learning (Tanwar 2019). In Grave and Whinston’s Algorithm, the Bellman Equation is employed to calculate the optimal solution of the prevailing k-partial TSP. The calculation is built upon the recorded optimal solution of previous (k - 1)-partial TSP. Then, the determination of k-partial mapping will serve as the "environment" for the subsequent (k+1)-partial mapping. The mean and variance in Section 3.2.2 serve as the value function in the modern reinforcement learning architecture. The key to solving the TSP is to minimize the mean distance of each k-partial mapping. Therefore, a large mean value will diminish the probability of taking a specific choice at the current step.

Unmistakably, there exist many distinctions between the approach from Grave and Whinston and modern reinforcement learning, as a result of the inadequacy of computing power in the 1960s. There is no adjustment on the network in the training procedure in Grave and Whinston’s architecture. The "learning" component in the model is not significant.

## 3.3. Ant-Q (1995)- A Classical RL Approach

The Ant-Q Algorithm was introduced by Luca M. Gambardella and Marco Dorigo in 1995, which presents many similarities with Q-learning (Watkins 1989). This algorithm was inspired by ant system (Dorigo, Maniezzo, and Colorni 1996), which is a distributed algorithm for combinatorial optimization. It was a notable algorithm that applied reinforcement learning on the

<!-- Page 7 -->
TSP. The Ant-Q algorithm gained competitive results on both symmetric and asymmetric TSP (Gambardella and Dorigo 1995).

### 3.3.1. Ant System (AS)

Ant System (AS) is the first Ant Colony Optimization algorithm, developed by Marco Dorigo (Gambardella and Dorigo 1995). The Ant Colony Optimization algorithm (ACO), also introduced by Dorigo, is a probabilistic technique for solving computational problems which can be reduced to finding good paths through graphs. Ant System uses an artificial ant— a computational agent— to find good solutions to graph-related optimization problems. The optimization problem that ACO can apply on is equivalent to the TSP, which finds the shortest path on a weighted graph. In the ACO algorithm, each artificial ant selects a path to constructs a solution arbitrarily for every iteration. Then, the solutions generated by the ants are compared and evaluated. The network will be adjusted based on the evaluations.

### 3.3.2. Q-Learning

Q-Learning is a reinforcement learning algorithm, which learns and records policies and takes action to maximize the expectation. The letter ‘Q’ stands for the quality of actions that are taken. Q-Learning is model-free, so that it does not require to build environmental networks. The algorithm centralizes on the quality of state-action combinations:

$$
Q: S \times A \to \mathbb{R}
$$

In the beginning, $Q$ is initialized randomly. At each time step $t$, action $a_t$ associates with a transformation cost/reward to the next stage $s_{t+1}$, the cost/reward is denoted as $r_t$. The quality $Q$ is updated by utilizing the Bellman Equation (Quality Value Function presented in Section 3.1):

$$
Q_{new}(s_t, a_t) = Q_t(s_t, a_t) + \alpha(r_t + \lambda \cdot max\{Q(s_{t+1}, a)\} - Q(s_t, a_t))
$$

where $\alpha$ is the learning rate and $\lambda$ is the discount factor.

### 3.3.3. Ant-Q Algorithm

Ant-Q Algorithm can be described as follows. For each iteration, there are four steps:

1. Initialize AQ-values, each agent $k$ is placed on a city $r_{k1}$ according to some policy. Initialize a set of to-be-visited cities $J_k(r_{k1})$.

2. Each agent makes a move and updates $AQ(r,s)$ if the move discounts the next state evaluation. The agents repeat moving and updating AQ-values until they are back to the starting city.

3. The length $L_k$ of the tour done by agent $k$ is computed, and $L_k$ is used to calculate the delayed reinforcements $\Delta AQ(r,s)$. Then, update the AQ-values base on $\Delta AQ(r,s)$.

4. Check whether the pre-defined termination condition is met. Return the approximated shortest path $L_k$.

The AQ-values are updates by the following rule:

$$
AQ(r,s) = (1 - \alpha) \cdot AQ(r,s) + \alpha \cdot (\Delta AQ(r,s) + \lambda \cdot Max_{z \in J_k(s)} AQ(s,z))
$$

where $J_k(s)$ is a function of the previous history of agent $k$. And $\Delta AQ(r,s)$ is calculated as follow:

$$
\Delta AQ(r,s) =
\begin{cases}
\frac{W}{L_k} & \text{if } (r,s) \in \text{tour done by agent } k \\
0 & \text{otherwise}
\end{cases}
$$

<!-- Page 8 -->
### 3.3.4. Evaluation

The approximation results of the Ant-Q algorithm are compared with the following approaches:

- Elastic Net: a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.
- Simulated Annealing: a probabilistic technique for approximating the global optimum of a given function. In the TSP, the difficulty with this approach is that while it rapidly finds a local minimum, it cannot get from there to the global minimum (Carr 2002).
- Self-Organizing Map: an artificial neural network for unsupervised learning.

There are five 50-city problems used in the comparison. The results show the Ant-Q algorithm achieves the minimum average distance among all the approaches in four of the five problems. This indicates reinforcement learning does have good performance in the TSP.

### 3.3.5. Analysis

The Ant-Q algorithm is a typical reinforcement learning algorithm that can be effectively applied to the TSP. It stores the AQ-values and utilizes the AQ-values to determine the optimal path. This is an effective model-free reinforcement learning algorithm. Therefore, the computations are not as complicated as the algorithms with network models.

However, the limitation on Q-learning also applies to the Ant-Q algorithm. A model-free algorithm is not compatible with a large number of environmental factors. The AQ-values are the only determining factors of the paths so that the algorithm is not able to take many states/actions into consideration. Therefore, the algorithm could possibly eliminate the path that is costly in a short run but efficient in a long run.

The limitation on Ant-Q is no longer a problem when the concept of deep reinforcement learning is introduced. Deep reinforcement learning integrates deep learning architectures (deep neural networks) with reinforcement learning algorithms (Q-learning, actor-critic, etc.) and is capable of scaling to previously unsolvable problems (Arulkumaran 2017).

## 3.4. REINFORCE (2019)- A Deep RL Approach

REINFORCE presents an idea to learn a heuristic for combinatorial optimization problems. The model is trained using a simple baseline based on a deterministic greedy rollout, which we find is more efficient than using a value function (Kool 2019). This algorithm significantly improve the performance on the TSP up to 100 cities.

### 3.4.1. Attention Model

takes graph structure into account by a masking procedure. The attention based encoder-decoder model defines a stochastic policy $p(\pi|s)$ for selecting a solution $\pi$ given a problem instance $s$. It is factorized and parameterized by $\theta$:

$$
p_\theta(\pi|s) = \prod_{t=1}^{n} p_\theta(\pi_t|s, \pi_{1:t-1})
$$

<!-- Page 9 -->
The encoder produces embeddings of all input nodes. Input nodes are embedded and processed by N sequential layers, each consisting of a multi-head attention layer and node-wise feed-forward sub-layer. The node embedding and graph embedding are produced as the inputs of the decoder. The decoder produces the sequence $\pi$ of input nodes. To produce the solution, a context vector that consists of the graph embedding of first, last and unvisited cities will be given to a decoder. The decoder will calculate the probability distribution of unvisited cities and output the next city to be visited.

### 3.4.2. Algorithm

The Attention Model obtains a solution (path) $\pi|s$ from a probability distribution $p_\theta(\pi|s)$. To train the model, the loss is defined as

$$
L(\theta|s) = \mathbb{E}_{p_\theta}[L(\pi)]
$$

The loss is adjusted by gradient descent, using the REINFORCE gradient estimator with baseline $b(s)$ (Williams 1992):

$$
\Delta L(\theta|s) = \mathbb{E}_{p_\theta}[(L(\pi) - b(s)) \Delta \log p_\theta(\pi|s)]
$$

With the greedy rollout as baseline $b(s)$, the function $L(\pi) - b(s)$ is negative if the sampled solution $\pi$ is better than the greedy rollout, causing actions to be reinforced. This way the model is trained to improve over itself.

### 3.4.3. Evaluation

For the TSP, the algorithm is compared with Nearest Insertion, Random Insertion, Farthest Insertion, as well as Nearest Neighbor.

- **Nearest Insertion** inserts the node $i$ that is nearest to the tour:
  $$
  i^* = \arg\min_{i \notin S} (\min_{j \in S} (d_{ij}))
  $$

- **Farthest Insertion** inserts the node $i$ so that the distance of the tour is maximized:
  $$
  i^* = \arg\max_{i \notin S} (\min_{j \in S} (d_{ij}))
  $$

- **Random Insertion** inserts a random node.

- **Nearest Neighbor** heuristic represents the partial solution as a path with a starting and ending node.

They are compared using 20, 50, and 100-city TSP, while the REINFORCE algorithm obtains the best performance among all the selected algorithms within a relatively short period of time.

### 3.4.4. Analysis

This algorithm presents an approach to utilizing RL techniques on the self-training of the attention model, or graph attention network. The attention model provides a probabilistic approach to estimating the path base on the given environment. The attention model is focusing on solving sub-problems and merging the outcomes in a probabilistic manner to form the final result. Thus it can reduce the overall computation complexity by adjusting the parameters in the attention network to set concentrations (sub-graph with high probability scores).

<!-- Page 10 -->
REINFORCE integrates deep neural networks with RL. They use a roll-out network to deterministically estimate the difficulty of the instance, and periodically update the roll-out network with the parameters of the policy network (Rivlin 2019). This algorithm achieves high performance in small scale problems (up to 100-city problem), but not capable of very large problems. The deterministic greedy roll-out reduces the complexity of the problem. However, as the scale of the problem increase, the roll-out will limit the potential to approach the optimal solution.

More recently, researchers train graph convolutional networks using a probabilistic greedy mechanism to predict the quality of a node and embed the Q-Learning framework into the network (Manchanda 2020). This more recent algorithm largely reduces the computation complexity and is capable of million-city problems.

## 4. Discussion

Through analyzing three RL approaches to the TSP, we argue that RL is a good technique to solve combinatorial optimization problems. All of the three algorithms we reviewed in Section 3 achieved good performance among the algorithms developed in those time periods. Taking into account the selection bias when comparing algorithms, we could at least argue that the RL approach in combinatorial optimization achieves above-average performance.

In addition to the performance, the RL approach has its own strength. In the modern RL algorithms that are introduced in the recent five years, there is no human knowledge required by those models. Which means the RL model starts from completely arbitrary values/states. After deep RL is widely used in approximating combinatorial optimization, the quality and capability of the RL approach are raised.

The potential of the RL approach approximation is greater than approximating arithmetically or algorithmically. With the rapid expansion of computing power, the training time and the quantity of training data will be less taken into consideration. Then, the performance and capability of the RL approach approximation will be enhanced continuously. This trend may be significant when quantum computing is widely used in machine learning.

### 4.1. Future Research

With the expansion of computing power, the RL network for combinatorial optimization gradually will grow more intricate in order to advance the performance. We intend to obtain a performance-complexity balance, where a comparatively good solution is found with less computation effort.

In the Graves and Whinston’s Algorithm presented in Section 3.2, they made decisions based on the combination of mean and variance (the calculation is shown in Section 3.2.1) and their decision on the next city would minimize the combination of mean and variance. A possible enhancement is embedding a simple feed-forward neural network.

The value function of Graves and Whinston’s algorithm could be updated to the following form:

<!-- Page 11 -->
$Q = \alpha \cdot \mu + \beta \cdot \sigma^2$

where $\mu$ and $\sigma^2$ represent the mean and variance, $\alpha$ and $\beta$ are the weights associated with $\mu$ and $\sigma^2$. $\alpha$ and $\beta$ are the constants that will be trained by the neural network during the training procedure. The loss can be calculated using the optimal distance in the sub-graph and the distance of the current decision. Then, we can apply stochastic gradient descent to adjust $\alpha$ and $\beta$.

The mean and variance can form a Gaussian distribution. Based on the given TSP structure and current state, the Gaussian distribution for choosing the path is adjusted. The network can generate various probability distributions that are used to determine the path.

This approach could lead to a much smaller computation scale compare to deep RL models in recent years. And the baseline performance will be the results presented in Graves and Whinston’s paper, therefore, the improved performance will be guaranteed.

## References

Anstreicher, K.M. 2003. *Recent advances in the solution of quadratic assignment problems.*. Mathematical Programming Series B 97, 27 - 42.

Koopmans, T. C. and M. J. Beckmann. 1957. *Assignment problems and the location of economic activities.*. Econometrica 25, 53 - 76.

Çela, E. 1998. *The Quadratic Assignment Problem: Theory and Algorithms.*. Kluwer Academic Publishers, Dordrecht.

Rainer E. burkard. 2013. *The Quadratic Assignment Problem: Theory and Algorithms.*. Handbook of Combinatorial Optimization pp 2741-2814.

Panel Christos H.Papadimitriou. 1977. *The Euclidean travelling salesman problem is NP-complete.*. Theoretical Computer Science, Volume 4, Issue 3, Pages 237-244

Vincent Francois-Lavet, Peter Henderson, Riashat Islam, Marc G. Bellemare, Joelle Pineau. 2018. *An Introduction to Deep Reinforcement Learning*. arXiv preprint at arXiv:1811.12560 [cs.LG].

Van Otterlo, M.; Wiering, M. 2012. *Reinforcement learning and markov decision processes*. Reinforcement Learning. Adaptation, Learning, and Optimization. 12. pp. 3–42. doi:10.1007/978-3-642-27645-3_1. ISBN 978-3-642-27644-6.

Kaelbling, Leslie P.; Littman, Michael L.; Moore, Andrew W. (1996). *Reinforcement Learning: A Survey*. Journal of Artificial Intelligence Research. 4: 237–285. arXiv:cs/9605103. doi:10.1613/jair.301. Archived from the original on 2001-11-20.

Bellman, R.E. 1957. *Dynamic Programming*. Princeton University Press, Princeton, NJ. Republished 2003: Dover, ISBN 0-486-42809-5.

Kirk, Donald E. (1970). *Optimal Control Theory: An Introduction*. Prentice-Hall. p. 55. ISBN 0-13-638098-0.

<!-- Page 12 -->
Graves, Josh. 2017. *Understanding RL: The Bellman Equations* [Blog post]. Retrieved from joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/

G.W. Graves, A.B. Whinston. 1970. *An Algorithm For The Quadratic Assignment Problem*. Management Science. Vol 17. pg 453.

Robert L. Karg and Gerald L. Thompson. 1964. *A Heuristic Approach to Solving Travelling Salesman Problems*. Management Science. Vol. 10, No. 2 (Jan., 1964), pp. 225-248

Sanchit Tanwar. 2019. *Bellman Equation and dynamic programming*. Medium [Blog post]. Retrieved from medium.com/analytics-vidhya/bellman-equation-and-dynamic-programming-773ce67fc6a7

Luca M. Gambardella, Marco Dorigo. 1995. *Ant-Q: A Reinforcement Learning approach to the TSP*. Proceedings of the Twelfth International Conference on Machine Learning, Tahoe City, California, July 9–12, 1995. Pages 252-260

Christopher J. C. H. Watkins. 1989. *Learning with delayed rewards*. Ph. D. dissertation, Psychology Department, University of Cambridge, England.

M. Dorigo ; V. Maniezzo ; A. Colorni. 1996. *Ant system: optimization by a colony of cooperating agents*. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) ( Volume: 26 , Issue: 1 , Feb 1996 ). Page 29 - 41

Arulkumaran, K.; Deisenroth, M. P.; Brundage, M.; Bharath, A. A. November 2017. *Deep Reinforcement Learning: A Brief Survey*. IEEE Signal Processing Magazine. 34 (6): 26–38.

Carr, Roger. 2002. “Simulated Annealing.” From MathWorld–A Wolfram Web Resource, created by Eric W. Weisstein. https://mathworld.wolfram.com/SimulatedAnnealing.html

Or Rivlin. 2019. *Reinforcement Learning for Combinatorial Optimization*. Towards Data Science [Blog post]. Retrieved from https://towardsdatascience.com/reinforcement-learning-for-combinatorial-optimization-d1402e396e91

Sahil Manchanda, Akash Mittal, Anuj Dhawan, Sourav Medya1, Sayan Ranu, Ambuj Singh. 2020. *Learning Heuristics over Large Graphs via Deep Reinforcement Learning*. arXiv:1903.03332v3

Wouter Kool, Herke van Hoof, Max Welling. 2019. *ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!* ICLR 2019.

Ronald J Williams. 1992. *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine learning, 8(3-4):229–256.