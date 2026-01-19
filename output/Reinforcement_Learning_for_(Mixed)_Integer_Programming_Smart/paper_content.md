<!-- Page 1 -->
# Reinforcement Learning for (Mixed) Integer Programming: Smart Feasibility Pump

Meng Qi$^{*1}$ Mengxin Wang$^{*1}$ Zuo-Jun (Max) Shen$^1$

## Abstract

Mixed integer programming (MIP) is a general optimization technique with various real-world applications. Finding feasible solutions for MIP problems is critical because many successful heuristics rely on a known initial feasible solution. However, it is in general NP-hard. In this work, we propose a deep reinforcement learning (DRL) model that efficiently finds a feasible solution for a general type of MIPs. In particular, we develop a smart feasibility pump (SFP) method empowered by DRL, inspired by Feasibility Pump (FP), a popular heuristic for searching feasible MIP solutions. Numerical experiments on various problem instances show that SFP significantly outperforms the classic FP in terms of the number of steps required to reach the first feasible solution. We consider two different structures for the policy network. The classic perception (MLP) and a novel convolution neural network (CNN) structure. The CNN captures the hidden information of the constraint matrix of the MIP problem and relieves the burden of calculating the projection of the current solution as the input at each time step. This highlights the representational power of the CNN structure.

## 1. Introduction

Integer programming (IP) and mixed integer programming (MIP) are mathematical optimization problems where all or some of the decision variables are restricted to be integers. IPs and MIPs are critical for solving discrete and combinatorial optimization problems in various applications, including production planning, scheduling, and vehicle routing (we refer to (Pochet and Wolsey, 2006), (Sawik, 2011), and (Malandraki and Daskin, 1992) for more details). The study of theory and computation methods dates back to several decades ago ((Dantzig et al., 1954), (Land and Doig, 2010), and (Padberg, 1973)).

Despite the importance of solving IPs and MIPs in application, they are generally very difficult to solve in theory (NP-hard) and in practice. There is no polynomial time algorithm that can guarantee to solve a general IP/MIP. Besides exact methods (Branch-and-bound, branch-and-cut, etc.), heuristic algorithms are widely adopted for their simplicity and efficiency. Many successful heuristics including local branching (Fischetti and Lodi, 2003) and RINS (Danna et al., 2005), require an initial feasible solution to start with. Finding a feasible solution is often the first step and one of the most critical steps of solving IP/MIPs. However, it is also NP-hard in general; even well-developed commercial solvers, such as CPLEX, may struggle or even fail.

In this study, our goal is to provide a deep reinforcement learning (DRL) model that efficiently finds a feasible solution for a general type of IP/MIPs. In particular, we consider IP/MIP problems with a linear objective, linear constraints, and integral constraints. Such formulation is quite general since any combinatorial optimization problem with finite feasible regions can be reformulated as an IP/MIP in this form (Tang et al., 2020). We aim at developing a DRL agent that smartly perform the idea of *Feasibility Pump*, which is one of the most well-known algorithms for finding a feasible solution for MIPs in optimization theory.

The main idea of FP is to decompose the original problem into two problems: one focuses on the feasibility of a continuous relaxation and the other retains integrality. The original work of FP was introduced by (Fischetti et al., 2005) for $0-1$ MIPs. The authors demonstrated that FP is very effective in finding feasible solutions. Later, (Bertacco et al., 2007) extended this algorithm to general MIPs and introduced a random component in the rounding function. Further extensions of FP include the following. (Fischetti and Salvagnin, 2009) adjusted the rounding method so that a feasible solution can be found in fewer steps. (Baena and Castro, 2011) and (Boland et al., 2014) investigated the idea of using integral reference points to further improve the

---

$^*$Equal contribution  
$^1$Department of Industrial Engineering and Operations Research, UC Berkeley. Correspondence to: Meng Qi <meng.qi@berkeley.edu>, Mengxin Wang <mengxin_wang@berkeley.edu>.

*Reinforcement Learning for Real Life (RL4RealLife) Workshop in the 38th International Conference on Machine Learning, 2021. Copyright 2021 by the author(s).*

<!-- Page 2 -->
rounding steps. (Achterberg and Berthold, 2007) proposed the objective feasibility pump, which includes the original objective in the objective function of FP to improve the quality of the feasible solution. (Boland et al., 2012) proposed a penalty system for non-integrality to prevent the FP from cycling. (Bonami et al., 2009) and (Bonami and Gonçalves, 2012) extended the FP algorithm to nonlinear MIPs.

Although this heuristic is widely adopted, it requires solving an optimization problem within each iteration, which is especially inefficient when extending to nonlinear cases. The efficiency may also be weakened because FP may loop between infeasible points.

Recently, several studies have been conducted exploring the topic of using machine learning methods to solve IP/MIPs. The main drawback of adopting supervised learning schemes is that they require known (near) optimal solutions in the training set, which is difficult to achieve for IP/MIPs. One of the most closely related work is (Fischetti and Jo, 2017), where they trained deep neural networks to find a feasible solution for 0-1 MIPs.

Some other works have aimed to use machine learning methods for branch-and-bound algorithms. (He et al., 2014) used imitation learning to learn an adaptive node searching order to achieve better solutions faster for MIPs. (Khalil et al., 2016) learned an easy-to-evaluate surrogate function that mimics the strong branching strategy for branch-and-bound. (Nair et al., 2020) proposed the methods of neural diving and neural branching, to improve the high-quality joint variable assignment and bounding the gap, respectively.

Other related works have studied combinatorial problems, which are a subset of IPs. The most related works include (Khalil et al., 2017), where the authors proposed a method based on RL to solve various combinatorial problems on graphs. (Bello et al., 2016) proposed a framework based on recurrent neural network (RNN) and deep reinforcement learning (DRL) to general combinatorial problems. (Nazari et al., 2018) proposed a simplified version of RNN and DRL to tackle the vehicle routing problem (VRP). (Li et al., 2018) combined deep learning techniques with a graph convolutional network to solve combinatorial problems based on graphs. (Delarue et al., 2020) investigated VRP and developed a framework for a value-function-based RL model where the action selection problem is formulated as a mixed-integer optimization problem. (Tang et al., 2020) designed RL methods to enhance the cutting plane method for solving IPs.

So far to our knowledge, there is no existing work that focuses on feasibility study for general MIPs using RL models. We believe that RL models can be utilized to learn feasible solutions for a general type of MIPs.

**Our contributions.** In this work, we propose a general framework, the *smart feasibility pump* (SFP) model, which utilizes the RL frameworks based on the spirit of FP. More specifically, we propose two different SFP models: SFP based on multi-layer perception (SFP-MLP) and SFP based on convolutional neural network (SFP-CNN), both find feasible solutions for IP/MIPs more efficiently, that is, with fewer search steps. Moreover, to better capture the structure of the constraint matrix, SFP-CNN adopts a novel CNN structure for policy learning networks that significantly improves the performance of SFP-CNN compared to SFP-MLP and the original FP. We summarize the contribution of our work as follows:

- **A RL model for feasible solutions of MIPs.** Different from existing literature which mostly focus on one specific application and aim for the optimal solution, our work is the first attempt to use (deep) RL methods for seeking feasible solutions for a class of general MIPs.

- **The spirit of a successful heuristic.** Our model is inspired by FP, one of the most popular heuristic algorithms for finding feasible solutions for IP/MIP. The effectiveness of decomposing the continuous relaxation and integrality has been verified by the success of FP. Adopting a similar idea, we propose the smart feasibility pump model because it is empowered by deep RL models.

- **A novel CNN for constraint matrix.** Besides a regular MLP, we innovatively adopt a convolutional structure for the policy network to capture the structure of constraint matrix of MIPs. The CNN provides us with a stronger representational power to capture the hidden structure of the constraint matrix.

- **Empirical evaluation.** We conduct numerical experiments with different problem dimensions and number of constraints. The results demonstrate the significant advantages of the SFP models compared to the original FP.

## 2. Background

### Mixed Integer Programming.

In this work, we aim at finding a feasible solution of a generic Mixed Integer Programming (MIP) problem. A generic MIP problem can be written as follows:

$$
\begin{aligned}
\min \quad & c^T x \\
\text{s.t.} \quad & Ax \leq b \\
& x_i \in \mathbb{Z}, \forall i \in S
\end{aligned}
\tag{1a}
\tag{1b}
\tag{1c}
$$

where $A \in \mathbb{R}^{m \times n}$, $c, b \in \mathbb{R}^n$, and $x \in \mathbb{R}^n$ denote the decision variables. A subset of the decision variables is

<!-- Page 3 -->
integral, and $I$ denotes the index set of the integer variables. When $I = \{1, 2, ..., n\}$, it becomes an IP problem. The MIP problem aims to minimize a linear objective function (1c) while satisfying a set of linear constraints (1b) and the integral constraints (1c). A feasible solution is an element in the feasible region $\mathcal{P} = \{x \in \mathbb{R} | Ax \leq b, x_i \in \mathbb{Z}, \forall i \in S\}$. To avoid triviality, we focus on the case when $\mathcal{P}$ is bounded and excludes the origin.

Solving an MIP problem is generally known to be NP-hard. Even well-developed commercial solvers (for instance, Cplex, Gurobi, and Mosek) may struggle or fail. Many successful heuristics for solving MIPs, such as local branching (Fischetti and Lodi, 2003) and RINS (Danna et al., 2005), only work with a known initial feasible solution. It is of great importance to find a good feasible solution for MIPs. However, finding a feasible solution for a MIP is also NP-hard, owing to the well-known equivalence of the optimization problem and the feasibility problem in terms of computational complexity.

## Feasibility Pump

The feasibility pump is one of the most popular approaches for finding the initial feasible solution of MIPs. The basic idea of the FP algorithm is to iteratively find and round a continuous relaxation solution for the MIP. Let $\mathcal{P}_R = \{x \in \mathbb{R} | Ax \leq b\}$ denote the continuous relaxation of $\mathcal{P}$. The overall procedure is shown in Algorithm 1 and illustrated in Figure 1. The FP algorithm starts with the rounded optimal continuous relaxation solution of the MIP and then searches for the nearest points in the relaxed feasible region. It continues perturbing and rounding the new point found at each step until a feasible solution is discovered or the limit of maximum number of steps is reached. Despite being a powerful heuristic, it requires solving an optimization problem within each iteration, which becomes especially inefficient when the problem size increases or extends to nonlinear constraint cases. In addition, the FP algorithm tends to converge to and loop between infeasible points with non-binary integer variables (Fischetti et al., 2005).

### Algorithm 1 Feasibility Pump

With a slight abuse of notation, $[x]$ denotes the rounding of $x$, i.e. $[x]_i = [x_i], \forall i \in I$ and $[x]_i = x_i \forall i \notin I$. The algorithm terminates if reaching the maximum number of iterations with no feasible solution found.

```
1: Initialization: $x^0 \leftarrow \arg\min_{x \in \mathcal{P}_R} c^T x$; $\bar{x}^0 \leftarrow [x^0]$; $k \leftarrow 0$
2: while $\bar{x}^k$ is not feasible do
3:     $x^{k+1} \leftarrow \arg\min_{x \in \mathcal{P}_R} \|x - \bar{x}^k\|$
4:     $\bar{x}^{k+1} \leftarrow [x^{k+1}]$
5:     if $\bar{x}^{k+1} = \bar{x}^k$ then
6:         random perturbation of $\bar{x}_j^k, \forall j \in I$.
7:     else
8:         $k \leftarrow k + 1$
9:     end if
10: end while
11: Return $\bar{x}^k$
```

**Figure 1**: Illustration of the Feasibility Pump

---

## 3. Smart Feasibility Pump: a RL Formulation

Here, we present our formulation of the SFP method as an RL problem. We aim to learn an RL policy that can find a feasible solution for any randomly generated IP/MIPs with the form of (1).

### 3.1. RL Formulation

In this section, we describe two slightly different formulations for the SFP-MLP and the SFP-CNN. For each of the formulations, we start by specifying the Markov decision process. At each time step $t$, we denote the state as $s_t \in \mathcal{S}$, the action as $a_t \in \mathcal{A}$, and the instant reward as $r_t \in \mathbb{R}$. A policy $\pi : \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})$ provides a mapping from any state to a distribution over actions $\pi(\cdot | s_t)$. Our goal is to learn a policy that maximizes the expected cumulative rewards over a horizon $T$, that is, $\max_\pi J(\pi) := \mathbb{E}_\pi[\sum_{t=0}^{T-1} \gamma^t r_t]$, where $\gamma \in (0, 1]$ is a discount factor. In the remaining of this section, we specify the state space $\mathcal{S}$, action space $\mathcal{A}$, reward $r_t$, and the transition from $s_t$ to $s_{t+1}$.

#### State Space $\mathcal{S}$ for SFP-MLP

In the case of MIP, we let the state at each time $t$ to be $s_t = (\text{Flat}(A), b, x_t, \tilde{x}_t, I)$, $\tilde{x}$ denotes the projection of the current solution to the feasible region defined by constraints (1b). The idea of including a projection $\tilde{x}_t$ is borrowed from the original FP algorithm, which provides useful information for the agent about the direction to move to achieve feasibility. It improves the learning ability of the agent yet with the cost of large computational efforts at each time step. Moreover, $\text{Flat}(A)$ is a vector that equals the flattened constraint matrix $A$ in the original problem (1a)-(1c), and $I$ is a binary vector that indicates whether the $i$th variable is an integer (whether $i \in I$). For IPs, as we know that all decision variables in (1a)-(1c) are integers, we set the state vector as

<!-- Page 4 -->
$s_t = (\text{Flat}(A), b, \tilde{x}, x)$.

As explained later in Section 3.2, if we use our proposed network structure with CNN, we can avoid calculating the projection of $x_t$ at each time step $t$ and instead use the initial solution $x_0$ as $\tilde{x}_t$ for all time steps.

**State Space $\mathcal{S}$ for SFP-CNN.** When we adopt our proposed CNN as the policy network, because of the ability of the CNN to capture the structure of the constraint matrix, the learning ability of the RL agent is brought up to a level that we can get rid of the projection of $x_t$ at each time step $t$. Instead, it only requires the projection to be performed once for the initial point $x_0$ and use it as $x_t$ for all $t > 0$. Therefore, for MIPs, we defined the state vector as $s_t = ([A, b], x_t, \tilde{x}_0, I)$, where $[A, b]$ denotes the matrix defined as matrix $A$ concatenated with column vector $b$. Similarly, for IPs, we have $s_t = ([A, b], x_t, \tilde{x}_0)$.

**Action Space and State Transition.** In each time step $t$, the agent draws an action $a_t$ as a movement of the current solution $x_t$, that is, $a_t \in \mathcal{A} = \mathbb{R}^n$. Then, $x_t$ is updated by rounding $x_t + a_t$, that is, $x_{t+1} = [x_t + a_t]$ and $x_{t+1}$ is integral but not guaranteed to satisfy the constraints (1b). The transition of state is deterministic and $s_{t+1} = (\text{Flat}(A), b, \tilde{x}_{t+1}, x_{t+1}, I)$ (for IP, $s_{t+1} = (\text{Flat}(A), b, \tilde{x}_{t+1}, x_{t+1})$), where $\tilde{x}_{t+1}$ is the projection of $x_{t+1}$ to the feasible region defined by (1b).

**Reward.** As our goal is to find a feasible solution, we use the total violation of constraints (1b) as the reward, that is, $r_t = -\|Ax_t - b\|$. As in each step, $x_t$ is guaranteed to be integral, it is reasonable to only consider the violation of constraints (1b) rather than constraints (1c).

## 3.2. Policy learning

In this part, we demonstrate our RL model. To train the policy network, we use actor-critic algorithms with proximal policy optimization (PPO) as proposed in (Schulman et al., 2017). Moreover, two different policy network structures: the classic MLP and a novel CNN are adopted for the policy network of SFP-MLP and SFP-CNN, respectively.

**Actor-Critic with PPO.** For policy gradient methods, we use PPO for actor-critic. PPO is a family of policy gradient methods that use different surrogate objective functions compared to standard policy gradient methods. In particular, we use the clipped surrogate objective method that proposed in (Schulman et al., 2017), that is, we use the following objective function while training the policy network:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)],
\tag{2}
$$

where $\hat{A}_t$ is an estimator of the advantage function at time step $t$. We refer to (Schulman et al., 2017) for more details about the PPO methods.

## Policy network structure.

- **SFP-MLP.** As illustrated in Figure 2, in the SFP-MLP model, the policy network is an MLP that takes the state vector as input. The state vector consists of a flattened constraint matrix $A$, constraint vector $b$, current solution $x_t$, and its projection $\tilde{x}_t$, as well as the integral indicator vector $I$.

- **SFP-CNN.** To better characterize the constraint matrix $(A, b)$ in IP/MIP defined in (1), we use a CNN to capture the policy $\pi_\theta(a|s)$, instead of an MLP. The network structure is illustrated in Figure 3. The inputs of the neural network include two parts, where **Constr_Mat** represents the constraint matrix $[A, b]$ and **Current_Sols** represents the current solution $x$ and the initial solution $x_0$. In the case of MIPs, there is an additional input **I** that indicates the index of the integral variables. Unlike using MLP to fit the policy, we keep using the initial solution (which is the rounded optimal solution of a continuous relaxation) instead of the projection of the current solution. Therefore, there is no need to calculate the projection at each time step, which makes our method more computationally efficient compared to using MLP for policy fitting.

Figure 2: Policy Network Structure of SFP-MLP.

Figure 3: Policy Network Structure of SFP-CNN.

<!-- Page 5 -->
# Reinforcement Learning for (Mixed) Integer Programming: Smart Feasibility Pump

## 4. Evaluation

### 4.1. Experiment Design

**Simulator Design and Experiment Setting.** We randomly generate a set of IP and MIP problem instances of form (1) based on the parameters listed in Table 1. The problem instances are small, moderate and large in terms of the dimension of the decision variable $n$ and the number of constraints $m$, including a 5-dimensional decision variable with six constraints; 7-dimensional decision variable with nine constraints; and a 9-dimensional decision variable with 18 constraints. We implemented a simulation environment for the SFP agent. In each training iteration, the agent interacts with one of the generated problems starting at the optimal continuous relaxation. For each problem size, the agent is trained for 50 iterations. The maximum number of training steps is 100. The agent either finds a feasible solution within 100 steps or stops at the 100-th step. Hence, step $< 100$ means that a feasible solution has been found, and step $= 100$ means that the agent has failed for this sample. The same problem instances are tested on the original FP algorithm for comparison.

**Performance Evaluation Metric.** The reward per iteration measures that the violation of the constraints is not a direct metric of the model performance in the context of our application since our primary goal is to find a feasible solution within a small number of steps. We consider the number of steps to reach a feasible solution as our performance evaluation metric instead of the reward per iteration. In particular, we evaluated the empirical mean and standard deviation of the number of steps, which we denote as **EpLenMean** and **EpLenStd**, respectively. Note that in this setting, it is *not* always better to have a *lower* variance of the number of steps. Because our agent interacts with the environment for at most 100 steps, a poorly performed model may have EpLenMean = 100 but EpLenStd = 0 if it fails to find any feasible solution. An ideal model has both a low EpLenMean and a low EpLenStd. A model outperforms model with higher EpLenMean and lower EpLenStd because the latter one is stable with more steps to reach a feasible solution.

---

**Figure 4:** Comparison of SFP-MLP and SFP-CNN. In each of the subfigures (a)–(c), we demonstrate the performance comparison of SFP-MLP and SFP-CNN in the average number of steps (denoted by EpLenMean) and the standard deviation of the number of steps (denoted by EpLenStd) in each iteration. Note that the agent stops if a feasible solution is found or if it reaches the maximum number of steps (100). Therefore, in our setting, it is not always better to have a lower variance in the number of steps. Since the agent interacts with the environment for at most 100 steps, a poorly performed model might have EpLenStd = 0 if it fails to find any feasible solution. An ideal model has both a low EpLenMean and a low EpLenStd. A model is dominated by another model if it has a higher EpLenMean and a lower EpLenStd.

---

**Table 1:** Problem Parameters

| Parameter | Distribution |
|-----------|--------------|
| $A_{ij}$  | $\text{randint}[-10,10]$ |
| $b$       | $A\xi + \epsilon$, where $\xi_j \sim \text{randint}[1,10]$ and $\epsilon_j \sim \text{randint}[1,10]$ for all $j = 1, ..., n$ |
| $I$       | $\text{randint}[0,1]$ |

<!-- Page 6 -->
# 4.2. Empirical Results

## Experiment 1: Comparison with FP

In this section, we compare our proposed SFP agents with the classic FP algorithm (Algorithm 1). The results are summarized in Table 2 for the IP problems and Table 3 for the MIP problems. EpLenMax, 90 Quant and 10 Quant denote the maximum, 90 quantile and 10 quantile of the number of steps, respectively. The result shows that the SFP agent finds a feasible solution to IP/MIPs faster than the FP algorithm. In addition, SFP-CNN dominates the FP algorithm and SFP-MLP, especially when the problem size increases. Therefore, the SFP agent is much more computationally efficient than the classic FP algorithm.

## Experiment 2: MLP and CNN

In this section, we present the empirical evaluation results for the SFP-MLP and SFP-CNN methods. Figure 4 shows the training curves of SFP-MLP and SFP-CNN with different problem scales. The results show that SFP-MLP and SFP-CNN are comparable when the problem size is small (Figure 4a). Both EpLenMean and EpLenStd decreases when the training iteration increases, which is how an ideal model would perform. SFP-CNN outperforms SFP-MLP in the sense that it converges faster to a lower EpLenMean with comparable EpLenStd when the problem size becomes larger (Figures 4b and 4c). Figure 4c provides an additional comparison of the SFP-MLP without the projection of the previous point. Interestingly, we observe that the performance of SFP-MLP is largely dampened without the projection information, while the performance of SFP-CNN without projection is better than that of SFP-MLP with projection. Thus, SFP-CNN can be more computationally efficient than SFP-MLP with larger problem scales. This also highlights the representational power of the CNN structure to capture hidden information in the constraint matrices.

In summary, the SFP agents find a feasible solution for IP/MIPs efficiently and outperforms the classic FP algorithm. SFP-CNN dominates the classic FP algorithm and SFP-MLP. In addition, SFP-CNN works without the projection of the current point. As noted in Section 2, it requires solving an optimization problem at each step to obtain the projection. Therefore, SFP-CNN achieves a significant improvement over the original FP method. This highlights the representational power of our proposed CNN structure to capture hidden information in the constraint matrix.

# 5. Conclusion

In this work, we propose an SFP, a reinforcement learning-based model for finding feasible solutions to generic IP/MIP problems. Unlike the supervised learning scheme, our method does not require the knowledge of a training set that includes feasible solutions. Our model is constructed based on the spirit of a well-known heuristic, the FP. Numerical experiments on different problem scales show that our proposed SFP agents can efficiently find a feasible solution for general IP/MIPs and improve the performance compared to the original FP algorithm.

To capture the inherent structure of the constraint matrix, we propose a novel CNN structure for the policy network in addition to the MLP policy network. SFP-CNN outperforms SFP-MLP and is more computationally efficient because it does not require finding the projection of the current point at each step. The CNN structure exhibits great representation power for optimization problems.
```

---

**Tables Transcribed Below**

### Table 2: Comparison with Feasibility Pump (IP)

|              | $n = 5, m = 6$ |                |                | $n = 7, m = 9$ |                |                | $n = 9, m = 18$ |                |                |
|--------------|----------------|----------------|----------------|----------------|----------------|----------------|-----------------|----------------|----------------|
|              | FP             | MLP            | CNN            | FP             | MLP            | CNN            | FP              | MLP            | CNN            |
| EpLenMean    | 43.2           | 15.0           | 11.1           | 65.5           | 32.8           | 28.2           | 90.0            | 90.3           | 54.0           |
| EpLenStd     | 48.6           | 34.6           | 29.0           | 46.8           | 46.4           | 44.2           | 29.7            | 30.9           | 49.6           |
| EpLenMax     | 100.0          | 100.0          | 100.0          | 100.0          | 100.0          | 100.0          | 100.0           | 100.0          | 100.0          |
| 90 Quant     | 100.0          | 100.0          | 29.5           | 100.0          | 100.0          | 100.0          | 100.0           | 100.0          | 100.0          |
| 10 Quant     | 1.0            | 1.0            | 1.0            | 1.0            | 1.0            | 1.0            | 61.2            | 51.5           | 1.0            |

### Table 3: Comparison with Feasibility Pump (MIP)

|              | $n = 5, m = 6$ |                |                | $n = 7, m = 9$ |                |                | $n = 9, m = 18$ |                |                |
|--------------|----------------|----------------|----------------|----------------|----------------|----------------|-----------------|----------------|----------------|
|              | FP             | MLP            | CNN            | FP             | MLP            | CNN            | FP              | MLP            | CNN            |
| EpLenMean    | 31.9           | 13.1           | 12.7           | 57.0           | 25.0           | 27.2           | 82.7            | 70.5           | 60.8           |
| EpLenStd     | 45.5           | 32.5           | 31.3           | 48.2           | 42.6           | 42.6           | 36.8            | 46.0           | 48.3           |
| EpLenMax     | 100.0          | 100.0          | 100.0          | 100.0          | 100.0          | 100.0          | 100.0           | 100.0          | 100.0          |
| 90 Quant     | 100.0          | 100.0          | 100.0          | 100.0          | 100.0          | 100.0          | 100.0           | 100.0          | 100.0          |
| 10 Quant     | 1.0            | 1.0            | 1.0            | 1.0            | 1.0            | 1.0            | 2.0             | 1.0            | 1.0            |

<!-- Page 7 -->
We suggest several opportunities for future research in this direction. One interesting topic would be to leverage the CNN structure, which can transfer useful information between different problem settings with different numbers of constraints. The other potential direction would be to take the objective value into account and trying to find a feasible solution with better quality.

## References

Tobias Achterberg and Timo Berthold. Improving the feasibility pump. *Discrete Optimization*, 4(1):77–86, 2007.

Daniel Baena and Jordi Castro. Using the analytic center in the feasibility pump. *Operations Research Letters*, 39(5):310–317, 2011.

Irwan Bello, Hieu Pham, Quoc V Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning. *arXiv preprint arXiv:1611.09940*, 2016.

Livio Bertacco, Matteo Fischetti, and Andrea Lodi. A feasibility pump heuristic for general mixed-integer problems. *Discrete Optimization*, 4(1):63–76, 2007.

Natashia L Boland, Andrew C Eberhard, F Engineer, and Angelos Tsoukalas. A new approach to the feasibility pump in mixed integer programming. *SIAM Journal on Optimization*, 22(3):831–861, 2012.

Natashia L Boland, Andrew C Eberhard, Faramroze G Engineer, Matteo Fischetti, Martin WP Savelsbergh, and Angelos Tsoukalas. Boosting the feasibility pump. *Mathematical Programming Computation*, 6(3):255–279, 2014.

Pierre Bonami and João PM Gonçalves. Heuristics for convex mixed integer nonlinear programs. *Computational Optimization and Applications*, 51(2):729–747, 2012.

Pierre Bonami, Gérard Cornuéjols, Andrea Lodi, and François Margot. A feasibility pump for mixed integer nonlinear programs. *Mathematical Programming*, 119(2):331–352, 2009.

Emilie Danna, Edward Rothberg, and Claude Le Pape. Exploring relaxation induced neighborhoods to improve mip solutions. *Mathematical Programming*, 102(1):71–90, 2005.

George Dantzig, Ray Fulkerson, and Selmer Johnson. Solution of a large-scale traveling-salesman problem. *Journal of the operations research society of America*, 2(4):393–410, 1954.

Arthur Delarue, Ross Anderson, and Christian Tjandraatmadja. Reinforcement learning with combinatorial actions: An application to vehicle routing. *arXiv preprint arXiv:2010.12001*, 2020.

Matteo Fischetti and Jason Jo. Deep neural networks as 0-1 mixed integer linear programs: A feasibility study. *arXiv preprint arXiv:1712.06174*, 2017.

Matteo Fischetti and Andrea Lodi. Local branching. *Mathematical programming*, 98(1-3):23–47, 2003.

Matteo Fischetti and Domenico Salvagnin. Feasibility pump 2.0. *Mathematical Programming Computation*, 1(2-3):201–222, 2009.

Matteo Fischetti, Fred Glover, and Andrea Lodi. The feasibility pump. *Mathematical Programming*, 104(1):91–104, 2005.

He He, Hal Daume III, and Jason M Eisner. Learning to search in branch and bound algorithms. *Advances in neural information processing systems*, 27:3293–3301, 2014.

Elias Boutros Khalil, Pierre Le Bodic, Le Song, George L Nemhauser, and Bistra N Dilkina. Learning to branch in mixed integer programming. In *AAAI*, pages 724–731, 2016.

Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. In *Advances in neural information processing systems*, pages 6348–6358, 2017.

Ailsa H Land and Alison G Doig. An automatic method for solving discrete programming problems. In *50 Years of Integer Programming 1958-2008*, pages 105–132. Springer, 2010.

Zhuwen Li, Qifeng Chen, and Vladlen Koltun. Combinatorial optimization with graph convolutional networks and guided tree search. In *Advances in Neural Information Processing Systems*, pages 539–548, 2018.

Chryssi Malandraki and Mark S Daskin. Time dependent vehicle routing problems: Formulations, properties and heuristic algorithms. *Transportation science*, 26(3):185–200, 1992.

Vinod Nair, Sergey Bartunov, Felix Gimeno, Ingrid von Glehn, Pawel Lichocki, Ivan Lobov, Brendan O’Donoghue, Nicolas Sonnerat, Christian Tjandraatmadja, Pengming Wang, et al. Solving mixed integer programs using neural networks. *arXiv preprint arXiv:2012.13349*, 2020.

MohammadReza Nazari, Afshin Oroojlooy, Lawrence Snyder, and Martin Takac. Reinforcement learning for solving the vehicle routing problem. In *Advances in Neural Information Processing Systems*, pages 9860–9870, 2018.

<!-- Page 8 -->
# Reinforcement Learning for (Mixed) Integer Programming: Smart Feasibility Pump

Manfred W Padberg. On the facial structure of set packing polyhedra. *Mathematical programming*, 5(1):199–215, 1973.

Yves Pochet and Laurence A Wolsey. *Production planning by mixed integer programming*. Springer Science & Business Media, 2006.

Tadeusz Sawik. *Scheduling in supply chains using mixed integer programming*. Wiley Online Library, 2011.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

Yunhao Tang, Shipra Agrawal, and Yuri Faenza. Reinforcement learning for integer programming: Learning to cut. In *International Conference on Machine Learning*, pages 9367–9376. PMLR, 2020.