<!-- Page 1 -->
# Integrating prediction in mean-variance portfolio optimization

Andrew Butler and Roy H. Kwon  
University of Toronto  
Department of Mechanical and Industrial Engineering  
December 1, 2022

## Abstract

Prediction models are traditionally optimized independently from their use in the asset allocation decision-making process. We address this shortcoming and present a framework for integrating regression prediction models in a mean-variance optimization (MVO) setting. Closed-form analytical solutions are provided for the unconstrained and equality constrained MVO case. For the general inequality constrained case, we make use of recent advances in neural-network architecture for efficient optimization of batch quadratic-programs. To our knowledge, this is the first rigorous study of integrating prediction in a mean-variance portfolio optimization setting. We present several historical simulations using both synthetic and global futures data to demonstrate the benefits of the integrated approach.

**Keywords:** Data driven optimization, mean-variance optimization, regression, differentiable neural networks

## 1 Introduction

Many problems in quantitative finance can be characterized by the following elements:

1. A sample data set $\mathbf{Y} = \{\mathbf{y}^{(1)}, ..., \mathbf{y}^{(m)}\} = \{\mathbf{y}^{(i)}\}_{i=1}^m$ of uncertain quantities of interest, $\mathbf{y}^{(i)} \in \mathbb{R}^{d_y}$, such as asset returns.

2. A decision, $\mathbf{z} \in \mathbb{R}^{d_z}$, often constrained to some feasible region $\mathbb{S} \subseteq \mathbb{R}^{d_z}$.

3. An *objective (cost) function*, $c\colon \mathbb{R}^{d_z} \times \mathbb{R}^{d_y} \to \mathbb{R}$, to be minimized over decision variable $\mathbf{z} \in \mathbb{S}$ in the context of the observed realization $\mathbf{y}^{(i)}$.

For example, in portfolio management we are often presented with the following problem: for a particular observation of asset returns, $\mathbf{y}^{(i)}$, the objective is to construct a vector of assets weights, $\mathbf{z}^*(\mathbf{y}^{(i)})$, that minimizes the cost, $c(\mathbf{z}, \mathbf{y}^{(i)})$ and adheres to the constraint set $\mathbb{S}$. A common choice for cost is the Markowitz mean-variance quadratic objective [35], with typical constraints being that the weights be non-negative and sum to one. Of course, the realization of asset returns, $\{\mathbf{y}^{(i)}\}_{i=1}^m$, are not directly observable at decision time and instead must be estimated through associated feature data $\mathbf{X} = \{\mathbf{x}^{(1)}, ..., \mathbf{x}^{(m)}\}$, of covariates of $\mathbf{Y}$. Let $f\colon \mathbb{R}^{d_x} \times \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_y}$ denote the $\boldsymbol{\theta}$-parameterized prediction model for estimating $\hat{\mathbf{y}}$. In this paper, we consider regression prediction models of the form:

$$
\hat{\mathbf{y}}^{(i)} = f(\mathbf{x}^{(i)}, \boldsymbol{\theta}) = \boldsymbol{\theta}^T \mathbf{x}^{(i)},
$$

<!-- Page 2 -->
with regression coefficient matrix $\boldsymbol{\theta} \in \mathbb{R}^{d_x \times d_y}$.

In most applications, estimating $\hat{\mathbf{y}}^{(i)}$ requires solving an independent *prediction optimization problem* over the prediction model parameter $\boldsymbol{\theta}$. Continuing with the example above; in order to generate mean-variance efficient portfolios we must supply, at a minimum, an estimate of expected asset returns and covariances. A prototypical framework would first estimate the conditional expectations of asset returns and covariances by ordinary least-squares (OLS) regression and then ‘plug-in’ those estimates to a mean-variance quadratic program (see for example Goldfarb and Iyengar [24], Clarke et al. [16] Chen et al. [14]).

As exemplified above, prediction and decision-based optimization are often decoupled processes; first predict, then optimize. Indeed a perfect prediction model ($\hat{\mathbf{y}}^{(i)} = \mathbf{y}^{(i)}$) would invariably lead to optimal decision-making. In reality, however, prediction models rarely have perfect accuracy and as such an inefficiency exists in the ‘predict, then optimize’ paradigm; prediction models are estimated in order to produce ‘optimal’ predictions, not ‘optimal’ decisions.

In this paper, we follow the work of Donti et al. [17], Elmachtoub and Grigas [19] and others, and propose the use of an integrated prediction and optimization (IPO) framework with direct applications to mean-variance portfolio optimization. Specifically, we estimate $\boldsymbol{\theta}$ such that the resulting optimal decisions, $\{\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})\}_{i=1}^m$, minimizes the expected realized decision cost:

$$
\begin{aligned}
& \underset{\boldsymbol{\theta} \in \Theta}{\text{minimize}} \quad L(\boldsymbol{\theta}) = \mathbb{E}[c(\mathbf{z}^*(\hat{\mathbf{y}}), \mathbf{y})] \\
& \text{subject to} \quad \mathbf{z}^*(\hat{\mathbf{y}}) = \underset{\mathbf{z} \in \mathcal{S}}{\text{argmin}} \, c(\mathbf{z}, \hat{\mathbf{y}}),
\end{aligned}
\tag{1}
$$

Solving Program (1) challenging for several reasons. First, even in the case where the decision program is convex, the resulting integrated program is likely not convex in $\boldsymbol{\theta}$ and therefore we have no guarantee that a particular local solution is globally optimal. Secondly, as outlined by Donti et al. [17], in the case where $L(\boldsymbol{\theta})$ is differentiable, computing the gradient, $\nabla_{\boldsymbol{\theta}} L$, remains difficult as it requires differentiation through the argmin operator. Moreover, solving program (1) through iterative descent methods can be computationally demanding as at each iteration we must solve several instances of $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$.

In this paper we address the aforementioned challenges and provide an efficient framework for integrating linear regression predictions into a mean-variance portfolio optimization. The remainder of the paper is outlined as follows. We first review the growing body of literature in the field of integrated methods and summarize our primary contributions. In Section 2 we present the mean-variance portfolio optimization problem and provide the IPO formulation. We review the current state-of-the-art approach for locally solving Program (1) in the presence of lower-level inequality constraints. We then consider several special instances of the IPO mean-variance optimization problem. In particular, we demonstrate that when the MVO program is either unconstrained or contains only linear equality constraints then the IPO problem can be recast as a convex quadratic program and solved analytically. We discuss the sampling distribution properties of the optimal IPO regression coefficients and demonstrate that the IPO solution explicitly minimizes the tracking-error to ex-post optimal mean-variance portfolios.

In Section 3.1 we perform several simulation studies, using synthetically generated data, and compare the IPO approach to a traditional ‘predict, then optimize’ framework with prediction models estimated by OLS regression. In Sections 3.2 we discuss the computational challenges of the state-of-the art iterative solution. We demonstrate the computational advantage of the closed-form IPO solution, which guarantees optimality and is approximately an order of magnitude more computationally efficient. In Section 3.3 we present a simulation that demonstrates that a heuristic analytical IPO solution, with inequality constraints

<!-- Page 3 -->
removed, can provide competitive out-of-sample performance and lower variance over a wide range of problem parameterizations. We conclude in Section 4 with a historical analysis using global futures data and demonstrate that the IPO framework can provide lower realized costs and improved economic outcomes in comparison to the ‘predict, then optimize’ alternative.

## 1.1 Existing Literature

In recent years there has been a growing body of research on methods for integrating prediction models with downstream decision-making processes. For example, Ban and Rudin [4] present a direct empirical risk minimization approach using nonparametric kernel regression as the core prediction method. They consider a data-driven newsvendor problem and demonstrate that their approach outperforms the ‘best-practice benchmark’ when evaluated out-of-sample. More recently, Kannan et al. [33] present three frameworks for integrating machine learning prediction models within a stochastic optimization setting. Their primary contribution is in using the out-of-sample residuals from leave-one-out prediction models to generate scenarios which are then optimized in the context of a sample average approximation program. Their frameworks are flexible and accommodate parametric and nonparametric prediction models, for which they derive convergence rates and finite sample guarantees.

Bertsimas and Kallus [6] present a general framework for optimizing a conditional stochastic approximation program whereby the conditional density is estimated through a variety of parametric and nonparametric machine learning methods. They generate locally optimal decision policies within the context of the decision optimization problem and consider the setting where the decision policy affects subsequent realizations of the uncertainty variable. They also consider an empirical risk minimization framework for generating predictive prescriptions and discuss the relative trade-offs of such an approach.

Recently, Elmachtoub and Grigas [19] proposed replacing the prediction-based loss function with a convex surrogate loss function that optimizes prediction variables based on the decision error induced by the prediction. They demonstrate that their ‘smart predict, then optimize’ (SPO) loss function attains Fisher consistency with the least-squares loss function and show through example that optimizing predictions in the context of decision objectives and constraints can lead to improved overall decision error. The SPO loss function however is limited to linear objective functions, and despite convexity can be computationally demanding due to repeatedly solving the decision program.

Our approach is most similar to, and is largely inspired by, the work of Amos and Kolter [2] and Donti et al. [17]. Recall that computing the Jacobian, $\partial \mathbf{z}^* / \partial \boldsymbol{\theta}$, is complicated by the bi-level structure of Program (1). Amos and Kolter [2] present an efficient framework for embedding quadratic programs as differentiable layers in a neural network. The author’s demonstrate that for linearly constrained quadratic programs, implicit differentiation of the KKT optimality conditions provides the necessary ingredients for computing the desired gradient, $\partial L / \partial \boldsymbol{\theta}$. Donti et al. [17] present the first direct application of the aforementioned work and propose an end-to-end stochastic programming approach for estimating the parameters of probability density functions in the context of their final task-based loss function. They consider applications from power scheduling and battery storage and demonstrate that their ‘task-based end-to-end’ approach can result in lower out-of-sample decision costs in comparison to traditional maximum likelihood estimation and a black-box neural network.

<!-- Page 4 -->
## 1.2 Main Contributions

While our methodology follows closely to that of Dondi et al. [17] and Elmachtoub and Grigas [19], in this paper we provide several notable differences and extensions.

1. We consider linear regression prediction models with a downstream quadratic MVO objective function. We demonstrate that when the MVO program is either unconstrained or contains only linear equality constraints then the integrated program can be recast as quadratic program. We discuss the necessary conditions for convexity and provide analytical solutions for the optimal IPO coefficients, $\boldsymbol{\theta}^*$. We present conditions for which $\boldsymbol{\theta}^*$ is an unbiased estimator of $\boldsymbol{\theta}$ and derive the analytical expression for the variance. We demonstrate that the IPO coefficients explicitly minimize the tracking error to the unconstrained ex-post optimal MVO portfolio and provide the equivalent minimum-tracking error optimization program.

2. We conduct three simulation studies based on synthetically generated data. The first simulation compares the out-of-sample performance of the IPO and OLS models under varying degrees of estimation error. We demonstrate that for unconstrained and equality constrained cases, the IPO model can produce consistently lower out-of-sample decision costs. The second simulation demonstrates the computational advantage of the analytical IPO solution over a wide range of asset universe sizes. The third simulation considers linear inequality constrained MVO program under varying degrees of model misspecification. We propose approximating the non-convex problem with the analytical IPO solution whereby the inequality constraints are ignored. We demonstrate the computational and performance advantage of the analytical IPO solution, which is on average 100x - 1000x times faster than the current state-of-the-art method and produces solutions with lower out-of-sample variance and, in some instances, improved MVO costs.

3. We perform several historical simulations using global futures data, considering both unconstrained and constrained MVO programs and univariate and multivariate regression models. Out-of-sample numerical results demonstrate that the IPO model can provide lower realized cost and superior economic performance in comparison to the traditional OLS ‘predict then optimize’ approach.

Finally we note that in this paper, asset mean returns are estimated through linear regression, whereas the asset covariance matrices are estimated by a traditional weighted moving average approach [5, 44]. This is supported by the observation that asset mean returns are both nonstationary and heterogeneous and are therefore likely to be dependent on feature data [20, 28, 37], whereas variance and covariances are typically much more stable and exhibit strong autocorrelation effects [8, 18, 42]. Moreover, Chopra and Ziemba [15] and Best and Grauer [7] report that MVO portfolio weights are an order of magnitude more sensitive to the estimate of asset mean returns compared to estimates of asset covariances. The choice of linear regression model is deliberate and motivated by the long established history of regression forecasting in the financial literature (see for example, [21, 22, 23]). Indeed, asset returns are often characterized as time-varying and reactive, and typically exhibit extremely low signal-to-noise ratios (SNRs) [30]. As a result, low variance models, like simple linear regression, tend to generalize out-of-sample and are often preferred over models of higher complexity.

<!-- Page 5 -->
# 2 Methodology

## 2.1 IPO: Mean-Variance Optimization

We begin with a brief introduction to mean-variance portfolio optimization. We denote the matrix of (excess) return observations as $\mathbf{Y} = [\mathbf{y}^{(1)}, \mathbf{y}^{(2)}, ..., \mathbf{y}^{(m)}] \in \mathbb{R}^{m \times d_z}$ and denote the portfolio at time $i$ as $\mathbf{z}^{(i)} \in \mathbb{R}^{d_z}$. Let $\mathbf{V}^{(i)} \in \mathbb{R}^{d_z \times d_z}$ denote the time-varying symmetric positive definite covariance matrix of asset returns. The mean variance cost function at time $i$ is given by:

$$
c(\mathbf{z}, \mathbf{y}^{(i)}) = -\mathbf{z}^T \mathbf{y}^{(i)} + \frac{\delta}{2} \mathbf{z}^T \mathbf{V}^{(i)} \mathbf{z}
\tag{2}
$$

with risk-aversion parameter $\delta \in \mathbb{R}_{+}$ and denote the optimal portfolio weights as:

$$
\mathbf{z}^{*}(\mathbf{y}^{(i)}) = \underset{\mathbf{z} \in \mathbb{S}}{\mathrm{argmin}} -\mathbf{z}^T \mathbf{y}^{(i)} + \frac{\delta}{2} \mathbf{z}^T \mathbf{V}^{(i)} \mathbf{z}.
\tag{3}
$$

In reality, we do not know the values $\mathbf{y}^{(i)}$ or $\mathbf{V}^{(i)}$ at decision time. In this paper we model the time-varying covariance matrix using a weighted moving average approach and denote the covariance estimate as $\hat{\mathbf{V}}^{(i)}$. Asset returns are modelled according to the following linear model:

$$
\mathbf{y}^{(i)} = \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \boldsymbol{\theta} + \boldsymbol{\epsilon}^{(i)}
\tag{4}
$$

with residuals $\boldsymbol{\epsilon}^{(i)} \sim \mathcal{N}(\mathbf{0}, \Sigma)$. Here $\operatorname{diag}(\cdot)$ denotes the usual diagonal operator and $\mathbf{P} \in \mathbb{R}^{d_y \times d_x}$ controls the regression design with each element $\mathbf{P}_{jk} \in \{0,1\}$. In particular, we assume that each asset has its own, perhaps unique, set of feature variables. For example, if the feature variables represent price-to-earnings (P/E) and debt-to-equity (D/E) ratios for each asset under consideration, then it would be unrealistic to model a particular assets return as a function of all available P/E and D/E ratios. Indeed, doing so would almost certainly lead to model overfit. Instead, we choose to model asset $j$’s return as a linear function of the P/E and D/E ratios relevant to asset $j$, specifically:

$$
\hat{\mathbf{y}}_j^{(i)} = \boldsymbol{\theta}_{\mathbf{a}(j)}^T \mathbf{x}_{\mathbf{a}(j)}^{(i)},
\tag{5}
$$

where $\mathbf{a}(j)$ denotes the indices of the feature variables relevant to asset $j$. Therefore,

$$
\mathbf{P}_{jk} =
\begin{cases}
1, & \text{if } k \in \mathbf{a}(j) \\
0, & \text{otherwise}
\end{cases}
\tag{6}
$$

and for observation $i$, the regression estimate of asset expected returns is given by:

$$
\hat{\mathbf{y}}^{(i)} = f(\mathbf{x}^{(i)}, \boldsymbol{\theta}) = \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \boldsymbol{\theta}.
\tag{7}
$$

Given that $\mathbf{y}^{(i)}$ and $\mathbf{V}^{(i)}$ are unobservable, it follows that in practice portfolio managers solve the MVO program under the estimation hypothesis:

$$
\underset{\mathbf{z} \in \mathbb{S}}{\mathrm{minimize}} c(\mathbf{z}, \hat{\mathbf{y}}^{(i)}) = -\mathbf{z}^T \hat{\mathbf{y}}^{(i)} + \frac{\delta}{2} \mathbf{z}^T \hat{\mathbf{V}}^{(i)} \mathbf{z} = -\mathbf{z}^T \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \boldsymbol{\theta} + \frac{\delta}{2} \mathbf{z}^T \hat{\mathbf{V}}^{(i)} \mathbf{z}
\tag{8}
$$

<!-- Page 6 -->
In a ‘predict, then optimize’ parameter estimation, $\theta$ would be chosen in order to minimize a prediction loss function $\ell \colon \mathbb{R}^{d_y} \times \mathbb{R}^{d_y} \to \mathbb{R}$, such as least-squares. We denote $\mathbb{E}_D$ as the expectation operator with respect to the training set $D = \{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^m$ and choose $\hat{\theta}$ such that:

$$
\hat{\theta} = \argmin_{\theta \in \Theta} \mathbb{E}_D[\ell(f(\mathbf{x}^{(i)}, \theta), \mathbf{y}^{(i)})],
\tag{9}
$$

A ‘predict, then optimize’ framework, would simply ‘plug-in’ the estimate, $\hat{\mathbf{y}}^{(i)} = \mathbf{P} \diag(\mathbf{x}^{(i)}) \hat{\theta}$, into program (8) in order to generate the optimal decisions $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$.

Conversely, in the IPO framework, the objective is to choose $\theta$ in order to minimize the average MVO cost induced by the optimal decisions $\{\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})\}_{i=1}^m$. Specifically, we solve the bi-level optimization program (1), presented in discrete form in program (10):

$$
\begin{aligned}
& \underset{\theta \in \Theta}{\text{minimize}} \quad L(\theta) = \frac{1}{m} \sum_{i=1}^m \left( - \mathbf{z}^*(\hat{\mathbf{y}}^{(i)})^T \mathbf{y}^{(i)} + \frac{\delta}{2} \mathbf{z}^*(\hat{\mathbf{y}}^{(i)})^T \mathbf{V}^{(i)} \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) \right) \\
& \text{subject to} \quad \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) = \argmin_{\mathbf{z} \in \mathbb{S}} - \mathbf{z}^T \mathbf{P} \diag(\mathbf{x}^{(i)}) \theta + \frac{\delta}{2} \mathbf{z}^T \hat{\mathbf{V}}^{(i)} \mathbf{z} \quad \forall i = 1, ..., m.
\end{aligned}
\tag{10}
$$

Note at this point we have not described the feasible region, $\mathbb{S}$, of the MVO program. In the following subsections we briefly discuss the general case where $\mathbb{S}$ describes a set of linear equality and inequality constraints and formalize the current state-of-the-art neural-network framework. We then discuss two special cases in which an analytical solution to the MVO problem is possible and derive the relevant theory.

## 2.2 Current state-of-the-art methodology

We begin with the general case whereby the feasible region of the MVO program is defined by both linear equality and inequality constraints. Specifically we consider the following MVO program:

$$
\begin{aligned}
& \underset{\mathbf{z}}{\text{minimize}} \quad - \mathbf{z}^T \hat{\mathbf{y}}^{(i)} + \frac{\delta}{2} \mathbf{z}^T \hat{\mathbf{V}}^{(i)} \mathbf{z} \\
& \text{subject to} \quad \mathbf{A} \mathbf{z} = \mathbf{b} \\
& \quad\quad\quad\quad \mathbf{G} \mathbf{z} \leq \mathbf{h}
\end{aligned}
\tag{11}
$$

where $\mathbf{A} \in \mathbb{R}^{d_{\text{eq}} \times d_z}$, $\mathbf{b} \in \mathbb{R}^{d_{\text{eq}}}$ and $\mathbf{G} \in \mathbb{R}^{d_{\text{iq}} \times d_z}$, $\mathbf{h} \in \mathbb{R}^{d_{\text{iq}}}$ describe the linear equality and inequality constraints, respectively.

In general, there is no known analytical solution to Program (11) and instead the solution, $\mathbf{z}^*$, is obtained through iterative optimization methods. Moreover, because of the inequality constraints, the IPO objective, $L(\theta)$, is generally not a convex function of $\theta$. Therefore, in the general case we follow Amos and Kolter [2] and Donti et al. [17] and compute locally optimal solutions, $\theta^*$, by restructuring Program (10) as an end-to-end neural network and apply (stochastic) gradient descent. The IPO equivalent neural-network structure is depicted in Figure 1. In the forward pass, the input layer takes the feature variables $\mathbf{x}^{(i)}$ and passes them to a simple linear layer to produce the estimates, $\hat{\mathbf{y}}^{(i)}$. The predictions are then passed to a differentiable quadratic programming layer which, for a given input $\hat{\mathbf{y}}^{(i)}$, solves the decision program and

<!-- Page 7 -->
returns the optimal portfolio weights $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$. Finally, the quality of the portfolio decisions are evaluated by the MVO cost function, $c(\mathbf{z}^*(\hat{\mathbf{y}}^{(i)}), \mathbf{y}^{(i)})$ in the context of the true return values $\mathbf{y}^{(i)}$. We refer the reader to Appendix A for more comprehensive implementation details.

```plaintext
Input layer          Linear layer                  QP layer                 Loss Function
    x^(i)       →   ŷ^(i) = P diag(x^(i)) θ   →   min_z c(z, ŷ^(i))   →   c(z*, y^(i))
                    ∂ŷ^(i)/∂θ                     ∂z*/∂ŷ^(i)             ∂c/∂z*
    θ ← θ - g_θ   ←
```

Figure 1: IPO program represented as an end-to-end neural-network with predictive linear layer, differentiable quadratic programming layer and realized cost loss function.

In the general case we compute a locally optimal solution, $\boldsymbol{\theta}^*$, by applying (stochastic) gradient descent. Prediction model parameters are updated by backpropagation, with descent direction, $\mathbf{g}_{\boldsymbol{\theta}}$, estimated over a randomly drawn sample batch, $B$:

$$
\mathbf{g}_{\boldsymbol{\theta}} = \sum_{i \in B} \left( \frac{\partial c}{\partial \boldsymbol{\theta}} \right)_{|(\mathbf{z}^*(\hat{\mathbf{y}}^{(i)}), \mathbf{y}^{(i)})} \approx \nabla_{\boldsymbol{\theta}} L.
$$

Note that each iteration of gradient descent therefore requires forward solving and backward differentiating through, at most, $m$ mean-variance optimization programs, $\{\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})\}_{i=1}^{m}$, which in some applications can be computationally expensive to compute.

## 2.3 Special case 1: $\mathbb{S} = \mathbb{R}^{d_z}$

We are motivated by Gould et al. [26] who demonstrate that under special constraint cases, an analytical solution for the gradient and Hessian of a bi-level optimization problem exists. We first consider the case where the MVO program is unconstrained and therefore an analytical solution is given by Equation (12).

$$
\mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) = \frac{1}{\delta} \hat{\mathbf{V}}^{(i)^{-1}} \hat{\mathbf{y}}^{(i)} = \frac{1}{\delta} \hat{\mathbf{V}}^{(i)^{-1}} \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \boldsymbol{\theta}.
\tag{12}
$$

**Proposition 1.** Let $\mathbb{S} = \mathbb{R}^{d_z}$ and $\Theta = \mathbb{R}^{d_\theta}$. We define

$$
\mathbf{d}(\mathbf{x}, \mathbf{y}) = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \operatorname{diag}(\mathbf{x}^{(i)}) \mathbf{P}^T \hat{\mathbf{V}}^{(i)^{-1}} \mathbf{y}^{(i)} \right)
\tag{13}
$$

and

$$
\mathbf{H}(\mathbf{x}) = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \operatorname{diag}(\mathbf{x}^{(i)}) \mathbf{P}^T \hat{\mathbf{V}}^{(i)^{-1}} \mathbf{V}^{(i)} \hat{\mathbf{V}}^{(i)^{-1}} \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \right).
\tag{14}
$$

Then the IPO program (10) is an unconstrained quadratic program (QP) given by:

<!-- Page 8 -->
$$
\minimize_{\boldsymbol{\theta} \in \Theta} \frac{1}{2} \boldsymbol{\theta}^T \mathbf{H}(\mathbf{x}) \boldsymbol{\theta} - \boldsymbol{\theta}^T \mathbf{d}(\mathbf{x}, \mathbf{y}).
\tag{15}
$$

Furthermore, if there exists an $\mathbf{x}^{(i)}$ such that $\mathbf{x}_j^{(i)} \neq 0 \quad \forall j \in 1, ..., d_x$ then $\mathbf{H}(\mathbf{x}) \succ 0$ and therefore program (15) is an unconstrained convex quadratic program with unique minimum:

$$
\boldsymbol{\theta}^* = \mathbf{H}(\mathbf{x})^{-1} \mathbf{d}(\mathbf{x}, \mathbf{y}).
\tag{16}
$$

All proofs are provided in Appendix A. We make a few important observations. The first, is that for the realistic case where there exists an $\mathbf{x}^{(i)}$ such that each $\mathbf{x}_j^{(i)}$ are not exactly zero, then the optimal IPO regression coefficients, $\boldsymbol{\theta}^*$, are unique. Furthermore, we observe that the solution is independent of the risk-aversion parameter. This is intuitive, as when the MVO program is unconstrained, then the risk-aversion parameter simply controls the scale of the resulting solutions $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$.

We note that the solution presented in Equation (16) requires the action of the inverse of the Hessian: $\mathbf{H}(\mathbf{x})$. In many applications of machine learning, such as computer vision or statistical meta-modelling, it is difficult, if not impossible, to solve the inverse problem without customized algorithms or prior knowledge of the data (see for example Jones and Taylor [32], Ranjan et al. [39], Ongie et al. [38]). In many cases, the dimension of the relevant Hessian is either too large for both forward-mapping and inversion in reasonable compute time or is computationally unstable due to near-singularity. In our IPO framework, we fortunately do not encounter these technical difficulties surrounding the action of the inverse. In most practical settings, the dimension of the Hessian matrix, is on the order of 10 or 100, whereas the number of observations, $m$, is on the order of 1000 or 10000. The Hessian is therefore likely to be computationally stable and the action of the inverse is computationally tractable. This is validated numerically in Section 3.2 and we demonstrate the computational advantage of the analytical solution over the iterative descent method.

Furthermore, while outside of the scope of the current paper, we note that under the QP formulation (15), it is trivial to incorporate both regularization and constraints on $\boldsymbol{\theta}$. This is demonstrated by Program (17):

$$
\begin{aligned}
& \minimize_{\mathbf{z}} \quad \frac{1}{2} \boldsymbol{\theta}^T \mathbf{H}(\mathbf{x}) \boldsymbol{\theta} - \boldsymbol{\theta}^T \mathbf{d}(\mathbf{x}, \mathbf{y}) + \Omega(|\boldsymbol{\theta}|) \\
& \text{subject to} \quad \mathbf{A}_{\boldsymbol{\theta}} \boldsymbol{\theta} = \mathbf{b}_{\boldsymbol{\theta}} \\
& \quad \quad \quad \quad \quad \mathbf{G}_{\boldsymbol{\theta}} \boldsymbol{\theta} \leq \mathbf{h}_{\boldsymbol{\theta}}
\end{aligned}
\tag{17}
$$

where $\Omega \colon \mathbb{R}^{d_\theta} \to \mathbb{R}$ is a convex regularization function. In most cases, Program (17) can be solved to optimality by standard quadratic programming techniques, whereas the incorporation of regularization and constraints in the current state-of-the-art solution is structurally more challenging.

We now discuss the properties of the sampling distribution of the IPO parameter estimate, $\boldsymbol{\theta}^*$, and derive an estimate of the variance, $\mathrm{Var}(\boldsymbol{\theta}^*)$. Recall, from Equation (4) we have: $\mathbf{y}^{(i)} \sim \mathcal{N}(\mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \boldsymbol{\theta}, \Sigma)$.

**Proposition 2.** Let

$$
\mathbf{d}_{\mathbf{u}}(\mathbf{x}) = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \operatorname{diag}(\mathbf{x}^{(i)}) \mathbf{P}^T \hat{\mathbf{V}}^{(i)^{-1}} \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \right),
\tag{18}
$$

then the optimal IPO estimate, $\boldsymbol{\theta}^*$, is a biased estimate of $\boldsymbol{\theta}$ with bias $\mathbf{H}(\mathbf{x})^{-1} \mathbf{d}_{\mathbf{u}}(\mathbf{x})$.

<!-- Page 9 -->
**Corollary 1.** Let $\theta_{\mathbf{u}}^* = \mathbf{d}_{\mathbf{u}}(\mathbf{x})^{-1} \mathbf{H}(\mathbf{x}) \theta^*$. Then $\theta_{\mathbf{u}}^*$ is an unbiased estimator of $\theta$.

**Corollary 2.** Let $\hat{\mathbf{V}}^{(i)} = \mathbf{V}^{(i)} \, \forall i \in \{1, ..., m\}$. Then $\theta^*$ is an unbiased estimator of $\theta$.

We observe from Proposition 2 that differences, or estimation errors, between $\hat{\mathbf{V}}^{(i)}$ and $\mathbf{V}^{(i)}$, make $\theta^*$ a biased estimator in $\theta$. In particular, the bias can be corrected by left multiplication of $\theta^*$ by $\mathbf{d}_{\mathbf{u}}(\mathbf{x})^{-1} \mathbf{H}(\mathbf{x})$. This observation leads to Corollary 2, which shows that when the estimation error in the covariance is zero then $\theta^*$ is an unbiased estimator of $\theta$. Moreover, unlike the OLS estimate, $\hat{\theta}$, the IPO estimate, $\theta^*$, incorporates estimation error in the sample covariance in the (likely) event that the estimation error is nonzero. This is discussed in more detail in Section 3.1.

**Proposition 3.** Let $\{\mathbf{y}^{(i)}\}_{i=1}^m$ be independent random variables with $\mathbf{y}^{(i)} \sim \mathcal{N}(\mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \theta, \Sigma)$. Let $\hat{\Sigma}$ be an unbiased estimate of the sample covariance of residuals, given by:

$$
\hat{\Sigma} = \frac{1}{m-1} \sum_{i=1}^m \left( \mathbf{y}^{(i)} - \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \theta \right)^2.
\tag{19}
$$

Let

$$
\mathbf{M} = \frac{1}{\delta^2 m^2} \left( \operatorname{diag}(\mathbf{x}^{(i)}) \mathbf{P}^T \hat{\mathbf{V}}^{(i)-1} \hat{\Sigma} \hat{\mathbf{V}}^{(i)-1} \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \right),
\tag{20}
$$

then the variance, $\operatorname{Var}(\theta^*)$, is given by:

$$
\operatorname{Var}(\theta^*) = \mathbf{H}(\mathbf{x})^{-1} \mathbf{M} \mathbf{H}(\mathbf{x})^{-1}
\tag{21}
$$

We conclude this section by providing an alternative, and perhaps more intuitive expression of the optimal IPO coefficients derived from portfolio tracking-error optimization. Let $\|\cdot\|_{\mathbf{V}}$ denote the elliptic norm with respect to the symmetric positive definite matrix $\mathbf{V}$, defined as:

$$
\|\mathbf{w}\|_{\mathbf{V}} = \sqrt{\mathbf{w}^T \mathbf{V} \mathbf{w}}.
\tag{22}
$$

More specifically, $\|\mathbf{z}^{(1)} - \mathbf{z}^{(2)}\|_{\mathbf{V}}^2$ measures the tracking-error between two portfolio weights with respect to the covariance $\mathbf{V}$.

**Proposition 4.** Let $\mathbf{z}^*(\mathbf{y}^{(i)})$ and $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$ be as defined in Equation (3) and Equation (12), respectively. Then the optimal IPO coefficients, $\theta^*$, minimizes the average tracking error between $\mathbf{z}^*(\mathbf{y}^{(i)})$ and $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$ with respect to the realized covariance $\mathbf{V}^{(i)}$:

$$
\theta^* = \underset{\theta \in \Theta}{\operatorname{argmin}} \frac{1}{2m} \sum_{i=1}^m \|\mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) - \mathbf{z}^*(\mathbf{y}^{(i)})\|_{\mathbf{V}^{(i)}}^2
\tag{23}
$$

Indeed, Proposition 4 states that the IPO coefficients $\theta^*$ minimizes the average tracking error between the estimated optimal weights, $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$ and the ex-post optimal weight $\mathbf{z}^*(\mathbf{y}^{(i)})$.

<!-- Page 10 -->
## 2.4 Special Case 2: $\mathbb{S} = \{\mathbf{A} \, \mathbf{z} = \mathbf{b}\}$

We now consider the case where the MVO program is constrained by a set of linear equality constraints:

$$
\mathbb{S} = \{\mathbf{A} \, \mathbf{z} = \mathbf{b}\},
$$

where $\mathbf{A} \in \mathbb{R}^{d_{eq} \times d_z}$ and $\mathbf{b} \in \mathbb{R}^{d_{eq}}$. We assume the non-trivial case where $\mathbf{A}$ is not full rank. Let the columns of $\mathbf{F}$ form a basis for the nullspace of $\mathbf{A}$ defined as:

$$
\text{Null}(\mathbf{A}) = \{\mathbf{z} \in \mathbb{R}^{d_z} \mid \mathbf{A} \, \mathbf{z} = 0\}.
$$

Let $\mathbf{z}_0$ be a particular element of $\mathbb{S}$. It follows that $\forall \, \mathbf{w} \in \mathbb{R}^{d_z - d_n}$ then $\mathbf{z} = \mathbf{F} \, \mathbf{w} + \mathbf{z}_0$ is also an element of $\mathbb{S}$, where $d_n = \text{nullity}(\mathbf{A})$. We follow Boyd and Vandenberghe [12] and recast the MVO program as an unconstrained convex quadratic program:

$$
\min_{\mathbf{w}} c(\mathbf{F} \, \mathbf{w} + \mathbf{z}_0, \hat{\mathbf{y}}^{(i)}),
\tag{24}
$$

with unique global minimum:

$$
\mathbf{w}^*(\hat{\mathbf{y}}^{(i)}) = \frac{1}{\delta} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \left( \hat{\mathbf{y}}^{(i)} - \delta \, \hat{\mathbf{V}}^{(i)} \, \mathbf{z}_0 \right)
\tag{25}
$$

The solution to the MVO Program (3) is then given by:

$$
\begin{aligned}
\mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) &= \frac{1}{\delta} \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T (\mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \, \boldsymbol{\theta} - \delta \, \hat{\mathbf{V}}^{(i)} \, \mathbf{z}_0) + \mathbf{z}_0 \\
&= \frac{1}{\delta} \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \, \boldsymbol{\theta} + (\mathbf{I} - \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \hat{\mathbf{V}}^{(i)}) \, \mathbf{z}_0
\end{aligned}
\tag{26}
$$

**Proposition 5.** Let $\mathbb{S} = \{\mathbf{A} \, \mathbf{z} = \mathbf{b}\}$ and $\Theta = \mathbb{R}^{d_\theta}$. Define

$$
\mathbf{d}_{eq}(\mathbf{x}, \mathbf{y}) = \frac{1}{m \delta} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T (\mathbf{y}^{(i)} - \mathbf{V}^{(i)} (\mathbf{I} - \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \hat{\mathbf{V}}^{(i)}) \, \mathbf{z}_0) \right)
\tag{27}
$$

and

$$
\mathbf{H}_{eq}(\mathbf{x}) = \frac{1}{m \delta} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \mathbf{V}^{(i)} \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \right).
\tag{28}
$$

Then the IPO program (10) is an unconstrained quadratic program given by:

$$
\minimize_{\boldsymbol{\theta} \in \Theta} \frac{1}{2} \, \boldsymbol{\theta}^T \, \mathbf{H}_{eq}(\mathbf{x}) \, \boldsymbol{\theta} - \boldsymbol{\theta}^T \, \mathbf{d}_{eq}(\mathbf{x}, \mathbf{y}).
\tag{29}
$$

Furthermore, if there exists an $\mathbf{x}^{(i)}$ such that $\mathbf{x}_j^{(i)} \neq 0 \quad \forall j \in 1, ..., d_x$ then $\mathbf{H}_{eq}(\mathbf{x}) \succ 0$ and therefore program (29) is an unconstrained convex quadratic program with unique minimum:

$$
\boldsymbol{\theta}_{eq}^* = \mathbf{H}_{eq}(\mathbf{x})^{-1} \, \mathbf{d}_{eq}(\mathbf{x}, \mathbf{y}).
\tag{30}
$$

<!-- Page 11 -->
As before we briefly discuss the properties of the sampling distribution of the equality constrained IPO parameter estimate, $\boldsymbol{\theta}_{\text{eq}}^*$, and derive an estimate of the variance, $\text{Var}(\boldsymbol{\theta}_{\text{eq}}^*)$.

**Proposition 6.** Let

$$
\mathbf{d_e(x)} = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \mathbf{P}^T \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \right),
\tag{31}
$$

then the optimal IPO estimate, $\boldsymbol{\theta}_{\text{eq}}^*$, is a biased estimate of $\boldsymbol{\theta}$ with bias $\mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \, \mathbf{d_e(x)}$.

**Corollary 3.** Let $\hat{\mathbf{V}}^{(i)} = \mathbf{V}^{(i)} \, \forall i \in \{1, ..., m\}$. Then $\boldsymbol{\theta}_{\text{eq}}^*$ is an unbiased estimator of $\boldsymbol{\theta}$.

Again we observe from Proposition 6 that in general $\boldsymbol{\theta}_{\text{eq}}^*$ a biased estimator of $\boldsymbol{\theta}$. In particular, the bias can be corrected by left multiplication of $\boldsymbol{\theta}_{\text{eq}}^*$ by $\mathbf{d_e(x)}^{-1} \, \mathbf{H}_{\text{eq}}(\mathbf{x})$. Furthermore when the estimation error in the covariance is zero then $\boldsymbol{\theta}_{\text{eq}}^*$ is an unbiased estimator of $\boldsymbol{\theta}$.

**Proposition 7.** Let $\{\mathbf{y}^{(i)}\}_{i=1}^{m}$ be independent random variables with $\mathbf{y}^{(i)} \sim \mathcal{N}(\mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \, \boldsymbol{\theta}, \Sigma)$. Let

$$
\mathbf{M}_{\text{eq}} = \frac{1}{\delta^2 m^2} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \mathbf{P}^T \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \hat{\Sigma} \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \right).
\tag{32}
$$

then the variance, $\text{Var}(\boldsymbol{\theta}_{\text{eq}}^*)$, is given by:

$$
\text{Var}(\boldsymbol{\theta}_{\text{eq}}^*) = \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \, \mathbf{M}_{\text{eq}} \, \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1}
\tag{33}
$$

As before, we conclude this subsection with the following proposition that states that the IPO coefficients $\boldsymbol{\theta}^*$ minimizes the average tracking error between the estimated optimal weights, $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$ and the ex-post optimal weight $\mathbf{z}^*(\mathbf{y}^{(i)})$.

**Proposition 8.** Let $\mathbf{z}^*(\mathbf{y}^{(i)})$ and $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$ be as defined in Equation (3) and Equation (12), respectively. Then the optimal IPO coefficients, $\boldsymbol{\theta}_{\text{eq}}^*$, minimizes the average tracking error between $\mathbf{z}^*(\mathbf{y}^{(i)})$ and $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$ with respect to the realized covariance $\mathbf{V}^{(i)}$:

$$
\begin{aligned}
\boldsymbol{\theta}_{\text{eq}}^* = \underset{\boldsymbol{\theta}}{\text{argmin}} & \quad \frac{1}{2m} \sum_{i=1}^{m} \| \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) - \mathbf{z}^*(\mathbf{y}^{(i)}) \|_{\mathbf{V}^{(i)}}^2 \\
\text{subject to} & \quad \mathbf{A} \, \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) = \mathbf{b} \\
\text{subject to} & \quad \mathbf{A} \, \mathbf{z}^*(\mathbf{y}^{(i)}) = \mathbf{b}
\end{aligned}
\tag{34}
$$

## 3 Simulated experiments

### 3.1 Simulation 1: estimation error in $\hat{\mathbf{V}}$

Elmachtoub and Grigas [19] consider the integration of predictive forecasting with downstream optimization problems that have linear cost functions. Their simulated experiments demonstrate that the benefit of the

<!-- Page 12 -->
‘smart predict, then optimize’ (SPO) framework increases as the amount of model misspecification increases. Specifically, model misspecification is introduced by synthetically generating cost vectors that are polynomial functions of the simulated feature data and modelling the relationship as though it is linear. In particular, they demonstrate that a linear forecasting model trained with SPO can outperform traditional prediction models, such as OLS and random forest, and the amount of outperformance increases as the degree of nonlinearity in the ground truth increases.

Here, we demonstrate that, for a mean-variance decision program, the IPO model can provide lower out-of-sample MVO costs in comparison to a traditional OLS-based ‘predict, then optimize’ model, even when the underlying ground truth is *linear* in the feature variables. In particular, we demonstrate that the OLS model is vulnerable to estimation error in $\hat{\mathbf{V}}^{(i)}$, resulting in sub-optimal decision making and increasing out-of-sample MVO costs as estimation error in $\hat{\mathbf{V}}^{(i)}$ increases. The IPO model, on the other hand, incorporates the impact of estimation in the covariance matrix. The simulated experiment below demonstrates that the IPO model consistently outperforms the OLS model in terms of minimizing the out-of-sample MVO cost. Moreover, the outperformance is shown to be consistent over a wide range of signal-to-noise ratios (SNRs) and asset correlation assumptions commonly observed in financial forecasting. In general we observe that the benefit of the IPO model increases as the estimation error in $\hat{\mathbf{V}}^{(i)}$ increases, even when the underlying ground truth is linear in the feature variables.

We follow an experimental design similar to Hastie et al. [27]. Asset returns are assumed to be normally distributed, $\mathbf{y}^{(i)} \sim \mathcal{N}(\text{diag}(\mathbf{x}^{(i)})\,\boldsymbol{\theta}_0, \mathbf{V})$ where $\mathbf{V} \in \mathbb{R}^{d_z \times d_z}$ has entry $(j,k)$ equal to $\sigma^2 \rho^{|j-k|}$, and $\sigma = 0.0125$ (20% annualized). Asset mean returns are modelled according to univariate model of the form:

$$
\mathbf{y}^{(i)} = \text{diag}(\mathbf{x}^{(i)})\,\boldsymbol{\theta}_0 + \tau\,\boldsymbol{\epsilon}^{(i)},
$$

where feature data, $\mathbf{x}^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_x})$, and residuals, $\boldsymbol{\epsilon}^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{V})$. The scalar value $\tau$ controls the SNR level, where $\text{SNR} = \text{Var}(f(\mathbf{x}, \boldsymbol{\theta}_0)) / \text{Var}(\boldsymbol{\epsilon})$. We consider asset correlation values in the range of: $\rho \in \{0, 0.25, 0.5, 0.75\}$, and SNR values: $\text{SNR} \in \{0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.05, 0.10\}$. Note that it may appear that these SNR values are extremely low. However, in most applications of asset return forecasting, the SNRs are typically found to be much less than 1%. Indeed, a moderate sized universe (25 assets) with each asset having SNRs of 1% can generate annualized Sharpe ratios in the low double digits — which is extremely rare — and SNRs of 10% are extremely unlikely at a daily trading frequency.

We introduce estimation error in $\hat{\mathbf{V}}^{(i)}$ by varying the sample size, $s = \text{res} * d_z$, used for estimation. We set the number of assets, $d_z = 10$, and consider resolutions, $\text{res} \in \{5, 10, 20\}$, thus giving covariance sample sizes of $s \in \{50, 100, 200\}$. In all experiments we set the risk aversion parameter $\delta = 1$.

The simulation process can be described as follows:

1. Generate the ground truth coefficients: $\boldsymbol{\theta}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_\theta})$.
2. Generate feature variables: $\mathbf{x}^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_x})$.
3. Generate 2000 return observations: $\mathbf{y}^{(i)} \sim \mathcal{N}(\text{diag}(\mathbf{x}^{(i)})\,\boldsymbol{\theta}, \tau\,\boldsymbol{\epsilon}^{(i)})$, where $\tau$ is chosen to meet the desired SNR.
4. Divide the sample data into two equally sized disjoint data sets: in-sample and out-of-sample.
5. Generate estimates $\hat{\mathbf{V}}^{(i)}$ using the chosen sample size, $s$.

<!-- Page 13 -->
6. Estimate the optimal OLS and IPO coefficients on the in-sample data.

7. Generate the optimal out-of-sample MVO decisions, $\mathbf{z}^*(\hat{\mathbf{y}})$, using the covariance estimates $\hat{\mathbf{V}}^{(i)}$ and corresponding optimal regression coefficients for predicting $\hat{\mathbf{y}}^{(i)}$.

8. Evaluate several performance metrics (described below) on the out-of-sample data.

9. Repeat steps 1-8 a total of 100 times and average the results.

**Performance metrics:** Let $\boldsymbol{\theta}_0$ be the ground truth and $\boldsymbol{\theta}$ denote an estimated (OLS or IPO) regression coefficient. Let $\mathbf{V}$ denote the true asset covariance and let $\{\mathbf{y}^{(i)}\}_{i=1}^m$ denote the realized return observations.

- **MVO Cost:** Let $\mathbf{z}^*(\hat{\mathbf{y}})$ be as defined in Equation (3). The out-of-sample MVO cost is then given by:
$$
c(\mathbf{z}^*(\hat{\mathbf{y}}), \mathbf{y}^{(i)}) = -\mathbf{z}^*(\hat{\mathbf{y}})^T \mathbf{y}^{(i)} + \frac{\delta}{2} \mathbf{z}^*(\hat{\mathbf{y}})^T \mathbf{V} \mathbf{z}^*(\hat{\mathbf{y}})
\quad (35)
$$

- **Proportion of variance explained:** a measure of return prediction accuracy on defined as:
$$
\text{PVE}(\boldsymbol{\theta}) = 1 - \mathbb{E}[(\mathbf{y}^{(i)} - \text{diag}(\mathbf{x}^{(i)}) \boldsymbol{\theta})^2] / \text{Var}(\mathbf{y}^{(i)}).
$$

We consider the case where the MVO program contains equality constraints. In particular, we enforce that the sum of the weights must be equal to one: $\mathbb{S} = \{\mathbf{z}^T \mathbf{1} = 1\}$. Figures 2 and 3 report the average and 95%-ile range of the out-of-sample MVO costs and PVE values, respectively, as a function of the SNR. Here, the covariance resolution is set to 20 and therefore the expected estimation error in $\hat{\mathbf{V}}^{(i)}$ is relatively low. As a result, we observe that the difference in both out-of-sample MVO cost and PVE is negligible, with the IPO model producing marginally lower MVO costs and the OLS model producing marginally higher PVE, as expected. Observe that even in the most optimistic case where estimation error in $\hat{\mathbf{V}}^{(i)}$ is low and the ground truth relationship is linear, there is no adverse repercussions in using the IPO model. Furthermore, in order to effectively eliminate estimation error we require a covariance resolution on the order of 20; which in practical terms implies that for a 100 asset portfolio we require a sample size of 2000 return observations. In many forecasting applications a sample size of this magnitude would be impractical and would potentially interfere with the observed time-varying dependency of asset volatilities and correlations [20, 8, 10, 9, 11]. Furthermore, as estimation error increases, we observe that for the majority of relevant SNRs, the IPO model produces a lower realized out-of-sample MVO cost. In particular Figures 4 and 6 demonstrate a statistically significant reduction in out-of-sample MVO costs as the covariance resolution decreases to 10 and 5, respectively. Interestingly, Figures 5 and 7 demonstrate that the IPO model produces lower realized MVO costs, despite providing lower average prediction accuracy, as measured by PVE. Note that this finding is consistent with the results presented in [19]. Finally, we observe in Figures 4 and 6 that the benefit of the IPO model is greatest when the ground truth correlation values, $\rho$, are closest to zero. Indeed this is intuitive as the extent of covariance estimation error in both magnitude and sign is largest when correlation values approach zero.

<!-- Page 14 -->
(a) $\rho = 0$, res = 20.

(b) $\rho = 0.25$, res = 20.

(c) $\rho = 0.50$, res = 20.

(d) $\rho = 0.75$, res = 20.

Figure 2: Out-of-sample MVO cost for IPO and OLS as of function of return signal-to-noise ratios.

(a) $\rho = 0$, res = 20.

(b) $\rho = 0.25$, res = 20.

(c) $\rho = 0.50$, res = 20.

(d) $\rho = 0.75$, res = 20.

Figure 3: Out-of-sample PVE for IPO and OLS as of function of return signal-to-noise ratios.

<!-- Page 15 -->
(a) $\rho = 0$, res = 10.

(b) $\rho = 0.25$, res = 10.

(c) $\rho = 0.50$, res = 10.

(d) $\rho = 0.75$, res = 10.

Figure 4: Out-of-sample MVO cost for IPO and OLS as of function of return signal-to-noise ratios.

(a) $\rho = 0$, res = 10.

(b) $\rho = 0.25$, res = 10.

(c) $\rho = 0.50$, res = 10.

(d) $\rho = 0.75$, res = 10.

Figure 5: Out-of-sample PVE for IPO and OLS as of function of return signal-to-noise ratios.

<!-- Page 16 -->
(a) $\rho = 0$, res = 5.

(b) $\rho = 0.25$, res = 5.

(c) $\rho = 0.50$, res = 5.

(d) $\rho = 0.75$, res = 5.

Figure 6: Out-of-sample MVO cost for IPO and OLS as of function of return signal-to-noise ratios.

(a) $\rho = 0$, res = 5.

(b) $\rho = 0.25$, res = 5.

(c) $\rho = 0.50$, res = 5.

(d) $\rho = 0.75$, res = 5.

Figure 7: Out-of-sample PVE cost for IPO and OLS as of function of return signal-to-noise ratios.

## 3.2 Simulation 2: computational efficiency

Here, we compare the computational efficiency of the analytical IPO solution with the current state-of-the-art method based on implicit differentiation and iterative gradient descent, from here on denoted as IPO-GRAD. Note that the IPO-GRAD implementation is optimized such that the matrix factorization (Equation (40)), required to compute gradient, is performed once at the initialization of the algorithm. The

<!-- Page 17 -->
IPO-GRAD coefficients are initialized by drawing from the standard normal distribution and the algorithm terminates when $\|\partial L/\partial \boldsymbol{\theta}\| < 10^{-6}$.

We generate synthetic asset returns, following the procedure outlined in Section 3.1, with $\rho = 0$, SNR $= 0.005$ and varying the number of assets, $d_z \in \{25, 50, 100, 250\}$. Each asset is assumed to have 3 unique features, and therefore $d_\theta = 3d_z$. Tables 1 and 2, report the time, in seconds, taken by each method to compute the optimal regression coefficients for the unconstrained and equality constrained cases, respectively. For the IPO-GRAD method we also report the number of iterations of gradient descent. For each portfolio size, we report the average and 95%-ile range over 100 instances of simulated data. Observe that for problems with 100 or fewer assets, the computation time required to compute the optimal IPO coefficients analytically is comparable to the computation time required to compute the optimal OLS coefficients. In contrast, the IPO-GRAD method typically requires over 100 iterations of gradient descent and is anywhere from 10x - 1000x slower than the corresponding IPO method. We note that for problems of larger scale, the analytical IPO solution remains tractable and is on average 6x faster than the IPO-GRAD method.

| No. Assets | OLS           | IPO           | IPO-GRAD       | Iterations    |
|------------|---------------|---------------|----------------|---------------|
| 25         | 0.029         | 0.071         | 4.333          | 178           |
|            | (0.028,0.032) | (0.07,0.08)   | (3.966,5.076)  | (164,210)     |
| 50         | 0.247         | 0.429         | 6.557          | 186           |
|            | (0.209,0.253) | (0.342,0.447) | (6.032,7.278)  | (173,207)     |
| 100        | 0.545         | 1.7           | 17.642         | 200           |
|            | (0.491,0.638) | (1.495,1.837) | (16.03,21.301) | (183,247)     |
| 250        | 2.89          | 17.961        | 123.975        | 208.5         |
|            | (2.75,3.335)  | (17.546,18.092)| (114.008,165.094)| (193,279)   |

Table 1: Time in seconds for computing the optimal OLS, IPO and IPO-GRAD coefficients for an unconstrained MVO problem. Results are averaged over 100 instances of simulated data.

| No. Assets | OLS           | IPO           | IPO-GRAD       | Iterations    |
|------------|---------------|---------------|----------------|---------------|
| 25         | 0.029         | 0.088         | 4.664          | 176           |
|            | (0.028,0.032) | (0.085,0.094) | (4.333,5.587)  | (163,211)     |
| 50         | 0.247         | 0.473         | 7.389          | 188           |
|            | (0.171,0.259) | (0.383,0.543) | (6.696,8.227)  | (172,208)     |
| 100        | 0.549         | 2.025         | 19.711         | 200           |
|            | (0.492,0.669) | (1.855,2.161) | (18.034,23.449)| (183,241)     |
| 250        | 2.815         | 22.378        | 129.348        | 208           |
|            | (2.71,3.315)  | (21.8,22.511) | (119.684,174.607)| (193,280)   |

Table 2: Time in seconds for computing the optimal OLS, IPO and IPO-GRAD coefficients for an equality constrained MVO problem. Results are averaged over 100 instances of simulated data.

<!-- Page 18 -->
## 3.3 Simulation 3: inequality constrained IPO

We now consider the more general case whereby the feasible region of the MVO program is defined by inequality constraints. In general, an analytical solution to the IPO Program (10) in the presence of lower-level inequality constraints is not possible. Furthermore, Program (10) is not convex in $\boldsymbol{\theta}$. As a result, the current state-of-the-art approach (IPO-GRAD), described in Section 2.2, is recommended in order to obtain locally optimal solutions.

The IPO-GRAD solution, however, is challenging for several reasons. First, in contrast to the traditional OLS approach, estimating the IPO coefficients by iterative methods can be computationally expensive. Specifically, at each iteration of gradient descent the IPO-GRAD method must solve at most $m$ constrained quadratic programs, where $m$ is the total number of training observations. Computation time therefore scales linearly with the number of training examples, $m$, and the number of iterations of gradient descent, $n$. Moreover, when solved by interior-point methods, convex quadratic programs have worst-case time complexity on the order of $\mathcal{O}(d_z^3)$ [25], and therefore the worst-case complexity for IPO-GRAD is on the order of $\mathcal{O}(mnd_z^3)$. Fortunately, in most practical settings, quadratic programs are solved in substantially fewer iterations than their worst-case bound [12]. Nonetheless, most real-world portfolio optimization problems involve portfolio sizes on the order of 10 or 100 and are trained using thousands of training observations. Estimating prediction model parameters by IPO-GRAD can therefore be computationally expensive. Secondly, because the inequality constrained IPO problem is not convex, we have no guarantee that any particular local solution is globally optimal. Moreover, if is difficult to estimate $\text{Var}(\boldsymbol{\theta})$ and compute confidence intervals by standard parametric methods, and instead expensive nonparametric bootstrap methods are required.

As a heuristic, we are interested in determining the out-of-sample efficacy of the analytical IPO solutions, presented in Sections 2.3 - 2.4, applied to the inequality constrained problem. Specifically, we compute the IPO optimal coefficients analytically by dropping the inequality constraints in the lower-level MVO problem. The realized policy, $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$, however, enforces the inequality constraints in the out-of-sample evaluation period.

We generate synthetic asset returns, following the procedure outlined in Section 3.1, with $\rho = 0$, SNR = 0.005 and $d_z = 10$. Each asset is assumed to have 3 unique features ($d_\theta = 3d_z$). The inequality constraints are standard box-constraints of the form:

$$
-\gamma \leq \mathbf{z}_j \leq \gamma, \quad \forall j \in \{1, ..., d_z\},
$$

and we consider several values of $\gamma \in \{0.05, 0.10, 0.25, 0.50, 0.75, 1, 2, 5, 10\}$. We also vary the risk aversion parameter $\delta \in \{1, 5, 10, 25\}$. Finally, asset mean returns are generated according to linear and nonlinear polynomial models of the form:

$$
\mathbf{y}^{(i)} = \sum_{q=1}^{p} \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)})^q \boldsymbol{\theta}_q + \tau \, \boldsymbol{\epsilon}^{(i)},
$$

with $p \in \{1, 2, 4\}$.

Figures 8 - 10 compare the out-of-sample MVO cost for the IPO and IPO-GRAD methods as of function of the box constraint value, $\gamma$, and risk-aversion parameter, $\delta$. For each value of $\gamma, \delta$ and $p$, we report the mean and 95%-ile range over 30 instances of simulated data. First, we would expect the out-of-sample performance of the IPO and IPO-GRAD methods to converge as $\gamma$ increases. Furthermore as $\delta$, increases, the point (along $\gamma$) at which the two solutions converge will naturally decrease. This effect is purely a consequence of the inequality constraints being non-active when either $\gamma$ and/or $\delta$ are sufficiently large.

<!-- Page 19 -->
In Figure 8 asset returns are generated according to a linear ground truth model ($p = 1$). In all cases we observe that the IPO-GRAD does provide improved out-of-sample MVO costs when $\gamma$ is sufficiently small ($\gamma < 0.5$). However, for moderate and large values of $\gamma$, the IPO-GRAD method provides no improvement in out-of-sample MVO costs in comparison to the IPO method. Furthermore, in Figures 9 and 10, asset returns are generated according to a quadratic ($p = 2$) and quartic ($p = 4$) ground truth model, respectively. We observe that over practically every value of $\gamma$ and $\delta$, the IPO method provides an equivalent, if not improved, out-of-sample MVO costs in comparison to the IPO-GRAD method. We note that, while not explicitly shown here, the IPO-GRAD method produces lower in-sample (training) MVO costs over every experiment instance, and is potentially overfitting the training data. Moreover, we note that in all experiments, the variance of the out-of-sample MVO costs generated by the IPO-GRAD method is substantially larger than that of the IPO method. The lack of convexity and uniqueness of solution in the IPO-GRAD formulation, along with the likelihood of model overfit, provides a potential explanation for this effect.

Finally, Table 3 reports the average time (in seconds) and 95%-ile range, taken by each method to compute the optimal regression coefficients. The results are averaged over all 360 instances of simulated data. For the IPO-GRAD method we also report the number of iterations of gradient descent. We observe that the IPO-GRAD method typically requires around 60 iterations of gradient descent and is on average 100x - 1000x slower than the corresponding IPO method. Note that the computation times reported here are for a relatively small portfolio and, given the computational complexity described above, we would expect the IPO method to provide an even larger computational advantage on medium and large sized portfolios. We therefore conclude that in the presence of inequality constraints, the IPO heuristic is a compelling alternative to the more computationally expensive IPO-GRAD solution.

Figure 8: Out-of-sample MVO costs as of function of box constraint value with $p = 1$.

<!-- Page 20 -->
Figure 9: Out-of-sample MVO costs as of function of box constraint value with $p = 2$.

Figure 10: Out-of-sample MVO costs as of function of box constraint value with $p = 4$.

## 4 Experiments

**Experiment Setup:**

We consider an asset universe of 24 commodity futures markets, described in Table 11. The daily price data

<!-- Page 21 -->
| IPO       | IPO-GRAD      | Iterations |
|-----------|---------------|----------|
| 0.023     | 14.389        | 60       |
| (0.022,0.024) | (6.9417,22.5718) | (29,94)  |

Table 3: Time in seconds for computing the optimal IPO and IPO-GRAD coefficients for an inequality constrained MVO problem. Results are averaged over 360 instances of simulated data.

is given from March 1986 through December 2020, and is provided by Commodity Systems Inc. Futures contracts are rolled on, at most, a monthly basis in order to remain invested in the most liquid contract, as measured by open-interest and volume. Arithmetic returns are computed directly from the roll-adjusted price data.

In each experiment we follow Zumbach [44] and estimate the covariance matrix using an exponential moving average with a decay rate of 0.94. We consider both univariate and multivariate prediction models. The feature, $\{\mathbf{x}^{(i)}\}$, for univariate models is the 252-day average return, or trend, for each market. The feature therefore represents a measure of the well-documented ‘trend’ factor, popular to many Commodity Trading Advisors (CTAs) and Hedge Funds (see for example [3], [13],[36]). The features for multivariate models are the 252-day trend and the carry for each market. We follow Koijen et al. [34] and define the carry as the expected convenience yield, or cost, for holding that commodity, and is estimated by the percent difference in price between the two futures contracts closest to expiry.

As we will see below, the majority of the IPO and OLS regression coefficients are not statistically significant at an individual market level. Indeed this is common and well document in many applications of financial forecasting (see for example [29, 36]). The lack of statistical significance may be indicative of low signal-to-noise levels and/or forecasting model misspecification. Furthermore, the absence of statistical significance does not prohibit the development of profitable portfolio level trading strategies and indeed we observe in Table 4 that the features are statistically significant at the 95%-ile level when evaluated across all markets.

| Feature | Coefficient | Std. Error | T-Statistic | P-Value |
|---------|-------------|------------|-------------|---------|
| Carry   | 0.3300      | 0.1654     | 1.9953      | 0.0460  |
| Trend   | 0.0942      | 0.0324     | 2.9101      | 0.0036  |

Table 4: Univariate regression coefficients and t-statistic summary aggregated across all available markets.

Each day we form the optimal portfolio weight, $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$ at the close of day $i$, and assume execution at the following close, $i+1$. In each experiment, described below, we consider two models for estimating asset returns:

1. **OLS**: ordinary-least squares with prediction coefficients, $\hat{\boldsymbol{\theta}}$.
2. **IPO**: integrated prediction and optimization, where $\boldsymbol{\theta}^*$ is determined by the IPO optimization framework described in Section 2.

We consider 6 experiments:

1. Unconstrained MVO program with univariate regression.

<!-- Page 22 -->
2. Unconstrained MVO program with multivariate regression.

3. Equality constrained MVO program with univariate regression.

4. Equality constrained MVO program with multivariate regression.

5. Inequality constrained MVO program with univariate regression.

6. Inequality constrained MVO program with multivariate regression.

The equality constrained MVO programs are market-neutral: $\mathbb{S} = \{\mathbf{z}^T \mathbf{1} = 0\}$, whereas the inequality constrained MVO programs are both market-neutral and include lower bound and upper bound market constraints:

$$
\mathbb{S} = \{\mathbf{z}^T \mathbf{1} = 0, -0.125 \leq \mathbf{z} \leq 0.125\}.
$$

Note that the results and discussion for the equality constrained MVO and multivariate models are very similar to that of the unconstrained and inequality constrained MVO with univariate prediction models and can be found in Appendix B. In order to provide realistic annualized volatilities in the $10\% - 20\%$ range, we fix the risk-aversion parameter to $\delta = 50$. For each experiment, the initial parameter estimation is performed using the first 14 years of data (March 1986 through December 1999). Out-of-sample testing begins in January 2000 and ends in December 2020. We apply a walk-forward training and testing framework whereby the optimal regressions coefficients are updated every 2 years using all available training data at that point in time. Performance is in excess of the risk-free rate and gross of trading costs.

Each model is evaluated on absolute and relative terms, with a focus on out-of-sample MVO cost and out-of-sample Sharpe ratio cost, provided by Equation (36).

$$
c_{\text{MVO}}(\mathbf{z}, \mathbf{y}) = -\mu(\mathbf{z}, \mathbf{y}) + \frac{\delta}{2} \sigma^2(\mathbf{z}, \mathbf{y}), \quad \text{and} \quad c_{\text{SR}}(\mathbf{z}, \mathbf{y}) = -\frac{\mu(\mathbf{z}, \mathbf{y})}{\sigma(\mathbf{z}, \mathbf{y})}
\tag{36}
$$

where

$$
\mu(\mathbf{z}, \mathbf{y}) = \frac{1}{m} \sum_{i=1}^{m} \mathbf{z}^{T^{(i)}} \mathbf{y}^{(i)} \quad \text{and} \quad \sigma^2(\mathbf{z}, \mathbf{y}) = \frac{1}{m} \sum_{i=1}^{m} (\mathbf{z}^{T^{(i)}} \mathbf{y}^{(i)} - \mu(\mathbf{z}, \mathbf{y}))^2,
$$

denote the mean and variance of realized daily returns. To quantify the consistency of observed performance metrics, we bootstrap the out-of-sample returns generated by each model using 1000 samples as follows:

1. For each $k \in \{1, 2, ..., 1000\}$, sample, without replacement, a batch, $B_k$, with $|B_k| = 252$ observations (1 year) from the out-of-sample period.

2. For each sample and model, compute the realized MVO and Sharpe ratio costs using Equation (36).

In order to fairly compare the realized costs we ensure that each model uses identical bootstrap observations. We report the dominance ratio (DR); namely the proportion of samples for which the realized cost of the IPO model is less than that of the OLS model.

Our experiments should be interpreted as a proof-of-concept, rather than a fully comprehensive financial study. That said, we believe that the results presented below provide compelling evidence for using IPO for estimating regression coefficients. In general, the IPO models exhibit lower out-of-sample MVO costs and improved economic outcomes in comparison to the traditional OLS-based ‘predict, then optimize’ approach.

<!-- Page 23 -->
## 4.1 Experiment 1: unconstrained with univariate predictions

Economic performance metrics and average out-of-sample MVO costs are provided in Table 5 for the time period of 2000-01-01 to 2020-12-31 for the unconstrained MVO portfolios with univariate prediction models. Equity growth charts for the same time period are provided in Figure 11. We first observe that the IPO model provides higher absolute and risk-adjusted performance, as measured by the MVO cost and Sharpe ratio. Indeed the IPO model produces an out-of-sample MVO cost that is approximately 50% lower and a Sharpe ratio that is approximately 100% larger than that of the OLS model. Furthermore, the IPO models provide more conservative risk metrics, as measured by portfolio volatility, value-at-risk (VaR), and average drawdown (Avg DD).

In Figure 12 we compare the realized MVO and Sharpe ratio costs across 1000 out-of-sample realizations. In general we observe that the IPO model exhibits consistently lower MVO costs and generally higher Sharpe ratios than the OLS model. In Figure 12(a) we report a dominance ratio of 97% meaning that the IPO model realizes a lower MVO cost in 97% of samples in comparison to the OLS model. Figure 12(b) reports a dominance ratio of 68%.

In Figure 13 we report the estimated univariate regression coefficients and $\pm 1$ standard error bar for the last out-of-sample data fold. As stated earlier, it is clear that the majority of the IPO and OLS regression coefficients are not statistically significant at an individual market basis. Note that for some markets, the IPO model provides very different regression coefficients, in both magnitude and sign, compared to the OLS coefficients. In particular we observe that, with the exception of Cocoa (CC), all IPO regression coefficients are positive. In contrast, 33% of OLS coefficients are negative.

![Figure 11: Out-of-sample log-equity growth for the unconstrained mean-variance program and multivariate IPO and OLS prediction model.](image_placeholder)

|            | Annual Return | Sharpe Ratio | Volatility | Avg Drawdown | Value at Risk | MVO Cost |
|------------|---------------|--------------|----------|--------------|---------------|----------|
| IPO        | 0.1026        | 0.7593       | 0.1352   | -0.0275      | -0.0107       | 0.3544   |
| OLS        | 0.0644        | 0.3735       | 0.1725   | -0.0426      | -0.0142       | 0.6792   |

Table 5: Out-of-sample MVO costs and economic performance metrics for unconstrained mean-variance portfolios with univariate IPO and OLS prediction models.

<!-- Page 24 -->
(a) Out-of-sample MVO cost.  
(b) Out-of-sample Sharpe ratio cost.

Figure 12: Realized out-of-sample MVO and Sharpe ratio costs for the unconstrained mean-variance program and univariate IPO and OLS prediction models.

Figure 13: Optimal IPO and OLS regression coefficients for the unconstrained mean-variance program and univariate prediction model.

## 4.2 Experiment 5: inequality constrained with univariate predictions

Economic performance metrics and average out-of-sample MVO costs are provided in Table 6 for the time period of 2000-01-01 to 2020-12-31 for the constrained MVO portfolios with univariate prediction models. Equity growth charts for the same time period are provided in Figure 14. First, observe that the annual returns, risk and MVO costs are substantially smaller in the presence of portfolio constraints. Indeed this is consistent with the fact that box constraints are themselves a form of portfolio model regularization [31]. Nonetheless, we observe that the IPO model produces an out-of-sample MVO cost that is approximately 50% lower and a Sharpe ratio that is approximately 85% larger than that of the OLS model. In Figure 25 we compare the realized MVO and Sharpe ratio costs across 1000 bootstrapped sample realizations. Again we observe that the IPO model produces lower MVO and Sharpe ratio costs on average. Observe, however, that the dominance ratios are more modest, with values in the 60%-70% range. This result is intuitive and we would expect the out-of-sample performance of the two models to converge as the portfolio constraints become more strict. Indeed the IPO and OLS model would yield identical results in the limit where the portfolio constraints define a single weight, irrespective of the mean and covariance estimation. Lastly, in Figure 16 we report the estimated univariate regression coefficients and $\pm 1$ standard error bar for the last out-of-sample data fold. Recall that the IPO coefficients are obtained by first dropping the inequality constraints and then solving analytically for $\theta^*$ by Equation (30). The observations and differences between the optimal IPO and OLS coefficients are similar to those discussed in Section 4.1.

<!-- Page 25 -->
Figure 14: Out-of-sample log-equity growth for the inequality constrained mean-variance program and multivariate IPO and OLS prediction model.

|          | Annual Return | Sharpe Ratio | Volatility | Avg Drawdown | Value at Risk | MVO Cost |
|----------|---------------|--------------|----------|--------------|---------------|----------|
| IPO      | 0.0324        | 0.6310       | 0.0513   | -0.0116      | -0.0052       | 0.0335   |
| OLS      | 0.0181        | 0.3421       | 0.0529   | -0.0174      | -0.0053       | 0.0520   |

Table 6: Out-of-sample MVO costs and economic performance metrics for inequality constrained mean-variance portfolios with univariate IPO and OLS prediction models.

(a) Out-of-sample MVO cost.  
(b) Out-of-sample Sharpe ratio cost.

Figure 15: Realized out-of-sample MVO and Sharpe ratio costs for the inequality constrained mean-variance program and univariate IPO and OLS prediction models.

Figure 16: Optimal IPO and OLS regression coefficients for the equality constrained mean-variance program and univariate prediction model.

<!-- Page 26 -->
# 5 Conclusion and future work

In this paper we proposed an integrated prediction and optimization (IPO) framework for optimizing regression coefficients in the context of a mean-variance portfolio optimization. We structured the integrated problem as a bi-level program with a series of lower-level mean-variance optimization programs. We investigated the IPO framework under both univariate and multivariate regression settings and considered the MVO program under various forms of constraints. In a general setting, we presented the current state-of-the-art approach (IPO-GRAD) and restructured the IPO problem as a neural network with a differentiable quadratic programming layer. Where possible, we provided closed-form analytical solutions for the optimal IPO regression coefficients, $\boldsymbol{\theta}^*$, and the sufficient conditions for uniqueness. We described the sampling distribution properties of $\boldsymbol{\theta}^*$ and provided the conditions for which $\boldsymbol{\theta}^*$ is an unbiased estimator of $\boldsymbol{\theta}$ and provided the expression for the variance.

Extensive numerical simulations demonstrate the computational and performance advantage of the analytical IPO methodology. We demonstrated that, over a wide range of realistic signal-to-noise ratios, the IPO model outperforms the OLS model in terms of minimizing out-of-sample MVO costs. This is true even when the underlying ‘ground-truth’ return generating process is linear in the feature variables. We demonstrated, for a wide range of portfolio sizes, the computational advantage of computing the IPO coefficients analytically, which is on average 10x–1000x faster than the IPO-GRAD methodology. We briefly discussed the computational complexity of the IPO-GRAD methodology and proposed a heuristic which drops the inequality constraints during parameter estimation and invokes the analytical IPO solution. We find that in many instances the IPO-GRAD model overfits the training data, whereas the analytical IPO model produces solutions with lower out-of-sample variance, and in some cases, improved out-of-sample MVO costs. We concluded with several experiments using global futures data, under various forms of constraints and prediction model specifications. Out-of-sample results demonstrate that the IPO model provided lower realized MVO costs and superior economic performance in comparison to the traditional OLS ‘predict then optimize’ approach.

In the presence of general inequality constraints we determined that the current state-of-the-art IPO model is computationally expensive and has a tendency to overfit the training data. We believe that methods for regularizing both the prediction and the decision optimization program, as well as methods for choosing the ‘best’ feature subsets in an integrated setting are interesting areas of future research.

# 6 Data Availability Statement

The data used for experiments was obtained from Commodity Systems Inc: https://www.csidata.com.

# References

[1] Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, and J. Zico Kolter. Differentiable convex optimization layers. In *Advances in Neural Information Processing Systems*, volume 32, pages 9562–9574. Curran Associates, Inc., 2019.

[2] Brandon Amos and J. Zico Kolter. Optnet: Differentiable optimization as a layer in neural networks, 2017. URL https://arxiv.org/abs/1703.00443.

<!-- Page 27 -->
[3] Nick Baltas and Robert Kosowski. Momentum strategies in futures markets and trend-following funds. *SSRN Electronic Journal*, 2012. doi: 10.2139/ssrn.1968996.

[4] Gah-Yi Ban and Cynthia Rudin. The big data newsvendor: Practical insights from machine learning. *Operations Research*, 67(1):90–108, 2019.

[5] L. Bauwens, S. Laurent, and J. Rombouts. Multivariate garch models: A survey. *Econometrics eJournal*, 2003.

[6] Dimitris Bertsimas and Nathan Kallus. From predictive to prescriptive analytics. *Management Science*, 66(3):1025–1044, 2020.

[7] Michael J. Best and Robert R. Grauer. On the sensitivity of mean-variance-efficient portfolios to changes in asset means: Some analytical and computational results. *The Review of Financial Studies*, 4(2):315–342, 1991.

[8] Tim Bollerslev. Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3):307–327, 1986. ISSN 0304-4076.

[9] Tim Bollerslev. Modelling the coherence in short-run nominal exchange rates: A multivariate generalized arch model. *The Review of Economics and Statistics*, 72(3):498–505, 1990. ISSN 00346535, 15309142. URL http://www.jstor.org/stable/2109358.

[10] Tim Bollerslev, Robert F. Engle, and Jeffrey M. Wooldridge. A capital asset pricing model with time-varying covariances. *Journal of Political Economy*, 96(1):116–131, 1988.

[11] Tim Bollerslev, Robert Engle, and Daniel B. Nelson. Arch models. In R. F. Engle and D. McFadden, editors, *Handbook of Econometrics*, volume 4, pages 2961–3031. Elsevier, 1 edition, 1994.

[12] Stephen Boyd and Lieven Vandenberghe. *Convex Optimization*. Cambridge University Press, 2004. doi: 10.1017/CBO9780511804441.

[13] Benjamin Bruder, Tung-Lam Dao, Jean-Charles Richard, and Thierry Roncalli. Trend filtering methods for momentum strategies. *SSRN Electronic Journal*, 12 2011. doi: 10.2139/ssrn.2289097.

[14] Binbin Chen, Shih-Feng Huang, and Guangming Pan. High dimensional mean variance optimization through factor analysis. *Journal of Multivariate Analysis*, 133:140–159, 2015. ISSN 0047-259X.

[15] Vijay Kumar. Chopra and William T. Ziemba. The effect of errors in means, variances, and covariances on optimal portfolio choice. *The Journal of Portfolio Management*, 19(2):6–11, 1993. ISSN 0095-4918. doi: 10.3905/jpm.1993.409440. URL https://jpm.pm-research.com/content/19/2/6.

[16] Roger G Clarke, Harindra de Silva, and Robert Murdock. A factor approach to asset allocation. *The Journal of Portfolio Management*, 32(1):10–21, 2005.

[17] Priya Donti, Brandon Amos, and J. Zico Kolter. Task-based end-to-end model learning in stochastic optimization. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 30, pages 5484–5494. Curran Associates, Inc., 2017.

<!-- Page 28 -->
[18] Holger Drees and Catalin Starica. A simple non-stationary model for stock returns, 2002.

[19] Adam Elmachtoub and Paul Grigas. Smart ‘predict, then optimize’. *Management Science*, 10 2017. doi: 10.1287/mnsc.2020.3922.

[20] Robert F. Engle. Autoregressive conditional heteroskedasticity with estimates of the variance of uk inflation. *Econometrica*, 50(1):987–1008, 1982.

[21] Eugene F. Fama and Kenneth R. French. The cross section of expected stock returns. *Journal of Financial Finance*, 47(2), 1992.

[22] Eugene F. Fama and Kenneth R. French. Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(3):3–56, 1993.

[23] Eugene F. Fama and Kenneth R. French. A five-factor asset pricing model. *Journal of Financial Economics*, 116(1):1–22, 2015. ISSN 0304-405X.

[24] D. Goldfarb and G. Iyengar. Robust portfolio selection problems. *Mathematics of Operations Research*, 28(1):1–38, 2003.

[25] D. Goldfarb and Shucheng Liu. An o(n3l) primal interior point algorithm for convex quadratic programming. *Mathematical Programming*, 49:325–340, 1991.

[26] Stephen Gould, Basura Fernando, Anoop Cherian, Peter Anderson, Rodrigo Santa Cruz, and Edison Guo. On differentiating parameterized argmin and argmax problems with application to bi-level optimization, 2016. URL http://arxiv.org/abs/1607.05447.

[27] Trevor Hastie, Robert Tibshirani, and Ryan J. Tibshirani. Extended comparisons of best subset selection, forward stepwise selection, and the lasso, 2017. URL https://arxiv.org/abs/1707.08692.

[28] Der-Ann Hsu, Robert B. Miller, and Dean W. Wichern. On the stable paretian behavior of stock-market prices. *Journal of the American Statistical Association*, 69(345):108–113, 1974.

[29] Dashan Huang, Jiangyuan Li, Liyao Wang, and Guofu Zhou. Time series momentum: Is it there? *Journal of Financial Economics*, 135(3):774–794, 2020. ISSN 0304-405X.

[30] Ronen Israel, Bryan Kelly, and Tobias Moskowitz. Can machines learn finance. *Journal Of Investment Management*, 18, 11 2020.

[31] Ravi Jagannathan and Tongshu Ma. Risk reduction in large portfolios: Why imposing the wrong constraints helps. *The Journal of Finance*, 58(4):1651–1683, 2003.

[32] A. G. Jones and C. Taylor. Solving inverse problems in computer vision by scale space reconstruction. In *MVA*, 1994.

[33] Rohit Kannan, Guzin Bayraksan, and James R. Luedtke. Data-driven sample average approximation with covariate information, 2022. URL https://arxiv.org/abs/2207.13554.

[34] Ralph S.J. Koijen, Tobias J. Moskowitz, Lasse Heje Pedersen, and Evert B. Vrugt. Carry. *Journal of Financial Economics*, 127(2):197–225, 2018.

<!-- Page 29 -->
[35] H. Markowitz. Portfolio selection. *Journal of Finance*, 7(1):77–91, 1952.

[36] Tobias J. Moskowitz, Yao Hua Ooi, and Lasse Pedersen. Time series momentum. *Journal of Financial Economics*, 104(2):228–250, 2012.

[37] R.R. Officer. A time series examination of the market factor of the new york stock exchange. *University of Chicago PhD dissertation*, 1971.

[38] Gregory Ongie, Ajil Jalal, Christopher A. Metzler, Richard G. Baraniuk, Alexandros G. Dimakis, and Rebecca Willett. Deep learning techniques for inverse problems in imaging, 2020.

[39] P. Ranjan, R. Haynes, and R. Karsten. A Computationally Stable Approach to Gaussian Process Interpolation of Deterministic Computer Simulation Data. *Technometrics*, 52(4):366–378, 2011.

[40] Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczynski. *Lectures on Stochastic Programming*. Society for Industrial and Applied Mathematics, 2009. doi: 10.1137/1.9780898718751.

[41] Ankur Sinha, Tanmay Khandait, and Raja Mohanty. A gradient-based bilevel optimization approach for tuning hyperparameters in machine learning, 2020. URL https://arxiv.org/abs/2007.11022.

[42] Catalin Starica and Clive Granger. Nonstationarities in stock returns. *The Review of Economics and Statistics*, 87(3):503–522, 2005.

[43] V. Vapnik. Principles of risk minimization for learning theory. In J. Moody, S. Hanson, and R. P. Lippmann, editors, *Advances in Neural Information Processing Systems*, volume 4, pages 831–838. Morgan-Kaufmann, 1992.

[44] Gilles Zumbach. The riskmetrics 2006 methodology. *Econometrics: Applied Econometrics & Modeling eJournal*, 2007.

<!-- Page 30 -->
# A Appendix

## A.1 Neural network implementation details

In the general case we seek to determine a locally optimal solution, $\boldsymbol{\theta}^*$, to the IPO program by restructuring Program (1) as an end-to-end neural network and applying (stochastic) gradient descent. For compactness, we have temporarily dropped the index notation. From the multivariate chain-rule, the gradient of the IPO objective, $\nabla_{\theta} L$, can be expressed as:

$$
\nabla_{\theta} L = \frac{\partial L}{\partial \mathbf{z}^*} \frac{\partial \mathbf{z}^*}{\partial \hat{\mathbf{y}}} \frac{\partial \hat{\mathbf{y}}}{\partial \boldsymbol{\theta}}.
\tag{37}
$$

In our case, the MVO cost function, $c$, is smooth and twice differentiable over decision variables, $\mathbf{z}$, and therefore it is relatively straightforward to compute the gradient $\partial L / \partial \mathbf{z}^*$. The Jacobian, $\partial \mathbf{z}^* / \partial \hat{\mathbf{y}}$, requires differentiation through the argmin operator. Amos and Kolter [2] demonstrate that rather than forming the Jacobian directly, we can instead compute $\partial L / \partial \hat{\mathbf{y}}$ by implicit differentiation of the system of equations provided by the Karush–Kuhn–Tucker (KKT) conditions at the optimal solution $\mathbf{z}^*$ to program (11).

We follow the work of Amos and Kolter [2] and begin by first writing the Lagrangian of program (11):

$$
\mathcal{L}(\mathbf{z}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = -\mathbf{z}^T \hat{\mathbf{y}} + \frac{\delta}{2} \mathbf{z}^T \hat{\mathbf{V}} \mathbf{z} + \boldsymbol{\lambda}^T (\mathbf{G} \mathbf{z} - \mathbf{h}) + \boldsymbol{\nu}^T (\mathbf{A} \mathbf{z} - \mathbf{b}),
\tag{38}
$$

where $\boldsymbol{\lambda} \in \mathbb{R}^{d_{iq}}$ and $\boldsymbol{\nu} \in \mathbb{R}^{d_{eq}}$ are the dual variables of the inequality and equality constraints, respectively. The KKT optimality conditions for stationarity, primal feasibility, and complementary slackness are given by equations (39).

$$
\begin{aligned}
-\hat{\mathbf{y}} + \frac{\delta}{2} \hat{\mathbf{V}} \mathbf{z}^* + \mathbf{G}^T \boldsymbol{\lambda}^{*T} + \mathbf{A}^T \boldsymbol{\nu}^* &= 0 \\
(\mathbf{G} \mathbf{z}^* - \mathbf{h}) &\leq 0 \\
\boldsymbol{\lambda}^* &\geq 0 \\
\boldsymbol{\lambda}^* \cdot (\mathbf{G} \mathbf{z}^* - \mathbf{h}) &= 0 \\
\mathbf{A} \mathbf{z}^* &= \mathbf{b}
\end{aligned}
\tag{39}
$$

Following Amos and Kolter [2], we take the differentials of these conditions to yield the following system of equations:

$$
\begin{bmatrix}
\delta \hat{\mathbf{V}} & \mathbf{G}^T & \mathbf{A}^T \\
\text{diag}(\boldsymbol{\lambda}^*) \mathbf{G} & \text{diag}(\mathbf{G} \mathbf{z}^* - \mathbf{h}) & 0 \\
\mathbf{A} & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\mathrm{d}\mathbf{z} \\
\mathrm{d}\boldsymbol{\lambda} \\
\mathrm{d}\boldsymbol{\nu}
\end{bmatrix}
= -
\begin{bmatrix}
\delta \mathrm{d}\mathbf{V} \mathbf{z}^* - \mathrm{d}\mathbf{y} + \mathrm{d}\mathbf{G}^T \boldsymbol{\lambda}^* + \mathrm{d}\mathbf{A}^T \boldsymbol{\nu}^* \\
\text{diag}(\boldsymbol{\lambda}^*) \mathrm{d}\mathbf{G} \mathbf{z}^* - \text{diag}(\boldsymbol{\lambda}^*) \mathrm{d}\mathbf{h} \\
\mathrm{d}\mathbf{A} \mathbf{z}^* - \mathrm{d}\mathbf{b}
\end{bmatrix}.
\tag{40}
$$

Amos and Kolter [2] make two important observations about the system of equations (40). The first, is that the left side matrix gives the optimality conditions of the convex quadratic problem, which, when solving by interior-point methods, must be factorized in order to obtain the solution to the decision program [12]. Secondly, the right side gives the differentials of the relevant functions at the achieved solution with respect to any of the input parameters. In particular, we seek to compute the Jacobian $\partial \mathbf{z}^* / \partial \hat{\mathbf{y}}$. As explained by Amos and Kolter [2], the Jacobian $\partial \mathbf{z}^* / \partial \hat{\mathbf{y}}$ is obtained by letting $\mathrm{d}\mathbf{y} = \mathbf{I}$ (setting all other differential terms to zero) and solving the system of equations for $\mathrm{d}\mathbf{z}$. From a computation standpoint, the required Jacobian

<!-- Page 31 -->
is therefore effectively obtained ‘for free’ upon factorization of the left matrix when obtaining the solution, $\mathbf{z}^*$, in the forward pass.

In practice it is inefficient to compute the Jacobian matrix $\partial \mathbf{z}^* / \partial \hat{\mathbf{y}}$. Instead we compute $\partial c / \partial \hat{\mathbf{y}}$ directly by multiplying the backward pass vector by the inverse of the transposed left-hand-side matrix, as shown in equation (41).

$$
\begin{bmatrix}
\bar{\mathbf{d}}_{\mathbf{z}} \\
\bar{\mathbf{d}}_{\boldsymbol{\lambda}} \\
\bar{\mathbf{d}}_{\boldsymbol{\nu}}
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial c}{\partial \hat{\mathbf{y}}} \\
- \\
-
\end{bmatrix}
= -
\begin{bmatrix}
\delta \hat{\mathbf{V}} & \mathbf{G}^T \operatorname{diag}(\boldsymbol{\lambda}^*) & \mathbf{A}^T \\
\mathbf{G} & \operatorname{diag}(\mathbf{G} \mathbf{z}^* - \mathbf{h}) & 0 \\
\mathbf{A} & 0 & 0
\end{bmatrix}^{-1}
\begin{bmatrix}
\left( \frac{\partial c}{\partial \mathbf{z}^*} \right)^T \\
0 \\
0
\end{bmatrix}.
\tag{41}
$$

More generally, equation (41) allows for efficient computation of the gradients with respect to any of the MVO input problem variables. For the reader’s interest, we state the gradients for all other problem variables and refer the reader to Amos and Kolter [2] for their derivation.

$$
\begin{aligned}
\frac{\partial c}{\partial \hat{\mathbf{V}}} &= \frac{1}{2} \left( \bar{\mathbf{d}}_{\mathbf{z}} {\mathbf{z}^*}^T + \mathbf{z}^* \bar{\mathbf{d}}_{\mathbf{z}}^T \right) \\
\frac{\partial c}{\partial \mathbf{A}} &= \bar{\mathbf{d}}_{\boldsymbol{\nu}} {\mathbf{z}^*}^T + \boldsymbol{\nu}^* \bar{\mathbf{d}}_{\mathbf{z}}^T \\
\frac{\partial c}{\partial \mathbf{G}} &= \operatorname{diag}(\boldsymbol{\lambda}^*) \bar{\mathbf{d}}_{\boldsymbol{\lambda}} {\mathbf{z}^*}^T + \boldsymbol{\lambda}^* \bar{\mathbf{d}}_{\mathbf{z}}^T
\end{aligned}
\quad
\begin{aligned}
\frac{\partial c}{\partial \hat{\mathbf{y}}} &= \bar{\mathbf{d}}_{\mathbf{z}} \\
\frac{\partial c}{\partial \mathbf{b}} &= -\bar{\mathbf{d}}_{\boldsymbol{\nu}} \\
\frac{\partial c}{\partial \mathbf{h}} &= -\operatorname{diag}(\boldsymbol{\lambda}^*) \bar{\mathbf{d}}_{\boldsymbol{\lambda}}
\end{aligned}
$$

## A.2 Proof of Proposition 1

We begin with the following proposition that will become useful later.

**Proposition 9.** Let $\mathbf{V} \in \mathbb{R}^{m \times m}$ be a symmetric positive definite matrix. Let $\mathbf{B} \in \mathbb{R}^{m \times n}$ and consider the quadratic form $\mathbf{A} = \mathbf{B}^T \mathbf{V} \mathbf{B}$. Then $\mathbf{A}$ is a symmetric positive definite matrix if $\mathbf{B}$ has full column rank.

*Proof.* The symmetry of $\mathbf{A}$ follows directly from the definition. To prove positive definiteness, let $\mathbf{x} \in \mathbb{R}^n$ be a non-zero vector and consider the quadratic form $\mathbf{x}^T \mathbf{A} \mathbf{x}$:

$$
\mathbf{x}^T \mathbf{A} \mathbf{x} = \mathbf{x}^T \mathbf{B}^T \mathbf{V} \mathbf{B} \mathbf{x} = \mathbf{y}^T \mathbf{V} \mathbf{y}.
$$

Clearly $\mathbf{y}^T \mathbf{V} \mathbf{y} > 0$ for all $\mathbf{y} \neq 0$ and $\mathbf{y}^T \mathbf{V} \mathbf{y} = 0 \iff \mathbf{B} \mathbf{x} = 0$. But $\mathbf{B}$ has full column rank and therefore the only solution to $\mathbf{B} \mathbf{x} = 0$ is the trivial solution $\mathbf{x} = 0$. It follows then that $\mathbf{x}^T \mathbf{B}^T \mathbf{V} \mathbf{B} \mathbf{x} > 0$ and therefore $\mathbf{A}$ is positive definite. $\square$

Let $\mathbb{S} = \mathbb{R}^{d_z}$, then the solution to the MVO Program (3) is given by:

$$
\mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) = \frac{1}{\delta} \hat{\mathbf{V}}^{(i)^{-1}} \hat{\mathbf{y}}^{(i)} = \frac{1}{\delta} \hat{\mathbf{V}}^{(i)^{-1}} \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \boldsymbol{\theta}.
\tag{42}
$$

Direct substitution of (42) into Equation (10) yields the following quadratic objective:

$$
L(\boldsymbol{\theta}) = \frac{1}{2} \boldsymbol{\theta}^T \mathbf{H}(\mathbf{x}) \boldsymbol{\theta} - \boldsymbol{\theta}^T \mathbf{d}(\mathbf{x}, \mathbf{y})
\tag{43}
$$

<!-- Page 32 -->
where

$$
\mathbf{d}(\mathbf{x}, \mathbf{y}) = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \hat{\mathbf{V}}^{(i)-1} \, \mathbf{y}^{(i)} \right)
\tag{44}
$$

and

$$
\mathbf{H}(\mathbf{x}) = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \hat{\mathbf{V}}^{(i)-1} \, \mathbf{V}^{(i)} \, \hat{\mathbf{V}}^{(i)-1} \, \mathbf{P} \, \mathrm{diag}(\mathbf{x}^{(i)}) \right).
\tag{45}
$$

Applying Proposition 9 it follows then that if there exists an $\mathbf{x}^{(i)}$ such that $\mathbf{x}_j^{(i)} \neq 0 \quad \forall j \in {1, ..., d_x}$ then $\mathbf{H}(\mathbf{x}, \mathbf{y}) \succ 0$ and therefore (46) is a convex quadratic function.

$$
\underset{\boldsymbol{\theta} \in \Theta}{\text{minimize}} \quad \frac{1}{2} \, \boldsymbol{\theta}^T \, \mathbf{H}(\mathbf{x}) \, \boldsymbol{\theta} - \boldsymbol{\theta}^T \, \mathbf{d}(\mathbf{x}, \mathbf{y}).
\tag{46}
$$

In the absence of constraints on $\boldsymbol{\theta}$, then the first-order conditions are necessary and sufficient for optimality, with optimal IPO coefficients given by:

$$
\boldsymbol{\theta}^* = \mathbf{H}(\mathbf{x})^{-1} \, \mathbf{d}(\mathbf{x}, \mathbf{y})
\tag{47}
$$

## A.3 Proof of Proposition 2

Let $\boldsymbol{\theta}^*$ and $\mathbf{d}_{\mathbf{u}}(\mathbf{x})$ be as defined by Equation (16) and Equation (18), respectively. It follows then that:

$$
\begin{aligned}
\mathbb{E}[\boldsymbol{\theta}^*] &= \mathbb{E}\left[ \mathbf{H}(\mathbf{x})^{-1} \, \mathbf{d}(\mathbf{x}, \mathbf{y}) \right] \\
&= \mathbf{H}(\mathbf{x})^{-1} \, \frac{1}{m\delta} \sum_{i=1}^{m} \left( \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \hat{\mathbf{V}}^{(i)-1} \, \mathbb{E}\left[ \mathbf{y}^{(i)} \right] \right) \\
&= \mathbf{H}(\mathbf{x})^{-1} \, \frac{1}{m\delta} \sum_{i=1}^{m} \left( \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \hat{\mathbf{V}}^{(i)-1} \, \mathbf{P} \, \mathrm{diag}(\mathbf{x}^{(i)}) \, \boldsymbol{\theta} \right) \\
&= \mathbf{H}(\mathbf{x})^{-1} \, \mathbf{d}_{\mathbf{u}}(\mathbf{x}) \, \boldsymbol{\theta}
\end{aligned}
\tag{48}
$$

Corollary 2 follows directly from Equation (48). Observe that when $\hat{\mathbf{V}}^{(i)} = \mathbf{V}^{(i)} \, \forall i \in \{1, ..., m\}$, then

$$
\mathbf{H}(\mathbf{x}) = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \hat{\mathbf{V}}^{(i)-1} \, \mathbf{P} \, \mathrm{diag}(\mathbf{x}^{(i)}) \right).
$$

It follows then that:

$$
\begin{aligned}
\mathbb{E}[\boldsymbol{\theta}^*] &= \mathbb{E}\left[ \mathbf{H}(\mathbf{x})^{-1} \, \mathbf{d}(\mathbf{x}, \mathbf{y}) \right] \\
&= \mathbf{H}(\mathbf{x})^{-1} \, \frac{1}{m\delta} \sum_{i=1}^{m} \left( \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \hat{\mathbf{V}}^{(i)-1} \, \mathbf{P} \, \mathrm{diag}(\mathbf{x}^{(i)}) \, \boldsymbol{\theta} \right) \\
&= \mathbf{H}(\mathbf{x})^{-1} \, \mathbf{H}(\mathbf{x}) \, \boldsymbol{\theta} \\
&= \boldsymbol{\theta} \, .
\end{aligned}
\tag{49}
$$

<!-- Page 33 -->
## A.4 Proof of Proposition 3

Let $\{\mathbf{y}^{(i)}\}_{i=1}^m$ be independent random variables with $\mathbf{y}^{(i)} \sim \mathcal{N}(\mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \boldsymbol{\theta}, \Sigma)$. Let $\hat{\Sigma}$ and $\mathbf{M}$ be as defined by Equation (19) and Equation (20), respectively. It follows then that:

$$
\begin{aligned}
\operatorname{Var}(\boldsymbol{\theta}^*) &= \operatorname{Var}\left(\mathbf{H}(\mathbf{x})^{-1} \mathbf{d}(\mathbf{x}, \mathbf{y})\right) \\
&= \mathbf{H}(\mathbf{x})^{-1} \operatorname{Var}\left(\frac{1}{m \delta} \sum_{i=1}^m \operatorname{diag}(\mathbf{x}^{(i)}) \mathbf{P}^T \hat{\mathbf{V}}^{(i)-1} \mathbf{y}^{(i)}\right) \mathbf{H}(\mathbf{x})^{-1} \\
&= \mathbf{H}(\mathbf{x})^{-1} \frac{1}{m^2 \delta^2} \sum_{i=1}^m \left( \operatorname{diag}(\mathbf{x}^{(i)}) \mathbf{P}^T \hat{\mathbf{V}}^{(i)-1} \operatorname{Var}(\mathbf{y}^{(i)}) \hat{\mathbf{V}}^{(i)-1} \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \right) \mathbf{H}(\mathbf{x})^{-1} \quad (50) \\
&= \mathbf{H}(\mathbf{x})^{-1} \frac{1}{m^2 \delta^2} \sum_{i=1}^m \left( \operatorname{diag}(\mathbf{x}^{(i)}) \mathbf{P}^T \hat{\mathbf{V}}^{(i)-1} \hat{\Sigma} \hat{\mathbf{V}}^{(i)-1} \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \right) \mathbf{H}(\mathbf{x})^{-1} \\
&= \mathbf{H}(\mathbf{x})^{-1} \mathbf{M} \mathbf{H}(\mathbf{x})^{-1}.
\end{aligned}
$$

## A.5 Proof of Proposition 4

Let $\mathbf{z}^*(\mathbf{y}^{(i)})$ and $\mathbf{z}^*(\hat{\mathbf{y}}^{(i)})$ be as defined in Equation (3) and Equation (12), respectively. Recall, the objective function of the minimum tracking-error representation of the IPO program is:

$$
L_{\mathrm{te}}(\boldsymbol{\theta}) = \frac{1}{2m} \sum_{i=1}^m \| \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) - \mathbf{z}^*(\mathbf{y}^{(i)}) \|_{\mathbf{V}^{(i)}}^2 \quad (51)
$$

The first-order necessary conditions for optimality of Program (3) state:

$$
\mathbf{V}^{(i)} \mathbf{z}^*(\mathbf{y}^{(i)}) = \mathbf{y}^{(i)} \quad (52)
$$

Expanding Equation (53) and substituting in Equation (52) completes the proof:

$$
\begin{aligned}
L_{\mathrm{te}}(\boldsymbol{\theta}) &= \frac{1}{2m} \sum_{i=1}^m \left( \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) - \mathbf{z}^*(\mathbf{y}^{(i)}) \right)^T \mathbf{V}^{(i)} \left( \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) - \mathbf{z}^*(\mathbf{y}^{(i)}) \right) \\
&= \frac{1}{m} \sum_{i=1}^m \frac{1}{2} \mathbf{z}^*(\hat{\mathbf{y}}^{(i)})^T \mathbf{V}^{(i)} \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) - \mathbf{z}^*(\hat{\mathbf{y}}^{(i)})^T \mathbf{V}^{(i)} \mathbf{z}^*(\mathbf{y}^{(i)}) + \frac{1}{2} \mathbf{z}^*(\mathbf{y}^{(i)})^T \mathbf{V}^{(i)} \mathbf{z}^*(\mathbf{y}^{(i)}) \quad (53) \\
&= \frac{1}{m} \sum_{i=1}^m \frac{1}{2} \mathbf{z}^*(\hat{\mathbf{y}}^{(i)})^T \mathbf{V}^{(i)} \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) - \mathbf{z}^*(\hat{\mathbf{y}}^{(i)})^T \mathbf{y}^{(i)} + \mathbf{z}^*(\mathbf{y}^{(i)})^T \mathbf{V}^{(i)} \mathbf{z}^*(\mathbf{y}^{(i)}).
\end{aligned}
$$

Note that the proof of Proposition 8 follows a similar argument for the case of equality constrained MVO portfolios.

## A.6 Proof of Proposition 5

In the presence of equality constraints then the solution to the MVO Program is given by:

$$
\mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) = \frac{1}{\delta} \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \mathbf{P} \operatorname{diag}(\mathbf{x}^{(i)}) \boldsymbol{\theta} + (\mathbf{I} - \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \hat{\mathbf{V}}^{(i)}) \mathbf{z}_0, \quad (54)
$$

<!-- Page 34 -->
where $\mathbf{z}_0$ be a particular element of $\mathbb{S} = \{\mathbf{A}\,\mathbf{z} = \mathbf{b}\}$ and $\mathbf{F}$ is a basis for the nullspace of $\mathbf{A}$.

Direct substitution of (54) into Equation (10) yields the following quadratic objective:

$$
L(\boldsymbol{\theta}) = \frac{\delta}{2} \sum_{i=1}^{m} \mathrm{L}_1^{(i)}(\boldsymbol{\theta}) - \sum_{i=1}^{m} \mathrm{L}_2^{(i)}(\boldsymbol{\theta}),
\tag{55}
$$

where

$$
\begin{aligned}
\mathrm{L}_1^{(i)}(\boldsymbol{\theta}) &= \mathbf{z}^*(\hat{\mathbf{y}}^{(i)})^T \, \mathbf{V}^{(i)} \, \mathbf{z}^*(\hat{\mathbf{y}}^{(i)}) \\
&= \frac{1}{\delta^2} \, \boldsymbol{\theta}^T \, \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \mathbf{V}^{(i)} \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \mathbf{P} \, \mathrm{diag}(\mathbf{x}^{(i)}) \, \boldsymbol{\theta} \\
&\quad + \frac{2}{\delta} \, \boldsymbol{\theta}^T \, \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \mathbf{V}^{(i)} (\mathbf{I} - \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \hat{\mathbf{V}}^{(i)}) \, \mathbf{z}_0 \\
&\quad + \mathbf{z}_0^T (\mathbf{I} - \hat{\mathbf{V}}^{(i)} \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T) \, \mathbf{V}^{(i)} (\mathbf{I} - \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \hat{\mathbf{V}}^{(i)}) \, \mathbf{z}_0
\end{aligned}
\tag{56}
$$

and

$$
\begin{aligned}
\mathrm{L}_2^{(i)}(\boldsymbol{\theta}) &= \mathbf{z}^*(\hat{\mathbf{y}}^{(i)})^T \, \mathbf{y}^{(i)} \\
&= \frac{1}{\delta} \, \boldsymbol{\theta}^T \, \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \mathbf{y}^{(i)} + \mathbf{z}_0^T (\mathbf{I} - \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \hat{\mathbf{V}}^{(i)}) \, \mathbf{y}^{(i)}
\end{aligned}
\tag{57}
$$

Simplifying Equation (55) and removing constant terms yields the following quadratic objective:

$$
L(\boldsymbol{\theta}) = \frac{1}{2} \, \boldsymbol{\theta}^T \, \mathbf{H}_{\mathrm{eq}}(\mathbf{x}) \, \boldsymbol{\theta} - \boldsymbol{\theta}^T \, \mathbf{d}_{\mathrm{eq}}(\mathbf{x}, \mathbf{y})
\tag{58}
$$

where

$$
\mathbf{d}_{\mathrm{eq}}(\mathbf{x}, \mathbf{y}) = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T (\mathbf{y}^{(i)} - \mathbf{V}^{(i)} (\mathbf{I} - \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \hat{\mathbf{V}}^{(i)}) \, \mathbf{z}_0) \right)
\tag{59}
$$

and

$$
\mathbf{H}_{\mathrm{eq}}(\mathbf{x}) = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \mathrm{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \mathbf{V}^{(i)} \, \mathbf{F} (\mathbf{F}^T \, \hat{\mathbf{V}}^{(i)} \, \mathbf{F})^{-1} \, \mathbf{F}^T \, \mathbf{P} \, \mathrm{diag}(\mathbf{x}^{(i)}) \right).
\tag{60}
$$

Again, applying Proposition 9 it follows then that if there exists an $\mathbf{x}^{(i)}$ such that $\mathbf{x}_j^{(i)} \neq 0 \quad \forall j \in 1,...,d_x$ then $\mathbf{H}_{\mathrm{eq}}(\mathbf{x}) \succ 0$ and therefore (61) is a convex quadratic program:

$$
\underset{\boldsymbol{\theta} \in \Theta}{\text{minimize}} \quad \frac{1}{2} \, \boldsymbol{\theta}^T \, \mathbf{H}_{\mathrm{eq}}(\mathbf{x}) \, \boldsymbol{\theta} - \boldsymbol{\theta}^T \, \mathbf{d}_{\mathrm{eq}}(\mathbf{x}, \mathbf{y}).
\tag{61}
$$

In the absence of constraints on $\boldsymbol{\theta}$, then the first-order conditions are necessary and sufficient for optimality, with optimal IPO coefficients given by:

$$
\boldsymbol{\theta}_{\mathrm{eq}}^* = \mathbf{H}_{\mathrm{eq}}(\mathbf{x})^{-1} \, \mathbf{d}_{\mathrm{eq}}(\mathbf{x}, \mathbf{y}).
\tag{62}
$$

<!-- Page 35 -->
## A.7 Proof of Proposition 6

Let $\boldsymbol{\theta}_{\text{eq}}^*$ and $\mathbf{d}_{\text{e}}(\mathbf{x})$ be as defined by Equation (30) and Equation (31), respectively. It follows then that:

$$
\begin{aligned}
\mathbb{E}[\boldsymbol{\theta}_{\text{eq}}^*] &= \mathbb{E}\left[\mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \, \mathbf{d}_{\text{eq}}(\mathbf{x}, \mathbf{y})\right] \\
&= \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \frac{1}{m\delta} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \, \mathbb{E}\left[\mathbf{y}^{(i)}\right] \right) \\
&= \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \frac{1}{m\delta} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \, \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \right) \\
&= \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \, \mathbf{d}_{\text{e}}(\mathbf{x}) \, \boldsymbol{\theta}
\end{aligned}
\quad (63)
$$

Corollary 3 follows directly from Equation (63). Observe that when $\hat{\mathbf{V}}^{(i)} = \mathbf{V}^{(i)} \, \forall i \in \{1, ..., m\}$, then

$$
\mathbf{H}_{\text{eq}}(\mathbf{x}) = \frac{1}{m\delta} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \, \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \right).
$$

It follows then that:

$$
\begin{aligned}
\mathbb{E}[\boldsymbol{\theta}_{\text{eq}}^*] &= \mathbb{E}\left[\mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \, \mathbf{d}_{\text{eq}}(\mathbf{x}, \mathbf{y})\right] \\
&= \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \frac{1}{m\delta} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \, \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \right) \\
&= \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \, \mathbf{H}_{\text{eq}}(\mathbf{x}) \, \boldsymbol{\theta} \\
&= \boldsymbol{\theta}
\end{aligned}
\quad (64)
$$

## A.8 Proof of Proposition 7

Let $\{\mathbf{y}^{(i)}\}_{i=1}^{m}$ be independent random variables with $\mathbf{y}^{(i)} \sim \mathcal{N}(\mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \, \boldsymbol{\theta}, \Sigma)$. Let $\hat{\Sigma}$ and $\mathbf{M}_{\text{eq}}$ be as defined by Equation (19) and Equation (32), respectively. It follows then that:

$$
\begin{aligned}
\text{Var}(\boldsymbol{\theta}_{\text{eq}}^*) &= \text{Var}\left(\mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \, \mathbf{d}_{\text{eq}}(\mathbf{x}, \mathbf{y})\right) \\
&= \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \, \text{Var}\left( \text{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T (\mathbf{y}^{(i)} - \mathbf{V}^{(i)} (\mathbf{I} - \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \hat{\mathbf{V}}^{(i)}) \, \mathbf{z}_0 ) \right) \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \\
&= \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \frac{1}{m^2 \delta^2} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \, \text{Var}(\mathbf{y}^{(i)}) \, \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \, \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \right) \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \\
&= \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \frac{1}{m^2 \delta^2} \sum_{i=1}^{m} \left( \text{diag}(\mathbf{x}^{(i)}) \, \mathbf{P}^T \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \, \hat{\Sigma} \, \mathbf{F} (\mathbf{F}^T \hat{\mathbf{V}}^{(i)} \mathbf{F})^{-1} \mathbf{F}^T \, \mathbf{P} \, \text{diag}(\mathbf{x}^{(i)}) \right) \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \\
&= \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1} \, \mathbf{M}_{\text{eq}} \, \mathbf{H}_{\text{eq}}(\mathbf{x})^{-1}.
\end{aligned}
\quad (65)
$$

# B Additional Experiments

## B.1 Experiment 2: unconstrained with multivariate predictions

Economic performance metrics and average out-of-sample MVO costs are provided in Table 7 for the time period of 2000-01-01 to 2020-12-31 for the unconstrained MVO portfolios with multivariate prediction mod-

<!-- Page 36 -->
els. Equity growth charts for the same time period are provided in Figure 17. Again we observe that the IPO model provides higher absolute and risk-adjusted performance and in general more conservative risk metrics. The IPO model produces an out-of-sample MVO cost that is approximately 50% lower and a Sharpe ratio that is approximately 100% larger than that of the OLS model. In Figure 18 we compare the realized MVO and Sharpe ratio costs across 1000 out-of-sample realizations. Again we observe that the IPO model exhibits consistently lower MVO costs with a dominance ratio of 99% and generally lower Sharpe ratio costs with a dominance ratio of 65%.

In Figure 19 we report the estimated regression coefficients and $\pm1$ standard error bar for the last out-of-sample data fold. As before, the majority of the IPO and OLS regression coefficients are not statistically significant at an individual market basis. Figures 19 (a) and 19 (b) report the estimated regression coefficients for the Carry and Trend features, respectively. Again we observe that the IPO model provides very different regression coefficients, in both magnitude and sign, compared to the OLS coefficients. Observe that in the multivariate regression model, 50% of the OLS Trend coefficients are negative. In contrast, the IPO model has only 3 (12.5%) negative coefficients: Cocoa (CC), Live Cattle (LC) and Platinum (PL). Furthermore, in many cases such as Feeder Cattle (FC) and Soymeal (SM), the OLS coefficients are relatively large ($> 0.30$) whereas the corresponding IPO coefficients are effectively zero. Lastly note that the magnitude of the coefficients is approximately 10x larger than the corresponding trend coefficients and is a result of the carry feature values being approximately an order of magnitude smaller.

![Figure 17: Out-of-sample log-equity growth for the unconstrained mean-variance program and multivariate IPO and OLS prediction model.](image_placeholder)

|          | Annual Return | Sharpe Ratio | Volatility | Avg Drawdown | Value at Risk | MVO Cost |
|----------|---------------|--------------|----------|--------------|---------------|----------|
| IPO      | 0.1416        | 0.8835       | 0.1603   | -0.0294      | -0.0138       | 0.5004   |
| OLS      | 0.1034        | 0.4477       | 0.2310   | -0.0438      | -0.0208       | 1.2308   |

Table 7: Out-of-sample MVO costs and economic performance metrics for unconstrained mean-variance portfolios with multivariate IPO and OLS prediction models.

## B.2 Experiment 3 and 4: equality constrained with univariate and multivariate predictions

Economic performance metrics and average out-of-sample MVO costs are provided in Tables 8 and 9 for the equality constrained MVO portfolios with univariate and multivariate prediction models, respectively.

<!-- Page 37 -->
(a) Out-of-sample MVO cost.  
(b) Out-of-sample Sharpe ratio cost.

Figure 18: Realized out-of-sample MVO and Sharpe ratio costs for the unconstrained mean-variance program and multivariate IPO and OLS prediction models.

Equity growth charts for the time period of 2000-01-01 to 2020-12-31 are provided in Figures 20 and 22. As in the unconstrained case, we observe that the IPO model provides higher absolute and risk-adjusted performance, and in general produces more conservative risk metrics. Figures 21 and 23 demonstrate that the IPO model produces consistently lower out-of-sample MVO costs, with dominance ratios of 93% and 99%, respectively, and generally lower Sharpe ratio costs with dominance ratios of 67% and 66%, respectively. The regression coefficients are identical to those provided in Figures 16 and 26, and we refer to Section 4 for relevant discussion.

|          | Annual Return | Sharpe Ratio | Volatility | Avg Drawdown | Value at Risk | MVO Cost |
|----------|---------------|--------------|----------|--------------|---------------|----------|
| IPO      | 0.1238        | 0.7665       | 0.1616   | -0.0290      | -0.0142       | 0.5288   |
| OLS      | 0.0713        | 0.3803       | 0.1876   | -0.0471      | -0.0170       | 0.8082   |

Table 8: Out-of-sample MVO costs and economic performance metrics for equality constrained mean-variance portfolios with univariate IPO and OLS prediction models.

|          | Annual Return | Sharpe Ratio | Volatility | Avg Drawdown | Value at Risk | MVO Cost |
|----------|---------------|--------------|----------|--------------|---------------|----------|
| IPO      | 0.1590        | 0.8851       | 0.1797   | -0.0339      | -0.0163       | 0.6482   |
| OLS      | 0.1151        | 0.4784       | 0.2406   | -0.0497      | -0.0215       | 1.3315   |

Table 9: Out-of-sample MVO costs and economic performance metrics for equality constrained mean-variance portfolios with multivariate IPO and OLS prediction models.

## B.3 Experiment 6: inequality constrained with multivariate predictions

Economic performance metrics and average out-of-sample MVO costs are provided in Table 10 for the time period of 2000-01-01 to 2020-12-31 for the inequality constrained MVO portfolios with multivariate prediction models. Equity growth charts for the same time period are provided in Figure 24. Once again we observe that the IPO model provides modestly higher absolute and risk-adjusted performance and in general more conservative risk metrics. The IPO model produces an out-of-sample MVO cost that is approximately 60% lower and a Sharpe ratio that is approximately 25% larger than that of the OLS model. In Figure 25 we

<!-- Page 38 -->
Figure 19: Optimal IPO and OLS regression coefficients for the unconstrained mean-variance program and multivariate prediction model.

compare the realized MVO and Sharpe ratio costs across 1000 out-of-sample realizations. Again we observe more modest dominance ratios with values in the 55%-65% range. We observe that the IPO model provides a modest improvement to performance in comparison to the OLS model; a likely result of lower prediction model misspecification and improved portfolio regularization by virtue of the box constraints. The estimated regression coefficients are provided in Figure 26 and the findings are similar to those described in Section B.1.

|          | Annual Return | Sharpe Ratio | Volatility | Avg Drawdown | Value at Risk | MVO Cost |
|----------|---------------|--------------|----------|--------------|---------------|----------|
| IPO      | 0.0456        | 0.7937       | 0.0574   | -0.0119      | -0.0057       | 0.0369   |
| OLS      | 0.0411        | 0.6488       | 0.0634   | -0.0145      | -0.0063       | 0.0593   |

Table 10: Out-of-sample MVO costs and economic performance metrics for inequality constrained mean-variance portfolios with multivariate IPO and OLS prediction models.

## C Experiment details

All experiments were conducted on an Apple Mac Pro computer (2.7 GHz 12-Core Intel Xeon E5,128 GB 1066 MHz DDR3 RAM) running macOS ‘Catalina’. The software was written using the R programming language (version 4.0.0) and torch (version 0.2.0).

<!-- Page 39 -->
Figure 20: Out-of-sample log-equity growth for the equality constrained mean-variance program and multivariate IPO and OLS prediction model.

(a) Out-of-sample MVO cost.  
(b) Out-of-sample Sharpe ratio cost.

Figure 21: Realized out-of-sample MVO and Sharpe ratio costs for the equality constrained mean-variance program and univariate IPO and OLS prediction models.

Figure 22: Out-of-sample log-equity growth for the equality constrained mean-variance program and multivariate IPO and OLS prediction model.

<!-- Page 40 -->
(a) Out-of-sample MVO cost.  
(b) Out-of-sample Sharpe ratio cost.

Figure 23: Realized out-of-sample MVO and Sharpe ratio costs for the equality constrained mean-variance program and multivariate IPO and OLS prediction models.

---

Figure 24: Out-of-sample log-equity growth for the inequality constrained mean-variance program and multivariate IPO and OLS prediction model.

---

(a) Out-of-sample MVO cost.  
(b) Out-of-sample Sharpe ratio cost.

Figure 25: Realized out-of-sample MVO and Sharpe ratio costs for the inequality constrained mean-variance program and multivariate IPO and OLS prediction models.

<!-- Page 41 -->
(a) Auxiliary feature: Carry.

(b) Auxiliary feature: Trend.

Figure 26: Optimal IPO and OLS regression coefficients for the equality constrained mean-variance program and multivariate prediction model.

| Asset Class | Market (Symbol) |
|-------------|-----------------|
| Energy      | WTI crude (CL)  | Heating oil (HO) | Gasoil (QS)     |
|             | RBOB gasoline (XB) |                |                 |
| Grain       | Bean oil (BO)   | Corn (C)         | KC Wheat (KW)   |
|             | Soybean (S)     | Soy meal (SM)    | Wheat (W)       |
| Livestock   | Feeder cattle (FC) | Live cattle (LC) | Lean hogs (LH) |
| Metal       | Gold (GC)       | Copper (HG)      | Palladium (PA)  |
|             | Platinum (PL)   | Silver (SI)      |                 |
| Soft        | Cocoa (CC)      | Cotton (CT)      | Robusta Coffee (DF) |
|             | Coffee (KC)     | Canola (RS)      | Sugar (SB)      |

Table 11: Futures market universe. Symbols follow Bloomberg market symbology. Data is provided by Commodity Systems Inc (CSI).