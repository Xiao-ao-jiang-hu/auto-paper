<!-- Page 1 -->
# Analytics and Machine Learning in Vehicle Routing Research

Ruibin Bai${}^{\text{a}}$ and Xinan Chen${}^{\text{a}}$ and Zhi-Long Chen${}^{\text{b}}$ and Tianxiang Cui${}^{\text{a}}$ and Shuhui Gong${}^{\text{c}}$ and Wentao He${}^{\text{a}}$ and Xiaoping Jiang${}^{\text{d}}$ and Huan Jin${}^{\text{a}}$ and Jiahuan Jin${}^{\text{a}}$ and Graham Kendall${}^{\text{e,f}}$ and Jiawei Li${}^{\text{a}}$ and Zheng Lu${}^{\text{a}}$ and Jianfeng Ren${}^{\text{a}}$ and Paul Weng${}^{\text{g,h}}$ and Ning Xue${}^{\text{i}}$ and Huayan Zhang${}^{\text{a}}$

${}^{\text{a}}$ School of Computer Science, University of Nottingham Ningbo China, Ningbo, China.  
${}^{\text{b}}$ Robert H. Smith School of Business, University of Maryland, MD 20742, USA.  
${}^{\text{c}}$ China University of Geosciences, Beijing, China.  
${}^{\text{d}}$ National University of Defence Technology, Hefei, China  
${}^{\text{e}}$ School of Computer Science, University of Nottingham, UK.  
${}^{\text{f}}$ School of Computer Science, University of Nottingham Malaysia, Malaysia  
${}^{\text{g}}$ UM-SJTU Joint Institute, Shanghai Jiao Tong University, Shanghai, China.  
${}^{\text{h}}$ Department of Automation, Shanghai Jiao Tong University, Shanghai, China.  
${}^{\text{i}}$ Faculty of Medicine and Health Sciences, University of Nottingham, UK.

## ARTICLE HISTORY

Compiled February 22, 2021

## ABSTRACT

The Vehicle Routing Problem (VRP) is one of the most intensively studied combinatorial optimisation problems for which numerous models and algorithms have been proposed. To tackle the complexities, uncertainties and dynamics involved in real-world VRP applications, Machine Learning (ML) methods have been used in combination with analytical approaches to enhance problem formulations and algorithmic performance across different problem solving scenarios. However, the relevant papers are scattered in several traditional research fields with very different, sometimes confusing, terminologies. This paper presents a first, comprehensive review of hybrid methods that combine analytical techniques with ML tools in addressing VRP problems. Specifically, we review the emerging research streams on ML-assisted VRP modelling and ML-assisted VRP optimisation. We conclude that ML can be beneficial in enhancing VRP modelling, and improving the performance of algorithms for both online and offline VRP optimisations. Finally, challenges and future opportunities of VRP research are discussed.

## KEYWORDS

vehicle routing; machine learning; data driven methods; uncertainties

## 1. Background and Motivation

The Vehicle Routing Problem (VRP) is one of the most studied problems in the field of operations research. A search using keyword “vehicle routing” on Clarivate’s Web of Science returns more than 8,000 papers, including 131 review papers. One reason for this significant research attention is due to the booming e-commerce industry that leads to exponential growth in transportation and logistics. With the advances in computing

---

*The authors are presented in alphabetic order. Contact R. Bai Email: ruibin.bai@nottingham.edu.cn for correspondence.*

<!-- Page 2 -->
power and progresses in modelling and solution methodologies, it is now possible to solve VRPs of much larger sizes in less time than we could in the past. There have been a number of survey papers related to VRP. Vidal, Laporte, and Matl (2020) provided a good overview of different VRP variants, including the emerging variants characterised by different objectives and performance metrics. Braysy and Gendreau (2005a,b) conducted comprehensive reviews on the heuristic methods for different VRPs. Most of the papers they reviewed focus on deterministic VRPs, in which the problem parameters are assumed to be deterministic and known prior to the problem solving. Gendreau, Laporte, and Seguin (1996) provided a review on stochastic vehicle routing in which some of the problem parameters are assumed stochastic, while Pillac et al. (2013) surveyed all the dynamic vehicle routing problems in which the problem parameters are revealed dynamically over time. Given their close relevancy between stochastic VRP and dynamic VRP, Ritzinger, Puchinger, and Hartl (2016) provided a combined review for both the dynamic and stochastic vehicle routing problems.

Although a tremendous amount of research has been devoted to VRP problems, it is still very difficult to tackle some practical VRP applications for the following reasons. Firstly, the majority of existing VRP research focuses on the analytical properties of different VRP variants and the corresponding solution methods. This type of research is often dominated by the use of mathematical models to define key objectives and constraints (Vidal, Laporte, and Matl 2020). However, for the convenience of theoretical analyses and problem solving, almost all mathematical models are associated with a number of assumptions, some of which may not be practical for real-life applications. It can also be challenging to estimate relevant problem parameters. For practitioners, it becomes a hurdle in translating the existing models and algorithms into successful real-life applications.

Secondly, numerous VRP models have been developed to mathematically formulate uncertainties pertaining to the VRPs. However, most of them are limited to theoretical or small-scale empirical studies. Implementation of these models and the proposed solution methods in real-world applications is rare and still faces considerable challenges. There is a growing demand for making these models more practically applicable.

To tackle some of these issues, such as unrealistic model assumptions, difficulties in parameter estimation and the practicality of solution algorithms, there is an emerging VRP research direction of using hybrid methods that combine data analytics and machine learning tools with conventional optimisation based techniques. With the assistance of analytics and ML, conventional VRP modelling and solution techniques can be significantly strengthened. This paper aims to provide a comprehensive review of such hybrid methods for VRP applications.

VRP research is traditionally limited to the Operations Research (OR) community. However, with the advances in machine learning methodologies, researchers in this community have recently made attempts to tackle combinatorial optimisation problems (including VRPs) solely using machine learning methods (i.e. without explicitly exploiting the structures of the mathematical models). These methods often, despite some progress, suffer from issues such as the lack of generalisation across different scenarios, inefficiency in data use, and the inability to discover insights and interpret solution structures. There seems to be very little interaction between the two communities. Related papers are scattered in journals of both research communities and cross-community paper citations are fewer than you would expect. This leads to the lagged acknowledgement of progress made across research communities. Furthermore, each community uses its own set of terminologies such that similar ideas are defined with different terms. This often causes considerable confusion for researchers and prac-

<!-- Page 3 -->
tioners. We believe that a thorough review of the VRP research that uses tools from these two research communities would be useful to both communities.

Industrial partners also require a holistic review of existing VRP modelling techniques that explain the practicability of these formulations in terms of quality, availability of data and how parameters can be estimated with required precision in order to satisfy the assumptions and other engineering requirements. In this review, we include a section to existing research studies on how machine learning has been used in achieving more practical VRP modelling and parameter estimation. We will also review existing studies that use ML to help build more efficient algorithms for VRP problems.

This review paper is not intended to give another comprehensive review of all VRP-related papers. Instead, the focus is on a new, fast-growing VRP research direction that investigates novel ways to integrate existing analytical methods based on mathematical models with advanced ML methodologies.

We recognise the long history of such research efforts in addressing VRP problems, but it has drawn particular research attention in the past three years, partly due to the growing popularity of analytics and ML research in the OR community, and partly due to a shift of focus in the ML community from single node intelligence to complex system intelligence. We believe that, to achieve major breakthroughs to realise full system intelligence, researchers from both the OR community and the ML community need to collaborate at a much deeper level to tackle the significant challenges that VRP presents. This review paper aims to bridge the gap and promote more interactions and collaborations between the two communities. We expect that our review will inspire more researchers to tackle complex VRP problems and make a positive impact on our daily life.

Figure 1 illustrates the overall classification of related research papers that hybridise ML with analytical approaches in the VRP. We broadly classify three major types of integration efforts. They are ML-assisted VRP modelling, ML-assisted offline and ML-assisted online optimisations, respectively. Most machine learning methodologies require historical data of some problem parameters. Some of the methods will also require meta-data generated by the optimisation methodologies.

The remainder of this review is organised as follows: In Section 2, we provide an introduction to VRP problems, its main variants and the main algorithms that have been utilized. Section 3 reviews modelling methodologies for VRPs with uncertainties, including stochastic programming, robust optimisation, chance constrained programming, and data analytics and forecast. In Section 4, ML-assisted VRP algorithms are reviewed according to how the machine learning is utilised in the solution methods, including decomposition based methods, adaptive neighbourhood search and machine learning trainable constructive methodologies. Section 5 provides some concluding remarks and discusses the challenges, prospects and opportunities for the future research.

## 2. Introduction

For the benefit of those new to VRP, in this section, we give a general introduction of VRP problem, its main variants, and commonly used algorithms.

<!-- Page 4 -->
# Figure 1. The proposed classifications of machine learning assisted VRP research.

## 2.1. Basic Vehicle Routing Problem

The basic VRP, proposed by Dantzig and Ramser (1959), can be defined as an optimisation problem comprising of a set of distributed customers, each with a freight demand, and a fleet of vehicles starting from the central depot. The objective is to find minimal travel cost (e.g. distance or time), such that each customer is visited and served by a vehicle exactly once. The VRP is related to one of the most extensively studied combinatorial optimisation problems, the Travelling Salesman Problem (TSP), which was first considered by Menger (1932). While TSP aims to find a circular shortest path to traverse all customers without consideration of several practical constraints related to capacity and time, the basic VRP problem also takes into account the capacity constraints related to vehicles and hence requires to find multiple vehicle routes with the minimum total cost. The basic VRP problem is also called the Capacitated VRP (CVRP). For a given set of customers $V$ and vehicle depot node 0, a commonly used mathematical model for CVRP is the following vehicle flow formulation:

$$
\min \sum_{i \in V} \sum_{j \in V} c_{ij} x_{ij}
\tag{1}
$$

<!-- Page 5 -->
subject to

$$
\sum_{i \in V} x_{ij} = 1 \quad \forall j \in V \setminus \{0\} \tag{2a}
$$

$$
\sum_{j \in V} x_{ij} = 1 \quad \forall i \in V \setminus \{0\} \tag{2b}
$$

$$
\sum_{i \in V} x_{i0} = K \tag{2c}
$$

$$
\sum_{j \in V} x_{0j} = K \tag{2d}
$$

$$
\sum_{i \notin S} \sum_{j \in S} x_{ij} \geq r(S) \quad \forall S \subseteq V \setminus \{0\}, S \neq \emptyset \tag{2e}
$$

where $x_{ij}$ is a binary decision variable that indicates whether the arc $(i,j)$ is part of the solution and $c_{ij}$ is the cost of using arc $(i,j)$. $K$ is the number of vehicles being used and $r(S)$ is the minimum number of vehicles required to serve customer set $S$. Constraints (2a,2b) make sure that each customer is visited exactly once and constraints (2c,2d) ensure the satisfaction of the number of vehicle routes. Finally constraint (2e) makes sure that the demands from all customers are fully satisfied. VRP is a classical NP-hard problem, thus most solution algorithms are heuristic-based.

## 2.2. Main VRP Variants

Due to the complexities in real-world VRP problems, the basic VRP problem has been extended into a number of variants. The review work by Braekers, Ramaekers, and Van Nieuwenhuyse (2016) refers to a taxonomy from Eksioglu, Vural, and Reisman (2009a), which provides a detailed categorisation of various VRP problems and their models. They classified VRPs with different constraints and objectives as ‘scenario characteristics’, and VRP for different real-world applications as ‘problem physical characteristics’. They also defined an ‘information characteristics’ category and a ‘data characteristics’ dimension, based on the nature of the problem data.

The focus of this paper is mainly on the potential benefits of using data analytics and machine learning for better VRP formulations and problem solving. Specifically, for some conventional VRP variants, machine learning models can potentially be applied as probabilistic modelling technique for customer arrivals, demand quantities, waiting times, etc. For example, when vehicles are constrained to deliver services to customers within certain time intervals, it is known as the Vehicle Routing Problem with Time Windows (VRPTW). If a number of goods are transported from certain pickup locations to other delivery locations, it is known as the Pickup and Delivery Problem (PDP). If part of the problem components remains uncertain and follows a probability distribution, it is defined as Stochastic Vehicle Routing Problem (SVRP). For different VRP modelling, machine learning methodologies can be implemented in historical data analysis to get the prior knowledge. We adopt the taxonomy proposed in Eksioglu, Vural, and Reisman (2009a) to highlight the VRP variants for which machine learning techniques could be used to enhance the practicality and quality of the problem modelling.

### (1) Scenario Characteristics

<!-- Page 6 -->
- Number of stops/customers on a route is partially known or partially probabilistic (Albareda-Sambola, Fernández, and Laporte 2014; Snoeck, Merchán, and Winkenbach 2020).
- Customer service demand quantity is stochastic or unknown. (Zhang et al. 2013; Dinh, Fukasawa, and Luedtke 2018; Ghosal and Wiesemann 2020; Markovic, Cavar, and Caric 2005).
- Request times of new customers are stochastic or unknown.
- Onsite service/waiting times are stochastic or unknown. (Li, Tian, and Leung 2010; Zhang, Lam, and Chen 2013).

(2) Problem Physical Characteristics
- Time window restrictions (on customers, roads, and at facilities) are stochastic or unknown (Chuah and Yingling 2005).
- Travel time is stochastic or unknown. (Musolino et al. 2013; Han, Lee, and Park 2014; Li, Tian, and Leung 2010; Taş et al. 2013, 2014).

(3) Information Characteristics
- Evolution of information is partially dynamic. (Sabar et al. 2014).
- Quality of information is stochastic or unknown. (Balaji et al. 2019).
- Availability of information is local to only a subset of participants concerned in the problem.

(4) Data Characteristics
- Data used is from the real world which can be noisy and messy. (Markovic, Cavar, and Caric 2005; Bent and Van Hentenryck 2005; Li et al. 2018; Žunić, Donko, and Buza 2020)

Another VRP variant is the automated guided vehicle (AGV) routing, which is essentially a driver-less transport system used in logistics warehouses and marine container terminals (Vis 2006). Because of the availability of advanced communication and control mechanisms in real-time, AGV routing problems are often modelled as a dynamic problem (Qiu et al. 2002). As a result, data analytical methods and machine learning play a major role in tackling these type of problems as an improved version over simple heuristic based methods (e.g. (Grunow, Günther, and Lehmann 2006; Wang et al. 2015; Zhang et al. 2019)). Readers can refer to detailed surveys (Qiu et al. 2002; Vis 2006; Hasan, Abidin, and MFMS 2019) for better understanding of AGV routing.

## 2.3. Types of VRP algorithms

Considering that the VRP and its variants belong to the NP-hard class of problems, exact algorithms are only applicable under certain circumstances and for small-scale problems. Exact methods treat VRP as integer or mixed-integer programs, and try to find a (near-)optimal solution. Since real-life VRPs are usually of large sizes, heuristic-based methods are considered more suitable. Elshaer and Awad (2020) states that more than 70% of solution methods in the literature are based on metaheuristics, which have various “meta” strategies capable of escaping from poor local optima but cannot guarantee optimality (Boussaïd, Lepagnot, and Siarry 2013). Representative exact algorithms for VRP include branch-and-price algorithm (Christiansen and Lysgaard 2007) and branch-and-cut algorithms (Augerat et al. 1995; Ralphs et al. 2003; Baldacci, Hadjiconstantinou, and Mingozzi 2004). Metaheuristics can be divided into two categories: single-point based heuristics, and population-based heuristics. The former includes classical heuristic methods, such as simulated annealing (SA) (Kirkpatrick, Gelatt, and Vecchi 1983), tabu search (TS) (Glover and Laguna 1997), GRASP (for greedy

<!-- Page 7 -->
randomized adaptive search procedure) (Feo and Resende 1995), variable neighborhood search (VNS) (Mladenović and Hansen 1997), guided local search (GLS) (Voudouris and Tsang 2003), iterated Local Search (ILS) (Stützle 1999), large neighborhood search (LNS) and adaptive large neighborhood search (ALNS) Pisinger and Ropke (2010). The latter includes two main types, evolutionary algorithms and swarm intelligence, which are mainly inspired by some natural phenomena. The readers may refer to Elshaer and Awad (2020) for a comprehensive review on metaheuristic for VRP. This review focuses on the machine learning assisted VRP algorithms and Section 4 discusses all the relevant papers.

## 3. ML-Assisted VRP Modelling

This section provides an overview of various VRP modelling methodologies that are supported by data analytics and machine learning. In particular, we focus on the modelling techniques for handling uncertain, incomplete, imprecise or ambiguous data in VRPs, including stochastic programming, robust optimisation, chance constrained programming and data forecast.

### 3.1. Stochastic Programming for VRP Modelling

Machine learning has long been used to assist the modelling of stochastic VRPs, where the uncertain data is represented by random variables with known probability distributions. A number of studies have modelled stochastic VRPs as a Markov Decision Process (MDP), which can be tackled by Neuro-Dynamic Programming (NDP). NDP is also referred to as reinforcement learning in the artificial intelligence literature (Bertsekas and Tsitsiklis 1996). The value-function approximation method and/or the policy-function approximation method of NDP is tailored in each of those studies. Secomandi (2000) compared two different NDP algorithms on solving VRPs with stochastic demands and showed that the NDP with rollout policy performs better than an approximate policy iteration. Godfrey and Powell (2002) formulated the dynamic fleet management problem as a multistage dynamic program and use gradient-based information to obtain nonlinear approximations of value functions. Zhang et al. (2013) used tabular value approximation and policy approximation to solve VRPs with stochastic demands. Ulmer et al. (2020) presented a route-based Markov decision process modelling framework for dynamic VRP that extends the conventional MDP model for dynamic and stochastic optimisation problems by redefining the conventional action space to operate on route plans. The proposed modelling framework makes it easier to integrate machine learning approaches with the existing route-based analytical methods.

Machine learning has also been used to estimate the probability distributions of uncertain data in stochastic VRPs. Ritzinger and Puchinger (2013) state that hybrid metaheuristics are often used to solve complex and large real-world optimisation problems, combining advantages from machine learning techniques and mathematical optimisation. The authors examined the hybrid metaheuristics for dynamic stochastic VRP, where not all information is available in advance. Bent and Van Hentenryck (2005) proposed a machine learning approach by sampling from historical data to get partial knowledge of stochastic distribution in online scheduling problems. They claim that their approach could help widen the application area of stochastic algorithms. Experiments are conducted on several problems including dynamic VRPTW from pre-

<!-- Page 8 -->
vious work Bent and Van Hentenryck (2004). Defourny (2010) generated and selected scenario trees with random branching structures for multi-stage stochastic programming over large planning horizons. This work combines the perturb-and-combine estimation methods from machine learning with stochastic programming techniques for sequential decision making under uncertainty. The performance has been shown to be competitive. Bai et al. (2014) studied a two-stage model for stochastic transportation network that supports vehicle rerouting in the second stage.

## 3.2. Robust Optimisation for VRP Modelling

Stochastic programming assumes that the probability distributions of uncertain parameters are known, and aims for the best average performance over all possible future scenarios. By contrast, robust optimisation assumes that the probability distributions of uncertain parameters can be unknown, but the range of the uncertain data is known, and aims for a solution that could work even in the worst-case scenario. In general, robust optimisation is considered as a risk-averse method.

Robust CVRPs require the vehicle routes to be feasible for all customer demands within a pre-specified uncertainty set (e.g., a box, polyhedron, or ellipsoid). Branch-and-cut schemes for the exact solution of the robust CVRPs have been proposed by (Sungur, Ordóñez, and Dessouky 2008) and (Gounaris, Wiesemann, and Floudas 2013). Robust CVRPs are amenable to solution schemes that appear to scale better than those for the chance constrained CVRPs (see Section 3.3). However, the solutions obtained from the robust CVRPs can be overly conservative since all demand scenarios within the uncertainty set are treated as equally likely, and the routes are selected solely in view of the worst demand scenario from the uncertainty set (Ghosal and Wiesemann (2020)).

For a comprehensive review of robust optimisation, we refer interested readers to (Bertsimas, Brown, and Caramanis 2010). More recent contributions are some simulation based approaches, which transform the complex stochastic VRPs into a set of deterministic CVRPs, for which fast and extensively tested meta-heuristics exist (Bernardo, Du, and Pannek 2020). To our knowledge, very little, if any, research has been conducted so far to apply machine learning techniques to robust VRPs, which offers many research opportunities.

## 3.3. Chance-constrained Programming for VRP Modelling

Chance-constrained programs are different from the stochastic programming and robust optimisation. It assumes that constraints are satisfied to some degree, measured by probabilities, rather than fully met. In VRPs, the chance-constrained CVRPs consider vehicle routes that satisfy the customer demands along each route with a pre-specified probability.

Dinh, Fukasawa, and Luedtke (2018) modelled the stochastic demand of customers as a joint normal distribution or a given discrete distribution. In particular, such modelling allows correlation between customer demands rather than independent demands. A branch-and-cut-and-price (BCP) algorithm is used to tackle the problem. Route relaxation is proposed in order to price through dynamic programming. The experiments demonstrated this approach’s ability to solve the distributionally robust chance constrained vehicle routing problems (CCVRPs).

<!-- Page 9 -->
Ghosal and Wiesemann (2020) focused on the distributionally robust chance-constrained CVRP, which assumes that the probability distribution of the customer demand is partially known. The customer demands are modelled as the distribution based on ambiguity sets. Several first-order and second-order moment ambiguity sets are investigated and branch-and-cut schemes are used to solve it. The experiments demonstrated the good scalability of distributionally robust CVRP.

Li, Tian, and Leung (2010) studied a version of VRP where the service and travel times are stochastic and time window constraints are involved. Such problem is then formulated as a chance constrained programming model and a stochastic programming model with recourse. The author assigned a confidence level for both time window and driving duration of each route. Tabu search is utilised as the algorithmic solver. Similarly, Gutierrez et al. (2018) also focused on VRP with stochastic service and travel times and time windows. The arrival times are estimated by a log-normal approximation. A multi-population memetic algorithm (MPMA) is then proposed as the algorithm to address the problem. Wu et al. (2020a) studied VRP for application of wet waste collections. In this work, chance constrained programming is used to model the uncertainty of waste generation rate and make it deterministic. A hybrid algorithm consisting of Particle Swarm Optimisation (PSO) and Simulated Annealing (SA) are used for the optimisation.

Similar to the robust optimisation methods, very little research has been conducted to explicitly combine machine learning with chance-constrained programming in solving VRP problems. However, one obvious way to hybridise them is to use ML to estimate probabilities or the parameters of the distributions. Another hybridisation opportunity can be using ML to speed up the various branch and pricing methods developed for chance-constrained programming.

## 3.4. Data Analytics and Forecast in VRP Modelling

In this subsection, we discuss a relatively loose connection between machine learning and the VRP. Research works in this area are not likely to use machine learning to address the related problem, but use it as a tool to reflect and analyse data that is related to problem solutions. A rough classification by Talbi (2016) divides such combination into the low-level integration, which is highly related to hyperparameter tuning; and the high-level integration, where machine learning is applied to help the modelling and provide certain contexts for metaheuristics. In this section, we mainly focus on machine learning assisted predictive models for handling uncertainties in VRP, and machine learning based model fine-tuning methods.

### 3.4.1. Predictive Models for Uncertainties in VRP

So far, a great number of methods have been proposed to solve classic VRP and many of them can generate high-quality solutions. However, most of these methods fail to work in realistic scenarios because of the over-simplification of the real-world constraints and uncertainties. For example, the elements such as customer occurrences and demands, and travel time may not be deterministic. Therefore, more sophisticated approaches are required to model and solve these problems.

Ce and Hui (2010) made a basic classification by analysing customer’s demands and introduced corresponding data mining techniques for each case. Markovic, Cavar, and Caric (2005) used a neural network to predict customers’ stochastic demand based on historical data, while Snoeck, Merchán, and Winkenbach (2020) applied probabilistic

<!-- Page 10 -->
graphical model to predict constrained customers for routing problems. Musolino et al. (2013) developed a simulation-based method to forecast the travel time for an emergency vehicles routing problem. Li et al. (2019) investigated various ensemble based machine learning methods to predict the travel time in vehicle routing applications.

Li et al. (2018) applied a data-driven search algorithm to improve traditional meta-heuristics methods for large-scale manufacturing logistic problems in real-world industry practice. In the VRP part, they learned a multinomial transition matrix by updating from historical expert data and generated valid solutions by sampling from the matrix.

Calvet et al. (2016a) considered a realistic VRP setting where customers have different demands to different assigned depots. The matching between heterogeneous depots and customers with stochastic demand could significantly affect the profitability of the company. Thus, several regression models which estimate the customer demands based on their assigned depot are used to support the market segmentation strategy.

In dynamic VRP, a customer’s availability indicates the available service time window of each customer, which is one of the important aspects that cause the dynamism in Dynamic VRP (DVRP). Kucharska (2019) focused on this dynamism and used algebraic-logical meta-model (ALMM) to solve dynamic VRP with predicted customer availability.

Another commonly used modelling approach addressing uncertainties in VRP is rolling horizon planning. Problems are solved iteratively over a planning horizon that shifts forward a constant or variable time span at each step. At each iteration, some of the uncertainties of the problem are estimated with historical data. Cordeau et al. (2015) used this mechanism for a dynamic multi-period auto-carrier transportation problem which considers balancing vehicle usage and demand in dynamic settings. Wen et al. (2010) studied dynamic multi-period vehicle routing problem with multi-objective of minimising customer waiting time and total travel costs as well as balancing the daily workload. Historical data are used to estimate the average daily workload over the rolling horizon.

Machine learning has also been applied to a wide variety of problems in transportation and logistics. Zhang et al. (2011) reviewed various data-driven traffic management systems, in which computer vision and other learning methods for environment sense and prediction are adopted. Zantalis et al. (2019) presented a review on machine learning and IoT in smart transportation, while Anuar et al. (2019) gave a review on machine learning and VRP methods in humanitarian logistics.

### 3.4.2. Learning to Configure Algorithms

There are a great number of hyper-parameters in VRP algorithms such as acceptance criteria of simulated annealing algorithm or population size of family of nature-inspired algorithms. Usually, these hyper-parameters need to be carefully adjusted manually. Good default parameters could significantly affect the performance of algorithms (Bengio, Lodi, and Prouvost 2020). Finding such suitable parameters is generally called the parameter setting problem (PSP). Machine learning methodologies could also be used to tackle the PSP problem as data-driven approaches. The general idea is to train regression models that map the specific problem instance features to the algorithm parameters where the training data are sets of pre-adjusted parameters and the corresponding performance.

Calvet et al. (2016b) proposed a novel statistical learning-based methodology for solving the PSP and used it to fine-tune the existing algorithm for multi-depot vehicle

<!-- Page 11 -->
routing problem. Kadaba, Nygard, and Juell (1991) proposed a hybrid framework including machine learning and knowledge-based systems to solve routing and scheduling applications. Neural networks are used for model selection and GA hyper-parameters tuning. Cooray and Rupasinghe (2017) used simple clustering techniques to improve the parameter selection for GA. Žunić, Donko, and Buza (2020) built a framework where the predictive models, such as support vector machine, were used to adaptively control the hyper-parameters for each instance based on the historical data. The performance of the framework was demonstrated by testing on heterogeneous fleet vehicle routing problem with time windows (HVRPTW) along with a set of realistic logistics constrains.

Clustering methodologies are also used in VRP modelling. Hu, Huang, and Zeng (2007) proposed a novel framework for solving CVRPTW. In the first stage of this framework, customers are clustered into different groups based on their features. The experiments demonstrated that such clustering could efficiently support the subsequent stages to generate feasible solutions. Costa, Mei, and Zhang (2020) used a Cluster-based Hyper-Heuristic (CbHH) that adaptively cluster the customers to determine the neighborhood searching space at each step. The genetic algorithm is used to evolve the sequence of low-level perturbative operator to improve the initial solution. The experiment showed the effectiveness of the clustering technique and GA-based heuristic search. Similarly, Göçmen and Erol (2019) used clustering method and upstream 3D-bin packing to provide context for VRP problem.

## 4. ML-Assisted VRP Algorithms

Machine learning can help address combinatorial optimisation problems in various ways (see Bengio, Lodi, and Prouvost (2020) for a more general discussion or Smith (1999) for a survey of older work). It can assist traditional solvers that utilise heuristics to make decisions. Those heuristics are created by experts, are generally hard to design, and may not transfer well from one problem (instance) to another. Machine learning can help learn such heuristics automatically. For instance, He, Daume III, and Eisner (2014) and Khalil et al. (2016) independently proposed learning searching in a branch-and-bound tree for mixed-integer linear programs (MILP). Li, Chen, and Koltun (2018) trained a graph convolutional network to guide a tree search, notably for the Boolean satisfiability (SAT) problem. Such approaches are useful as they can benefit those aiming to solve problems that can be formulated as MILP or SAT. See Lodi and Zarpellon (2017) for a recent survey on machine learning applied to branch and bound. In this section, we review various strategies that combine machine learning with traditional VRP solution methods.

### 4.1. ML-Guided VRP Decomposition Strategies

As a classic NP-hard combinatorial optimisation problem, VRP’s search space grows exponentially with respect to the problem size, causing significant challenges to solve to optimality as the problem size increases. A common strategy is to decompose the original large scale problem into a number of smaller sub-problems. However, how the problem is decomposed becomes another difficult problem. In this section, we review decomposition based VRP studies that adopt machine learning to guide the decomposition.

In general, in a decomposition approach, the VRP problem is divided into smaller

<!-- Page 12 -->
and simpler sub-problems and a specific solution method is applied to each sub-problem (Archetti and Speranza 2014). We classify the ML-guided VRP decomposition into **hierarchical** and **integrated** approaches. Hierarchical approaches are high-level combinations of ML and OR algorithms (e.g. either exact methods or (meta-)heuristics), while the integrated approaches are low-level combinations because one algorithm is embedded within another.

### 4.1.1. Hierarchical approaches

The ML-guided hierarchical decomposition in VRP can be classified into two types: Route First and Cluster Second (RFCS) and Cluster First and Route Second (CFRS). RFCS decomposes a VRP problem into two sub-problems: 1) Generate a TSP tour from the depot around all the customers and back to the depot; 2) Partition the TSP tour into a set of vehicle routes. Conversely, the CFRS starting from assigning customers to vehicles, then vehicle tours are constructed.

The OR approaches of solving RFCS (e.g. Beasley (1983), Montoya et al. (2014)) and CFRS (e.g. Fisher and Jaikumar (1981), Dondo and Cerdá (2007)) have been applied to many VRP problems.

**ML-guided CFRS** Erdoğan and Miller-Hooks (2012) solved the Green Vehicle Routing Problem (G-VRP) using Modified Clarke and Wright Savings (MCWS) heuristic and an unsupervised clustering algorithm, which is built on the concepts from the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm for exploiting the spatial properties of the G-VRP. The VRP is decomposed into two sub-problems: 1) Cluster customers using DBSCAN; 2) run MCWS to construct vehicle tours.

Comert et al. (2017) proposed a two-stage solution method for the Vehicle Routing Problem with Time Windows (VRPTW). In the first stage, customers were assigned to vehicles using the best-performing clustering algorithms among K-means, K-medoids, and DBSCAN. In the second stage, a VRPTW was solved using Linear Programming (LP) to construct vehicle tours.

Göçmen and Erol (2019) studied intermodal network problems combining pick-up routing problems with three-dimensional loading constraints, clustered backhauls at the operational level, and train loading at a tactical level. The authors first used the clustering of backhauls, then packed using K-means for a feasible loading pattern with four loading dimensions, and finally utilised capacitated VRP formulation which can be solved to optimally for small problem instances. The K-means clustering ensures the assignments of tasks into several clusters, where the initial cluster number is defined before the solution procedure starts. The task assignment to which of the clusters is not part of the decision variables but is an input to the intermodal network problem.

The scientific literature shows that the ML-guided CFRS follows a similar solution procedure where an unsupervised cluster algorithm is first adopted to cluster customers to vehicles, followed by vehicle tours construction using (meta)-heuristics or LP methods. Readers are referred to He et al. (2009), Nallusamy et al. (2010), Korayem, Khorsid, and Kassem (2015), Reed, Yiannakou, and Evering (2014), Comert et al. (2018), Xu, Pu, and Duan (2018), Geetha, Poonthalir, and Vanathi (2013), Rautela, Sharma, and Bhardwaj (2019), Geetha, Vanathi, and Poonthalir (2012), Yücenur and Demirel (2011), Qi et al. (2011), Luo and Chen (2014), Gao et al. (2016), Miranda-Bront et al. (2017) for similar and more complete references.

**ML guided RFCS** We found only one study using ML in RFCS. Kubra, Muhammet, and Ozcan (2019) investigated the problem of determining service routes for the

<!-- Page 13 -->
staff of a company. The authors first implemented a GA to create a TSP tour, and then partitioned and compared the tours by considering different scenarios using clustering methods including K-means, K-medoids and K-modes.

### 4.1.2. Integrated approaches

Integrated approaches of the decomposition have been applied to enhance the performance of LP algorithms. For example, the approximation of strong branching (Alvarez, Louveaux, and Wehenkel 2017), learning techniques introduced in the branch-and-bound algorithm (Lodi and Zarpellon 2017). A recent survey of utilising ML to solve combinatorial optimisation problems can be found in Bengio, Lodi, and Prouvost (2020). However, the integrated approaches tailored to the VRP are somewhat limited.

Desaulniers, Lodi, and Morabit (2020) employed a Graph Neural Networks (GNN) to select a subset of the columns generated at each iteration of the column generation process. The input of the GNN is a set of promising columns generated by solving an LP model to minimise the number of columns and ensure a maximal decrease of the objective value. The learned GNN model is to reduce the computational time spent re-optimising the restricted master problem (RMP) at each iteration by selecting the most promising columns. Computational results on two types of VRP indicate that average computational time reductions of 20 to 30% are achievable when solving the RMP.

Kruber, Lübbecke, and Parmentier (2017) applied a K-nearest-neighbour classifier to decide whether or not a Dantzig-Wolfe decomposition should be applied to a given problem, and which decomposition to choose. This method is not limited to VRP though.

Yao et al. (2019) proposed a decomposition framework based on Alternating Direction Method of Multipliers (ADMM) to iteratively improve the primal and dual solution quality simultaneously. ADMM has been utilised in distributed convex optimisation problems arising in statistics and machine learning (Boyd et al. (2010)). The authors first constructed a multi-dimensional commodity flow formulation for the VRP. Then, ADMM was applied to develop a decomposition framework, in which the original model was decomposed into a series of least-cost path problems which can be solved by the dynamic programming.

## 4.2. ML-Guided Perturbative VRP Algorithms

When VRPs are modelled as offline optimisation problems (i.e. the problem related parameters are known and given prior to problem solving), perturbation based meta-heuristics and evolutionary algorithms are among the most popular solution methods. The simplest perturbation search method is the basic local search. It searches a predefined neighbourhood of candidate solutions to find solutions which are superior to the current ones according to the objective function. The neighbourhood of a solution is defined as the set of solutions that can be derived by perturbing the incumbent solution according to certain transition rules. For example, 2-opt neighbourhood function for VRP operates in the neighbourhood of solutions solutions that only differ in the order of two connecting arcs. Since global search is generally impractical for NP-hard problems, local search can provide satisfactory, although not optimal, solutions within a reasonable time when well designed. See Gendreau and Potvin (2010) for more detailed discussions of meta-heuristics and local search algorithms.

Local search starts from an initial solution (either randomly generated or heuris-

<!-- Page 14 -->
tically constructed), and moves to a neighbour solution better than the current one continuously until no improvement can be made or the stopping criteria are met. Local search may be trapped in local optima. To overcome this shortcoming, a number of meta-heuristics approaches are introduced to improve the neighbourhood search strategies, for example simulated annealing, tabu search, guided neighbourhood search, variable neighbourhood search and adaptive large neighbourhood search. The idea of getting away from local optimum is to introduce a perturbation so that the search process can accept inferior solutions and ‘jump’ out poor local optima by following some guiding strategies.

So far, ML has been used in different parts of perturbation based search approaches to improve its performance: initial solution generation, adaptive selection of neighbourhoods at different search stages, generation of neighbourhood functions and solution evaluations. We review ML guided initial solution generation in Section 4.3. In the following we focus on the intelligent neighbourhood selection, neighbourhood function generation and solution evaluations assisted by machine learning.

### 4.2.1. Learning to Select Perturbation Heuristics

The concept of using machine learning techniques to guide the neighbourhood search is well documented. Initially, various adaptive mechanisms in the principle of basic reinforcement learning are used in the form of perturbative hyper-heuristics (Burke et al. 2019) and adaptive large neighbourhood search (ALNS) (Ropke and Pisinger 2006). The main idea of perturbative hyper-heuristic is to intelligently select a set of perturbative low-level heuristics to adapt to different problem solving scenarios. As such, the learning based components are crucial parts of this type of hyper-heuristic methods. The literature of applying learning assisted perturbative hyper-heuristics for various combinatorial optimisation problems is rich. Here, we shall focus mainly on the papers that are developed for vehicle routing.

Bai et al. (2007) addressed a VRPTW problem with a hyper-heuristic method that uses the concept of reinforcement learning to guide the selection of low-level heuristics and simulated annealing as the perturbation acceptance criteria. The study investigated how the memory length of the adaptive learning mechanism affects the performance of the algorithm. Sabar et al. (2015) proposed a new hyper-heuristic method for two VRP problems by using an improved reward scheme (dynamic multiarmed bandit-extreme value-based reward) in the learning mechanism, coupled with gene expression programming for acceptance criteria generation. Soria-Alcaraz et al. (2017) extended this learning reward scheme with additional information from non-parametric statistics and fitness landscape measurements.

Garrido and Cristina Riff (2010) investigated evolutionary-based hyper-heuristic approaches for a dynamic vehicle routing problem. The results showed that evolutionary based learning mechanisms improve the algorithm performance by adapting to different dynamic environments. Asta and Ozcan (2014) proposed to use an apprenticeship learning in the hyper-heuristics for vehicle routing. The trained algorithm is able to produce high quality solutions for test instances which are not seen during the hyper-heuristic training stage.

The learning assisted perturbative hyper-heuristics have also been used in other VRP variants, including railway maintenance service routing (Pour, Drake, and Burke 2018), multi-depot m-TSP problem (Pandiri and Singh 2018), multi-objective routing planning (Yao, Peng, and Xiao 2018), mixed-shift full truckload routing (Chen et al. 2018, 2020a), energy-aware routing (Leng et al. 2019), urban transport routing (Ahmed,

<!-- Page 15 -->
Mumford, and Kheiri 2019; Heyken Soares et al. 2020).

One emerging direction for perturbative heuristic hyper-heuristics is the use of deep reinforcement learning (DRL) for heuristics selection. A DRL combines deep learning with aforementioned reinforcement learning. Tyasnurita, Ozcan, and John (2017) used a time delay neural network (TDNN) as a classifier to select the low-level heuristics to solve open vehicle routing problem. Parameters of TNDD were trained through the experience replay from an ‘expert’ hyper-heuristic algorithm called MCF-AM (Modified Choice Function - All Moves) from Drake, Özcan, and Burke (2012). Chen and Tian (2019) proposed a general deep reinforcement learning based hyper-heuristic framework for combinatorial optimisation problems called local rewriting framework (Neuwriter) which generates a solution iteratively. In this work, a region picker and a rule picker were defined and trained separately. At each rewrite step, two pickers selected a sub-area of current solution and its rewrite rule one behind the other. Then the modified sub-area will be put back to generate a new solution. The model repeated this process until convergence. The online job scheduling problem, expression simplification problem, and capacitated vehicle routing problem were used to test this method. A bi-directional LSTM layer was used to embed the input nodes in CVRP problem, and a similar pointer network mechanism was used to select a node through a probability distribution. Wu et al. (2020b) extended the Neuwriter framework by integrating the region picker and rule picker policy networks into one. Specifically, the compatibility computation was adopted in the model to produce a probability matrix of node pairs whose element specified the two corresponding nodes. In order to capture the node position information, sinusoidal positional encoding was introduced to the embedding layer. The results showed that this method outperformed Neuwriter.

Lu, Zhang, and Yang (2020) proposed an iterative improvement method called ‘Learn to Improve (L2I)’ based on that. Notably, this method outperformed LKH3 (Helsgaun 2017) which was the state-of-the-art method of VRP on both speed and solution quality. The model started with a random initial solution and used two classes of predefined low-level heuristics, namely, improvement operators and perturbation operators which were used to local search the solution and destroy part of the solution respectively. A double layer controller guided by reinforcement learning was used to select the corresponding heuristics. When generating the action, the model would also consider the history actions and their effects. A policy ensemble mechanism was used to improve the generalisation.

### 4.2.2. Learning to Adapt Neighbourhood Choices

Most real-life combinatorial optimisation problems involve complex objectives and constraints which often lead to very different solution space landscapes. For example, some solution spaces are very rugged with a lot of local optima while some other solution spaces contain plateaus that are insensitive to neighbourhood moves. These challenges lead to considerable research efforts in embedding learning mechanisms for more efficient neighbourhood search through adaptive neighbourhood selection. Since the early work on adaptive large neighbourhood search (ALNS) from (Ropke and Pisinger 2006), a number of follow-up research efforts have been made in either directly applying or extending the methods to solve different VRP variants. Papers that used similar algorithms for different VRP variants include Laporte, Musmanno, and Vocaturo (2010); Ribeiro and Laporte (2012); Kovacs et al. (2012); Hemmelmayr, Cordeau, and Crainic (2012); Demir, Bektaş, and Laporte (2012); Masson, Lehuéde, and Peton (2013); Azi, Gendreau, and Potvin (2014); Aksen et al. (2014); Belo-Filho, Amorim, and Almada-

<!-- Page 16 -->
Lobo (2015).

There have been some research to further enhance ALNS method by hybridising with other methods. Qu and Bard (2012) combined ALNS with a GRASP approach in solving a pickup and delivery problem with transshipment. Parragh and Cordeau (2017) combined a branch and price method with the ALNS method for a truck and trailer routing problem considering the time window constraints. Zulj, Kramer, and Schneider (2018) integrated tabu search in a ALNS framework for solving an order-batching problem. Lahyani, Gouguenheim, and Coelho (2019) used a hybrid ALNS method to successfully address a multi-depot open vehicle routing problem. Ha et al. (2020) combined constraint programming with ALNS to solve VRP with synchronisation constraints.

Another learning-based ALNS algorithm proposed by Hottung and Tierney (2019) was called neural large neighborhood search (NLNS). The method was specifically adapted to support parallel computing, which is one of the contributions of this method. Such features could support two implementation patterns: batch search which solved a set of instances simultaneously and the single instance search which solved only one instance concurrently. The potential of the method was demonstrated through experiments for capacitated vehicle routing problem (CVRP) and the split delivery vehicle routing problem (SDVRP).

### 4.2.3. Learning to generate perturbation heuristics

A new exciting direction of using machine learning in solving VRP problems is that machine-learning models could be used to generate new heuristics to perturb solutions as apposed to using manually-crafted perturbation heuristics. For example, a neural network model takes an incumbent solution as inputs and outputs the indices of nodes that are modified. In this case, the neural network model itself acts as perturbative heuristic in the traditional optimisation algorithm.

da Costa et al. (2020) focused on improvement (or perturbative) heuristics that could refine a given solution iteratively until reaching a near-optimal solution. In his work, a method that could learn a policy to generate the 2-opt heuristic for TSP was proposed. A similar encoder-decoder framework was used to encode the graph and generate a sequence of action distribution but the mechanism was modified to make it easy to extend to k-opt operations. The difference to the initial pointer network is that at each decoder step, the model outputs an action (two nodes for 2-opt) rather than one specific node to construct the final solution.

Gao et al. (2020) proposed a method for learning the local search heuristics. Similarly, an encoder-decoder was used and trained by actor-critic algorithm. Inspired by the ALNS algorithm for vehicle routing problems, the authors defined the local search heuristic as a destroy operator and a repair operator. The destroy operator removed a subset node of the current solution and the repair operator was used to generate a permutation of the selected elements and insert them back. Motivated by the graph attention network (GAT) mechanism (Veličković et al. 2018) which is an effective method to represent the graph topology by propagating the neighbour node information through the attention mechanism, the authors proposed a modified version called Element-wise GAT with Edge-embedding (EGATE) which not only considered the information of nodes but also the arc between the nodes. The attention mask was generated by softmax the concatenation of embedding of arc and two nodes it connected with. The decoder acted as destroy and repair operation.

<!-- Page 17 -->
Chen et al. (2020b) were also motivated by ALNS algorithm, and proposed a similar method called dynamic particle removal (DPR) using Hierarchical Recurrent Graph Convolutional Network (HRGCN). Similar to the method of Gao et al. (2020), the authors also defined the local search as a destroy and repair operator. The degree (size and the allocation of the sub-nodes) of the destroy operator was dynamically determined. The HRGCN is able to be aware of spatial (graph topology) and temporal (embedding in previous iterations) context information.

### 4.2.4. Learning to Speedup Solution Evaluations

For many real world scheduling and routing problems, computing evaluation functions is expensive. ML has been used in evaluating solutions to reduce computational complexity. Boyan and Moore (2000) developed a general algorithm (which was called STAGE) to learn an evaluation function that predicted the outcome of a local search. The learned evaluation function was then used to guide the future search trajectories toward better optima on the same problem. Moll et al. (1998) introduced an offline reinforcement learning phase to STAGE and compared using of learned evaluation function with the original evaluation function. The proposed algorithm was applied to the Dial-A-Ride Problem, a variant of TSP, to show how well learning an instance-independent evaluation function could guide local search for additional instances.

## 4.3. Learning to Construct VRP Solutions

In this section we discuss some representative works that exploit machine learning in the constructive approach for vehicle routing related problems. Recall that such an approach consists of building iteratively a complete solution from scratch (e.g., in TSP, it sequentially selects unvisited cities until a tour is formed) in contrast to approaches based on iterative perturbative search.

The most common approach is to learn a probability distribution over solutions, which can then guide a tree search (e.g., greedy search, beam search) to generate a full solution. In contrast, another possibility is to learn directly a constructive solver. Indeed, such a solver can be seen as a sequential decision-maker, which can be trained via an evolutionary or a reinforcement learning process.

Using machine learning to solve routing problems has a long history (e.g., Hopfield and Tank (1985); Potvin, Lapalme, and Rousseau (1990)). We refer the reader to other surveys on historical developments (e.g., Smith (1999)). We will focus mostly on more recent applications of machine learning to these routing problems. We discuss first the works that focus on (mostly planar) TSP or variants, and then turn to those on VRP and its variants. We mention work on online VRP and variants at the end.

For TSP, Vinyals, Fortunato, and Jaitly (2015) proposed Pointer Networks, which is a novel neural network architecture with two components. First, an encoder was implemented as a recurrent neural network sequentially reading the positions of the cities. Second, a decoder was realized as a recurrent neural network iteratively outputting a probability distribution over remaining cities, using an attention mechanism (Bahdanau, Cho, and Bengio 2015). The whole model was trained in a supervised fashion. For testing, beam search was used to ensure that only valid tours are output. The technique was applied to TSP instances with up to 50 nodes.

Bello et al. (2017) extended the previous approach to train the network using reinforcement learning with an actor-critic scheme. They used the cost length of a generated tour as an unbiased estimate of the value of a policy. The authors found out that pre-

<!-- Page 18 -->
training on a set of instances in addition to training on the particular instance to be solved yielded the best performance on TSP instances with up to 100 nodes. Some more recent investigation (Joshi, Laurent, and Bresson 2019b) suggests that RL may lead to better generalization capability compared to supervised learning.

Khalil et al. (2017) proposed a general method to solve graph-based combinatorial optimization problems based on graph embedding to encode a partial solution and reinforcement learning to learn a greedy policy. In contrast to the previous approaches, they used a value-based RL method, fitted Q-learning.

Deudon et al. (2018) proposed a graph attention network architecture to improve over Pointer Networks to obtain invariance over input order. Also, the authors introduced the idea of preprocessing the inputs with PCA to obtain rotation invariance. With a critic using a similar architecture, the policy was trained in an actor-critic architecture. They used a mask to remove already visited cities. In their recent hybrid method, the authors proposed further to improve the solutions sampled from the trained policy using the 2-OPT heuristics.

In contrast to other approaches mentioned here, Nowak et al. (2017) proposed a non-autoregressive model based on graph neural networks, i.e., their model does not select cities sequentially. Instead, the model trained in a supervised way outputs an adjacency matrix representing a distribution over tours, from which they extract a full solution via beam search. Although the method is not competitive with other deep learning methods, Joshi, Laurent, and Bresson (2019a) improved this approach by using notably graph convolutional networks. They showed that their approach compares favourably with other previous autoregressive approaches.

Yang et al. (2018) proposed to perform the dynamic programming update of the Bellman-Held-Karp algorithm (Bellman 1962; Held and Karp 1962) for solving TSP in an approximate way using neural networks. Doing so allows their proposition to tackle much larger TSP instances.

Ma et al. (2020) proposed Graph Pointer Network (GPN), an extension of Pointer Network, with a graph embedding of the input. They demonstrated promising generalisability of GPN, which can be trained on small TSP instances (up to 100 cities) and then solve larger instances up to 1000 cities. Besides, they suggest using a hierarchy of GPNs to take into account additional constraints on the TSP problem. They tried their ideas on TSP with time windows and showed that their approach performs well.

Some works considered the multiple TSP (mTSP), where several traveling salesmen need to visit all the cities exactly once in a cooperative manner. This problem can be seen as a relaxed VRP problem. Kaempfer and Wolf (2019) adapted PointNet (Qi et al. 2017), which deals with sets of points, to this mTSP problem. Hu, Yao, and Lee (2020) solved it by first using a cooperative multi-agent deep reinforcement learning for agent-to-city assignment, and then computing the tour of each agent using a classic TSP solver.

While the previous works only focus on TSP, recent works started to consider the harder problems of VRPs and variants. Nazari et al. (2018) proposed a simplified version of Pointer Networks to solve capacitated VRPs, split-delivery VRPs (SDVRP) and stochastic VRPs. In order to make the input invariant to sequence order (e.g., order of customers), they replaced the RNN encoder of Pointer Networks by simple embedding maps. The resulting model can then handle changes in the input (e.g., customer demand after being visited). An actor-critic scheme was used for training and beam search, which tracks the most probable paths, was used to generate the final best solution.

Kool, van Hoof, and Welling (2019) proposed a transformer-based model to solve

<!-- Page 19 -->
routing problems, such as TSP, CVRP, or split delivery VRP. The model is similar to that of Deudon et al. (2018) with a few simplifications and improvements. In particular, in contrast to that work, Kool, van Hoof, and Welling (2019) did not use 2-OPT. Another novelty is that policy gradient with a self-critic baseline (estimated with greedy policy rollouts) was used for training. Peng, Wang, and Zhang (2020) generalized this approach to use a dynamic attention model so that state features can be updated during the construction of a solution.

Sheng, Ma, and Xia (2020) proposed a variation of Pointer Network to solve VRP with Task Priority and Limited Resources. The model was trained in the RL setting. They showed that this approach is comparable to Genetic Algorithm (GA) for medium-sized instances ($\approx 50$ cities), but its performance becomes better than GA for larger-sized instances ($> 100$ cities) while taking much less computational time.

Duan et al. (2020) proposed a technique that combines training with reinforcement learning and supervised learning. The method was based on graph convolutional network to encode a problem instance with node and edge features. Node features were used as input of an RNN policy to output a solution, which was used to train a classifier taking edge features as inputs to predict the probability of selecting an edge in a solution. The method was evaluated on real-world data sets and shown to generalize well.

Apart from the work by Nazari et al. (2018), who considered SVRP, there is scarce literature on dynamic and stochastic VRP using modern machine learning techniques. One exception is Balaji et al. (2019) that considers stochastic and dynamic capacitated VRP problems with pickup and delivery, time windows and service guarantee. The authors showed that deep RL algorithms can directly be trained to solve them and they are competitive or superior to classic baselines.

On the other hand, traditional evolutionary algorithms have advantages in dynamic capacitated VRP problems, especially with the assistance of heuristic-based methods. Sabar et al. (2013) investigated a hyper-heuristic method assisted by a grammatical evolutionary method for capacitated VRP problem.

Jacobsen-Grocott et al. (2017) attempted to use a hyper-heuristic method to solve Dynamic Vehicle Routing with Time Windows (DVRPTW). Such problems require the acceptance or rejection decisions for dynamically arriving customer requests. The genetic programming evolved heuristics were used to determine whether or not to accept new requests and add them to the current routes. The results showed that with the dynamism degree increasing, the GP-evolved heuristics significantly outperformed the handcrafted heuristics.

Liu et al. (2017) developed a new Genetic Programming-based Hyper-Heuristic (GPHH) for automated heuristic design for Uncertain Capacitated Arc Routing Problem (UCARP), and designed a novel effective meta-algorithm. Their experimental results showed that the proposed GPHH significantly outperforms the existing GPHH methods and manually designed heuristics.

To gain more interpretable routing policies, Wang et al. (2019) used three ensemble genetic programming methods, namely, Bagging GP, Boosting GP, and Cooperative Co-evolution GP, to solve uncertain capacitated arc routing problem. Evolved depth-limited tree expressions correspond to an ensemble of these methods to represent the priorities of each task in an instance. The results showed that an ensemble of simple policies is able to compete with the complex policies while maintaining their high interpretability.

MacLachlan et al. (2020) proposed to evolve collaborative routing policies within a data-driven genetic programming hyper-heuristic algorithm for a capacitated arc

<!-- Page 20 -->
routing problem with uncertainties.

On the application side, Chen et al. (2020c) introduced genetic programming based hyper-heuristic method to solve a realistic VRP in marine container port with various uncertainty. The results confirmed the effectiveness of GP for this kind of dynamic uncertain problem.

## 5. Concluding Remarks and Future Directions

The vehicle routing problem and its variants are one of the most studied combinatorial optimisation problems in the research community because of its close relevance to transportation problems in industrial and societal activities, and yet few known algorithms have solved them to a satisfactory level. The main challenges lie in the scale of real-world problems, complexity in objectives and constraints (non-linearity, dynamic nature) and uncertainties. This literature review focuses on research efforts in integrating data analytics and machine learning in addressing these challenging VRP problems. A number of observations can be made.

### 5.1. Problem diversity and dynamics

One of most challenging aspects of VRP is its diverse and dynamic nature. As indicated in Eksioglu, Vural, and Reisman (2009b), the varieties of VRP can be attributed to the diverse problem scenario characteristics, its physical characteristics, as well as its information characteristics. This leads to numerous VRP models that have been developed to capture the main structures of various real-world problems. Although some research efforts have been made to generalise these models, big challenges still exist in terms of solution methods because these generalised models may not be able to take advantage of the underlying special structures that can be exploited for the development of more efficient algorithms. Therefore, a growing research direction is to automate (at least partially) the modelling process of real-life VRP problems by integrating data analytics and machine learning methods so that the key patterns and structures of the problems can be automatically identified and the most suitable VRP variants can be matched to the problem at hand. For practical applications, it, therefore, makes sense to develop a VRP expert system with a repository of parameterized VRP models and their corresponding algorithms so that the real-world problems can be automatically analysed, clustered and matched with one of existing model-algorithm pairs and readily solved.

Nevertheless, one should also recognise the dynamic nature of the VRP problems in real life. The problem can diverge from one variant or type to another over time. Therefore, such an expert system must be adaptive to the dynamic changes of the environments, leading to the requirements of using more advanced AI methods so that the system can evolve and improve automatically. Therefore, the ability to self-learn and self-evolve over time will be a key feature of the next generation VRP expert systems.

### 5.2. Perturbative improvements vs. generative approaches

As a classic NP-hard problem, VRP problems are often solved heuristically via iterative procedures, which can be broadly categorised in two different ways, namely perturbative and generative. A perturbative method is a kind of create-improve pro-

<!-- Page 21 -->
cess that assumes a deterministic (i.e. offline) model and aims to improve the quality of the solution incrementally while having ability to escaping from poor local optima. Various learning mechanisms can be used to exploit the structures of the solution landscapes. Machine learning has proved to be beneficial in helping select among a set of pre-defined perturbative heuristics or neighbourhoods in the most appropriate way. It can also be used for the automation at lower level. For example, ML can generate perturbation heuristics automatically. It can also be used to approximate the expensive solution evaluations to speed up the local search process.

Generative VRP approaches seek to build a high quality solution from scratch. Because the training process is often done offline, this type of methods has advantages compared with perturbative algorithms in terms of solution time and ability to handle uncertainties and problem related dynamics. Its main weakness is the quality of the resulting solutions which are often inferior compared with those from perturbative methods. However, with the advances in the computing power and deep learning (in particular the deep reinforcement learning), generative VRP algorithms have gained increasing popularity in recent years and have achieved more successes in solving practical VRP problems.

## 5.3. Model driven vs. data driven

Although in the early stage of the VRP research, significant research attentions have been paid on the perturbative methods that strive to search for the global optimality to a given problem formulation. However, it is increasingly recognised that, due to the problem uncertainties and dynamics, the perceived optimality is very rarely realised in practice. It is often impractical (if ever possible) to formulate the real-life VRP problems exactly because the resulting models will most likely be intractable. A compromise has to be made between the accuracy of the models and their tractability. This can be very challenging for practitioners.

As an alternative, data driven methods have been proposed as end-to-end solvers to VRP problems to reduce (or remove) the requirement of mathematical models. These methods train deep neural networks to produce solutions directly without the mathematical formulations, potentially making them easier to be applied in the real world. Therefore, these data driven methods often serve as black-box solvers and are often criticised for their poor interpretation abilities. Another stream of data-driven methods are based on the principles of genetic programming that evolves decision trees for solution constructions for VRP problems based on historical data. When suitable constraints and requirements are enforced during the evolution process, the resulting GP trees tend to have better interpretation abilities than the solutions from the deep neural networks.

## 5.4. Challenges and Prospects

Despite the fast growing popularity of integrating ML in VRP research, the research community still faces a number of challenges.

Firstly, the VRP is often part of a larger complex system that requires a good level of reliability or robustness across all possible scenarios to ensure the system does not reach certain undesirable (or disastrous) states. In addition, a good level of interpretability of the adopted algorithms is also required. Although there has been some research efforts in the area of explainable AI and verification methods, more theoretic research must

<!-- Page 22 -->
be conducted to lay solid foundations for the future research and broader applications.

Secondly, there lacks a high-quality ML-assisted VRP research platform for training and testing purposes. The required resources and libraries are very scattered in two different research communities in machine learning and operations research, respectively. To help grow the ML-assisted VRP research, the two research communities must work together towards an integrated platform with libraries and tools for both ML and optimisation. Of course, this would also lead to opportunities for multidisciplinary research.

Thirdly, many of the current ML-assisted VRP research requires huge amount of data which is normally not available directly. The trained models do not generalise well across different instances, scenarios and problem domains. This lack of data and model generalisation is largely caused by the trial-and-error nature of the traditional deep reinforcement learning. Important information regarding the objective function(s) and constraints is not fully exploited. More novel approaches are required to provide a better fusion of the current analytical methods based on mathematical models and neural network methods driven by data.

Last but not least, more real-world applications must be encouraged to address the criticisms of the existing VRP research. One must balance the adoption costs and the performance of the solution methods. In particular, a much more powerful VRP simulation software is required to help practitioners to build customised VRP environment with high performance in speed at a low cost. In addition, the research community should also consider building pre-trained VRP models/libraries to reduce the training costs further.

## Acknowledgement

This work is supported by the National Natural Science Foundation of China (grant number 72071116, 71471092), the Zhejiang Natural Science Foundation (grant number LR17G010001) and the Ningbo Science and Technology Bureau (grant numbers 2019B10026, 2017D10034).

## References

Ahmed, Leena, Christine Mumford, and Ahmed Kheiri. 2019. “Solving urban transit route design problem using selection hyper-heuristics.” *European Journal of Operational Research* 274 (2): 545–559.

Aksen, Deniz, Onur Kaya, F. Sibel Salman, and Ozge Tuncel. 2014. “An adaptive large neighborhood search algorithm for a selective and periodic inventory routing problem.” *European Journal of Operational Research* 239 (2): 413–426.

Albareda-Sambola, Maria, Elena Fernández, and Gilbert Laporte. 2014. “The dynamic multi-period vehicle routing problem with probabilistic information.” *Computers & Operations Research* 48: 31–39.

Alvarez, Alejandro Marcos, Quentin Louveaux, and Louis Wehenkel. 2017. “A Machine Learning-Based Approximation of Strong Branching.” *INFORMS Journal on Computing* 29 (1): 185–195.

Anuar, Wadi Khalid, M Moll, L S Lee, S Pickl, and H V. Seow. 2019. “Vehicle Routing Optimization for Humanitarian Logistics in Disaster Recovery: A Survey.” In *Proceedings of the International Conference on Security and Management (SAM2019)*, Athens, .

<!-- Page 23 -->
Archetti, Claudia, and M. Grazia Speranza. 2014. “A survey on matheuristics for routing problems.” *EURO Journal on Computational Optimization* 2 (4): 223–246.

Asta, Shahriar, and Ender Ozcan. 2014. *An Apprenticeship Learning Hyper-Heuristic for Vehicle Routing in HyFlex*. New York: Ieee.

Augerat, Ph, Jose Manuel Belenguer, Enrique Benavent, A Corberán, D Naddef, and G Rinaldi. 1995. *Computational results with a branch and cut code for the capacitated vehicle routing problem*. Vol. 34. IMAG.

Azi, Nabila, Michel Gendreau, and Jean-Yves Potvin. 2014. “An adaptive large neighborhood search for a vehicle routing problem with multiple routes.” *Computers & Operations Research* 41: 167–173.

Bahdanau, Dzmitry, Kyung Hyun Cho, and Yoshua Bengio. 2015. “Neural machine translation by jointly learning to align and translate.” In *3rd international conference on learning representations, ICLR 2015 - conference track proceedings*, .

Bai, Ruibin, Edmund K. Burke, Michel Gendreau, Graham Kendall, and Barry McCollum. 2007. *Memory length in hyper-heuristics: An empirical study*. New York: Ieee.

Bai, Ruibin, Stein W Wallace, Jingpeng Li, and Alain Yee-Loong Chong. 2014. “Stochastic service network design with rerouting.” *Transportation Research Part B: Methodological* 60: 50–65.

Balaji, Bharathan, Jordan Bell-Masterson, Enes Bilgin, Andreas Damianou, Pablo Moreno Garcia, Arpit Jain, Runfei Luo, Alvaro Maggiar, Balakrishnan Narayanaswamy, and Chun Ye. 2019. “ORL: Reinforcement Learning Benchmarks for Online Stochastic Optimization Problems.” In *AAAI*, Nov. http://arxiv.org/abs/1911.10641.

Baldacci, Roberto, Eleni Hadjiconstantinou, and Aristide Mingozzi. 2004. “An exact algorithm for the capacitated vehicle routing problem based on a two-commodity network flow formulation.” *Operations research* 52 (5): 723–738.

Beasley, JE. 1983. “Route first—Cluster second methods for vehicle routing.” *Omega* 11 (4): 403–408.

Bellman, Richard. 1962. “Dynamic programming treatment of the travelling salesman problem.” *Journal of the ACM (JACM)* .

Bello, Irwan, Hieu Pham, Quoc V. Le, Mohammad Norouzi, and Samy Bengio. 2017. “Neural Combinatorial Optimization with Reinforcement Learning.” In *ICLR Workshop*, Nov., 473–474. http://arxiv.org/abs/1611.09940.

Belo-Filho, M. a. F., P. Amorim, and B. Almada-Lobo. 2015. “An adaptive large neighbourhood search for the operational integrated production and distribution problem of perishable products.” *International Journal of Production Research* 53 (20): 6040–6058.

Bengio, Yoshua, Andrea Lodi, and Antoine Prouvost. 2020. “Machine Learning for Combinatorial Optimization: a Methodological Tour d’Horizon.” *European Journal of Operational Research* S0377221720306895.

Bent, Russell, and Pascal Van Hentenryck. 2005. “Online Stochastic Optimization Without Distributions.” In *ICAPS*, Vol. 5, 171–180.

Bent, Russell W., and Pascal Van Hentenryck. 2004. “Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic Customers.” *Operations Research* 52 (6): 977–987.

Bernardo, Marcella, Bo Du, and Jürgen Pannek. 2020. “A simulation-based solution approach for the robust capacitated vehicle routing problem with uncertain demands.” *Transportation Letters* 0 (0): 1–10.

Bertsekas, Dimitri P., and John N. Tsitsiklis. 1996. *Neuro-dynamic programming*. Belmont, Mass., United States: Athena Scientific.

Bertsimas, Dimitris, David Brown, and Constantine Caramanis. 2010. “Theory and Applications of Robust Optimization.” *SIAM Review* 53.

Boussaid, Ilhem, Julien Lepagnot, and Patrick Siarry. 2013. “A survey on optimization metaheuristics.” *Information sciences* 237: 82–117.

Boyan, J. A., and A. W. Moore. 2000. “Learning evaluation functions to improve optimization by local search.” *Journal of Machine Learning Research* 1 ((NOV)): 77–112.

Boyd, Stephen, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. 2010. “Dis-

<!-- Page 24 -->
tributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers.” *Foundations and Trends® in Machine Learning* 3 (1): 1–122.

Braekers, Kris, Katrien Ramaekers, and Inneke Van Nieuwenhuyse. 2016. “The vehicle routing problem: State of the art classification and review.” *Computers & Industrial Engineering* 99: 300–313.

Bräysy, O., and M. Gendreau. 2005a. “Vehicle routing problem with time windows, part 1: Route construction and local search algorithms.” *Transportation Science* 39 (1): 104–118.

Bräysy, O., and M. Gendreau. 2005b. “Vehicle routing problem with time windows, part II: Metaheuristics.” *Transportation Science* 39 (1): 119–139.

Burke, Edmund K., Matthew R. Hyde, Graham Kendall, Gabriela Ochoa, Ender Özcan, and John R. Woodward. 2019. “A Classification of Hyper-Heuristic Approaches: Revisited.” In *Handbook of Metaheuristics*, edited by Michel Gendreau and Jean-Yves Potvin, International Series in Operations Research & Management Science, 453–477. Cham: Springer International Publishing.

Calvet, Laura, Albert Ferrer, M Isabel Gomes, Angel A Juan, and David Masip. 2016a. “Combining statistical learning with metaheuristics for the multi-depot vehicle routing problem with market segmentation.” *Computers & Industrial Engineering* 94: 93–104.

Calvet, Laura, Angel A Juan, Carles Serrat, and Jana Ries. 2016b. “A statistical learning based approach for parameter fine-tuning of metaheuristics.” *SORT-Statistics and Operations Research Transactions* 201–224.

Ce, Fu, and Wang Hui. 2010. “The solving strategy for the real-world vehicle routing problem.” In *2010 3rd International Congress on Image and Signal Processing*, .

Chen, Binhui, Rong Qu, Ruibin Bai, and Wasakorn Laesanklang. 2018. “A hyper-heuristic with two guidance indicators for bi-objective mixed-shift vehicle routing problem with time windows.” *Applied Intelligence* 48 (12): 4937–4959.

Chen, Binhui, Rong Qu, Ruibin Bai, and Wasakorn Laesanklang. 2020a. “A variable neighborhood search algorithm with reinforcement learning for a real-life periodic vehicle routing problem with time windows and open routes.” *RAIRO - Operations Research* 54 (5): 1467–1494.

Chen, Mingxiang, Lei Gao, Qichang Chen, and Zhixin Liu. 2020b. “Dynamic Partial Removal: A Neural Network Heuristic for Large Neighborhood Search.” *arXiv* .

Chen, Xinan, Ruibin Bai, Rong Qu, Haibo Dong, and Jianjun Chen. 2020c. “A Data-Driven Genetic Programming Heuristic for Real-World Dynamic Seaport Container Terminal Truck Dispatching.” In *2020 IEEE Congress on Evolutionary Computation (CEC)*, 1–8. IEEE.

Chen, Xinyun, and Yuandong Tian. 2019. “Learning to Perform Local Rewriting for Combinatorial Optimization.” In *NeurIPS*, 6281–6292. https://github.com/facebookresearch/neural-rewriter.

Christiansen, Christian H, and Jens Lysgaard. 2007. “A branch-and-price algorithm for the capacitated vehicle routing problem with stochastic demands.” *Operations Research Letters* 35 (6): 773–781.

Chuah, Keng Hoo, and Jon C. Yingling. 2005. “Routing for a Just-in-Time Supply Pickup and Delivery System.” *Transportation Science* 39 (3): 328–339.

Comert, Serap Ercan, Harun Resit Yazgan, Sena Kir, and Furkan Yener. 2018. “A cluster first-route second approach for a capacitated vehicle routing problem: a case study.” *International Journal of Procurement Management* 11 (4): 399.

Comert, Serap Ercan, Harun Resit Yazgan, Irem Sertvuran, and Hanife Sengul. 2017. “A new approach for solution of vehicle routing problem with hard time window: an application in a supermarket chain.” *Sādhanā* 42 (12): 2067–2080.

Cooray, P. L. N. U., and Thashika D. Rupasinghe. 2017. “Machine Learning-Based Parameter Tuned Genetic Algorithm for Energy Minimizing Vehicle Routing Problem.” *Journal of Industrial Engineering* 2017: 3019523.

Cordeau, Jean-François, Mauro Dell’Amico, Simone Falavigna, and Manuel Iori. 2015. “A rolling horizon algorithm for auto-carrier transportation.” *Transportation Research Part B: Methodological* 76: 68–80.

<!-- Page 25 -->
Costa, Joao Guilherme Cavalcanti, Yi Mei, and Mengjie Zhang. 2020. “Cluster-based Hyper-Heuristic for Large-Scale Vehicle Routing Problem.” In *2020 IEEE Congress on Evolutionary Computation (CEC)*, 1–8. IEEE.

da Costa, Paulo R. de O., Jason Rhuggenaath, Yingqian Zhang, and Alp Akcay. 2020. “Learning 2-opt Heuristics for the Traveling Salesman Problem via Deep Reinforcement Learning.” *arXiv:2004.01608 [cs, stat]*.

Dantzig, George B, and John H Ramser. 1959. “The truck dispatching problem.” *Management science* 6 (1): 80–91.

Defourny, Boris. 2010. “Machine Learning Solution Methods for Multistage Stochastic Programming.” PhD diss., University of Liege. https://www.lehigh.edu/defourny/PhDthesis_B_Defourny.pdf.

Demir, Emrah, Tolga Bektaş, and Gilbert Laporte. 2012. “An adaptive large neighborhood search heuristic for the Pollution-Routing Problem.” *European Journal of Operational Research* 223 (2): 346–359.

Desaulniers, Guy, Andrea Lodi, and Mouad Morabit. 2020. *Machine-learning-based column selection for column generation*. Technical Report G-2020-29. GERAD. ISSN: 0771-2440 Publication Title: Les Cahiers du GERAD, Accessed 2020-09-29. https://www.gerad.ca/en/papers/G-2020-29.

Deudon, Michel, Pierre Cournut, Alexandre Lacoste, Yossiri Adulyasak, and Louis Martin Rousseau. 2018. “Learning heuristics for the tsp by policy gradient.” In *CPAIOR*, Vol. 10848 LNCS, 170–181. Springer Verlag. http://halog.polymtl.ca/wp-content/uploads/2018/11/cpaior-learning-heuristics-6.pdf.

Dinh, Thai, Ricardo Fukasawa, and James Luedtke. 2018. “Exact algorithms for the chance-constrained vehicle routing problem.” *Mathematical Programming* 172 (1-2): 105–138.

Dondo, Rodolfo, and Jaime Cerdá. 2007. “A cluster-based optimization approach for the multi-depot heterogeneous fleet vehicle routing problem with time windows.” *European Journal of Operational Research* 176 (3): 1478–1507.

Drake, John H., Ender Özcan, and Edmund K. Burke. 2012. “An Improved Choice Function Heuristic Selection for Cross Domain Heuristic Search.” In *Parallel Problem Solving from Nature - PPSN XII*, edited by Carlos A. Coello Coello, Vincenzo Cutello, Kalyanmoy Deb, Stephanie Forrest, Giuseppe Nicosia, and Mario Pavone, Lecture Notes in Computer Science, Berlin, Heidelberg, 307–316. Springer.

Duan, Lu, Yang Zhan, Haoyuan Hu, Yu Gong, Jiangwen Wei, Xiaodong Zhang, and Yinghui Xu. 2020. “Efficiently Solving the Practical Vehicle Routing Problem: A Novel Joint Learning Approach.” In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, KDD ’20, New York, NY, USA, Aug., 3054–3063. Association for Computing Machinery.

Eksioglu, Burak, Arif Volkan Vural, and Arnold Reisman. 2009a. “The vehicle routing problem: A taxonomic review.” *Computers & Industrial Engineering* 57 (4): 1472–1483.

Eksioglu, Burak, Arif Volkan Vural, and Arnold Reisman. 2009b. “The vehicle routing problem: A taxonomic review.” *Computers & Industrial Engineering* 57 (4): 1472–1483.

Elshaer, Raafat, and Hadeer Awad. 2020. “A taxonomic review of metaheuristic algorithms for solving the vehicle routing problem and its variants.” *Computers & Industrial Engineering* 140: 106242.

Erdoğan, Sevgi, and Elise Miller-Hooks. 2012. “A Green Vehicle Routing Problem.” *Transportation Research Part E: Logistics and Transportation Review* 48 (1): 100–114.

Feo, Thomas A, and Mauricio GC Resende. 1995. “Greedy randomized adaptive search procedures.” *Journal of global optimization* 6 (2): 109–133.

Fisher, Marshall L., and Ramchandran Jaikumar. 1981. “A generalized assignment heuristic for vehicle routing.” *Networks* 11 (2): 109–124.

Gao, Lei, Mingxiang Chen, Qichang Chen, Ganzhong Luo, Nuoyi Zhu, and Zhixin Liu. 2020. “Learn to Design the Heuristics for Vehicle Routing Problem.” In *arXiv*, Feb. Accessed 2020-10-03. http://arxiv.org/abs/2002.08539.

Gao, Shangce, Yirui Wang, Jiujun Cheng, Yasuhiro Inazumi, and Zheng Tang. 2016. “Ant

<!-- Page 26 -->
colony optimization with clustering for solving the dynamic location routing problem.” *Applied Mathematics and Computation* 285: 149–173.

Garrido, Pablo, and Maria Cristina Riff. 2010. “DVRP: a hard dynamic combinatorial optimisation problem tackled by an evolutionary hyper-heuristic.” *Journal of Heuristics* 16 (6): 795–834.

Geetha, S., G. Poonthalir, and P.T. Vanathi. 2013. “Nested particle swarm optimisation for multi-depot vehicle routing problem.” *International Journal of Operational Research* 16 (3): 329.

Geetha, S., P. T. Vanathi, and G. Poonthalir. 2012. “Metaheuristic Approach For The Multi Depot Vehicle Routing Problem.” *Applied Artificial Intelligence* 26 (9): 878–901.

Gendreau, M., G. Laporte, and R. Seguin. 1996. “Stochastic vehicle routing.” *European Journal of Operational Research* 88 (1): 3–12.

Gendreau, M., and J. Potvin. 2010. *Handbook of metaheuristics*. International series in operations research & management science, Vol 46. Springer.

Ghosal, Shubhechyya, and Wolfram Wiesemann. 2020. “The distributionally robust chance constrained vehicle routing problem.” *Operations Research* .

Glover, Fred W., and Manuel Laguna. 1997. *Tabu Search*. Springer US.

Godfrey, Gregory A., and Warren B. Powell. 2002. “An Adaptive Dynamic Programming Algorithm for Dynamic Fleet Management, I: Single Period Travel Times.” *Transportation Science* 36 (1): 21–39.

Gounaris, Chrysanthos E., Wolfram Wiesemann, and Christodoulos A. Floudas. 2013. “The Robust Capacitated Vehicle Routing Problem Under Demand Uncertainty.” *Operations Research* 61 (3): 677–693.

Grunow, Martin, Hans-Otto Günther, and Matthias Lehmann. 2006. “Strategies for dispatching AGVs at automated seaport container terminals.” *OR spectrum* 28 (4): 587–610.

Gutierrez, Andres, Laurence Dieulle, Nacima Labadie, and Nubia Velasco. 2018. “A multi-population algorithm to solve the VRP with stochastic service and travel times.” *Computers & Industrial Engineering* 125: 144–156.

Göğmen, Elifcan, and Rızvan Erol. 2019. “Transportation problems for intermodal networks: Mathematical models, exact and heuristic algorithms, and machine learning.” *Expert Systems with Applications* 135: 374–387.

Ha, Minh Hoang, Tat Dat Nguyen, Thinh Nguyen Duy, Hoang Giang Pham, Thuy Do, and Louis-Martin Rousseau. 2020. “A new constraint programming model and a linear programming-based adaptive large neighborhood search for the vehicle routing problem with synchronization constraints.” *Computers & Operations Research* 124: 105085.

Han, Jinil, Chungmok Lee, and Sungsoo Park. 2014. “A robust scenario approach for the vehicle routing problem with uncertain travel times.” *Transportation science* 48 (3): 373–390.

Hasan, Hameedah Sahib, M Shukri Zainal Abidin, and MSA Mahmud MFMS. 2019. “Automated guided vehicle routing: Static, dynamic and free range.” *Int J Eng Adv Technol* 8 (5C): 1–7.

He, He, Hal Daume III, and Jason M Eisner. 2014. “Learning to Search in Branch and Bound Algorithms.” In *Advances in Neural Information Processing Systems 27*, edited by Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, 3293–3301. Curran Associates, Inc.

He, Ruhan, Weibin Xu, Jiaxia Sun, and Bingqiao Zu. 2009. “Balanced K-Means Algorithm for Partitioning Areas in Large-Scale Vehicle Routing Problem.” In *2009 Third International Symposium on Intelligent Information Technology Application*, NanChang, China, 87–90. IEEE.

Held, Michael, and Richard M. Karp. 1962. “A dynamic programming approach to sequencing problems.” *Journal of the Society for Industrial and Applied Mathematics* .

Helsgaun, Keld. 2017. “An extension of the Lin-Kernighan-Helsgaun TSP solver for constrained traveling salesman and vehicle routing problems.” *Roskilde: Roskilde University* .

Hemmelmayr, Vera C., Jean-Francois Cordeau, and Teodor Gabriel Crainic. 2012. “An adaptive large neighborhood search heuristic for Two-Echelon Vehicle Routing Problems arising in

<!-- Page 27 -->
city logistics.” *Computers & Operations Research* 39 (12): 3215–3228.

Heyken Soares, Philipp, Leena Ahmed, Yong Mao, and Christine L. Mumford. 2020. “Public transport network optimisation in PTV Visum using selection hyper-heuristics.” *Public Transport*.

Hopfield, J. J., and D. W. Tank. 1985. “‘Neural’ computation of decisions in optimization problems.” *Biological Cybernetics* 52 (3): 141–152.

Hottung, André, and Kevin Tierney. 2019. “Neural large neighborhood search for the capacitated vehicle routing problem.” In *arXiv*, https://arxiv.org/abs/1911.09539.

Hu, Xiangpei, Minfang Huang, and Amy Z Zeng. 2007. “An intelligent solution system for a vehicle routing problem in urban distribution.” *International Journal of Innovative Computing, Information and Control* 3 (1): 189–198.

Hu, Yujiao, Yuan Yao, and Wee Sun Lee. 2020. “A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs.” *Knowledge-Based Systems* 204: 106244.

Jacobsen-Grocott, Josiah, Yi Mei, Gang Chen, and Mengjie Zhang. 2017. “Evolving heuristics for Dynamic Vehicle Routing with Time Windows using genetic programming.” In *2017 IEEE Congress on Evolutionary Computation (CEC)*, Jun., 1948–1955.

Joshi, Chaitanya K., Thomas Laurent, and Xavier Bresson. 2019a. “An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem.” http://arxiv.org/abs/1906.01227.

Joshi, Chaitanya K., Thomas Laurent, and Xavier Bresson. 2019b. “On Learning Paradigms for the Travelling Salesman Problem.” http://arxiv.org/abs/1910.07210.

Kadaba, Nagesh, Kendall E. Nygard, and Paul L. Juell. 1991. “Integration of adaptive machine learning and knowledge-based systems for routing and scheduling applications.” *Expert Systems with Applications* 2 (1): 15–27.

Kaempfer, Yoav, and Lior Wolf. 2019. “Learning the Multiple Traveling Salesmen Problem with Permutation Invariant Pooling Networks.” Feb. http://arxiv.org/abs/1803.09621.

Khalil, Elias B., Pierre Le Bodic, Le Song, George Nemhauser, and Bistra Dilkina. 2016. “Learning to branch in mixed integer programming.” In *30th AAAI Conference on Artificial Intelligence, AAAI 2016*, https://www.cc.gatech.edu/~lsong/papers/KhaLebSonNemDil16.pdf.

Khalil, Elias B., Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. 2017. “Learning Combinatorial Optimization Algorithms over Graphs.” *Advances in Neural Information Processing Systems* DECEM2017: 6349–6359.

Kirkpatrick, Scott, C Daniel Gelatt, and Mario P Vecchi. 1983. “Optimization by simulated annealing.” *science* 220 (4598): 671–680.

Kool, Wouter, Herke van Hoof, and Max Welling. 2019. “Attention, Learn to Solve Routing Problems!” In *7th International Conference on Learning Representations, ICLR 2019*, International Conference on Learning Representations, ICLR. http://arxiv.org/abs/1803.08475.

Korayem, L, M Khorsid, and S S Kassem. 2015. “Using Grey Wolf Algorithm to Solve the Capacitated Vehicle Routing Problem.” *IOP Conference Series: Materials Science and Engineering* 83: 012014.

Kovacs, Attila A., Sophie N. Parragh, Karl F. Doerner, and Richard F. Hartl. 2012. “Adaptive large neighborhood search for service technician routing and scheduling problems.” *Journal of Scheduling* 15 (5): 579–600.

Kruber, Markus, Marco E. Lübbecke, and Axel Parmentier. 2017. “Learning When to Use a Decomposition.” In *Integration of AI and OR Techniques in Constraint Programming*, edited by Domenico Salvagnin and Michele Lombardi, Vol. 10335, 202–210. Cham: Springer International Publishing.

Kubra, Ekiz Melike, Bozdemir Muhammet, and Turkkan Burcu Ozcan. 2019. “Route First-Cluster Second Method For Personal Service Routing Problem.” *Journal of Engineering Studies and Research* 25 (2): 18–24.

Kucharska, Edyta. 2019. “Dynamic vehicle routing problem—Predictive and unexpected cus-

<!-- Page 28 -->
tomer availability.” *Symmetry* 11 (4): 546.

Lahyani, Rahma, Anne-Lise Gouguenheim, and Leandro C. Coelho. 2019. “A hybrid adaptive large neighbourhood search for multi-depot open vehicle routing problems.” *International Journal of Production Research* 57 (22): 6963–6976.

Laporte, Gilbert, Roberto Musmanno, and Francesca Vocaturo. 2010. “An Adaptive Large Neighbourhood Search Heuristic for the Capacitated Arc-Routing Problem with Stochastic Demands.” *Transportation Science* 44 (1): 125–135.

Leng, Longlong, Yanwei Zhao, Zheng Wang, Jingling Zhang, Wanliang Wang, and Chunmiao Zhang. 2019. “A Novel Hyper-Heuristic for the Biobjective Regional Low-Carbon Location-Routing Problem with Multiple Constraints.” *Sustainability* 11 (6): 1596.

Li, Xia, Ruibin Bai, Peer-Olaf Siebers, and Christian Wagner. 2019. “Travel time prediction in transport and logistics.” *VINE Journal of Information and Knowledge Management Systems* 49 (3): 277–306.

Li, Xiangyong, Peng Tian, and Stephen CH Leung. 2010. “Vehicle routing problems with time windows and stochastic travel and service times: Models and algorithm.” *International Journal of Production Economics* 125 (1): 137–145.

Li, Xijun, Mingxuan Yuan, Di Chen, Jianguo Yao, and Jia Zeng. 2018. “A data-driven three-layer algorithm for split delivery vehicle routing problem with 3D container loading constraint.” In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 528–536.

Li, Zhuwen, Qifeng Chen, and Vladlen Koltun. 2018. “Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search.” *Advances in Neural Information Processing Systems* 2018-Decem: 539–548.

Liu, Yuxin, Yi Mei, Mengjie Zhang, and Zili Zhang. 2017. “Automated heuristic design using genetic programming hyper-heuristic for uncertain capacitated arc routing problem.” In *Proceedings of the Genetic and Evolutionary Computation Conference*, 290–297.

Lodi, Andrea, and Giulia Zarpellon. 2017. “On learning and branching: a survey.” *TOP* 25 (2): 207–236.

Lu, Hao, Xingwen Zhang, and Shuang Yang. 2020. “A Learning-based Iterative Method for Solving Vehicle Routing Problems.” In *ICLR*, 15.

Luo, Jianping, and Min-Rong Chen. 2014. “Multi-phase modified shuffled frog leaping algorithm with extremal optimization for the MDVRP and the MDVRPTW.” *Computers & Industrial Engineering* 72: 84–97.

Ma, Qiang, Suwen Ge, Danyang He, Darshan Thaker, and Iddo Drori. 2020. “Combinatorial Optimization by Graph Pointer Networks and Hierarchical Reinforcement Learning.” In *AAAI Workshop on Deep Learning on Graphs: Methodologies and Applications*, https://arxiv.org/pdf/1911.04936.pdf.

MacLachlan, Jordan, Yi Mei, Juergen Branke, and Mengjie Zhang. 2020. “Genetic Programming Hyper-Heuristics with Vehicle Collaboration for Uncertain Capacitated Arc Routing Problems.” *Evolutionary Computation* 28 (4): 563–593.

Markovic, H, I Cavar, and T Caric. 2005. “Using data mining to forecast uncertain demands in stochastic vehicle routing problem.” In *13th International Symposium on Electronics in Transport (ISEP), Slovenia*, 1–6.

Masson, Renaud, Fabien Lehuede, and Olivier Peton. 2013. “An Adaptive Large Neighborhood Search for the Pickup and Delivery Problem with Transfers.” *Transportation Science* 47 (3): 344–355.

Menger, Karl. 1932. “Das botenproblem.” *Ergebnisse eines mathematischen kolloquiums* 2 (4): 11–12.

Miranda-Bront, Juan José, Brian Curcio, Isabel Méndez-Díaz, Agustín Montero, Federico Pousa, and Paula Zabala. 2017. “A cluster-first route-second approach for the swap body vehicle routing problem.” *Annals of Operations Research* 253 (2): 935–956.

Mladenović, Nenad, and Pierre Hansen. 1997. “Variable neighborhood search.” *Computers & operations research* 24 (11): 1097–1100.

Moll, R. N., A. G. Barto, T. J. Perkins, and R. S. Sutton. 1998. “Learning Instance-Independent

<!-- Page 29 -->
Value Functions to Enhance Local Search.” In *Neural information processing systems*, 1017–1023.

Montoya, Jose A., Christelle Guéret, Jorge E. Mendoza, and Juan G. Villegas. 2014. “A route-first cluster-second heuristic for the Green Vehicle Routing Problem.” In *ROADEF2014*, Bordeaux, France, Feb. Accessed 2020-10-02. https://hal.archives-ouvertes.fr/hal-00946492.

Musolino, Giuseppe, Antonio Polimeni, Corrado Rindone, and Antonino Vitetta. 2013. “Travel time forecasting and dynamic routes design for emergency vehicles.” *Procedia-Social and Behavioral Sciences* 87: 193–202.

Nallusamy, R., K. Duraiswamy, R. Dhanalaksmi, and P. Parthiban. 2010. “Optimization of Multiple Vehicle Routing Problems using Approximation Algorithms.” http://arxiv.org/abs/1001.4197.

Nazari, Mohammadreza, Afshin Oroojlooy, Lawrence V. Snyder, and Martin Takac. 2018. “Reinforcement Learning for Solving the Vehicle Routing Problem.” https://arxiv.org/abs/1802.04240.

Nowak, Alex, Soledad Villar, Afonso S. Bandeira, and Joan Bruna. 2017. “A Note on Learning Algorithms for Quadratic Assignment with Graph Neural Networks.” https://arxiv.org/abs/1706.07450v1.

Pandiri, Venkatesh, and Alok Singh. 2018. “A hyper-heuristic based artificial bee colony algorithm for k-Interconnected multi-depot multi-traveling salesman problem.” *Information Sciences* 463: 261–281.

Parragh, Sophie N., and Jean-Francois Cordeau. 2017. “Branch-and-price and adaptive large neighborhood search for the truck and trailer routing problem with time windows.” *Computers & Operations Research* 83: 28–44.

Peng, Bo, Jiahai Wang, and Zizhen Zhang. 2020. “A Deep Reinforcement Learning Algorithm Using Dynamic Attention Model for Vehicle Routing Problems.” In *Artificial Intelligence Algorithms and Applications*, edited by Kangshun Li, Wei Li, Hui Wang, and Yong Liu, Communications in Computer and Information Science, Singapore, 636–650. Springer.

Pillac, Victor, Michel Gendreau, Christelle Gueret, and Andres L. Medaglia. 2013. “A review of dynamic vehicle routing problems.” *European Journal of Operational Research* 225 (1): 1–11.

Pisinger, David, and Stefan Ropke. 2010. “Large neighborhood search.” In *Handbook of metaheuristics*, 399–419. Springer.

Potvin, Jean-Yves, Guy Lapalme, and Jean-Marc Rousseau. 1990. “Integration of AI and OR techniques for computer-aided algorithmic design in the vehicle routing domain.” *Journal of the Operational Research Society* 41 (6): 517–525.

Pour, Shahrzad M., John H. Drake, and Edmund K. Burke. 2018. “A choice function hyper-heuristic framework for the allocation of maintenance tasks in Danish railways.” *Computers & Operations Research* 93: 15–26.

Qi, Charles R., Hao Su, Kaichun Mo, and Leonidas J. Guibas. 2017. “PointNet: Deep learning on point sets for 3D classification and segmentation.” In *Proceedings - 30th IEEE conference on computer vision and pattern recognition, CVPR 2017*, .

Qi, Mingyao, Guoxiang Ding, You Zhou, and Lixin Miao. 2011. “Vehicle Routing Problem with Time Windows Based on Spatiotemporal Distance.” *Journal of Transportation Systems Engineering and Information Technology* 11 (1): 85–89.

Qiu, Ling, Wen-Jing Hsu, Shell-Ying Huang, and Han Wang. 2002. “Scheduling and routing algorithms for AGVs: a survey.” *International Journal of Production Research* 40 (3): 745–760.

Qu, Yuan, and Jonathan F. Bard. 2012. “A GRASP with adaptive large neighborhood search for pickup and delivery problems with transshipment.” *Computers & Operations Research* 39 (10): 2439–2456.

Ralphs, Ted K, Leonid Kopman, William R Pulleyblank, and Leslie E Trotter. 2003. “On the capacitated vehicle routing problem.” *Mathematical programming* 94 (2-3): 343–359.

Rautela, Anubha, S.K. Sharma, and P. Bhardwaj. 2019. “Distribution planning using capaci-

<!-- Page 30 -->
tated clustering and vehicle routing problem: A case of Indian cooperative dairy.” *Journal of Advances in Management Research* 16 (5): 781–795.

Reed, Martin, Aliki Yiannakou, and Roxanne Evering. 2014. “An ant colony algorithm for the multi-compartment vehicle routing problem.” *Applied Soft Computing* 15: 169–176.

Ribeiro, Glaydston Mattos, and Gilbert Laporte. 2012. “An adaptive large neighborhood search heuristic for the cumulative capacitated vehicle routing problem.” *Computers & Operations Research* 39 (3): 728–735.

Ritzinger, Ulrike, and Jakob Puchinger. 2013. “Hybrid Metaheuristics for Dynamic and Stochastic Vehicle Routing.” In *Hybrid Metaheuristics*, edited by El-Ghazali Talbi, Vol. 434, 77–95. Berlin, Heidelberg: Springer Berlin Heidelberg.

Ritzinger, Ulrike, Jakob Puchinger, and Richard F. Hartl. 2016. “A survey on dynamic and stochastic vehicle routing problems.” *International Journal of Production Research* 54 (1): 215–231.

Ropke, Stefan, and David Pisinger. 2006. “An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows.” *Transportation Science* 40 (4): 455–472.

Sabar, Nasser R., Masri Ayob, Graham Kendall, and Rong Qu. 2013. “Grammatical Evolution Hyper-Heuristic for Combinatorial Optimization Problems.” *Ieee Transactions on Evolutionary Computation* 17 (6): 840–861.

Sabar, Nasser R., Masri Ayob, Graham Kendall, and Rong Qu. 2014. “A dynamic multiarmed bandit-gene expression programming hyper-heuristic for combinatorial optimization problems.” *IEEE transactions on cybernetics* 45 (2): 217–228.

Sabar, Nasser R., Masri Ayob, Graham Kendall, and Rong Qu. 2015. “A Dynamic Multiarmed Bandit-Gene Expression Programming Hyper-Heuristic for Combinatorial Optimization Problems.” *IEEE Transactions on Cybernetics* 45 (2): 217–228.

Secomandi, Nicola. 2000. “Comparing neuro-dynamic programming algorithms for the vehicle routing problem with stochastic demands.” *Operations Research* 25.

Sheng, Yuxiang, Huawei Ma, and Wei Xia. 2020. “A Pointer Neural Network for the Vehicle Routing Problem with Task Priority and Limited Resources.” *Information Technology And Control* 49 (2): 237–248.

Smith, Kate A. 1999. “Neural networks for combinatorial optimization: A review of more than a decade of research.” *INFORMS Journal on Computing* .

Snoeck, André, Daniel Merchán, and Matthias Winkenbach. 2020. “Route learning: a machine learning-based approach to infer constrained customers in delivery routes.” *Transportation Research Procedia* 46: 229–236.

Soria-Alcaraz, Jorge A., Gabriela Ochoa, Marco A. Sotelo-Figeroa, and Edmund K. Burke. 2017. “A methodology for determining an effective subset of heuristics in selection hyper-heuristics.” *European Journal of Operational Research* 260 (3): 972–983.

Stützle, Thomas. 1999. *Local Search Algorithms for Combinatorial Problems: Analysis, Improvements, and New Applications*. IOS Press.

Sungur, Ilgaz, Fernando Ordóñez, and Maged Dessouky. 2008. “A robust optimization approach for the capacitated vehicle routing problem with demand uncertainty.” *IIE Transactions* 40 (5): 509–523.

Talbi, El-Ghazali. 2016. “Combining metaheuristics with mathematical programming, constraint programming and machine learning.” *Annals of Operations Research* 240 (1): 171–215.

Taş, D, Michel Gendreau, Nico Dellaert, Tom Van Woensel, and AG De Kok. 2014. “Vehicle routing with soft time windows and stochastic travel times: A column generation and branch-and-price solution approach.” *European Journal of Operational Research* 236 (3): 789–799.

Taş, Duygu, Nico Dellaert, Tom Van Woensel, and Ton De Kok. 2013. “Vehicle routing problem with stochastic travel times including soft time windows and service costs.” *Computers & Operations Research* 40 (1): 214–224.

Tyasnurita, Raras, Ender Ozcan, and Robert John. 2017. “Learning heuristic selection using a time delay neural network for open vehicle routing.” In *2017 IEEE Congress on Evolutionary*

<!-- Page 31 -->
Computation (CEC 2017), Jun.

Ulmer, Marlin W, Justin C Goodson, Dirk C Mattfeld, and Barrett W Thomas. 2020. “On Modeling Stochastic Dynamic Vehicle Routing Problems.” *EURO Journal on Transportation and Logistics* 9 (2).

Veličković, Petar, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. “Graph Attention Networks.” http://arxiv.org/abs/1710.10903.

Vidal, Thibaut, Gilbert Laporte, and Piotr Matl. 2020. “A concise guide to existing and emerging vehicle routing problem variants.” *European Journal of Operational Research* 286 (2): 401–416.

Vinyals, Oriol, Meire Fortunato, and Navdeep Jaitly. 2015. “Pointer Networks.” In *Advances in Neural Information Processing Systems 28*, edited by C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, 2692–2700. Curran Associates, Inc. http://papers.nips.cc/paper/5866-pointer-networks.pdf.

Vis, Iris FA. 2006. “Survey of research in the design and control of automated guided vehicle systems.” *European Journal of Operational Research* 170 (3): 677–709.

Voudouris, Christos, and Edward PK Tsang. 2003. “Guided local search.” In *Handbook of metaheuristics*, 185–218. Springer.

Wang, Chunbao, Lin Wang, Jian Qin, Zhengzhi Wu, Lihong Duan, Zhongqiu Li, Mequn Cao, et al. 2015. “Path planning of automated guided vehicles based on improved A-Star algorithm.” In *2015 IEEE International Conference on Information and Automation*, 2071–2076. IEEE.

Wang, Shaolin, Yi Mei, John Park, and Mengjie Zhang. 2019. “Evolving Ensembles of Routing Policies using Genetic Programming for Uncertain Capacitated Arc Routing Problem.” In *2019 IEEE Symposium Series on Computational Intelligence (SSCI)*, Dec., 1628–1635.

Wen, Min, Jean-François Cordeau, Gilbert Laporte, and Jesper Larsen. 2010. “The dynamic multi-period vehicle routing problem.” *Computers & Operations Research* 37 (9): 1615–1623.

Wu, Hailin, Fengming Tao, Qingqing Qiao, and Mengjun Zhang. 2020a. “A chance-constrained vehicle routing problem for wet waste collection and transportation considering carbon emissions.” *International Journal of Environmental Research and Public Health* 17 (2): 458.

Wu, Yaoxin, Wen Song, Zhiguang Cao, Jie Zhang, and Andrew Lim. 2020b. “Learning Improvement Heuristics for Solving Routing Problems.” *arXiv:1912.05784 [cs]*.

Xu, Haitao, Pan Pu, and Feng Duan. 2018. “Dynamic Vehicle Routing Problems with Enhanced Ant Colony Optimization.” *Discrete Dynamics in Nature and Society* 2018: 1–13.

Yang, Feidiao, Tiancheng Jin, Tie-Yan Liu, Xiaoming Sun, and Jialin Zhang. 2018. “Boosting Dynamic Programming with Neural Networks for Solving NP-hard Problems.” In *ICML*, Vol. 95, 726–739.

Yao, Yu, Xiaoning Zhu, Hongyu Dong, Shengnan Wu, Hailong Wu, Lu Carol Tong, and Xuesong Zhou. 2019. “ADMM-based problem decomposition scheme for vehicle routing problem with time windows.” *Transportation Research Part B: Methodological* 129: 156–174.

Yao, Yuan, Zhe Peng, and Bin Xiao. 2018. “Parallel Hyper-Heuristic Algorithm for Multi-Objective Route Planning in a Smart City.” *IEEE Transactions on Vehicular Technology* 67 (11): 10307–10318.

Yücenur, G. Nilay, and Nihan Çetin Demirel. 2011. “A new geometric shape-based genetic clustering algorithm for the multi-depot vehicle routing problem.” *Expert Systems with Applications* 38 (9): 11859–11865.

Zantalìs, Fotios, Grigorios Koulouras, Sotiris Karabetsos, and Dionisis Kandris. 2019. “A Review of Machine Learning and IoT in Smart Transportation.” *Future Internet* 11 (4): 94.

Zhang, C, NP Dellaert, L Zhao, T Van Woensel, Derya Sever, et al. 2013. “Single vehicle routing with stochastic demands: approximate dynamic programming.” *Dept. Ind. Eng., Tsinghua Univ., Beijing, China, Tech. Rep* 425.

Zhang, Junlong, William HK Lam, and Bi Yu Chen. 2013. “A stochastic vehicle routing problem with travel time uncertainty: trade-off between cost and customer service.” *Networks and Spatial Economics* 13 (4): 471–496.

<!-- Page 32 -->
Zhang, Junping, Fei-Yue Wang, Kunfeng Wang, Wei-Hua Lin, Xin Xu, and Cheng Chen. 2011. “Data-Driven Intelligent Transportation Systems: A Survey.” *IEEE Transactions on Intelligent Transportation Systems* 12 (4): 1624–1639.

Zhang, Yan, Ling-ling Li, Hsiung-Cheng Lin, Zewen Ma, and Jiang Zhao. 2019. “Development of path planning approach using improved A-star algorithm in AGV system.” *Journal of Internet Technology* 20 (3): 915–924.

Zulj, Ivan, Sergej Kramer, and Michael Schneider. 2018. “A hybrid of adaptive large neighborhood search and tabu search for the order-batching problem.” *European Journal of Operational Research* 264 (2): 653–664.

Žunić, Emir, Dženana Đonko, and Emir Buza. 2020. “An adaptive data-driven approach to solve real-world vehicle routing problems in logistics.” *Complexity* 2020.