<!-- Page 1 -->
This CVPR 2021 paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore.

# Deep Graph Matching under Quadratic Constraint

Quankai Gao$^{1}$, Fudong Wang$^{1}$, Nan Xue$^{1}$, Jin-Gang Yu$^{2}$, Gui-Song Xia$^{1}$  
$^{1}$Wuhan University, Wuhan, China.  
$^{2}$South China University of Technology, Guangzhou, China.  
{quankaigao, fudong-wang, xuenan, guisong.xia}@whu.edu.cn, jingangyu@scut.edu.cn

## Abstract

Recently, deep learning based methods have demonstrated promising results on the graph matching problem, by relying on the descriptive capability of deep features extracted on graph nodes. However, one main limitation with existing deep graph matching (DGM) methods lies in their ignorance of explicit constraint of graph structures, which may lead the model to be trapped into local minimum in training. In this paper, we propose to explicitly formulate pairwise graph structures as a quadratic constraint incorporated into the DGM framework. The quadratic constraint minimizes the pairwise structural discrepancy between graphs, which can reduce the ambiguities brought by only using the extracted CNN features. Moreover, we present a differentiable implementation to the quadratic constrained-optimization such that it is compatible with the unconstrained deep learning optimizer. To give more precise and proper supervision, a well-designed false matching loss against class imbalance is proposed, which can better penalize the false negatives and false positives with less overfitting. Exhaustive experiments demonstrate that our method achieves competitive performance on real-world datasets. The code is available at: [https://github.com/Zerg-Overmind/QC-DGM](https://github.com/Zerg-Overmind/QC-DGM).

## 1. Introduction

Graph matching aims to find an optimal one-to-one node correspondence between graph-structured data, which has been widely used in many tasks [3, 6, 9, 14, 20, 37]. By integrating the similarity between nodes and edges in a combinatorial fashion, graph matching is often mathematically formulated as a quadratic assignment problem (QAP) [29].

The study of this paper is funded by the National Natural Science Foundation of China (NSFC) under grant contracts No.61922065, No.61771350 and No.41820104006 and 61871299. It is also supported by Supercomputing Center of Wuhan University.  
Corresponding author: Gui-Song Xia (guisong.xia@whu.edu.cn).

<!-- Image: Figure 1 -->
Figure 1. Deep graph matching with/without quadratic constraint. Matching results are in left and the predicted (black) v.s. the ground truth (red) correspondence matrices are in right.

QAP is known to be NP-hard [16], and various approximation techniques [25, 26, 28, 34] have been proposed to make it computationally tractable.

Until recently, deep graph matching (DGM) methods give birth to many more flexible formulations [13, 32, 39, 45] besides traditional QAP. DGM aims to learn the meaningful node affinity by using deep features extracted from convolutional neural network. To this end, many existing DGM methods [32, 39, 45] primarily focus on the feature modeling and refinement for more accurate affinity construction. The feature refinement step is expected to capture the implicit structure information [39] encoded in learnable parameters. However, nodes with similar deep features are hard to distinguish from each other in deep graph matching, while their structure contexts may be very different. Moreover, the implicit structure information is not specific enough, which is insufficient to clearly represent the structural discrepancy over graphs (e.g., Fig. 1(a)).

In traditional graph matching, it is common to incorporate pairwise structures into the formulation to enhance matching accuracy [25], which inspired us to consider quadratic structural constraint in deep graph matching to maximize the adjacency consensus and achieve global consistency. More precisely, we use the pairwise term of

<!-- Page 2 -->
Figure 2. Overview of our proposed architecture for deep graph matching under quadratic constraint. Node attributes consisting geometric prior and deep features are refined to build the initial node affinity matrix, which is followed by a Sinkhorn layer and then further optimized under quadratic constraint (QC). Loss between the prediction and the ground truth (GT) is calculated by the proposed false matching loss (FM-Loss).

Koopmans-Beckmann’s QAP [29] as our quadratic constraint to minimize the adjacency discrepancy of graphs to be matched (e.g., Fig. 1(b)). To this end, we present a modified Frank-Wolfe algorithm [23], which is a differentiable optimization scheme w.r.t. learnable parameters in our model and the relaxed Koopmans-Beckmann’s QAP.

Another important issue of deep graph matching is class imbalance. Concretely, the result of a graph matching task is usually represented as a permutation matrix, where only a small portion of the entries take the value of one representing the pairs to be matched while the rest are zero-valued, leading to the imbalance between matched and unmatched entries. In case of such class imbalance, it will be problematic to establish the loss function between predicted matching matrices and ground truth matrices by using the conventional cross-entropy-type loss functions (see Section 3.4 for details). To our best knowledge, there is no loss function specifically designed for deep graph matching to take care of the class imbalance issue so far. To this end, we design a novel loss function for deep graph matching, called False Matching Loss, which will be experimentally shown to be better for dealing with class imbalance and overfitting in compared with previous works.

Our main contributions are highlighted as follows:

- We explicitly introduce quadratic constraint with our constructed geometric structure into deep graph matching, which can further revise wrong matches by minimizing structure discrepancy over graphs.
- We present a differentiable optimization scheme in training to approach the objective such that it is compatible with the unconstrained deep learning optimizer.
- We propose a novel loss function focusing on false matchings, i.e. false negatives and false positives, to better lead the parameter update against class imbalance and overfitting.

## 2. Preliminaries and Related Work

For better understanding, this section will revisit some preliminaries and related works on both traditional combinatorial graph matching and deep graph matching.

### 2.1. Combinatorial graph matching

Graph matching aims to build the node-to-node correspondence between the given two graphs $\mathcal{G}_A = \{V_A, E_A\}$ with $|V_A| = n$ and $\mathcal{G}_B = \{V_B, E_B\}$ with $|V_B| = m$, where $V$ we denote as the set of nodes and $E$ as the set of edges. By denoting $\mathbf{X}$ as the correspondence matrix indicating the matching between two graphs $\mathcal{G}_A$ and $\mathcal{G}_B$, i.e., $X_{ij} = 1$ means the $i$-th node in $V_A$ matches to the $j$-th node in $V_B$, $X_{ij} = 0$ otherwise, one well-known form of graph matching with combinatorial nature can be written as:

$$
\min_{\mathbf{X}} \quad \| \mathbf{A} - \mathbf{X} \mathbf{B} \mathbf{X}^T \|_F^2 - \text{tr}(\mathbf{X}_u^T \mathbf{X}) \tag{1}
$$
$$
\mathbf{X} \in \{0, 1\}^{n \times m}, \mathbf{X} \mathbf{1}_n = \mathbf{1}_m, \mathbf{X}^T \mathbf{1}_m \leq \mathbf{1}_n
$$

where $\mathbf{A} \in \mathbb{R}^{n \times n}, \mathbf{B} \in \mathbb{R}^{m \times m}$ are adjacency matrices encoding the pairwise information of edges in graphs $\mathcal{G}_A$ and $\mathcal{G}_B$, respectively. $\mathbf{X}_u \in \mathbb{R}^{n \times m}$ measures the node similarities between two graphs. $\|\cdot\|_F$ is the Frobenius norm. Generally, Eq. (1) can be cast as a quadratic assignment problem called Koopmans-Beckmann’s QAP [29]. One can find more details in [43].

In the previous works [15, 38, 47] following this combinatorial graph matching formulation, the node similarity $\mathbf{X}_u$ and adjacency matrices $\mathbf{A}, \mathbf{B}$ are usually calculated with some specifically designed handcrafted features like SIFT [30], Shape Context [2], etc. And then they will solve the objective functions (e.g., Eq. (1)) with different discrete or continuous constrained-optimization algorithms [8, 38, 10, 44]. Until recently, the deep learning based graph matching framework has been developed consisting of learned features and unconstrained optimizer, which will be detailed in next section.

<!-- Page 3 -->
## 2.2. Deep graph matching

**Feature extraction.** Recently, many works [13, 39, 41, 45, 32] on deep graph matching have been proposed to take advantages of the descriptive capability of high-dimensional deep features as node or edge attributes, which can collect visual information of background images. As a basic setting of these works, the output of CNN layers relu4_2 and relu5_1 are commonly used, denoted as $\mathbf{U} \in \mathbb{R}^{n \times d}$ and $\mathbf{F} \in \mathbb{R}^{n \times d}$,

$$
\mathbf{U} = \text{align}(\text{CNN}_{\text{relu4.2}}(I), V)
\tag{2}
$$

$$
\mathbf{F} = \text{align}(\text{CNN}_{\text{relu5.1}}(I), V)
\tag{3}
$$

where $I$ denotes the input image and $V$ denotes the annotated keypoints. CNN here is a widely used architecture VGG16 [35] initially pretrained on the ImageNet [33]. "align" in Eq. (2) and Eq. (3) is bi-linear interpolation to approximately assign the output features of a convolution layer to the annotated keypoints on the input image pairs.

**Feature modeling and refinement.** Since the extracted raw features are independent with the graph structure, various refinement strategies on the raw features are adopted in deep graph matching trying to implicitly utilize the information of graph structure. As a typical example of non-Euclidean data, a graph with its node and edge attributes can be processed by graph convolutional network (GCN) [21] under the message passing scheme to update its attributes. Each node attribute is updated by aggregating its adjacency node attributes so that GCN is expected to implicitly capture the structure contexts of each node to some extent. There are also some works [32, 45] using unary node features to model pairwise edge features for matching.

**Differentiable optimization.** Deep graph matching asks for the model fully-differentiable so that many combinatorial solvers and methods (e.g., Hungarian algorithm [22] and IPFP [26]) are not recommended. Thus, various relaxation approaches become popular, which refer to the recent progress [32, 39, 42].

Under these settings, deep graph matching can be reformulated as maximizing the node affinity based on the extracted features.

**Loss function for deep graph matching.** Though many works about deep graph matching have been proposed, there are few thorough discussions about loss functions. The prototype of cross entropy loss has been widely used in deep graph matching, e.g. permutation loss [39] and its improved version [45]. Instead of directly calculating the linear assignment cost, GMN [46] uses “displacement loss” measuring pixel-wise offset on the image but is shown to have a weaker supervision than cross entropy loss [39]. However, none of the existing works consider the class imbalance that naturally exists in deep graph matching. Besides, overfitting and gradient explosion are always conspicuous on models trained with cross-entropy-type loss functions. For the above reasons, we propose a novel loss function specifically designed for deep graph matching, which not only considers numerical issue but also shows promising performance against overfitting and class imbalance.

## 3. DGM under Quadratic Constraint

We briefly demonstrate our method overview here. As shown in Fig. 2, given the input two images with detected or annotated keypoints as graph nodes, we firstly adopt the CNN features and the coordinates of keypoints to calculate both the initial node attributes and the pairwise structural context as weighted adjacency matrices. By this end, we establish graphs with structural attributes, based on which we explicitly use the weighted adjacency matrices to construct the quadratic constraint, and design a differentiable constrained-optimization algorithm to achieve compatibility with the deep learning optimizer. Since the quadratic constrained-optimization is non-convex and needs a proper initialization, we update the node attributes with weighted adjacency matrices through a GCN module and a differentiable Sinkhorn layer [1, 36] respectively, to obtain a node affinity matrix between two graphs as the initialization. Finally, the solved optimum will be compared with the ground truth by the proposed false matching loss, which addresses the issue of class imbalance to achieve better performance.

### 3.1. Geometric structure for DGM

**Unary geometric prior** Since our work doesn’t focus on deep feature extraction, we follow the previous deep graph matching works to use CNN features described in Eq. (2) and Eq. (3). Moreover, features from different layers of CNN are expected to incorporate both the local and global semantic information of images, we concatenate $\mathbf{U}$ and $\mathbf{F}$ together as $\mathbf{P}_r = \text{cat}(\mathbf{U}; \mathbf{F})$ to be the initial node attributes of two graphs.

Since the extracted raw features associated to nodes only have visual information of local patches, to make raw node attributes more discriminative, we add the normalized 2D Cartesian coordinate $[\hat{\mathbf{x}}, \hat{\mathbf{y}}]$ of each node as $\mathbf{P} = \text{cat}(\mathbf{P}_r; [\hat{\mathbf{x}}, \hat{\mathbf{y}}])$, which provides a unary geometric prior that can better describe the locations of nodes as a complement to the CNN features.

**Pairwise structural context** In deep graph matching, the graph construction is usually based on node coordinates and never consider the visual meaningful features of the background image. For this reason, we introduce deep feature weighted adjacency matrices $\mathbf{A}_D$ and $\mathbf{B}_D$ of the two graphs to learn more proper relations among graph nodes, which are defined as

$$
\mathbf{A}_D = f(\mathbf{P}_\mathbf{A}) \odot \mathbf{A}, \quad \mathbf{B}_D = f(\mathbf{P}_\mathbf{B}) \odot \mathbf{B}
\tag{4}
$$

<!-- Page 4 -->
Figure 3. Illustration of the geometric relationship between two nodes and their edge connection in attribute space. Normalized attributes of nodes are represented as coloured squares.

where $\mathbf{A} \in \mathbb{R}^{n \times n}, \mathbf{B} \in \mathbb{R}^{m \times m}$ are the binary adjacency matrices built on coordinates of nodes in two graphs, $\mathbf{P}_A \in \mathbb{R}^{n \times (d+2)}, \mathbf{P}_B \in \mathbb{R}^{m \times (d+2)}$ are the above-mentioned node attributes of two graphs, $\odot$ denotes element-wise product. $f(\mathbf{P})$ can be various commutative function and here we use a linear kernel $f_{i,j} = \mathbf{p}_i^T \mathbf{p}_j$ for simplicity, where $\mathbf{p}_i$ is the $i$-th row of $\mathbf{P}$. As illustrated in Fig. 3, the geometric meaning of the function $f$ is related to the cosine of the angle between two normalized node attributes, which can be detailed as $\cos\theta_{ij} = \frac{\langle \mathbf{p}_i, \mathbf{p}_j \rangle}{|\mathbf{p}_i||\mathbf{p}_j|} = \frac{1}{\sqrt{3}}\frac{1}{\sqrt{3}}\langle \mathbf{p}_i, \mathbf{p}_j \rangle = \frac{1}{3}\mathbf{p}_i^T \mathbf{p}_j$. By this definition, each element of $\mathbf{A}_D$ and $\mathbf{B}_D$ represents the feature distance between the corresponding nodes while preserving the topology constraints provided by $\mathbf{A}$ and $\mathbf{B}$.

**Attributes fusion with GCN** There are many convolutional architectures [12, 48] for processing irregular data. As a typical example of non-Euclidean data, a graph with its node and edge attributes can be processed by GCN under the message passing scheme to update its attributes. Each node attribute is updated by aggregating its adjacency node attributes so that GCN is expected to capture the structure contexts of each node in an implicit way. With the above interpretation, we adopt GCN that incorporates both update from neighbors and self-update, which can be written as:

$$
\mathbf{P}^{l+1} = \sigma(\mathbf{A}_D \mathbf{P}^l \mathbf{W}_r^l + \mathbf{P}^l \mathbf{W}_s^l) \tag{5}
$$

where $\mathbf{W}_r^l, \mathbf{W}_s^l \in \mathbb{R}^{(d+2) \times (d+2)}$ denote learnable parameters of GCN at $l$-th layer. $\sigma$ is the activation function. The updated attributes $\mathbf{P}^{l+1}$ is then used to update $\mathbf{A}_D$ and $\mathbf{B}_D$ by Eq. (4).

**Node affinity** Since we have the refined attributes of two graphs, the node affinity $\mathbf{K}_p^l \in \mathbb{R}^{n_A \times n_B}$ at $l$-th iteration can be built by a learnable metric:

$$
\mathbf{K}_p^l = \exp\{\mathbf{P}_A^l \mathbf{W}_{\text{aff}}^l {\mathbf{P}_B^l}^T\} \tag{6}
$$

where $\mathbf{W}_{\text{aff}}^l$ is a matrix containing learnable parameters. We then adopt Sinkhorn layer taking $\mathbf{K}_p^l$ to the set of doubly-stochastic matrices $\mathcal{D}$, which is the convex hull of the set of permutation matrices $\mathcal{P}$.

## 3.2. Quadratic constraint for DGM

To explicitly utilize the information of graph structures, we formulate the objective as to minimize the pairwise structure discrepancy between the graphs to be matched. As Eq. (1), Koopmans-Beckmann’s QAP explicitly involves the second-order geometric context in its pairwise term and the optimal solution will minimize the two terms simultaneously. The learnable metric $\mathbf{K}_p^l$ we obtained from Eq. (6) is considered as the initial point of $\mathbf{X}$, i.e., $\mathbf{X}_0 = \mathbf{K}_p^l$. We rewrite Eq. (1) as:

$$
\min_{\mathbf{X}} g(\mathbf{X}) = \min_{\mathbf{X}} ||\mathbf{A}_D - \mathbf{X} \mathbf{B}_D \mathbf{X}^T||_F^2 - \text{tr}(\mathbf{X}_u^T \mathbf{X}) \tag{7}
$$

We specify unary affinity matrix $\mathbf{X}_u$ as the obtained node

---

**Algorithm 1: DGM under Quadratic Constraint**

**Input:** Nodes of graph pairs $V_s$; two input images $I_s$, where $s=A, B$; the ground truth $\mathbf{X}^*$; initial parameters $\mathbf{W} = \{\mathbf{W}_r^l, \mathbf{W}_s^l, \mathbf{W}_{aff}^l\}$

**Output:** prediction $\mathbf{X}_P \in \mathcal{P}$

//feature extraction and alignment  
$\mathbf{U}_s \leftarrow \text{align}(\text{CNN}_{\text{relu4\_2}}(I_s), V_s)$;  
$\mathbf{F}_s \leftarrow \text{align}(\text{CNN}_{\text{relu5\_1}}(I_s), V_s)$;  
//node attributes  
$\mathbf{P}_s \leftarrow \text{cat}(\mathbf{U}_s; \mathbf{F}_s; [\hat{\mathbf{x}}_s, \hat{\mathbf{y}}_s])$;

**Training stage :**  
for epoch $k \leq n$ do  
 $\mathbf{P}_s^l \leftarrow \text{GCN}((\mathbf{A}_D)_s, \mathbf{P}_s^{l-1})$;  
 $(\mathbf{A}_D^l)_s \leftarrow f(\mathbf{P}_s^l) \odot \mathbf{A}_s$;  
 $\mathbf{K}_p^l \leftarrow \exp\{\mathbf{P}_A^l \mathbf{W}_{\text{aff}}^l {\mathbf{P}_B^l}^T\}$;  
 $\mathbf{X} \leftarrow \mathbf{K}_p^l$;  
 for iter = 1:$m_1$ do  
  for k = 1:$m_2$ do  
   $\mathbf{y} \leftarrow \arg\min_{\mathbf{y}} \nabla g(\mathbf{X})^T \mathbf{y}$;  
   // $f_\mathcal{D}$ is Sinkhorn normalization  
   $\mathbf{s} \leftarrow f_\mathcal{D}(\mathbf{y})$;  
   $\overline{\mathbf{X}} \leftarrow \mathbf{X} - \epsilon_k (\mathbf{X} - \mathbf{s})$;  
  end  
  $\mathbf{X} \leftarrow f_\mathcal{D}(\overline{\mathbf{X}})$;  
 end  
 $\mathbf{W} \leftarrow L_{fm}(\mathbf{X}, \mathbf{X}^{\text{GT}}, \mathbf{W})$;  
end  

**Inference stage :**  
$\mathbf{X}_P \leftarrow \mathbf{X}$;  
for iter = 1:m do  
 repeat  
  $\mathbf{y} \leftarrow \arg\min_{\mathbf{y}} \nabla g(\mathbf{X}_P)^T \mathbf{y}$;  
  // $f_\mathcal{P}$ is Hungarian algorithm  
  $\mathbf{s} \leftarrow f_\mathcal{P}(\mathbf{y})$;  
  $\overline{\mathbf{X}}_P \leftarrow \mathbf{X}_P - \epsilon_k (\mathbf{X}_P - \mathbf{s})$;  
 until $\mathbf{X}_P$ converges;  
 $\mathbf{X}_P \leftarrow f_\mathcal{P}(\overline{\mathbf{X}}_P)$;  
end

<!-- Page 5 -->
affinity matrix $\mathbf{K}_p^l$ in both the training stage and the inference stage.

Due to the paradox between the combinatorial nature of QAP formulation and the differentiable requirement of deep learning framework, we consider a relaxed version of Eq. (7): $\mathbf{X} \in [0,1]^{n_A \times n_B}, \mathbf{X}\mathbf{1}_{n_B} = \mathbf{1}, \mathbf{X}^T\mathbf{1}_{n_A} \leq \mathbf{1}_{n_A}$. The value of $\mathbf{X}$ is continuous and satisfies normalization constraints at the same time. By minimizing the objective function, the solution will close to the ground truth in the direction of minimizing the adjacency inconsistency.

## 3.3. Quadratic constrained-optimization

We adopt a differentiable Frank-Wolfe algorithm [23] for $g(\mathbf{X})$ to obtain an approximate solution:

$$
\mathbf{y}_k = \arg\min_{\mathbf{y}} \nabla g(\mathbf{X}_k)^T \mathbf{y} \tag{8}
$$

$$
\mathbf{s} = f_{proj}(\mathbf{y}_k) \tag{9}
$$

$$
\bar{\mathbf{X}}_{k+1} = \mathbf{X}_k - \epsilon_k (\mathbf{X}_k - \mathbf{s}) \tag{10}
$$

$$
\mathbf{X}_{k+1} = f_{proj}(\bar{\mathbf{X}}_{k+1}) \tag{11}
$$

where $\epsilon_k$ is a parameter representing the step size. Usually, we set $\epsilon_k = \frac{2}{k+2}$ [18] in implementation. In training stage, $f_{proj}$ in Eq. (9) and Eq. (11) is the Sinkhorn layer to project the positive variable into the set of doubly stochastic matrices $\mathcal{D}$, while $f_{proj}$ is the Hungarian algorithm in the inference stage for obtaining a discrete solution. We write down the gradient of $g(\mathbf{X})$ as:

$$
\nabla g(\mathbf{X}) = -2[\mathbf{U}^T\mathbf{X}\mathbf{B}_D^l + \mathbf{U}\mathbf{X}\mathbf{B}_D^{l\ T}] - \mathbf{X}_u \tag{12}
$$

where $\mathbf{U} = \mathbf{A}_D^l - \mathbf{X}\mathbf{B}_D^l\mathbf{X}^T$. In training stage, the variable $\mathbf{X}$ is associated with the learnable parameters $\mathbf{W} = \{\mathbf{W}_r^l, \mathbf{W}_s^l, \mathbf{W}_{aff}^l\}$ and every iteration of Frank-Wolfe algorithm is actually going with the parameters fixed before backpropagation [17]. From Eq. (10) and Eq. (11), $\mathbf{X}_{k+1}$ is differentiable with respect to $\mathbf{X}_k$ so $\mathbf{X}_{k+1}$ is differentiable with respect to $\mathbf{X}_0 = \mathbf{K}_p^l$ by the recursive relations:

$$
\mathbf{X}_{k+1} = f_r(\mathbf{X}_k) = f_r \circ f_r(\mathbf{X}_{k-1}) = \dots = f_r^k(\mathbf{X}_0) \tag{13}
$$

where $f_r(\mathbf{X}) \triangleq f_{proj}(\mathbf{X} - \epsilon_k(\mathbf{X} - \mathbf{s}))$ is a differentiable function w.r.t. $\mathbf{X}$ in training stage. Since the goal of the optimization is to utilize the information of the pairwise term in Eq. (7), only few iterations roughly approaching to a local minimum in training stage can fulfill our purpose. Though there is no guarantee of global minimum, it is actually not necessary because the pairwise term will be noisy with outliers and the global minimum may not be the desired matching. Besides, few iterations in training stage encourage the model to learn a relatively short path to approximate the object so that it is easier to convergence within fewer iterations in inference stage.

<!-- Image: Figure 4 -->
Figure 4. An illustration of the work flow about our proposed false matching loss. We take both the false positive and false negative of the predicted doubly stochastic matrix into consideration. $\alpha$ and $\beta$ are two parameters to weight the two terms.

## 3.4. False-matching loss

**Class imbalance.** The elements of correspondence matrix can be divided into two classes with clear different meaning, one of which represents matched pairs and the other represents unmatched pairs. Given two graphs with equal number of nodes $n$, there are $n$ elements of correspondence matrix are 1 while the rest $n^2 - n$ are 0. Under the constraint of one-to-one matching, the unmatched pairs clearly take the majority. Though the values of the soft correspondence matrix are all between 0 and 1, the two classes should be separately treated in the calculation of loss during training.

The cross-entropy-type loss functions such as permutation loss [39] and its improved version [45] achieve the state-of-the-art performance. This type of loss functions directly measures the linear assignment cost between the ground truth $\mathbf{X}^* \in \{0,1\}^{n \times n}$ and the prediction $\mathbf{X} \in [0,1]^{n \times n}$ like

$$
L_{ce} = -\sum_{i,j} \mathbf{X}_{ij}^* \log \mathbf{X}_{ij} + (1 - \mathbf{X}_{ij}^*) \log(1 - \mathbf{X}_{ij}) \tag{14}
$$

with $i \in \mathcal{G}_A$, $j \in \mathcal{G}_B$. Cross entropy loss does not consider the class imbalance in deep graph matching because the vast majority of $1 - \mathbf{X}^*$ are preserved by multiplying $1 - \mathbf{X}^*$ while at most $n$ elements of $\mathbf{X}$ are kept by multiplying $\mathbf{X}^*$. Similarly, cross entropy loss with Hungarian attention [45] can not solve this issue either and the so called Hungarian attention in loss function brings discontinuity in learning.

**Numerical stability.** Since the optimization of deep learning is unconstraint, the local minimum with bad properties can not be avoid. Once a bad case or a bad local minimum occurs, e.g. at least one element of the prediction $\mathbf{X}$ approaches to 0 while the corresponding element of the ground truth $\mathbf{X}^*$ is 1, the cross-entropy-type loss will be invalid. Though permutation loss with Hungarian attention mechanism is more focus on a specific portion of the prediction $\mathbf{X}$, it only works without severe bad cases either.

<!-- Page 6 -->
(a) cross entropy loss $L_{ce}$  
(b) false matching loss $L_{fm}$

Figure 5. Comparison between cross entropy loss and false matching loss on a toy example, where $\mathbf{X} \in \mathbb{R}^{1\times3}$ and $\mathbf{X}^* = (0,0,1)$. We mark the three extreme points of $\mathbf{X}$: $\mathbf{A} = (0,0,1)$, $\mathbf{B} = (0,1,0)$ and $\mathbf{C} = (1,0,0)$. A cooler color means a smaller value. The comparison shows that false matching loss has a higher value than cross entropy loss when it close to the ground truth. Besides, false matching loss has a high but limited value when $\mathbf{X}$ close to the extreme points, which are the bad cases for cross entropy loss and make the loss function output infinite values. The curve of cross entropy loss on points between $\mathbf{B}$ and $\mathbf{C}$ are below the red line connected to $\mathbf{B}$ and $\mathbf{C}$, while the curve of false matching loss coincides with the red line.

Even with gradient clipping, overfitting is still a problem to be addressed.

Facing the problems above, we propose a new loss function called false matching loss:

$$
L_{fm} = \underbrace{e^{\alpha \sum_{i,j} [\mathbf{X} \odot (1 - \mathbf{X}^*)]_{ij}}}_{L_+} + \underbrace{e^{\beta \sum_{i,j} [\mathbf{X}^* \odot (1 - \mathbf{X})]_{ij}}}_{L_-}
\quad (15)
$$

where $i \in \mathcal{G}_A$, $j \in \mathcal{G}_B$. $L_+$ and $L_-$ of our false matching loss penalize false positive matches and false negative matches, which are the main indexes describing the false matchings in statistics. To address the issue of class imbalance, we draw on the experience of focal loss [27] and take two hyperparameters $\alpha$ and $\beta$ for weighting.

As illustrated in Fig. 5, false matching loss is more smooth and always has a relatively large value than cross entropy loss even when the prediction $\mathbf{X}$ is close to the ground truth. While on extreme points that are away from the ground truth, false matching loss will have more proper outputs (high but limited). Besides, the values of cross entropy loss on points between $\mathbf{B}$ and $\mathbf{C}$ are lower than that on $\mathbf{B}$ and $\mathbf{C}$, which is not proper because points between $\mathbf{B}$ and $\mathbf{C}$ are still far away from the ground truth.

Our method is summarized in Algorithm 1.

## 4. Experiments and Analysis

In this section, we evaluate our method on two challenging datasets. We compare the proposed method with several state-of-the-art deep graph matching methods: GLM-Net [19], PCA [39], NGM [40], CIE$_1$ [45], GMN [46], LCSGM [42] and BB-GM [32]. To compare the proposed false matching loss with the previous work: permutation loss with Hungarian attention [45], we follow [45] to implement the loss function for the source code is not publicly available. Given graphs with $n$ nodes, the matching accuracy is computed as

$$
\text{accuracy} = \frac{1}{n} \sum_{i,j} [\mathbf{X}^* \odot \text{Hungarian}(\mathbf{X})]_{ij}
\quad (16)
$$

Stochastic gradient descent (SGD) [4] is employed as the optimizer with an initial learning rate $10^{-3}$. In our false matching loss, we set $\alpha = 2$ and $\beta = 0.1$. In our optimization step, we set the number of iterations $m_1 = 3$ and $m_2 = 5$ for the consideration of both computational efficiency and convergence.

Since the feature refinement step is not limited to Eq. (5), we additionally deploy another novel architecture SplineCNN [12] to replace Eq. (5) as a comparison with [32]. Thus, two versions of our method are provided: qc-DGM$_1$ and qc-DGM$_2$. In qc-DGM$_1$, the node attributes are refined by a two-layer GCN as Eq. (5) with ReLU [31] activation function. While in qc-DGM$_2$, the refinement step is done by a two-layer SplineCNN [12] with max aggregation. In our implementation, all the input graphs are constructed by Delaunay triangulation. Specifically for GMN [46], two input graphs are constructed by fully-connected topology and Delaunay triangulation respectively. While GMN$_D$ is another version of GMN that two input graphs are both constructed by Delaunay triangulation.

### 4.1. Results on Pascal VOC Keypoints

Pascal VOC dataset [11] with Berkeley annotations of keypoints [5] has 20 classes of instance images with annotated keypoints. There are 7,020 annotated images for training and 1,682 for testing. Before training, each instance is cropped around its bounding box and is re-scaled to $256 \times 256$ as the input for VGG16. We follow the previous works to filter the truncated, occluded and difficult images. The training process will be performed on all 20 classes. Because of the large variation of pose, scale, and appearance, Pascal VOC Keypoints is considerably a challenging dataset.

Experimental results on 20 classes are given in Table 1. Since the peer work [32] suggests to compare F1 score (the harmonic mean of precision and recall) on the dataset without intersection filtering, we provide F1 score of our method to compare with BB-GM [32] and its ablation BB-GM-Max as shown in Table 2.

To show the robustness of the deep graph matching models against outliers, we add outliers to the original set of keypoints. The 2D Cartesian coordinates of the outliers are generated by Gaussian distribution $\mathcal{N}(0, 10)$. The outliers exist only in inference stage to challenge the model trained

<!-- Page 7 -->
| Method | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GMN [46] | 35.5 | 50.0 | 52.2 | 45.2 | 75.5 | 69.1 | 59.0 | 61.1 | 34.4 | 51.9 | 66.4 | 53.4 | 53.9 | 50.7 | 31.1 | 75.8 | 59.2 | 47.9 | 86.1 | 89.6 | 57.4 |
| GMN$_D$ [46] | 40.8 | 57.1 | 56.2 | 48.2 | 75.5 | 71.6 | 64.1 | 63.3 | 36.3 | 54.3 | 50.2 | 57.6 | 60.3 | 55.4 | 35.5 | 85.2 | 62.7 | 51.8 | 86.5 | 87.4 | 60.0 |
| qc-GMN | 37.3 | 52.2 | 54.3 | 47.2 | 76.4 | 70.4 | 61.2 | 61.7 | 34.5 | 53.1 | 69.1 | 55.4 | 56.0 | 52.4 | 31.6 | 77.3 | 59.7 | 49.1 | 87.4 | 89.7 | 58.8 |
| PCA [39] | 40.9 | 55.0 | 65.8 | 47.9 | 76.9 | 77.9 | 63.5 | 67.4 | 33.7 | 66.5 | 63.6 | 61.3 | 58.9 | 62.8 | 44.9 | 77.5 | 67.4 | 57.5 | 86.7 | 90.9 | 63.8 |
| qc-PCA | 42.5 | 58.5 | 66.1 | 51.3 | 79.6 | 78.2 | 65.8 | 68.7 | 35.1 | 66.8 | 65.6 | 62.5 | 62.1 | 63.1 | 45.1 | 80.7 | 67.7 | 59.1 | 87.0 | 91.1 | 64.8 |
| PCA-H [39] | 50.8 | 61.7 | 62.6 | 56.4 | 80.0 | 75.6 | 72.4 | 74.0 | 38.5 | 64.3 | 49.9 | 63.8 | 65.2 | 63.5 | 46.0 | 78.5 | 68.0 | 41.5 | 82.2 | 90.8 | 64.3 |
| PCA-F | 50.0 | 66.7 | 61.8 | 55.1 | 81.5 | 75.5 | 70.1 | 70.4 | 39.7 | 64.8 | 60.3 | 65.5 | 67.6 | 64.2 | 45.6 | 84.4 | 68.6 | 56.5 | 88.7 | 91.1 | 66.3 |
| NGM [40] | 50.8 | 64.5 | 59.5 | 57.6 | 79.4 | 76.9 | 74.4 | 69.9 | 41.5 | 62.3 | 68.5 | 62.2 | 62.4 | 64.7 | 47.8 | 78.7 | 66.0 | 63.3 | 81.4 | 89.6 | 66.1 |
| GLMNet [19] | 52.0 | 67.3 | 63.2 | 57.4 | 80.3 | 74.6 | 70.0 | 72.6 | 38.9 | 66.3 | 77.3 | 65.7 | 67.9 | 64.2 | 44.8 | 86.3 | 69.0 | 61.9 | 79.3 | 91.3 | 67.5 |
| LCSGM [42] | 46.9 | 58.0 | 63.6 | **69.9** | **87.8** | 79.8 | 71.8 | 60.3 | **44.8** | 64.3 | 79.4 | 57.5 | 64.4 | 57.6 | **52.4** | **96.1** | 62.9 | 65.8 | **94.4** | 92.0 | 68.5 |
| CIE$_1$-H [45] | 51.2 | **69.2** | **70.1** | 55.0 | 82.8 | 72.8 | 69.0 | **74.2** | 39.6 | 68.8 | 71.8 | 70.0 | **71.8** | 66.8 | 44.8 | 85.2 | **69.9** | 65.4 | 85.2 | **92.4** | 68.9 |
| qc-DGM$_1$(ours) | 48.4 | 61.6 | 65.3 | 61.3 | 82.4 | 79.6 | 74.3 | 72.0 | 41.8 | **68.8** | 65.0 | 66.1 | 70.9 | 69.6 | 48.2 | 92.1 | 69.0 | 66.7 | 90.4 | 91.8 | 69.3 |
| qc-DGM$_2$(ours) | 49.6 | 64.6 | 67.1 | 62.4 | 82.1 | **79.9** | **74.8** | 73.5 | 43.0 | 68.4 | **66.5** | 67.2 | 71.4 | **70.1** | 48.6 | 92.4 | 69.2 | **70.9** | 90.9 | 92.0 | **70.3** |

Table 1. Accuracy (%) of 20 classes and average accuracy on Pascal VOC. Bold numbers represent the best performing of the methods to be compared. Method with “-H” and “-F” denotes it with permutation loss with Hungarian attention and false matching loss, respectively. Suffix “-QC” denotes the method with the proposed quadratic constraint.

| Method | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BB-GM-Max [32] | 35.5 | 68.6 | 46.7 | 36.1 | 85.4 | 58.1 | 25.6 | 51.7 | 27.3 | 51.0 | 46.0 | 46.7 | 48.9 | 58.9 | 29.6 | 93.6 | 42.6 | 35.3 | 70.7 | 79.5 | 51.9 |
| BB-GM [32] | 42.7 | 70.9 | 57.5 | 46.6 | 85.8 | 64.1 | 51.0 | 63.8 | 42.4 | 63.7 | 47.9 | 61.5 | 63.4 | 69.0 | 46.1 | 94.2 | 57.4 | 39.0 | 78.0 | 82.7 | 61.4 |
| qc-DGM$_1$(ours) | 30.1 | 59.1 | 48.6 | 40.0 | 79.7 | 51.6 | 32.4 | 55.4 | 26.1 | 52.1 | 47.0 | 50.1 | 56.8 | 59.9 | 27.6 | 90.4 | 50.9 | 33.1 | 71.3 | 78.8 | 52.0 |
| qc-DGM$_2$(ours) | 30.9 | 59.8 | 48.8 | 40.5 | 79.6 | 51.7 | 32.5 | 55.8 | 27.5 | 52.1 | 48.0 | 50.7 | 57.3 | 60.3 | 28.1 | 90.8 | 51.0 | 35.5 | 71.5 | 79.9 | 52.6 |

Table 2. F1 score (%) of matching and mean over 20 classes on Pascal VOC without intersection filtering.

| Method | face | mbike | car | duck | wbottle | Mean |
|---|---|---|---|---|---|---|
| HARG-SSVM [7] | 91.2 | 44.4 | 58.4 | 55.2 | 66.6 | 63.2 |
| GMN [46] | 98.1 | 65.0 | 72.9 | 74.3 | 70.5 | 76.2 |
| GMN$_D$ | 100.0 | 82.4 | 84.4 | 84.0 | 91.2 | 88.4 |
| qc-GMN | 100.0 | 65.7 | 75.3 | 81.1 | 90.5 | 82.5 |
| PCA [39] | 100.0 | 76.7 | 84.0 | 93.5 | 96.9 | 90.2 |
| qc-PCA | 100.0 | 83.3 | 87.3 | 93.8 | 97.1 | 92.3 |
| PCA-H [39] | 100.0 | 76.9 | 88.9 | 89.7 | 92.9 | 89.7 |
| PCA-F [39] | 100.0 | 78.4 | 86.8 | 93.2 | 97.2 | 91.1 |
| NGM [19] | 99.2 | 82.1 | 84.1 | 77.4 | 93.5 | 87.2 |
| GLMNet [19] | 100.0 | 89.7 | 93.6 | 85.4 | 93.4 | 92.4 |
| LCSGM [42] | 100.0 | **99.4** | 91.2 | 86.2 | 97.9 | 94.9 |
| CIE$_1$-H [45] | 100.0 | 90.0 | 82.2 | 81.2 | 97.6 | 90.2 |
| BB-GM [32] | 100.0 | **99.2** | 96.9 | 89.0 | 98.8 | 96.8 |
| qc-DGM$_1$(ours) | **100.0** | 95.0 | 93.8 | **93.8** | 97.6 | 96.0 |
| qc-DGM$_2$(ours) | **100.0** | 98.8 | **98.0** | 92.8 | **99.0** | **97.7** |

Table 3. Accuracy (%) of 5 classes and the average on Willow Object Class.

on the clean data (without outliers). This robustness test is more challenging than that on synthetic graphs because all the inliers and outliers have their extracted CNN features and thus, more close to the real-world scenes. Experimental results are shown in Fig. 6. With our quadratic constraint, the improvement of overall accuracy is witnessed on the clean data and the robustness of deep graph matching models have been significantly improved against outliers.

By comparing PCA and qc-PCA or GMN and qc-GMN in Table 1, the effectiveness of our quadratic constraint is shown to be general and the matching accuracy is improved over all 20 classes by considering our quadratic constraint.

## 4.2. Results on Willow Object Class

Willow Object Class dataset [7] contains 5 classes with 256 images in total. Three classes (face, duck and wine bottle) of the dataset are from Caltech-256 and the rest two classes (car and motorbike) are from Pascal VOC 2007. We resize all the image pairs to $256 \times 256$ for VGG16 backbone and crop the images around the bounding box of the objects. The variations of the pose, scale, background and illumination of the images on Willow Object Class are small, and thus, graph matching tasks on this dataset are much more easier.

As shown in Table 3, the proposed method achieves the competitive performance. Comparing the methods with and without quadratic constraint (PCA and qc-PCA or GMN and qc-GMN), the performance improvements on Willow dataset are more prominent than those on Pascal VOC, which because the structure variations of graph pairs on the easier dataset are relatively small and quadratic constraint contributes to matching more soundly. Since BB-GM adopts SplineCNN to refine the features, we provide

Figure 6. Robustness analysis against outliers on Pascal VOC (a) and Willow Object Class (b). Method with “+” means *with* quadratic constraint while “-” means *without* quadratic constraint.

<!-- Page 8 -->
Figure 7. Matching examples on Pascal VOC and Willow Object Class. Nodes with the same color indicate the correspondence of a graph pair. All the visualized graphs are constructed by Delaunay triangulation in yellow.

qc-DGM$_2$ for fair comparison.

## 4.3. Further study

Table 4 shows the usefulness of the proposed components. CNN feature vector of each node has d=1024 dimensions, which are supposed to be better than the two-dimensional coordinates but the geometric prior encoded in coordinates provides a more precise description of graph nodes. By simply combining the unary geometric prior with the extracted CNN features, the matching accuracy is improved, which supports our point of view, i.e., CNN features are indeed useful but not discriminative enough to depict pixel-wise graph nodes.

**Quadratic constraint.** There are various forms of quadratic constraint that are not limited to ours. Global affinity matrix $\mathbf{K}$ is constrained by graph incidence matrices in the factorized form of Lawler’s QAP [24, 47], which can be considered as another form of quadratic constraint adopted in GMN. As shown in Table 1 and Table 3, GMN performs significantly worse than GMN$_D$ on both two datasets by replacing similar graph topology with the completely different one, while the extracted deep features of both settings remain the same. This implies the main limitation of quadratic constraint, i.e., only graphs with similar topology contribute to matching. Besides, the fact shows the independent relationship between the graph structure and raw deep features for the change of graph topology can not be reflected by raw deep features (before being refined).

**False-matching loss vs. cross entropy loss.** The two main loss functions to be compared with ours are permutation loss [39] and permutation loss with Hungarian attention [45]. We report accuracy/loss with training epoch in Fig. 8. In experiments, the model with cross-entropy-type loss functions (Loss-P and Loss-H) always encounter the bad cases leading to gradient explosion in training stage, while our false matching loss (Loss-F) can avoid this issue. Besides, false matching loss is shown to do better against overfitting than the compared two loss functions.

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{fig8_accuracy_loss_vs_epoch.png}
    \caption{Accuracy/loss vs. training epoch on Pascal VOC. As the training goes, the loss functions try to drag the output to binary and the local minimum with bad properties makes the cross-entropy-type loss explosion. Since the accuracy will be very close to 0 after gradient explosion, we truncate the descending curves and keep them unchanged for better visualization.}
\end{figure}
```

```latex
\begin{table}[h]
    \centering
    \begin{tabular}{ccccc}
        \toprule
        raw attributes & unary geometric prior & pairwise structure context & QC optimization & accuracy \\
        \midrule
        $\surd$ & $\surd$ & $\surd$ & $\surd$ & 69.3 \\
        $\surd$ & $\surd$ & $\surd$ & & 68.5 \\
        $\surd$ & $\surd$ & & & 68.3 \\
        $\surd$ & & & & 67.8 \\
        \bottomrule
    \end{tabular}
    \caption{Ablation study of qc-DGM$_1$ on Pascal VOC. The component been adopted is marked by a tick. “QC optimization” is quadratic constrained-optimization.}
\end{table}
```

## 5. Conclusion

In this work, we explicitly introduce quadratic constraint of graph structure into deep graph matching. To this end, unary geometric prior and pairwise structural context are considered for objective construction and a differentiable optimization scheme is provided to approach the problem. Moreover, we focus on class imbalance that naturally exists in deep graph matching to propose our false matching loss. Experimental results show the competitive performance of our method. In future work, we plan to seek a more general form of quadratic constraint to the learning-based optimization for better matching.

<!-- Page 9 -->
# References

[1] Ryan Prescott Adams and Richard S Zemel. Ranking via sinkhorn propagation. *arXiv preprint arXiv:1106.1925*, 2011.

[2] Serge Belongie, Jitendra Malik, and Jan Puzicha. Shape matching and object recognition using shape contexts. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(4):509–522, 2002.

[3] Alexander C Berg, Tamara L Berg, and Jitendra Malik. Shape matching and object recognition using low distortion correspondences. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, volume 1, pages 26–33. IEEE, 2005.

[4] Léon Bottou. Large-scale machine learning with stochastic gradient descent. In *Proceedings of COMPSTAT’2010*, pages 177–186. Springer, 2010.

[5] Lubomir Bourdev and Jitendra Malik. Poselets: Body part detectors trained using 3d human pose annotations. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 1365–1372. IEEE, 2009.

[6] William Brendel and Sinisa Todorovic. Learning spatiotemporal graphs of human activities. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 38(9):1774–1789, 2016.

[7] Minsu Cho, Karteek Alahari, and Jean Ponce. Learning graphs to match. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 25–32, 2013.

[8] Minsu Cho, Jian Sun, Olivier Duchenne, and Jean Ponce. Finding matches in a haystack: A max-pooling strategy for graph matching in the presence of outliers. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 2083–2090, 2014.

[9] Donatello Conte, Pasquale Foggia, Carlo Sansone, and Mario Vento. Thirty years of graph matching in pattern recognition. *International Journal of Pattern Recognition and Artificial Intelligence*, 18(03):265–298, 2004.

[10] Amir Egozi, Yosi Keller, and Hugo Guterman. A probabilistic approach to spectral graph matching. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 35(1):18–27, 2012.

[11] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. *International Journal of Computer Vision*, 88(2):303–338, 2010.

[12] Matthias Fey, Jan Eric Lenssen, Frank Weichert, and Heinrich Müller. Splinecnn: Fast geometric deep learning with continuous b-spline kernels. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 869–877, 2018.

[13] Matthias Fey, Jan E Lenssen, Christopher Morris, Jonathan Masci, and Nils M Kriege. Deep graph matching consensus. *arXiv preprint arXiv:2001.09621*, 2020.

[14] Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, and Andrea Lodi. Exact combinatorial optimization with graph convolutional neural networks. In *Proceedings of the Advances in Neural Information Processing Systems*, pages 15580–15592, 2019.

[15] Steven Gold and Anand Rangarajan. A graduated assignment algorithm for graph matching. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 18(4):377–388, 1996.

[16] Juris Hartmanis. Computers and intractability: a guide to the theory of np-completeness. *Siam Review*, 24(1):90, 1982.

[17] Catalin Ionescu, Orestis Vantzos, and Cristian Sminchisescu. Training deep networks with structured layers by matrix backpropagation. *arXiv preprint arXiv:1509.07838*, 2015.

[18] Martin Jaggi. Revisiting frank-wolfe: Projection-free sparse convex optimization. In *Proceedings of the International Conference on Machine Learning*, number CONF, pages 427–435, 2013.

[19] Bo Jiang, Pengfei Sun, Jin Tang, and Bin Luo. Glm-net: Graph learning-matching networks for feature matching. *arXiv preprint arXiv:1911.07681*, 2019.

[20] Hao Jiang, X Yu Stella, and David R Martin. Linear scale and rotation invariant matching. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(7):1339–1355, 2010.

[21] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*, 2016.

[22] Harold W Kuhn. The hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1-2):83–97, 1955.

[23] Simon Lacoste-Julien and Martin Jaggi. On the global linear convergence of frank-wolfe optimization variants. In *Proceedings of the Advances in Neural Information Processing Systems*.

[24] Eugene L Lawler. The quadratic assignment problem. *Management science*, 9(4):586–599, 1963.

[25] Marius Leordeanu and Martial Hebert. A spectral technique for correspondence problems using pairwise constraints. In *Proceedings of the IEEE International Conference on Computer Vision*, volume 2, pages 1482–1489. IEEE, 2005.

[26] Marius Leordeanu, Martial Hebert, and Rahul Sukthankar. An integer projected fixed point method for graph matching and map inference. In *Advances in neural information processing systems*, pages 1114–1122. Citeseer, 2009.

[27] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 2980–2988, 2017.

[28] Zhi-Yong Liu, Hong Qiao, Xu Yang, and Steven CH Hoi. Graph matching by simplified convex-concave relaxation procedure. *International Journal of Computer Vision*, 109(3):169–186, 2014.

[29] Eliane Maria Loiola, Nair Maria Maia de Abreu, Paulo Oswaldo Boaventura-Netto, Peter Hahn, and Tania Querido. A survey for the quadratic assignment problem. *European Journal of Operational Research*, 176(2):657–690, 2007.

[30] David G Lowe. Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2):91–110, 2004.

<!-- Page 10 -->
[31] Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In *Proceedings of the International Conference on Machine Learning*, 2010. 6

[32] Michal Rolínek, Paul Swoboda, Dominik Zietlow, Anselm Paulus, Vít Musil, and Georg Martius. Deep graph matching via blackbox differentiation of combinatorial solvers. In *Proceedings of the European Conference on Computer Vision*, pages 407–424. Springer, 2020. 1, 3, 6, 7

[33] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. *International Journal of Computer Vision*, 115(3):211–252, 2015. 3

[34] Christian Schellewald, Stefan Roth, and Christoph Schnörr. Evaluation of convex optimization techniques for the weighted graph-matching problem in computer vision. In *Joint Pattern Recognition Symposium*, pages 361–368. Springer, 2001. 1

[35] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*, 2014. 3

[36] Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and doubly stochastic matrices. *Pacific Journal of Mathematics*, 21(2):343–348, 1967. 3

[37] Richard Szeliski. *Computer vision: algorithms and applications*. Springer Science & Business Media, 2010. 1

[38] F. D. Wang, N. Xue, Y. Zhang, G. S. Xia, and M. Pelillo. A functional representation for graph matching. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(11):2737–2754, 2020. 2

[39] Runzhong Wang, Junchi Yan, and Xiaokang Yang. Learning combinatorial embedding networks for deep graph matching. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 3056–3065, 2019. 1, 3, 5, 6, 7, 8

[40] Runzhong Wang, Junchi Yan, and Xiaokang Yang. Neural graph matching network: Learning lawler’s quadratic assignment problem with extension to hypergraph and multiple-graph matching. *arXiv preprint arXiv:1911.11308*, 2019. 6, 7

[41] Runzhong Wang, Junchi Yan, and Xiaokang Yang. Combinatorial learning of robust deep graph matching: an embedding based approach. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2020. 3

[42] Tao Wang, He Liu, Yidong Li, Yi Jin, Xiaohui Hou, and Haibin Ling. Learning combinatorial solver for graph matching. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 7568–7577, 2020. 3, 6, 7

[43] Junchi Yan, Xu-Cheng Yin, Weiyao Lin, Cheng Deng, Hongyuan Zha, and Xiaokang Yang. A short survey of recent advances in graph matching. In *Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval*, pages 167–174, 2016. 2

[44] Junchi Yan, Chao Zhang, Hongyuan Zha, Wei Liu, Xiaokang Yang, and Stephen M Chu. Discrete hyper-graph matching. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 1520–1528, 2015. 2

[45] Tianshu Yu, Runzhong Wang, Junchi Yan, and Baoxin Li. Learning deep graph matching with channel-independent embedding and hungarian attention. In *Proceedings of the International Conference on Learning Representations*, volume 20, 2020. 1, 3, 5, 6, 7, 8

[46] Andrei Zanfir and Cristian Sminchisescu. Deep learning of graph matching. In *Proceedings of the IEEE Conference on Computer Vision and Pattern recognition*, pages 2684–2693, 2018. 3, 6, 7

[47] Feng Zhou and Fernando De la Torre. Factorized graph matching. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 38(9):1774–1789, 2015. 2, 8

[48] Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. Graph neural networks: A review of methods and applications. *arXiv preprint arXiv:1812.08434*, 2018. 4