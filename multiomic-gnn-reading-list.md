# Multiomic GNN Reading List

*Single-cell multi-omic integration with graph neural networks — 241 papers in 23 sub-categories, each with a short critical introduction. Version 2, 10 September 2026.*

An annotated bibliography in two parts, compiled for the single-cell multi-omic integration / graph neural network project. **Part I** is the applied literature: graph neural networks on single-cell and spatial multi-omics, together with the non-graph integration methods that every one of those papers benchmarks against. **Part II** is the machine-learning canon those models are built from.

Citation counts come from OpenAlex, retrieved 9 September 2026. Every entry links to its DOI, and every DOI was resolved against OpenAlex rather than generated, so these are safe to cite.

## Contents

**Part I — Applied: single-cell and spatial multi-omics** (182 papers)

- [1. Reference points: integration without graphs](#1-reference-points-integration-without-graphs) — 21 papers
    - [1.1 Anchors, projections and manifold alignment](#11-anchors-projections-and-manifold-alignment) — 10
    - [1.2 Probabilistic latent-variable models](#12-probabilistic-latent-variable-models) — 6
    - [1.3 Factor models and topic models](#13-factor-models-and-topic-models) — 5
- [2. Graph learning on single cells](#2-graph-learning-on-single-cells) — 34 papers
    - [2.1 Cell-graph clustering and representation learning](#21-cell-graph-clustering-and-representation-learning) — 19
    - [2.2 Cell type annotation and label transfer](#22-cell-type-annotation-and-label-transfer) — 8
    - [2.3 Imputation and denoising](#23-imputation-and-denoising) — 3
    - [2.4 Trajectories, velocity and cell state dynamics](#24-trajectories-velocity-and-cell-state-dynamics) — 3
    - [2.5 Higher-order graphs and the single-cell 3D genome](#25-higher-order-graphs-and-the-single-cell-3d-genome) — 1
- [3. Graph-based multi-omic integration](#3-graph-based-multi-omic-integration) — 9 papers
    - [3.1 Guidance graphs and cell-feature graphs](#31-guidance-graphs-and-cell-feature-graphs) — 7
    - [3.2 Cross-modality translation and prediction](#32-cross-modality-translation-and-prediction) — 2
- [4. Spatial omics](#4-spatial-omics) — 43 papers
    - [4.1 Spatial domain identification](#41-spatial-domain-identification) — 22
    - [4.2 Cell-type deconvolution](#42-cell-type-deconvolution) — 3
    - [4.3 Histology-to-expression prediction and resolution enhancement](#43-histology-to-expression-prediction-and-resolution-enhancement) — 6
    - [4.4 Cross-slice alignment and 3D reconstruction](#44-cross-slice-alignment-and-3d-reconstruction) — 2
    - [4.5 Spatial multi-omic integration](#45-spatial-multi-omic-integration) — 5
    - [4.6 Tissue-level phenotype and clinical prediction](#46-tissue-level-phenotype-and-clinical-prediction) — 3
    - [4.7 Segmentation and spatially variable genes](#47-segmentation-and-spatially-variable-genes) — 2
- [5. Cell-cell communication](#5-cell-cell-communication) — 5 papers
- [6. Gene regulatory network inference](#6-gene-regulatory-network-inference) — 9 papers
- [7. Patient-level multi-omics](#7-patient-level-multi-omics) — 48 papers
    - [7.1 Patient-similarity networks: classification and subtyping](#71-patient-similarity-networks-classification-and-subtyping) — 23
    - [7.2 Cancer gene and driver gene discovery on biological networks](#72-cancer-gene-and-driver-gene-discovery-on-biological-networks) — 15
    - [7.3 Survival and prognosis](#73-survival-and-prognosis) — 5
    - [7.4 Drug response prediction](#74-drug-response-prediction) — 5
- [8. Single-cell foundation models](#8-single-cell-foundation-models) — 4 papers
- [9. Reviews, benchmarks and software](#9-reviews-benchmarks-and-software) — 9 papers

**Part II — Methods: the classical GNN canon** (59 papers)

- [II.1 Origins and surveys](#ii1-origins-and-surveys) — 10 papers
- [II.2 Shallow node embeddings](#ii2-shallow-node-embeddings) — 5 papers
- [II.3 Convolution and message passing](#ii3-convolution-and-message-passing) — 9 papers
- [II.4 Attention and graph transformers](#ii4-attention-and-graph-transformers) — 2 papers
- [II.5 Heterogeneous and knowledge graphs](#ii5-heterogeneous-and-knowledge-graphs) — 6 papers
- [II.6 Scalability: sampling and batching](#ii6-scalability-sampling-and-batching) — 5 papers
- [II.7 Depth and over-smoothing](#ii7-depth-and-over-smoothing) — 5 papers
- [II.8 Pooling and graph-level readout](#ii8-pooling-and-graph-level-readout) — 2 papers
- [II.9 Expressivity and theory](#ii9-expressivity-and-theory) — 2 papers
- [II.10 Self-supervised and contrastive learning](#ii10-self-supervised-and-contrastive-learning) — 2 papers
- [II.11 Dynamic and spatio-temporal graphs](#ii11-dynamic-and-spatio-temporal-graphs) — 2 papers
- [II.12 Explainability](#ii12-explainability) — 1 paper
- [II.13 Domain landmarks](#ii13-domain-landmarks) — 6 papers
- [II.14 Libraries and benchmarks](#ii14-libraries-and-benchmarks) — 2 papers

- [How this list was built](#how-this-list-was-built)

---

## Part I — Applied: single-cell and spatial multi-omics

182 papers, screened from a citation-sorted OpenAlex sweep plus a hand-curated set of canonical methods. Citation range 17,390 – 16.

### 1. Reference points: integration without graphs

Nothing in this reading list is evaluated in a vacuum. These are the methods every graph paper benchmarks against, and several of them remain the better choice in practice. Read this section first: it defines the tasks, the metrics and the performance that a graph formulation has to beat in order to justify itself.

#### 1.1 Anchors, projections and manifold alignment

*Methods that align datasets by finding correspondences between cells and correcting in a shared low-dimensional space. Fast, linear or near-linear, and still the default in most working pipelines.*

**1. Comprehensive Integration of Single-Cell Data (Seurat v3)**  
*Cell*, 2019 · 17,390 citations · citation rank 1/182 · [10.1016/j.cell.2019.05.031](https://doi.org/10.1016/j.cell.2019.05.031)  
`CCA + MNN anchors; the reference baseline`

The paper that made cross-dataset single-cell integration routine. Stuart, Butler and colleagues project pairs of datasets into a shared space with diagonalized canonical correlation analysis, identify mutual nearest neighbours in that space as *anchors*, and score each anchor by how consistently its neighbourhood is preserved — the scoring step is what makes the method robust to cell types present in only one dataset. Anchors then drive either batch correction or the transfer of labels and continuous data across modalities, which is how Seurat v3 became a multi-omic tool rather than only a batch corrector. Almost every method later in this list benchmarks against it, so it is worth reading for the anchor formalism even if you never run it.

**2. Integrated analysis of multimodal single-cell data (Seurat v4, WNN)**  
*Cell*, 2021 · 16,751 citations · citation rank 2/182 · [10.1016/j.cell.2021.04.048](https://doi.org/10.1016/j.cell.2021.04.048)  
`weighted nearest neighbour graph across modalities`

Seurat v4 introduces weighted nearest neighbour (WNN) analysis, the most widely used non-graph answer to the question this whole reading list circles: how do you combine modalities that differ in noise level and information content? For each cell, WNN computes within-modality and cross-modality predicted profiles, converts the difference into a per-cell, per-modality affinity, and builds a single weighted graph whose edges blend RNA and protein (or ATAC) distances in proportions the data chooses. The result is a cell-level graph that downstream clustering and UMAP consume unchanged, which is why WNN is the default baseline for CITE-seq and multiome analyses. Conceptually it is a hand-designed version of the attention weighting that GNN-based integration methods later learn.

**3. Fast, sensitive and accurate integration of single-cell data with Harmony**  
*Nature Methods*, 2019 · 11,210 citations · citation rank 3/182 · [10.1038/s41592-019-0619-0](https://doi.org/10.1038/s41592-019-0619-0)  
`iterative soft-kmeans batch correction`

Harmony is the batch-correction baseline that graph methods are measured against on speed as much as accuracy. It runs in a low-dimensional PCA space and alternates two steps: soft k-means clustering with a diversity penalty that pushes each cluster to contain cells from all batches, and a linear, cluster-specific correction of the cell embeddings. Because both steps are cheap and operate on the embedding rather than the expression matrix, Harmony scales to millions of cells on a laptop and converges in seconds to minutes. Its limitation is the linearity of the correction, which is precisely the gap deep and graph-based integration models claim to fill.

**4. Single-cell chromatin state analysis with Signac**  
*Nature Methods*, 2021 · 1,938 citations · citation rank 5/182 · [10.1038/s41592-021-01282-5](https://doi.org/10.1038/s41592-021-01282-5)  
`scATAC processing + RNA/ATAC bridging`

Signac is the scATAC-seq counterpart to Seurat and the practical entry point for any RNA + chromatin project. It handles the peculiarities of accessibility data — fragment files, peak calling, per-cell quality metrics, term frequency–inverse document frequency normalization followed by latent semantic indexing — and computes gene activity scores that let ATAC cells be embedded alongside RNA cells through Seurat anchors. It also supports motif enrichment, footprinting and per-peak links to nearby genes, which is where the peak–gene edges used by many GNN methods in Section 6 come from. Read it as infrastructure: much of the "guidance graph" that GLUE and related models rely on is Signac output.

**5. scJoint integrates atlas-scale scRNA-seq and scATAC-seq with transfer learning**  
*Nature Biotechnology*, 2022 · 179 citations · citation rank 36/182 · [10.1038/s41587-021-01161-6](https://doi.org/10.1038/s41587-021-01161-6)  
`semi-supervised co-embedding`

scJoint tackles atlas-scale RNA + ATAC integration with a deliberately simple architecture: a shared-weight neural network trained under three objectives — supervised classification on labelled RNA cells, a cosine-similarity loss that aligns the two modalities' embeddings, and a self-training step that propagates labels to ATAC cells. There is no graph and no generative model, which is the point: the authors show that a semi-supervised co-embedding with careful loss design matches or beats heavier methods while running on a million cells. It is the strongest evidence in this list that architectural complexity is not automatically worth its cost, and a useful control when evaluating the GNN integration methods in Section 3.

**6. Integration of spatial and single-cell data across modalities with weakly linked features (MaxFuse)**  
*Nature Biotechnology*, 2023 · 110 citations · citation rank 50/182 · [10.1038/s41587-023-01935-0](https://doi.org/10.1038/s41587-023-01935-0)  
`iterative fuzzy smoothed matching`

MaxFuse addresses the hardest version of the integration problem: modalities whose features barely overlap, such as spatial protein panels and scRNA-seq, where the usual shared-feature bridge is only a few dozen weakly correlated markers. It iterates between smoothing each modality over its own within-modality nearest-neighbour graph, matching cells with linear assignment on the weak shared features, and refining the match with canonical correlation analysis fitted on the current pivots. Each round of fuzzy smoothing raises the signal-to-noise of the linkage, so the matching improves even when the initial correlation is near zero. This is the method to reach for when CODEX or MIBI data must be matched to transcriptomes.

**7. Bi-order multimodal integration of single-cell data (bindSC)**  
*Genome Biology*, 2022 · 76 citations · citation rank 64/182 · [10.1186/s13059-022-02679-x](https://doi.org/10.1186/s13059-022-02679-x)  
`bi-CCA for unmatched features`

bindSC generalizes canonical correlation analysis to the *bi-order* case, where the two datasets share neither cells nor features and the correspondence between features must itself be estimated. It introduces a gene-activity-like coupling matrix as a free parameter and optimizes it jointly with the CCA loadings, so the RNA–ATAC feature mapping is learned from the data rather than fixed by a heuristic gene-body-plus-promoter rule. That single change matters most for CITE-seq and for RNA–ATAC pairs where gene activity scores are poor proxies. It is a clean illustration of the idea that the cross-modal graph is something to infer, not something to assume.

**8. Effective and scalable single-cell data alignment with non-linear canonical correlation analysis**  
*Nucleic Acids Research*, 2021 · 30 citations · outside the top 100 on citations · [10.1093/nar/gkab1147](https://doi.org/10.1093/nar/gkab1147)  
`deep CCA alignment`

This paper replaces the linear projections of classical CCA with neural encoders, giving a non-linear canonical correlation objective that scales to atlas-sized single-cell data. Two modality-specific networks are trained to maximize correlation between their outputs subject to a whitening constraint, with stochastic optimization replacing the eigendecomposition that limits CCA to modest matrix sizes. The result aligns datasets that linear CCA under-fits while keeping the "one shared latent space, two encoders" structure that makes CCA interpretable. It sits usefully between Seurat's anchors and the fully generative models of the next section.

**9. MOJITOO: a fast and universal method for integration of multimodal single-cell data**  
*Bioinformatics*, 2022 · 26 citations · outside the top 100 on citations · [10.1093/bioinformatics/btac220](https://doi.org/10.1093/bioinformatics/btac220)  
`closed-form CCA-style multimodal integration`

MOJITOO's contribution is speed through closed form. Rather than training anything, it applies CCA to the modality-specific low-dimensional representations and keeps only the canonical components whose correlation is significant, giving a single joint embedding in seconds with no hyperparameters to tune and no random seed to worry about. On standard CITE-seq and multiome benchmarks it is competitive with methods that take hours. Include it in any benchmark as the "how much does the deep model actually buy you" control; its limits show up when modalities are related non-linearly.

**10. Moving towards genome-wide data integration for patient stratification (Integrate Any Omics)**  
*Nature Machine Intelligence*, 2025 · 22 citations · outside the top 100 on citations · [10.1038/s42256-024-00942-3](https://doi.org/10.1038/s42256-024-00942-3)  
`any-omics integration without feature matching`

This work pushes integration toward the setting where the omics layers have no matched features at all — methylation, expression, copy number, proteomics measured on overlapping but distinct patient sets. It learns a shared representation without requiring feature correspondence, using the sample axis rather than the feature axis as the anchor, and demonstrates stratification on cohorts where feature-matching methods cannot be run. For a textbook this is the bridge chapter between single-cell integration and the patient-level multi-omics of Section 7, where the sample-similarity graph is the natural object.

#### 1.2 Probabilistic latent-variable models

*Variational autoencoders with count-appropriate likelihoods. This is the backbone that most deep single-cell models — graph-based included — are built on top of.*

**1. Deep generative modeling for single-cell transcriptomics (scVI)**  
*Nature Methods*, 2018 · 2,811 citations · citation rank 4/182 · [10.1038/s41592-018-0229-2](https://doi.org/10.1038/s41592-018-0229-2)  
`VAE backbone reused by most integration models`

scVI is the architectural ancestor of most deep single-cell models. It is a variational autoencoder whose decoder emits the parameters of a zero-inflated negative binomial likelihood per gene, with library size modelled as its own latent variable and batch supplied as a conditioning covariate, so normalization, batch correction and dimensionality reduction happen inside one probabilistic model rather than as a pipeline of heuristics. Because the latent space is a posterior rather than a point estimate, downstream tasks — differential expression, imputation, transfer — inherit calibrated uncertainty. Every "VAE plus something" method later in this list, and the whole scvi-tools ecosystem, starts here.

**2. Mapping single-cell data to reference atlases by transfer learning (scArches)**  
*Nature Biotechnology*, 2021 · 671 citations · citation rank 15/182 · [10.1038/s41587-021-01001-7](https://doi.org/10.1038/s41587-021-01001-7)  
`architectural surgery for reference mapping`

scArches solves the reference-mapping problem without retraining the reference. Given a pretrained scVI-style model, it freezes the existing weights and inserts a small set of new, trainable adaptor weights ("architectural surgery") that absorb the query dataset's batch effect, so a new sample is mapped onto an atlas in minutes on a CPU and the reference's coordinates never move. This matters practically — atlases are expensive to build and must stay stable — and conceptually, because it is transfer learning for single-cell data years before foundation models made the framing standard. It is also the mechanism behind most published atlas-projection workflows.

**3. Joint probabilistic modeling of single-cell multi-omic data with totalVI**  
*Nature Methods*, 2021 · 649 citations · citation rank 17/182 · [10.1038/s41592-020-01050-x](https://doi.org/10.1038/s41592-020-01050-x)  
`CITE-seq RNA+protein joint VAE`

totalVI extends the scVI framework to CITE-seq by adding a second likelihood branch for antibody-derived tags, modelled as a negative binomial mixture that explicitly separates true protein signal from ambient antibody background. Both modalities inform one latent space, so protein measurements sharpen cell-state resolution where RNA is ambiguous, and the model can impute unmeasured proteins and denoise the measured ones. The background component is the part worth studying: it is a principled treatment of a noise process that ad hoc normalization handles badly. totalVI is the standard probabilistic baseline for RNA + protein integration.

**4. MultiVI: deep generative model for the integration of multimodal data**  
*Nature Methods*, 2023 · 363 citations · citation rank 23/182 · [10.1038/s41592-023-01909-9](https://doi.org/10.1038/s41592-023-01909-9)  
`joint RNA+ATAC+protein latent space`

MultiVI handles the mosaic case that real multiome experiments produce: some cells measured for RNA only, some for ATAC only, some for both. It trains modality-specific encoders into a shared latent space, penalizes the distance between the two posteriors for jointly profiled cells, and uses the resulting alignment to place single-modality cells in the same coordinates. Missing modalities can then be imputed by decoding through the other branch. This "align where you have pairs, generalize where you do not" recipe is the probabilistic counterpart of the mosaic graph methods that appear later in the spatial section.

**5. Cobolt: integrative analysis of multimodal single-cell sequencing data**  
*Genome Biology*, 2021 · 187 citations · citation rank 34/182 · [10.1186/s13059-021-02556-z](https://doi.org/10.1186/s13059-021-02556-z)  
`multimodal variational autoencoder`

Cobolt models paired and unpaired multi-omic cells in one multimodal variational autoencoder built on a multinomial (latent Dirichlet allocation-style) likelihood suited to sparse count data. Its product-of-experts posterior lets any subset of modalities be observed, so joint cells train the alignment and single-modality cells still receive a coherent embedding. The paper is a clear worked example of how the choice of likelihood — multinomial rather than Gaussian — changes what the latent space represents for very sparse ATAC counts. It benchmarks directly against MultiVI and scMM and is best read alongside them.

**6. A mixture-of-experts deep generative model for single-cell multiomics (scMM)**  
*Cell Reports Methods*, 2021 · 158 citations · citation rank 39/182 · [10.1016/j.crmeth.2021.100071](https://doi.org/10.1016/j.crmeth.2021.100071)  
`MoE-VAE for paired modalities`

scMM uses a mixture-of-experts variational autoencoder for paired modalities, in which each modality's encoder proposes a posterior and the joint posterior is their mixture, allowing generation in either direction from either input alone. The mixture form, in contrast to a product of experts, keeps each modality's contribution identifiable, and the authors exploit this to perform cross-modal generation and to traverse the latent space along interpretable pseudo-axes. It is compact, well-documented and a good teaching example of how the choice of multimodal fusion operator — mixture versus product — changes the model's behaviour.

#### 1.3 Factor models and topic models

*Interpretable linear and quasi-linear decompositions that separate shared from modality-private variation. Slower to be displaced than the deep-learning literature suggests.*

**1. Multi-Omics Factor Analysis (MOFA)**  
*Molecular Systems Biology*, 2018 · 1,578 citations · citation rank 6/182 · [10.15252/msb.20178124](https://doi.org/10.15252/msb.20178124)  
`Bayesian group factor analysis`

MOFA is Bayesian group factor analysis for multi-omics: it decomposes several data matrices measured on the same samples into a shared low-dimensional factor matrix and per-view loadings, with automatic relevance determination sparsity priors deciding which factors are active in which view. The output is directly interpretable — a factor is either shared across omics layers or private to one, and its loadings name the driving features — which is why MOFA remains the reference for unsupervised multi-omic variance decomposition. It also handles missing views gracefully. Read it for the framing of "shared versus private variation", a distinction that recurs throughout the graph literature.

**2. Single-Cell Multi-omic Integration Compares and Contrasts Brain Cell Identity (LIGER)**  
*Cell*, 2019 · 1,401 citations · citation rank 8/182 · [10.1016/j.cell.2019.05.006](https://doi.org/10.1016/j.cell.2019.05.006)  
`integrative NMF across modalities`

LIGER applies integrative non-negative matrix factorization, splitting each dataset's factor loadings into a shared metagene component and a dataset-specific one, then building a shared factor neighbourhood graph and quantile-normalizing cells within matched clusters. The dataset-specific term is the key idea: it gives batch and modality effects somewhere to live so they do not contaminate the shared factors. Applied to single-nucleus RNA and DNA methylation from brain, it demonstrated that cell identity can be recovered across radically different measurement types. It remains one of the strongest non-deep baselines for unpaired integration.

**3. MOFA+: comprehensive integration of multi-modal single-cell data**  
*Genome Biology*, 2020 · 1,118 citations · citation rank 12/182 · [10.1186/s13059-020-02015-1](https://doi.org/10.1186/s13059-020-02015-1)  
`multi-group, multi-view factor model`

MOFA+ rebuilds MOFA on a stochastic variational inference backend and adds a second structured axis — groups of samples — so variance can be decomposed simultaneously across views and across conditions, donors or time points. The scalability change is what made the framework usable on single-cell-scale matrices rather than bulk cohorts. In practice MOFA+ is often the first thing to run on a new multi-omic dataset: it is fast, needs no labels, and its factor-by-view variance table tells you immediately whether the modalities carry shared signal at all.

**4. scAI: integrative analysis of parallel single-cell transcriptomic and epigenomic profiles**  
*Genome Biology*, 2020 · 209 citations · citation rank 30/182 · [10.1186/s13059-020-1932-8](https://doi.org/10.1186/s13059-020-1932-8)  
`aggregated epigenomic factorization`

scAI addresses a specific difficulty of paired single-cell epigenomic data: individual cells' chromatin profiles are so sparse that factorization on the raw matrix mostly fits dropout. The method aggregates epigenomic profiles across similar cells — with the similarity itself learned inside the optimization rather than fixed beforehand — and factorizes the aggregated matrix jointly with expression. The alternating scheme means the cell-cell similarity and the factors refine one another. It is an early and clear statement of the aggregation-versus-imputation trade-off that recurs in every sparse-modality method.

**5. MIRA: joint regulatory modeling of multimodal expression and chromatin accessibility**  
*Nature Methods*, 2022 · 88 citations · citation rank 58/182 · [10.1038/s41592-022-01595-z](https://doi.org/10.1038/s41592-022-01595-z)  
`topic modelling across modalities`

MIRA models expression and accessibility with separate topic models and then links them: topics learned on RNA and on ATAC are related through a shared cell representation, and regulatory potential models connect accessible loci to genes to yield locus-to-gene and topic-to-topic interpretations. The topic formulation is well matched to accessibility data, where a cell's profile is naturally a mixture over regulatory programmes. MIRA's cis-regulatory scoring also produces one of the more defensible peak–gene link sets available, which matters for anyone building the prior graphs used by the GRN methods in Section 6.

---

### 2. Graph learning on single cells

The largest and most crowded part of the field: build a graph over cells (or over cells and genes) and learn on it. Most of these papers address clustering or annotation and differ in how the graph is constructed, how attributes and structure are fused, and what self-supervision is applied. Read a few carefully rather than all of them — the design space is small and the papers overlap heavily.

#### 2.1 Cell-graph clustering and representation learning

*Graph autoencoders, attention encoders and contrastive objectives over cell or cell-gene graphs. The central recurring questions are where attribute and structural information should meet, and whether the graph should be fixed or refined during training.*

**1. scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses**  
*Nature Communications*, 2021 · 485 citations · citation rank 21/182 · [10.1038/s41467-021-22197-x](https://doi.org/10.1038/s41467-021-22197-x)  
`cell graph + multi-modal autoencoders`

scGNN is the paper that established the template for the whole sub-field: build a cell–cell graph, learn on it, and let the graph and the representation refine one another. It stacks three components — a feature autoencoder, a graph autoencoder over an iteratively pruned cell graph, and a set of cluster-specific autoencoders — and cycles through them so that clustering improves the graph, the graph improves the embedding, and the embedding improves the clustering. Gene expression is modelled with a left-truncated mixture Gaussian to handle dropout explicitly rather than imputing it away. It remains the canonical citation for graph-based scRNA-seq analysis and the reference implementation most later methods compare against.

**2. ZINB-based graph embedding autoencoder for single-cell RNA-seq interpretation**  
*AAAI*, 2022 · 108 citations · citation rank 52/182 · [10.1609/aaai.v36i4.20392](https://doi.org/10.1609/aaai.v36i4.20392)  
`count-aware graph autoencoder (AAAI)`

This AAAI paper makes the point that a graph autoencoder applied to count data should not use a Gaussian reconstruction loss. It couples a graph embedding autoencoder over the cell kNN graph with a zero-inflated negative binomial decoder, so structural information from the graph and the discrete, over-dispersed, dropout-heavy nature of UMI counts are handled by the same model. The ablations are the useful part: replacing the ZINB head with mean-squared error costs substantially more than most architectural changes elsewhere in this section. A good example to teach the general principle that likelihood choice usually dominates architecture choice on count data.

**3. Deep structural clustering for scRNA-seq jointly through autoencoder and GNN (scDSC)**  
*Briefings in Bioinformatics*, 2022 · 108 citations · citation rank 53/182 · [10.1093/bib/bbac018](https://doi.org/10.1093/bib/bbac018)  
`ZINB AE coupled to GNN layers`

scDSC couples a ZINB-based denoising autoencoder with a GNN module and trains them jointly under a self-supervised clustering objective, transferring the autoencoder's learned representations into the GNN layer by layer rather than concatenating them at the end. A mutual-supervision scheme aligns the two branches' cluster assignments, and a KL self-optimizing target sharpens them over training. The design question it answers — where in the network should attribute information and structure information meet — is the recurring one across this whole sub-category. It is a strong, well-benchmarked representative of the "AE + GNN, jointly trained" family.

**4. scGAC: a graph attentional architecture for clustering scRNA-seq data**  
*Bioinformatics*, 2022 · 103 citations · citation rank 54/182 · [10.1093/bioinformatics/btac099](https://doi.org/10.1093/bioinformatics/btac099)  
`self-optimizing GAT clustering`

scGAC replaces the fixed kNN graph with attention: after building an initial cell graph it learns edge weights with graph attention, so neighbours that turn out to belong to other cell types are down-weighted rather than being trusted equally. Clustering is self-optimizing, with the attention network and the cluster centres updated in alternation. This directly targets the weakest assumption in most cell-graph methods — that a Euclidean kNN graph built on noisy, high-dimensional counts is trustworthy. Compare it with scGNN's graph pruning as two different fixes for the same problem.

**5. GNN-based embedding for clustering scRNA-seq data (graph-sc)**  
*Bioinformatics*, 2021 · 100 citations · citation rank 55/182 · [10.1093/bioinformatics/btab787](https://doi.org/10.1093/bioinformatics/btab787)  
`cell-gene bipartite graph autoencoder`

graph-sc takes the bipartite view: rather than a cell–cell graph, it builds a cell–gene graph in which edges carry expression values, and runs a graph autoencoder over it to embed cells. This avoids the arbitrariness of choosing k and a distance metric for a kNN graph, because the graph is simply the data. The paper's systematic comparison across many datasets and clustering algorithms is unusually thorough for this literature and worth reading for its evaluation design as much as its method. The cell–gene bipartite construction reappears in scMoGNN, DeepMAPS and SIMBA.

**6. A topology-preserving dimensionality reduction method for scRNA-seq using graph autoencoder (scGAE)**  
*Scientific Reports*, 2021 · 85 citations · citation rank 59/182 · [10.1038/s41598-021-99003-7](https://doi.org/10.1038/s41598-021-99003-7)  
`preserves manifold topology`

scGAE is aimed at a specific failure of standard dimensionality reduction: t-SNE and UMAP distort global topology, so trajectories and inter-cluster relationships read differently depending on hyperparameters. It trains a graph autoencoder that reconstructs both the feature matrix and the adjacency of the cell graph, with the dual objective preserving local neighbourhoods and the coarse shape of the manifold at once. The embeddings are consequently more stable for downstream trajectory work. Read it as the topology-preservation argument for graph autoencoders.

**7. CellVGAE: unsupervised scRNA-seq analysis with graph attention networks**  
*Bioinformatics*, 2021 · 45 citations · citation rank 86/182 · [10.1093/bioinformatics/btab804](https://doi.org/10.1093/bioinformatics/btab804)  
`variational graph AE on kNN cell graph`

CellVGAE runs a variational graph autoencoder with graph attention layers over a kNN cell graph, using the attention coefficients as an interpretability handle: the edges the model relies on can be inspected, and the genes driving them traced. Being variational, it also gives a probabilistic embedding rather than a point estimate. The authors emphasize that using highly variable genes as node features and attention as the aggregator recovers structure that PCA-based pipelines miss on small, heterogeneous datasets. It is one of the more readable implementations in this group.

**8. scCDG: a method based on denoising autoencoder and GCN for scRNA-seq analysis**  
*IEEE/ACM Trans. Comput. Biol. Bioinform.*, 2021 · 40 citations · citation rank 98/182 · [10.1109/tcbb.2021.3126641](https://doi.org/10.1109/tcbb.2021.3126641)  
`DAE features feeding a GCN`

scCDG is a two-stage design: a denoising autoencoder first produces a robust low-dimensional feature per cell, and a graph convolutional network then propagates those features over a cell similarity graph before clustering. Separating the two stages makes the pipeline simple to train and to reason about, at the cost of the mutual refinement that jointly trained models such as scDSC achieve. It is a useful control in that comparison, and its ablations quantify how much the denoising step alone contributes.

**9. scEGG: exogenous gene-guided clustering for single-cell transcriptomic data**  
*Briefings in Bioinformatics*, 2024 · 35 citations · outside the top 100 on citations · [10.1093/bib/bbae483](https://doi.org/10.1093/bib/bbae483)  
`external gene knowledge in the graph`

scEGG argues that the cell graph should be informed by biology rather than derived only from the expression matrix. It brings in exogenous gene knowledge — curated gene sets and interactions — to guide graph construction and to weight features before clustering, on the reasoning that similarity computed over functionally coherent gene groups is less dominated by technical variation. The gain is largest on datasets with subtle sub-types where unsupervised distances are close to noise. It pairs naturally with scPriorGraph in the annotation sub-section.

**10. Deep scRNA-seq clustering with graph prototypical contrastive learning (scGPCL)**  
*Bioinformatics*, 2023 · 33 citations · outside the top 100 on citations · [10.1093/bioinformatics/btad342](https://doi.org/10.1093/bioinformatics/btad342)  
`prototype-level contrastive objective`

scGPCL applies graph contrastive learning with prototypes: instead of contrasting individual cells, which forces apart cells that genuinely belong to the same type, it contrasts cells against learned cluster prototypes, so the objective no longer fights itself. The graph is a cell–gene bipartite graph and the encoder a GNN. This is the cleanest treatment in the single-cell literature of the false-negative problem in contrastive learning, which is a real obstacle when the number of true classes is small relative to the number of samples.

**11. Advancing scRNA-seq analysis through fusion of multi-layer perceptron and graph neural network**  
*Briefings in Bioinformatics*, 2023 · 30 citations · outside the top 100 on citations · [10.1093/bib/bbad481](https://doi.org/10.1093/bib/bbad481)  
`MLP+GNN hybrid encoder`

This paper fuses a multi-layer perceptron branch with a GNN branch, on the argument that message passing over a noisy cell graph can wash out the cell-intrinsic signal that a plain MLP preserves. The two representations are combined adaptively, so the model can fall back toward the MLP where the graph is unreliable. It is worth reading next to the over-smoothing literature in Part II, Section 7: the phenomenon it is compensating for is exactly the one PairNorm and JKNet address architecturally.

**12. scGNN 2.0: a graph neural network tool for imputation and clustering**  
*Bioinformatics*, 2022 · 29 citations · outside the top 100 on citations · [10.1093/bioinformatics/btac684](https://doi.org/10.1093/bioinformatics/btac684)  
`faster, more usable scGNN`

scGNN 2.0 is the engineering follow-up to scGNN, and the version to actually use. It rewrites the iterative training loop for speed, reduces memory use enough to handle atlas-scale inputs, adds proper hyperparameter handling and a much clearer interface, and improves both imputation accuracy and clustering over the original. The paper is short and largely about usability, which is precisely why it belongs in a textbook's discussion of what it takes to move a method from publication to practice.

**13. A new graph autoencoder-based consensus-guided model for scRNA-seq cell type detection**  
*IEEE Trans. Neural Netw. Learn. Syst.*, 2022 · 26 citations · outside the top 100 on citations · [10.1109/tnnls.2022.3190289](https://doi.org/10.1109/tnnls.2022.3190289)  
`consensus-guided graph AE`

This model uses consensus clustering to supervise a graph autoencoder: multiple base clusterings are aggregated into a consensus matrix, which then acts as a target for the graph autoencoder's embedding, so the representation is pulled toward structure that is stable across clustering runs rather than structure that any single algorithm happens to find. The idea addresses the reproducibility complaint about unsupervised single-cell clustering directly. It is the scRNA-seq analogue of the consensus-guided cancer subtyping model in Section 7.1.

**14. scGCC: graph contrastive clustering with neighborhood augmentations**  
*IEEE J. Biomed. Health Inform.*, 2023 · 25 citations · outside the top 100 on citations · [10.1109/jbhi.2023.3319551](https://doi.org/10.1109/jbhi.2023.3319551)  
`augmented cell graph contrastive clustering`

scGCC combines graph contrastive learning with augmentations tailored to single-cell data — neighbourhood-based edge and feature perturbations rather than the image-style crops that generic graph contrastive methods inherit. A momentum encoder supplies stable negative representations. The augmentation design is the contribution worth studying: the wrong augmentation on a cell graph destroys the very biological signal the embedding is supposed to capture, and the paper's ablations show how sensitive results are to this choice.

**15. Attention-based deep clustering method for scRNA-seq cell type identification**  
*PLOS Computational Biology*, 2023 · 23 citations · outside the top 100 on citations · [10.1371/journal.pcbi.1011641](https://doi.org/10.1371/journal.pcbi.1011641)  
`attention graph clustering`

An attention-based deep clustering method that learns the cell graph and the cluster assignment together, using attention both to weight neighbours during message passing and to weight the contribution of different feature blocks. The paper's value for a course is its careful comparison against non-graph deep clustering on the same datasets, which isolates how much the graph itself contributes once attention and self-supervision are held constant. The answer is smaller than the graph-methods literature usually implies.

**16. scDFN: enhancing scRNA-seq clustering with deep fusion networks**  
*Briefings in Bioinformatics*, 2024 · 23 citations · outside the top 100 on citations · [10.1093/bib/bbae486](https://doi.org/10.1093/bib/bbae486)  
`fuses attribute and structure branches`

scDFN fuses an attribute branch and a structure branch through a deep fusion network with information-transfer modules between corresponding layers, rather than concatenating the two representations at the output. A triplet self-supervised objective on the fused representation provides the clustering signal. It sits in the same design space as scDSC; the two together make a good pairing for teaching how much of a method's performance comes from where fusion happens versus from the fusion operator itself.

**17. scLEGA: attention-based deep clustering for low-expression genes**  
*Briefings in Bioinformatics*, 2024 · 23 citations · outside the top 100 on citations · [10.1093/bib/bbae371](https://doi.org/10.1093/bib/bbae371)  
`low-expression-aware graph clustering`

scLEGA targets the genes that most pipelines discard. Standard highly-variable-gene selection removes low-expression genes, but many lineage-defining transcription factors sit exactly there, so scLEGA uses a ZINB-based attention mechanism that keeps low-expression genes in the model and weights them explicitly, combined with a GNN clustering module. The paper shows recovery of rare populations that HVG-based pipelines merge. It is a useful corrective to the reflex of filtering aggressively before any graph is built.

**18. ScGSLC: unsupervised graph similarity learning framework for scRNA-seq clustering**  
*Computational Biology and Chemistry*, 2020 · 21 citations · outside the top 100 on citations · [10.1016/j.compbiolchem.2020.107415](https://doi.org/10.1016/j.compbiolchem.2020.107415)  
`graph similarity learning`

ScGSLC is an early entry in this family, framing clustering as graph similarity learning: cells are grouped by comparing graph structures built from gene networks rather than by distances in expression space. It predates most of the methods above and is technically simpler, but it stated the case that graph structure carries clustering-relevant information beyond the feature matrix. Read it for historical framing rather than for performance.

**19. Graph contrastive learning as a versatile foundation for advanced scRNA-seq analysis**  
*Briefings in Bioinformatics*, 2024 · 18 citations · outside the top 100 on citations · [10.1093/bib/bbae558](https://doi.org/10.1093/bib/bbae558)  
`general-purpose contrastive graph pretraining`

This paper treats graph contrastive learning as a general-purpose pretraining strategy for scRNA-seq rather than as a clustering method, showing that one contrastively pretrained graph encoder transfers across clustering, annotation, imputation and batch correction. That framing — self-supervised graph pretraining as a foundation, task heads on top — is the graph-native counterpart to the transformer foundation models in Section 8, and it is considerably cheaper to train. It is the natural closing paper for this sub-category.

#### 2.2 Cell type annotation and label transfer

*Supervised and semi-supervised node classification, including cross-dataset, cross-species and cross-modality transfer. This is where pretraining first appeared in single-cell work.*

**1. scDeepSort: pre-trained cell-type annotation using a weighted graph neural network**  
*Nucleic Acids Research*, 2021 · 173 citations · citation rank 37/182 · [10.1093/nar/gkab775](https://doi.org/10.1093/nar/gkab775)  
`cell-gene weighted graph, pretrained`

scDeepSort was the first pretrained GNN annotator for scRNA-seq: it builds a weighted cell–gene graph, trains a GNN on a large curated corpus of human and mouse tissue atlases, and then annotates new datasets without any reference dataset at prediction time. The weighted bipartite construction lets gene nodes act as shared context linking cells across datasets, which is what makes the pretrained weights transferable. It is the clearest early demonstration that annotation can be a transfer-learning problem rather than a per-dataset nearest-neighbour problem, and it predates the transformer foundation models by two years.

**2. scGCN is a graph convolutional networks algorithm for knowledge transfer in single cell omics**  
*Nature Communications*, 2021 · 127 citations · citation rank 44/182 · [10.1038/s41467-021-24172-y](https://doi.org/10.1038/s41467-021-24172-y)  
`cross-dataset/cross-omic label transfer`

scGCN transfers labels across datasets, species and modalities by constructing a hybrid graph containing both intra-dataset and inter-dataset cell edges — the latter found through mutual nearest neighbours in a CCA-aligned space — and then running semi-supervised graph convolution over the union graph so labels propagate from annotated to unannotated cells. Framing label transfer as node classification on a joined graph is a genuinely different formulation from the anchor-transfer of Seurat, and it handles the case where the query contains types absent from the reference more gracefully. It is the reference GNN method for knowledge transfer in single-cell omics.

**3. Cross-species cell-type assignment by a heterogeneous graph neural network (CAME)**  
*Genome Research*, 2022 · 57 citations · citation rank 75/182 · [10.1101/gr.276868.122](https://doi.org/10.1101/gr.276868.122)  
`heterogeneous cell-gene graph across species`

CAME performs cross-species cell type assignment on a heterogeneous graph whose nodes are cells and genes from two species and whose edges include within-species expression links and cross-species homology links. Because homology is many-to-many and imperfect, the model learns how much weight to give each ortholog edge instead of assuming one-to-one correspondence. Beyond transferring labels it produces aligned gene embeddings that suggest which orthologs are functionally conserved. This is the most convincing use of heterogeneous graphs in the single-cell annotation literature.

**4. Single-cell classification using graph convolutional networks (sigGCN)**  
*BMC Bioinformatics*, 2021 · 55 citations · citation rank 76/182 · [10.1186/s12859-021-04278-2](https://doi.org/10.1186/s12859-021-04278-2)  
`gene-interaction graph classifier`

sigGCN classifies cells with a GCN operating on a gene–gene interaction network, where each cell's expression vector supplies the node features. Inverting the usual construction — genes as nodes, cells as graph signals — means the prior biological network does the regularizing, and the model needs far fewer parameters than a dense classifier over 20,000 genes. It combines this graph branch with a standard neural network branch on the raw features. A clean demonstration that prior networks act as a structural prior on the classifier.

**5. scRGCL: cell type annotation with residual GCN and contrastive learning**  
*Briefings in Bioinformatics*, 2024 · 53 citations · citation rank 79/182 · [10.1093/bib/bbae662](https://doi.org/10.1093/bib/bbae662)  
`deep residual GCN avoids oversmoothing`

scRGCL builds a deep residual GCN for annotation, using residual connections to allow more propagation layers without the representation collapse that afflicts plain GCNs beyond two or three hops, and adds a contrastive objective with a cell-type-aware negative sampling scheme. It is the single-cell paper most directly engaged with the over-smoothing literature, and the ablations showing accuracy against depth are the ones to reproduce in a course exercise. Read it immediately after DeepGCNs and PairNorm in Part II.

**6. A robust and scalable graph neural network for accurate single-cell classification**  
*Briefings in Bioinformatics*, 2021 · 42 citations · citation rank 93/182 · [10.1093/bib/bbab570](https://doi.org/10.1093/bib/bbab570)  
`scalable GNN cell classifier`

This paper focuses on the scalability and robustness side of GNN-based cell classification: a lightweight architecture with sampling-based training that handles atlas-scale inputs, evaluated under batch effects, dropout and unbalanced cell-type frequencies rather than only on clean benchmarks. The stress-test evaluation is what distinguishes it from the many similar annotators. Useful as a practical guide to whether a GNN annotator will survive contact with a real dataset.

**7. scGraph: a graph neural network-based approach to automatically identify cell types**  
*Bioinformatics*, 2022 · 35 citations · outside the top 100 on citations · [10.1093/bioinformatics/btac199](https://doi.org/10.1093/bioinformatics/btac199)  
`gene-interaction graph per cell`

scGraph represents each cell as its own graph over a gene-interaction network, with expression as node attributes, and classifies cells by graph-level readout. Making the cell the graph rather than a node in a graph changes the inductive bias entirely: the model learns which sub-networks are active, and its attention weights point to the interacting gene modules responsible for a call. It is the natural bridge from this section to the graph-classification and pooling literature in Part II, Section 8.

**8. scPriorGraph: biosemantic cell-cell graphs with prior gene set selection**  
*Genome Biology*, 2024 · 30 citations · outside the top 100 on citations · [10.1186/s13059-024-03357-w](https://doi.org/10.1186/s13059-024-03357-w)  
`pathway-informed cell graph`

scPriorGraph constructs "biosemantic" cell–cell graphs: rather than a single kNN graph on all genes, it selects biologically coherent gene sets, computes cell similarity within each, and fuses the resulting graphs, so the edges reflect shared pathway activity rather than global expression distance. Dual-channel message passing then aggregates over both the fused cell graph and the gene-set features. The paper argues, with ablations, that most annotation errors trace to bad graphs rather than weak models — an argument worth taking seriously across this entire section.

#### 2.3 Imputation and denoising

*Recovering dropout using cell neighbourhoods, gene-gene structure, or both. Evaluate these on downstream tasks, not on held-out entry reconstruction.*

**1. Imputing scRNA-seq data by combining graph convolution and autoencoders (GraphSCI)**  
*iScience*, 2021 · 112 citations · citation rank 49/182 · [10.1016/j.isci.2021.102393](https://doi.org/10.1016/j.isci.2021.102393)  
`gene-gene graph guided imputation`

GraphSCI imputes dropout by combining a graph convolutional network over a gene–gene relation graph with an autoencoder over the expression matrix, so a missing value is inferred both from similar cells and from co-expressed genes. Training alternates between the two components, each conditioning on the other's current estimate. Using gene-level graph structure — rather than only cell neighbourhoods, as most imputation methods do — is what distinguishes it, and it helps most for genes expressed in few cells.

**2. scGGAN: single-cell RNA-seq imputation by graph-based generative adversarial network**  
*Briefings in Bioinformatics*, 2023 · 34 citations · outside the top 100 on citations · [10.1093/bib/bbad040](https://doi.org/10.1093/bib/bbad040)  
`gene graph + GAN imputation`

scGGAN pairs a graph representation of gene–gene relationships with a generative adversarial objective for imputation, so the imputed matrix is judged by a discriminator on whether it looks like real data rather than only by a reconstruction loss. The graph constrains the generator to respect known co-expression structure, which is what keeps GAN-based imputation from hallucinating plausible-looking but biologically arbitrary values. Read it as a case study in the risks as well as the benefits of adversarial objectives on count data.

**3. An efficient scRNA-seq dropout imputation method using graph attention network**  
*BMC Bioinformatics*, 2021 · 32 citations · outside the top 100 on citations · [10.1186/s12859-021-04493-x](https://doi.org/10.1186/s12859-021-04493-x)  
`GAT imputation over cell graph`

A compact graph-attention approach to dropout imputation: a cell graph is built, attention weights the contribution of each neighbour, and missing entries are recovered from attention-weighted neighbourhood profiles. The method is deliberately lightweight and fast, and the paper is honest about the central hazard of all imputation — over-smoothing away genuine biological zeros — reporting results both with and without the imputation step on downstream tasks. That evaluation design is the part to emulate.

#### 2.4 Trajectories, velocity and cell state dynamics

*Graphs used to model change rather than identity — differentiation, RNA velocity and metabolic flux.*

**1. A graph neural network model to estimate cell-wise metabolic flux (scFEA)**  
*Genome Research*, 2021 · 255 citations · citation rank 26/182 · [10.1101/gr.271205.120](https://doi.org/10.1101/gr.271205.120)  
`flux balance as graph constraint`

scFEA estimates metabolic flux per cell by treating the metabolic network as the graph: metabolites are nodes, reaction modules are edges, and a neural network predicts module fluxes from expression subject to a flux-balance constraint that inflow must match outflow at each metabolite. This turns a classical constraint-based modelling problem into a differentiable one solvable at single-cell resolution, where flux balance analysis on individual cells would otherwise be badly under-determined. It is the best example in this list of encoding a mechanistic biological constraint directly into the loss function.

**2. DeepVelo: deep learning extends RNA velocity to multi-lineage systems**  
*Genome Biology*, 2024 · 58 citations · citation rank 74/182 · [10.1186/s13059-023-03148-9](https://doi.org/10.1186/s13059-023-03148-9)  
`GCN over cell neighbourhood graph`

DeepVelo generalizes RNA velocity to systems where a single global kinetic rate per gene does not hold — multi-lineage tissues in which the same gene is spliced and degraded at different rates in different lineages. It uses a graph convolutional network over the cell neighbourhood graph to predict cell-specific kinetic parameters, so velocity vectors adapt to local context. The paper also gives a clearer treatment of velocity model diagnostics than most. It is the natural bridge between the graph literature and the dynamical-systems view of single-cell data.

**3. Unsupervised generative and graph representation learning for modelling cell differentiation**  
*Scientific Reports*, 2020 · 24 citations · outside the top 100 on citations · [10.1038/s41598-020-66166-8](https://doi.org/10.1038/s41598-020-66166-8)  
`early graph representation learning for trajectories`

An early combination of generative modelling and graph representation learning for cell differentiation, predating most of this section. Its contribution is conceptual rather than competitive: it framed differentiation as structure to be learned in a graph embedding rather than as a curve fitted in a reduced space, and showed that the embedding recovers branching. Include it for the historical line from manifold-learning trajectory inference to graph-based approaches.

#### 2.5 Higher-order graphs and the single-cell 3D genome

*Where pairwise edges are the wrong abstraction and hypergraphs are the right one.*

**1. Hyper-SAGNN: a self-attention based graph neural network for hypergraphs**  
*preprint (arXiv)*, 2019 · 31 citations · outside the top 100 on citations · [10.48550/arxiv.1911.02613](https://doi.org/10.48550/arxiv.1911.02613)  
`hypergraph model used for scHi-C`

Hyper-SAGNN generalizes graph neural networks from edges to hyperedges — relations among arbitrarily many nodes — using self-attention to produce both static and dynamic node embeddings whose agreement scores a candidate hyperedge. The motivating biological application is single-cell Hi-C and multi-way chromatin contacts, where an interaction genuinely involves more than two loci and forcing it into pairwise edges loses information. It is the entry point for higher-order structure in single-cell data and pairs directly with the hypergraph neural network paper in Part II.

---

### 3. Graph-based multi-omic integration

The methods this project is actually about: graph formulations in which the graph carries the cross-modal information. The key distinction to hold onto is whether the graph links *cells* (similarity), *features* (prior regulatory knowledge), or both.

#### 3.1 Guidance graphs and cell-feature graphs

*Models in which features from different omics layers are nodes joined by prior regulatory edges, or in which cells and features are co-embedded in one structure.*

**1. Multi-omics single-cell data integration and regulatory inference with graph-linked embedding (GLUE)**  
*Nature Biotechnology*, 2022 · 646 citations · citation rank 18/182 · [10.1038/s41587-022-01284-4](https://doi.org/10.1038/s41587-022-01284-4)  
`guidance graph links features across omics`

GLUE is the most influential graph formulation of multi-omic integration and the one to teach first. Its insight is that unpaired modalities cannot be aligned in feature space directly, but their *features* are related by prior knowledge — a peak overlaps a promoter, a gene body contains an ATAC region — so it builds a "guidance graph" whose nodes are features from every omics layer and whose edges encode those regulatory relations. Modality-specific variational autoencoders share a latent space; a graph autoencoder embeds the guidance graph; and an adversarial discriminator aligns the modalities while the feature embeddings tie the decoders together. Because the guidance graph is itself learned and refined during training, GLUE also outputs updated regulatory links, making integration and regulatory inference a single problem.

**2. Single-cell biological network inference using a heterogeneous graph transformer (DeepMAPS)**  
*Nature Communications*, 2023 · 193 citations · citation rank 33/182 · [10.1038/s41467-023-36559-0](https://doi.org/10.1038/s41467-023-36559-0)  
`cell-gene heterogeneous graph transformer`

DeepMAPS builds a heterogeneous graph containing both cells and genes and applies a heterogeneous graph transformer, whose type-specific attention lets cell-to-gene and gene-to-cell messages be weighted by learned, node-type-aware parameters. From the trained attention it reads off cell-type-specific gene modules — biological networks — rather than only an embedding, so clustering and regulon inference come out of the same model. It handles scRNA, scATAC and CITE-seq in one framework and ships with a web server. Read it directly after the HGT paper in Part II, of which it is the clearest biological application.

**3. Explainable multi-task learning for multi-modality biological data (UnitedNet)**  
*Nature Communications*, 2023 · 114 citations · citation rank 48/182 · [10.1038/s41467-023-37477-x](https://doi.org/10.1038/s41467-023-37477-x)  
`joint fusion + explainable modality contribution`

UnitedNet frames multi-modal single-cell analysis as explainable multi-task learning: one network is trained jointly on cross-modal prediction, fusion and clustering, and then interrogated with feature attribution to say how much each modality and each feature contributed to a given cell's representation. The explainability is not an afterthought bolted on — the multi-task structure is what makes attribution meaningful, because each head provides a different probe of the shared representation. For a textbook it is the best worked example of moving past "the model integrates well" to "here is which modality carried the information".

**4. Graph Neural Networks for Multimodal Single-Cell Data Integration (scMoGNN)**  
*KDD*, 2022 · 78 citations · citation rank 61/182 · [10.1145/3534678.3539213](https://doi.org/10.1145/3534678.3539213)  
`NeurIPS-competition-winning cell-feature graph (KDD)`

scMoGNN won the NeurIPS 2021 Open Problems multimodal single-cell competition, and the KDD paper explains why. It represents the data as a cell–feature bipartite graph, so modality integration, modality prediction and matching all become message-passing problems over one structure, and it adds feature-level auxiliary tasks that regularize the cell embeddings. The competition setting means the comparisons are unusually honest: identical data, identical metrics, many strong baselines. It is the strongest available evidence that graph formulations beat matrix-factorization and plain autoencoder approaches on paired multi-omic tasks.

**5. SIMBA: single-cell embedding along with features**  
*Nature Methods*, 2023 · 78 citations · citation rank 62/182 · [10.1038/s41592-023-01899-8](https://doi.org/10.1038/s41592-023-01899-8)  
`co-embeds cells and features in one graph`

SIMBA co-embeds cells and features — genes, peaks, motifs, proteins — in a single space using a knowledge-graph embedding objective borrowed from the relational learning literature, so a cell and a gene can be compared by distance directly. This makes several tasks that usually require separate methods into nearest-neighbour queries: marker discovery, cell annotation, and identification of the features defining a cluster. The paper also introduces a softmax-based normalization to correct for the degree bias that otherwise dominates such embeddings. It is a genuinely different framing from the encoder–decoder mainstream.

**6. A universal framework for single-cell multi-omics data integration with graph convolutional networks**  
*Briefings in Bioinformatics*, 2023 · 41 citations · citation rank 97/182 · [10.1093/bib/bbad081](https://doi.org/10.1093/bib/bbad081)  
`modality-agnostic GCN integration`

This paper proposes a modality-agnostic GCN framework for single-cell multi-omic integration: rather than hard-coding a pair of modalities, it defines a general construction for cell graphs across an arbitrary number of omics layers and shares a graph convolutional encoder across them. The generality is the contribution — most methods in this section are built for RNA + ATAC or RNA + protein and do not extend — and the paper evaluates on several combinations to support the claim. Useful as a survey-by-construction of what the common structure of these methods actually is.

**7. Single-cell multimodal prediction via transformers (scMoFormer)**  
*CIKM*, 2023 · 21 citations · outside the top 100 on citations · [10.1145/3583780.3615061](https://doi.org/10.1145/3583780.3615061)  
`transformer over cell-gene-protein graph (CIKM)`

scMoFormer places a transformer over a heterogeneous cell–gene–protein graph, using neighbourhood attention to combine the strengths of graph structure and transformer capacity for multimodal prediction. It is a direct successor to scMoGNN on the same competition tasks and reports improved performance, mainly through better handling of long-range dependencies among features that message passing over few hops cannot capture. Read the two together as a case study in when attention adds value over fixed-neighbourhood aggregation.

#### 3.2 Cross-modality translation and prediction

*Integration posed as translation: predict one modality from another rather than align them.*

**1. BABEL enables cross-modality translation between multiomic profiles**  
*PNAS*, 2021 · 184 citations · citation rank 35/182 · [10.1073/pnas.2023070118](https://doi.org/10.1073/pnas.2023070118)  
`encoder-decoder RNA<->ATAC translation`

BABEL trains paired encoder–decoder networks such that either modality's encoder can feed either modality's decoder, so an scRNA-seq profile can be translated into a predicted chromatin accessibility profile and vice versa. The interchangeable-decoder design forces a genuinely shared latent representation rather than two spaces glued together by a penalty. Trained on multiome data, it generalizes to samples in which only one modality was measured, including patient tumours. It is the cleanest statement of integration-as-translation, an alternative framing to integration-as-alignment.

**2. PIKE-R2P: PPI network-based knowledge embedding with GNN for single-cell RNA to protein prediction**  
*BMC Bioinformatics*, 2021 · 17 citations · outside the top 100 on citations · [10.1186/s12859-021-04022-w](https://doi.org/10.1186/s12859-021-04022-w)  
`cross-modality prediction with prior graph`

PIKE-R2P predicts protein abundance from scRNA-seq by embedding prior knowledge from a protein–protein interaction network into a GNN, so the prediction for one protein draws on the expression of its network neighbours' genes rather than on its own transcript alone. This directly addresses the weak and non-linear transcript-to-protein correlation that makes naive regression fail. It is a small paper but a good illustration of prior-graph regularization in the cross-modal prediction setting, and a natural comparison point for totalVI's imputation.

---

### 4. Spatial omics

The fastest-growing sub-field, and the one where graph neural networks have the clearest natural justification: the graph is given by physical geometry rather than inferred from expression. The sub-categories below correspond to genuinely different tasks, and methods are rarely comparable across them.

#### 4.1 Spatial domain identification

*The dominant task: partition a tissue section into coherent regions. Twenty-two papers here differ mostly in graph construction, attention mechanism and self-supervised objective. Note MENDER, which matches the deep models without training anything.*

**1. SpaGCN: identify spatial domains and spatially variable genes by graph convolutional network**  
*Nature Methods*, 2021 · 1,203 citations · citation rank 10/182 · [10.1038/s41592-021-01255-8](https://doi.org/10.1038/s41592-021-01255-8)  
`first widely used spatial GCN`

SpaGCN is the paper that started graph learning on spatial transcriptomics. It constructs an undirected weighted graph in which spot-to-spot edge weights combine physical distance with histology-image colour similarity, runs a graph convolutional network to aggregate expression over that neighbourhood, and clusters the resulting embeddings into spatial domains. Adding a histology term is the key move: it lets tissue morphology break ties that expression alone cannot, so domain boundaries follow anatomy. The method also detects spatially variable genes and meta-genes enriched in each domain, and its simplicity has made it the universal baseline for everything below.

**2. Deciphering spatial domains with an adaptive graph attention auto-encoder (STAGATE)**  
*Nature Communications*, 2022 · 762 citations · citation rank 13/182 · [10.1038/s41467-022-29439-6](https://doi.org/10.1038/s41467-022-29439-6)  
`GAT autoencoder on spatial neighbour graph`

STAGATE is the reference graph-attention method for spatial domains. It builds a spatial neighbour graph, then trains a graph attention autoencoder that learns edge weights rather than fixing them by distance, with an optional cell-type-aware pruning step that removes edges between spots the model judges to belong to different domains. Learning attention lets the receptive field expand inside a domain and contract at boundaries, which produces sharper, less speckled segmentations than SpaGCN. It scales to Stereo-seq and Slide-seq resolutions and can align consecutive sections into a 3D domain model. Most later spatial autoencoders are variations on this design.

**3. Spatially informed clustering, integration and deconvolution with GraphST**  
*Nature Communications*, 2023 · 686 citations · citation rank 14/182 · [10.1038/s41467-023-36796-3](https://doi.org/10.1038/s41467-023-36796-3)  
`graph contrastive learning; batch integration`

GraphST combines a graph autoencoder over the spatial neighbour graph with self-supervised contrastive learning, where positive pairs are a spot and its corrupted-graph counterpart and negatives come from a shuffled feature matrix. Beyond domain identification it handles two tasks most competitors treat separately: batch integration across slices, and cell-type deconvolution by mapping single-cell references onto spots. Treating all three as consequences of one representation is the paper's argument, and the benchmarks support it across 10x Visium, Stereo-seq and Slide-seq. It is the strongest all-round general-purpose spatial method in this list.

**4. Unsupervised spatially embedded deep representation of spatial transcriptomics (SEDR)**  
*Genome Medicine*, 2024 · 392 citations · citation rank 22/182 · [10.1186/s13073-024-01283-x](https://doi.org/10.1186/s13073-024-01283-x)  
`deep autoencoder + VGAE spatial embedding`

SEDR learns a spatially embedded representation by training a deep autoencoder on expression and a variational graph autoencoder on the spatial graph, then concatenating and jointly refining the two embeddings under a clustering objective. Keeping the expression and spatial encoders separate before fusion means the model can weigh how much spatial smoothing to apply, rather than baking it into every layer. The paper is thorough on benchmarking across platforms and includes trajectory and batch-correction analyses. It is a good example of the "two encoders, late fusion" design point in this family.

**5. Cell clustering for spatial transcriptomics data with graph neural networks (CCST)**  
*Nature Computational Science*, 2022 · 241 citations · citation rank 28/182 · [10.1038/s43588-022-00266-5](https://doi.org/10.1038/s43588-022-00266-5)  
`global graph-attention cell clustering`

CCST departs from purely local message passing by constructing a global cell graph — edges between spatially adjacent cells, but embedding learned with Deep Graph Infomax so each node's representation is contrasted against a global summary of the whole tissue. This gives every cell a representation that reflects both its neighbourhood and its position relative to the tissue as a whole, which helps identify domains that recur in disconnected regions. It was among the first to work at true single-cell spatial resolution rather than spot resolution. Read it alongside the DGI paper in Part II.

**6. Spatial-MGCN: multi-view graph convolutional network for spatial domains**  
*Briefings in Bioinformatics*, 2023 · 89 citations · citation rank 57/182 · [10.1093/bib/bbad262](https://doi.org/10.1093/bib/bbad262)  
`expression + spatial views with attention`

Spatial-MGCN builds two graphs — one from expression similarity, one from spatial proximity — and fuses their convolutions with attention, so the model can decide per spot whether transcriptional or spatial evidence should dominate. A ZINB decoder handles the count distribution. The multi-view construction is the recurring answer in this sub-section to the observation that a single graph forces a fixed trade-off between spatial smoothness and expression fidelity; Spatial-MGCN is the cleanest implementation of it.

**7. Define and visualize pathological architectures from spatial transcriptomics (RESEPT)**  
*Comput. Struct. Biotechnol. J.*, 2022 · 53 citations · citation rank 78/182 · [10.1016/j.csbj.2022.08.029](https://doi.org/10.1016/j.csbj.2022.08.029)  
`embedding-to-RGB image segmentation`

RESEPT takes an unusual route: it embeds spots with a graph autoencoder, maps the three leading embedding dimensions to RGB channels to produce a synthetic image of the tissue, and then applies a standard image segmentation network to that image. Converting the problem into one that mature computer-vision architectures already solve is a genuinely clever reduction, and it makes the result visually interpretable at every step. The paper also uses the RGB representation to assess segmentation quality without ground truth.

**8. MENDER: fast and scalable tissue structure identification in spatial omics**  
*Nature Communications*, 2024 · 49 citations · citation rank 83/182 · [10.1038/s41467-023-44367-9](https://doi.org/10.1038/s41467-023-44367-9)  
`multi-range neighbourhood context, no GNN training`

MENDER is the deliberate non-deep-learning entry: it identifies tissue structure by computing multi-range cell-type context vectors — the composition of each cell's neighbourhood at several radii — and clustering those directly, with no network to train. It runs orders of magnitude faster than the graph autoencoders above and is competitive on accuracy across many benchmarks. Every course on this topic should include it as the question it poses: how much of the performance of spatial GNNs comes from neighbourhood aggregation itself rather than from learning?

**9. SpaMask: dual masking graph autoencoder with contrastive learning**  
*PLOS Computational Biology*, 2025 · 46 citations · citation rank 85/182 · [10.1371/journal.pcbi.1012881](https://doi.org/10.1371/journal.pcbi.1012881)  
`masked graph modelling for spatial domains`

SpaMask applies masked graph autoencoding to spatial data, masking both node features and edges and training the model to reconstruct them, with a contrastive term supplying additional signal. Dual masking makes the representation robust to the dropout and irregular sampling typical of spatial platforms, since the model is trained under exactly that kind of corruption. It is one of two masked-autoencoder entries here — MAEST is the other — and the pairing shows how quickly the self-supervised recipes from general graph learning transfer.

**10. Identifying spatial domains via multi-view graph convolutional networks**  
*Briefings in Bioinformatics*, 2023 · 43 citations · citation rank 90/182 · [10.1093/bib/bbad278](https://doi.org/10.1093/bib/bbad278)  
`complementary graph views`

This method constructs complementary graph views of a spatial slide and runs multi-view graph convolution across them, with a consistency objective encouraging the views to agree on domain assignments while retaining view-specific detail. It is one of the clearer treatments of how to combine views without collapsing them into an average, which is the failure mode of naive multi-view fusion. Compare with Spatial-MGCN and SpaNCMG for three different fusion operators applied to the same idea.

**11. Latent feature extraction with a prior-based self-attention framework (PAST)**  
*Genome Research*, 2023 · 38 citations · outside the top 100 on citations · [10.1101/gr.277891.123](https://doi.org/10.1101/gr.277891.123)  
`Bayesian prior + self-attention for ST`

PAST introduces a Bayesian prior into spatial representation learning: a prior-based self-attention framework where reference data or known tissue structure constrains the latent space, letting the model borrow strength when a slide is small or shallowly sequenced. It also scales to large datasets through a memory-efficient attention implementation. The paper's argument — that spatial datasets are individually too sparse to learn from scratch and that priors, not more parameters, are the fix — anticipates the reference-based methods that follow it.

**12. STGNNks: cell types in spatial transcriptomics with GNN, denoising AE and k-sums**  
*Computers in Biology and Medicine*, 2023 · 37 citations · outside the top 100 on citations · [10.1016/j.compbiomed.2023.107440](https://doi.org/10.1016/j.compbiomed.2023.107440)  
`graph embedding + k-sums clustering`

STGNNks combines a graph neural network embedding with a denoising autoencoder and a k-sums clustering step, the last chosen because standard k-means is unstable on the elongated, unequal-density clusters that spatial embeddings produce. The clustering algorithm rather than the encoder is the contribution, which makes it a useful reminder that the final unsupervised step is often where spatial methods lose accuracy. Its benchmarks isolate that effect explicitly.

**13. Graph attention auto-encoder with contrastive learning for spatial domain recognition**  
*Communications Biology*, 2024 · 25 citations · outside the top 100 on citations · [10.1038/s42003-024-07037-0](https://doi.org/10.1038/s42003-024-07037-0)  
`augmentation-based contrastive spatial GAT`

A graph attention autoencoder with augmentation-based contrastive learning for spatial domain recognition: augmentations perturb the spatial graph and the feature matrix, and the contrastive objective requires the representation to be stable under those perturbations. The paper carefully studies which augmentations preserve biological meaning on spatial data — edge dropping is safe, feature shuffling less so — which is the practically useful part, since generic graph augmentations are often inappropriate here.

**14. ADEPT: autoencoder with differentially expressed genes and imputation for spatial clustering**  
*iScience*, 2023 · 23 citations · outside the top 100 on citations · [10.1016/j.isci.2023.106792](https://doi.org/10.1016/j.isci.2023.106792)  
`imputation-aware graph AE`

ADEPT interleaves imputation with clustering: it selects differentially expressed genes, imputes the expression matrix with a graph autoencoder, re-selects genes on the imputed matrix, and iterates, so gene selection and denoising improve one another. This targets a real problem — DE gene selection on raw sparse spatial data is noisy, and clustering inherits that noise — and the paper shows the iteration converging to more stable domains. It is the spatial analogue of the iterative refinement in scGNN.

**15. Scan-IT: domain segmentation of spatial transcriptomics images by graph neural network**  
*BMVC*, 2021 · 22 citations · outside the top 100 on citations · [10.5244/c.35.320](https://doi.org/10.5244/c.35.320)  
`BMVC spatial segmentation GNN`

Scan-IT, published at BMVC rather than in a biology venue, reformulates spatial domain identification as image segmentation over a graph derived from a deep spatial cell graph, importing conventions from the computer-vision segmentation literature. Its value in this list is partly as evidence of how the two communities approached the same task independently, and partly because its evaluation protocol is stricter than the biology-venue norm. Read it next to RESEPT.

**16. Graph deep learning enabled spatial domains identification**  
*Briefings in Bioinformatics*, 2023 · 21 citations · outside the top 100 on citations · [10.1093/bib/bbad146](https://doi.org/10.1093/bib/bbad146)  
`graph AE for spatial domain calling`

A graph autoencoder for spatial domain calling that focuses on the reconstruction objective and the construction of the neighbourhood graph rather than on adding architectural components, with careful ablations on neighbourhood radius and graph sparsity. Papers that isolate these choices are rare and useful: the results show that the neighbourhood definition often matters more than the encoder, which reframes how to read the rest of this sub-section.

**17. SpaNCMG: neighborhood-complementary mixed-view graph convolutional network**  
*Briefings in Bioinformatics*, 2024 · 21 citations · outside the top 100 on citations · [10.1093/bib/bbae259](https://doi.org/10.1093/bib/bbae259)  
`complementary near/far neighbourhood graphs`

SpaNCMG builds two complementary neighbourhood graphs — a near-neighbour graph capturing local tissue context and a farther-range graph capturing regional structure — and mixes their convolutions, so domains that are locally homogeneous but regionally distinct are separated correctly. It addresses the fixed-radius limitation shared by most methods above, in which one choice of neighbourhood must serve both fine boundaries and large domains. Compare with MENDER, which solves the same problem without learning.

**18. Attention-guided variational graph autoencoders reveal heterogeneity in spatial transcriptomics**  
*Briefings in Bioinformatics*, 2024 · 20 citations · outside the top 100 on citations · [10.1093/bib/bbae173](https://doi.org/10.1093/bib/bbae173)  
`attention-weighted VGAE`

An attention-guided variational graph autoencoder for spatial transcriptomics: attention weights the spatial neighbours during encoding while the variational formulation gives a distribution over each spot's embedding, so the model reports uncertainty about domain assignment. That uncertainty is genuinely useful at domain boundaries and in low-count regions, where every deterministic method returns a confident and often arbitrary label. It is one of the few spatial methods to take calibration seriously.

**19. Integrating multi-modal information to detect spatial domains by graph attention network**  
*Journal of Genetics and Genomics*, 2023 · 19 citations · outside the top 100 on citations · [10.1016/j.jgg.2023.06.005](https://doi.org/10.1016/j.jgg.2023.06.005)  
`histology + expression + location`

This method integrates three sources — gene expression, spatial coordinates and histology image features — in a graph attention network, learning how much each contributes per spot rather than combining them with a fixed weight as SpaGCN does. The ablation isolating the histology contribution is informative: image features help substantially on tissues with clear morphology and can hurt on tissues without it, which is worth knowing before adding an image encoder to a pipeline.

**20. SpaICL: image-guided curriculum graph contrastive learning for spatial clustering**  
*Briefings in Bioinformatics*, 2025 · 19 citations · outside the top 100 on citations · [10.1093/bib/bbaf433](https://doi.org/10.1093/bib/bbaf433)  
`curriculum learning on spatial graphs`

SpaICL applies curriculum learning to graph contrastive learning on spatial data, ordering training samples from easy to hard using histology-image guidance so the model learns confident regions first and ambiguous boundary regions later. Curriculum ordering is a rare idea in this literature and addresses a real difficulty: boundary spots are both the hardest and the most influential examples. The paper reports that the ordering, not the contrastive objective alone, produces most of the gain.

**21. MAEST: spatial domain detection with graph masked autoencoder**  
*Briefings in Bioinformatics*, 2025 · 19 citations · outside the top 100 on citations · [10.1093/bib/bbaf086](https://doi.org/10.1093/bib/bbaf086)  
`masked graph autoencoding`

MAEST brings graph masked autoencoding — the graph analogue of masked language modelling — to spatial domain detection: a large fraction of node features is masked and reconstructed, forcing the encoder to model spatial context rather than copy inputs. Masked modelling generally transfers better across slides than contrastive objectives, and the paper's cross-dataset experiments support this. Together with SpaMask it represents the current self-supervised state of the art in this sub-section.

**22. stAA: adversarial graph autoencoder for spatial clustering**  
*Briefings in Bioinformatics*, 2023 · 17 citations · outside the top 100 on citations · [10.1093/bib/bbad500](https://doi.org/10.1093/bib/bbad500)  
`adversarial regularization of the embedding`

stAA regularizes a spatial graph autoencoder adversarially: a discriminator forces the aggregated latent distribution to match a chosen prior, preventing the degenerate, highly anisotropic embeddings that plain graph autoencoders produce and that make downstream clustering unstable. It is the adversarial-autoencoder recipe applied to spatial data, and the paper's comparison against the same architecture without the adversarial term quantifies exactly what the regularization buys.

#### 4.2 Cell-type deconvolution

*Recovering per-spot cell-type composition from spot-level measurements, with the spatial graph as a regularizer on an ill-posed inverse problem.*

**1. SPACEL: deep-learning characterization of spatial transcriptome architectures**  
*Nature Communications*, 2023 · 125 citations · citation rank 45/182 · [10.1038/s41467-023-43220-3](https://doi.org/10.1038/s41467-023-43220-3)  
`deconvolution + domain + 3D alignment`

SPACEL is a three-module suite rather than a single model: Spoint deconvolves spot-level expression into cell-type proportions with a probabilistic model, Splane identifies spatial domains consistent across multiple slices using a graph network over the deconvolved compositions, and Scube aligns consecutive sections and reconstructs a 3D architecture. Doing deconvolution before domain identification means domains are defined by cell-type composition rather than by raw expression, which makes them comparable across slides and platforms. It is the most complete end-to-end spatial pipeline in this list.

**2. STdGCN: spatial transcriptomic cell-type deconvolution using graph convolutional networks**  
*Genome Biology*, 2024 · 43 citations · citation rank 92/182 · [10.1186/s13059-024-03353-0](https://doi.org/10.1186/s13059-024-03353-0)  
`expression + spatial GCN deconvolution`

STdGCN performs deconvolution on two graphs at once: an expression-similarity graph linking spots to pseudo-spots simulated from a single-cell reference, and a spatial adjacency graph linking real neighbouring spots. Message passing over the joint structure lets a spot's composition be informed both by transcriptionally similar reference mixtures and by what surrounds it physically, which regularizes the notoriously ill-posed deconvolution problem. The paper benchmarks against a large set of non-graph deconvolution tools and reports consistent gains from the spatial term.

**3. SD2: spatial transcriptomics deconvolution through dropout and spatial information**  
*Bioinformatics*, 2022 · 37 citations · outside the top 100 on citations · [10.1093/bioinformatics/btac605](https://doi.org/10.1093/bioinformatics/btac605)  
`dropout-aware graph deconvolution`

SD2 targets dropout in deconvolution: standard methods treat zeros in spot expression as low abundance, when many are technical, which biases proportion estimates toward cell types with high-expressing markers. It models dropout explicitly and uses spatial information to borrow strength from neighbouring spots when estimating the missing signal. The paper is a focused treatment of one well-defined bias rather than a general framework, which makes it a good illustration of how measurement-model detail affects downstream biological conclusions.

#### 4.3 Histology-to-expression prediction and resolution enhancement

*Predicting molecular measurements from morphology, and pushing spot-level data toward single-cell resolution. Read SEPAL's critique of the standard evaluation before trusting any reported correlation.*

**1. Spatial transcriptomics prediction from histology through Transformer and GNN (Hist2ST)**  
*Briefings in Bioinformatics*, 2022 · 212 citations · citation rank 29/182 · [10.1093/bib/bbac297](https://doi.org/10.1093/bib/bbac297)  
`image-to-expression with spot graph`

Hist2ST predicts spot-level gene expression from H&E histology by combining a convolutional module for within-patch image features, a transformer for long-range dependencies among patches, and a graph neural network over the spot adjacency graph to enforce spatial coherence of the predictions. The three-stage design reflects the structure of the problem: local morphology, global tissue context, and spatial smoothness are different kinds of information. It is the standard reference for image-to-expression prediction and the baseline the papers below compare against.

**2. Spatial transcriptomics gene expression prediction using exemplar guided GNN**  
*Pattern Recognition*, 2023 · 44 citations · citation rank 88/182 · [10.1016/j.patcog.2023.109966](https://doi.org/10.1016/j.patcog.2023.109966)  
`histology exemplars + spot graph`

This method guides expression prediction with exemplars: for a query histology patch it retrieves similar patches with known expression from a reference set and conditions the graph neural network's prediction on them, so the model interpolates among observed profiles rather than regressing from pixels alone. Retrieval-augmented prediction is well suited to a setting where the mapping from morphology to expression is many-to-one and poorly identified. It is a notably different inductive bias from Hist2ST's end-to-end regression.

**3. Transformer with convolution and graph-node co-embedding (TCGN)**  
*Medical Image Analysis*, 2023 · 42 citations · citation rank 94/182 · [10.1016/j.media.2023.103040](https://doi.org/10.1016/j.media.2023.103040)  
`interpretable expression prediction from histology`

TCGN co-embeds convolutional image features and graph nodes inside a transformer, so image patches and their spatial graph relations are processed in one attention mechanism rather than in sequential stages. Published in a medical imaging venue, it is stronger than most on evaluation rigour and on interpretability, tracing predictions back to image regions. Read it against Hist2ST as the "one joint architecture" alternative to the "three specialized modules" design.

**4. Inferring single-cell resolution spatial gene expression by fusing spots, location and histology with GCN**  
*Briefings in Bioinformatics*, 2024 · 20 citations · outside the top 100 on citations · [10.1093/bib/bbae630](https://doi.org/10.1093/bib/bbae630)  
`super-resolution via GCN`

This paper infers single-cell-resolution expression from spot-level data by fusing three sources in a graph convolutional network — the measured spot expression, the physical coordinates, and the histology image at sub-spot resolution — to distribute a spot's signal among the cells it contains. Super-resolution of this kind is under-determined and depends heavily on the image prior, which the paper is careful to acknowledge and to test by ablation. It is the natural companion to the deconvolution methods above.

**5. Integrating spatial transcriptomics and bulk RNA-seq with graph attention networks**  
*Briefings in Bioinformatics*, 2024 · 20 citations · outside the top 100 on citations · [10.1093/bib/bbae316](https://doi.org/10.1093/bib/bbae316)  
`enhanced-resolution expression prediction`

A graph attention approach that combines spatial transcriptomics with matched bulk RNA-seq, using the bulk profile — which is deep but unlocalized — to correct the shallow, noisy per-spot measurements while the spatial graph distributes the correction. Using bulk data as a depth prior for spatial data is an idea that deserves to be more common, since bulk profiles are cheap and often already available for the same tissue block.

**6. SEPAL: spatial gene expression prediction from local graphs**  
*ICCV Workshops*, 2023 · 17 citations · outside the top 100 on citations · [10.1109/iccvw60793.2023.00243](https://doi.org/10.1109/iccvw60793.2023.00243)  
`local-graph prediction head (ICCVW)`

SEPAL predicts spatial gene expression from local graph neighbourhoods, arguing that expression should be predicted relative to the tissue mean rather than in absolute terms, which removes the dominant and uninformative global component that inflates naive accuracy metrics. That evaluation point is the paper's main contribution and it applies to every method in this sub-section: reported correlations in the image-to-expression literature are frequently driven by mean expression rather than by spatial variation.

#### 4.4 Cross-slice alignment and 3D reconstruction

*Registration and matching — the unglamorous prerequisite for multi-slice and developmental work.*

**1. Integrating spatial transcriptomics across conditions, technologies and stages (STAligner)**  
*Nature Computational Science*, 2023 · 150 citations · citation rank 41/182 · [10.1038/s43588-023-00528-w](https://doi.org/10.1038/s43588-023-00528-w)  
`GAT + triplet loss for slice alignment`

STAligner integrates spatial slices across conditions, technologies and developmental stages by combining a graph attention autoencoder with a triplet loss over mutual nearest neighbours found between slices, so cells of the same type in different slices are pulled together while spatial structure within each slice is preserved. This makes it possible to compare domains between a healthy and a diseased section, or across time points, in a common embedding. It is the reference method for multi-slice spatial integration and a prerequisite for any developmental atlas work.

**2. Search and match across spatial omics samples at single-cell resolution (SANTO)**  
*Nature Methods*, 2024 · 48 citations · citation rank 84/182 · [10.1038/s41592-024-02410-7](https://doi.org/10.1038/s41592-024-02410-7)  
`cross-sample alignment and matching`

SANTO performs coarse-to-fine alignment and matching between spatial omics samples at single-cell resolution, handling both rigid transformations between serial sections and non-rigid deformation from tissue handling. It supports two distinct tasks — aligning slices into a 3D stack, and matching cells between samples measured on different platforms — under one optimization. Registration is the unglamorous prerequisite for most multi-slice analysis, and this is the most careful treatment of it in the list.

#### 4.5 Spatial multi-omic integration

*Slides carrying two or more modalities at the same coordinates. The most direct antecedents of this project's own problem.*

**1. Deciphering spatial domains from spatial multi-omics with SpatialGlue**  
*Nature Methods*, 2024 · 197 citations · citation rank 32/182 · [10.1038/s41592-024-02316-4](https://doi.org/10.1038/s41592-024-02316-4)  
`dual-attention across modality + spatial graphs`

SpatialGlue is the reference method for spatial multi-omics — slides on which expression and chromatin, or expression and protein, are measured at the same coordinates. It builds two graphs per modality, a spatial-proximity graph and a feature-similarity graph, and uses a dual-attention mechanism that first integrates within each modality and then across modalities, so the model learns both which neighbours matter and which modality is informative at each spot. The two-level attention is the contribution: single-level fusion cannot separate spatial smoothing from modality weighting. Benchmarked on Stereo-CITE-seq, SPOTS and spatial epigenome-transcriptome data.

**2. Cooperative integration of spatially resolved multi-omics data with COSMOS**  
*Nature Communications*, 2025 · 51 citations · citation rank 81/182 · [10.1038/s41467-024-55204-y](https://doi.org/10.1038/s41467-024-55204-y)  
`graph-based joint spatial multi-omic embedding`

COSMOS produces a joint embedding of spatially resolved multi-omic data through cooperative graph learning, in which modality-specific graph encoders exchange information during training instead of being fused only at the output. The cooperative scheme lets a modality with strong spatial structure inform the segmentation of a modality with weaker signal. The paper demonstrates spatial domain identification and trajectory analysis on spatial multi-omic slides where single-modality methods disagree with each other.

**3. Multi-omics integration for single-cell and spatially resolved data via dual-path graph attention auto-encoder**  
*Briefings in Bioinformatics*, 2024 · 31 citations · outside the top 100 on citations · [10.1093/bib/bbae450](https://doi.org/10.1093/bib/bbae450)  
`dual-path GAT for paired modalities`

A dual-path graph attention autoencoder for paired single-cell and spatially resolved multi-omics: two parallel attention paths, one per modality, with cross-path connections that let each modality's encoder attend to the other's intermediate representations. It is a lighter-weight alternative to SpatialGlue with a similar goal, and its ablations on the cross-path connections quantify how much cross-modal message passing contributes relative to independent encoding plus late fusion.

**4. MultiGATE: integrative analysis and regulatory inference in spatial multi-omics**  
*Nature Communications*, 2025 · 18 citations · outside the top 100 on citations · [10.1038/s41467-025-63418-x](https://doi.org/10.1038/s41467-025-63418-x)  
`graph attention across spatial modalities`

MultiGATE performs integration and regulatory inference jointly on spatial multi-omics, using graph attention across spatial and modality graphs to produce a shared embedding while simultaneously estimating peak-to-gene regulatory links that are specific to spatial domains. Making regulatory inference spatially aware is the novel part — regulatory relationships are usually estimated globally and then mapped onto space, which misses domain-specific regulation. It is one of the most recent and most ambitious entries in this section.

**5. Mosaic integration of spatial multi-omics with SpaMosaic**  
*Nature Genetics*, 2026 · 16 citations · outside the top 100 on citations · [10.1038/s41588-026-02573-3](https://doi.org/10.1038/s41588-026-02573-3)  
`mosaic (partially overlapping) spatial modalities`

SpaMosaic handles the mosaic case for spatial multi-omics: a study in which different slides carry different, only partially overlapping modality combinations. It uses the shared modalities as bridges to build a graph linking cells across slides, then learns a common embedding through contrastive learning, allowing missing modalities to be imputed spatially. Mosaic designs are increasingly what real experiments look like, since profiling every modality on every section is rarely affordable, and this is the first method built directly for that reality.

#### 4.6 Tissue-level phenotype and clinical prediction

*Graph-level rather than node-level tasks: predict an outcome for the tissue from the organization of its cells.*

**1. Graph deep learning for tumour microenvironments from spatial protein profiles (SPACE-GM)**  
*Nature Biomedical Engineering*, 2022 · 155 citations · citation rank 40/182 · [10.1038/s41551-022-00951-w](https://doi.org/10.1038/s41551-022-00951-w)  
`cellular-neighbourhood graphs from imaging`

SPACE-GM builds cellular-neighbourhood graphs from spatial protein imaging — each cell a node, edges to physical neighbours, node features from marker intensities — and trains a graph neural network to predict patient-level outcomes such as recurrence and response from the microenvironment structure. The prediction target is the tissue, not the cell, so the model must learn which cellular neighbourhood motifs are prognostic; the learned motifs are then extracted and interpreted. This is the clearest example in the list of graph learning as a tool for tumour-microenvironment biology rather than for data processing.

**2. Spatially resolved transcriptomics and graph-based deep learning for CNS tumor diagnostics**  
*Nature Cancer*, 2025 · 30 citations · outside the top 100 on citations · [10.1038/s43018-024-00904-z](https://doi.org/10.1038/s43018-024-00904-z)  
`clinical deployment of a spatial GNN`

A clinical deployment of spatial graph deep learning: spatially resolved transcriptomics combined with a graph-based model for central nervous system tumour diagnostics, evaluated against neuropathological diagnosis. It is included here less for methodological novelty than because it is one of the few papers in this literature that confronts the requirements of clinical use — reproducibility across sites, turnaround time, and failure-mode analysis. Any chapter arguing that these methods matter clinically should cite it and read its limitations section carefully.

**3. Graph neural networks learn emergent tissue properties from spatial molecular profiles**  
*Nature Communications*, 2025 · 17 citations · outside the top 100 on citations · [10.1038/s41467-025-63758-8](https://doi.org/10.1038/s41467-025-63758-8)  
`tissue-level emergent property prediction`

This paper asks whether graph neural networks trained on spatial molecular profiles learn *emergent* tissue-level properties — features of the tissue that are not present in any individual cell but arise from cellular organization. The authors design experiments to distinguish genuine emergent structure from aggregated single-cell signal, which is a methodological question most application papers skip. It is the most conceptually interesting entry in this sub-section and a good discussion piece on what these models actually learn.

#### 4.7 Segmentation and spatially variable genes

*Upstream and downstream of everything else in this section, and both more consequential than their citation counts suggest.*

**1. STAMarker: spatial domain-specific variable genes with saliency maps**  
*Nucleic Acids Research*, 2023 · 49 citations · citation rank 82/182 · [10.1093/nar/gkad801](https://doi.org/10.1093/nar/gkad801)  
`saliency over STAGATE-style encoder`

STAMarker identifies spatial-domain-specific variable genes using saliency maps computed from a STAGATE-style graph attention encoder: after domains are identified, gradients of the domain assignment with respect to input genes rank each gene's contribution. This is a different definition of "spatially variable" from the usual spatial-autocorrelation statistics — it asks which genes the model used to draw a boundary, not which genes vary smoothly in space — and the two often disagree, which the paper discusses. A good introduction to attribution methods applied to spatial models.

**2. Segger: fast and accurate cell segmentation of imaging-based spatial transcriptomics**  
*preprint (bioRxiv)*, 2025 · 17 citations · outside the top 100 on citations · [10.1101/2025.03.14.643160](https://doi.org/10.1101/2025.03.14.643160)  
`transcript-level graph segmentation; preprint`

Segger performs cell segmentation for imaging-based spatial transcriptomics — Xenium, MERFISH, CosMx — by treating individual transcripts as nodes in a heterogeneous graph with nuclei, and framing segmentation as a transcript-to-cell assignment problem solved by graph neural network link prediction. This replaces the standard approach of expanding nuclear masks by a fixed radius, which systematically misassigns transcripts in dense tissue. Segmentation errors propagate into every downstream analysis, so this is a more consequential paper than its citation count suggests.

---

### 5. Cell-cell communication

Inferring which cells signal to which, and through what. The graph methods here differ from the familiar ligand-receptor scoring tools in being spatially aware and, in several cases, supervised — which lets them propose interactions rather than only rank known ones.

**1. GCNG: graph convolutional networks for inferring gene interaction from spatial transcriptomics**  
*Genome Biology*, 2020 · 204 citations · citation rank 31/182 · [10.1186/s13059-020-02214-w](https://doi.org/10.1186/s13059-020-02214-w)  
`supervised extracellular gene interaction`

GCNG reframes cell–cell communication as supervised learning rather than as ligand–receptor score aggregation. It builds a graph over cells from spatial coordinates, places expression on the nodes, and trains a graph convolutional network to classify whether a given gene pair is an extracellular interacting pair, using known ligand–receptor databases as labels. Because the model is supervised and spatially aware, it can propose novel interacting pairs and can distinguish extracellular signalling from intracellular co-expression — a distinction that correlation-based methods cannot make. It is the foundational paper for graph-based communication inference.

**2. Deciphering cell-cell communication at single-cell resolution with subgraph-based graph attention**  
*Nature Communications*, 2024 · 73 citations · citation rank 66/182 · [10.1038/s41467-024-51329-2](https://doi.org/10.1038/s41467-024-51329-2)  
`subgraph GAT over ligand-receptor edges`

This method decodes communication at single-cell resolution using subgraph-based graph attention: for each candidate sending–receiving cell pair it extracts the local subgraph of ligand–receptor edges and scores it with an attention network, so the inference is about a specific pair in a specific neighbourhood rather than about average signalling between cell types. This resolves the main limitation of cluster-level tools such as CellPhoneDB and CellChat, which cannot say which individual cells are actually communicating. The attention weights also identify which ligand–receptor pairs drive each interaction.

**3. De novo reconstruction of cell interaction landscapes with DeepLinc**  
*Genome Biology*, 2022 · 69 citations · citation rank 69/182 · [10.1186/s13059-022-02692-0](https://doi.org/10.1186/s13059-022-02692-0)  
`VGAE recovers unseen cell interactions`

DeepLinc reconstructs the cell interaction landscape *de novo* with a variational graph autoencoder trained on the observed spatial adjacency graph, then uses the model's link-prediction scores to propose interactions not present in the observed graph — including long-range signalling between cells that are not physically adjacent. Treating the observed proximity graph as an incomplete sample of a latent interaction graph is the conceptual move worth teaching, and it generalizes well beyond this application. The paper also filters predictions by expression evidence to control false positives.

**4. CLARIFY: cell-cell interaction and GRN refinement from spatial transcriptomics**  
*Bioinformatics*, 2023 · 43 citations · citation rank 91/182 · [10.1093/bioinformatics/btad269](https://doi.org/10.1093/bioinformatics/btad269)  
`nested graphs, cell graph + gene graph`

CLARIFY runs two nested graphs simultaneously: a cell-level graph capturing spatial interactions, and per-cell gene regulatory graphs, with information flowing between the levels so that inferred communication refines the intracellular networks and vice versa. This closes a loop that most tools leave open — signalling received by a cell should change its regulatory state, and its regulatory state determines what it can send. The nested-graph formulation is elegant and computationally demanding, and the paper is honest about the scale limits.

**5. Learning cell communication from spatial graphs of cells (NCEM)**  
*preprint (bioRxiv)*, 2021 · 36 citations · outside the top 100 on citations · [10.1101/2021.07.11.451750](https://doi.org/10.1101/2021.07.11.451750)  
`node-centric expression models; preprint`

NCEM (node-centric expression models) asks a sharper question than most communication tools: how much of a cell's expression variance is explained by the identity of its spatial neighbours? It fits models predicting each cell's expression from its own type and its neighbourhood composition, using graph neural networks for the non-linear variants, and reports the variance attributable to the niche. Framing communication as a variance-decomposition problem gives a quantitative, falsifiable answer rather than a ranked list of ligand–receptor pairs. Still a preprint, but widely used and conceptually important.

---

### 6. Gene regulatory network inference

GRN inference posed as link prediction on a graph. The recurring difficulties are directionality, extreme class imbalance, and the fact that transcript co-expression is a weak proxy for regulation — which is the argument for the chromatin-based methods in this section.

**1. Inferring gene regulatory network from single-cell transcriptomes with graph autoencoder**  
*PLOS Genetics*, 2023 · 146 citations · citation rank 42/182 · [10.1371/journal.pgen.1010942](https://doi.org/10.1371/journal.pgen.1010942)  
`VGAE link prediction on TF-gene graph`

This paper casts GRN inference as link prediction on a graph autoencoder: candidate transcription factor–target edges are scored by the decoder of a variational graph autoencoder trained on an initial network built from expression, so the model learns network topology rather than only pairwise association. The formulation naturally incorporates prior known edges as supervision and handles the extreme class imbalance of GRN inference — real edges are a tiny fraction of candidate pairs — through negative sampling. A clear and well-written introduction to the graph-autoencoder view of regulatory inference.

**2. Graph attention network for link prediction of gene regulations from scRNA-seq (GENELink)**  
*Bioinformatics*, 2022 · 115 citations · citation rank 47/182 · [10.1093/bioinformatics/btac559](https://doi.org/10.1093/bioinformatics/btac559)  
`GAT on candidate regulatory graph`

GENELink applies graph attention to link prediction on a candidate regulatory graph, learning low-dimensional representations of genes such that regulatory relationships correspond to a learned asymmetric scoring function. The asymmetry matters: regulation is directional, and symmetric embeddings cannot represent it, which is a defect of several earlier methods. GENELink is one of the most-used GNN baselines for GRN inference from scRNA-seq and is evaluated on the standard BEELINE benchmark suite.

**3. Predicting gene regulatory links from scRNA-seq data using graph neural networks (GNNLink)**  
*Briefings in Bioinformatics*, 2023 · 79 citations · citation rank 60/182 · [10.1093/bib/bbad414](https://doi.org/10.1093/bib/bbad414)  
`transformer-augmented GCN encoder`

GNNLink augments a graph convolutional encoder with a transformer component so that dependencies between genes far apart in the candidate network can be modelled, rather than only those within a few message-passing hops. Regulatory cascades are exactly the kind of long-range structure that shallow GNNs miss, so the combination is well motivated. The paper evaluates across several benchmark networks and cell types and reports where the transformer helps and where it merely adds parameters.

**4. Inferring transcription factor regulatory networks from scATAC-seq based on GNNs**  
*Nature Machine Intelligence*, 2022 · 73 citations · citation rank 65/182 · [10.1038/s42256-022-00469-5](https://doi.org/10.1038/s42256-022-00469-5)  
`peak-gene graph for TF networks`

This work infers transcription factor regulatory networks directly from scATAC-seq using a graph neural network over a peak–gene graph, so the evidence for an edge is chromatin accessibility at a motif-containing regulatory element linked to a target gene, rather than transcript co-expression. Working in the accessibility domain sidesteps the fundamental weakness of expression-based GRN inference, where correlation between a TF's mRNA and its targets is often absent because regulation is post-transcriptional. It is the strongest argument in this section for multi-omic rather than transcriptome-only regulatory inference.

**5. scMGATGRN: multiview graph attention network for GRN inference**  
*Briefings in Bioinformatics*, 2024 · 63 citations · citation rank 71/182 · [10.1093/bib/bbae526](https://doi.org/10.1093/bib/bbae526)  
`multi-view attention over cell/gene graphs`

scMGATGRN uses multiple views — cell-level and gene-level graphs — with graph attention on each, then fuses them for regulatory link prediction, on the reasoning that the cell graph tells you which cells share a regulatory programme while the gene graph tells you which genes co-vary within it. Multi-view fusion here plays the same role as in the spatial section: it avoids committing to a single, necessarily lossy graph construction. Among the better-benchmarked recent GRN methods.

**6. Gene knockout inference with variational graph autoencoder (scTenifoldKnk-style)**  
*Nucleic Acids Research*, 2023 · 45 citations · citation rank 87/182 · [10.1093/nar/gkad450](https://doi.org/10.1093/nar/gkad450)  
`VGAE for perturbation inference`

This paper uses a variational graph autoencoder to predict the transcriptome-wide consequences of a gene knockout: the regulatory network is embedded, the target node is removed or perturbed, and the model's reconstruction of the remaining network reveals the propagated effect. Perturbation prediction from observational data alone is a strong claim, and the paper validates against real knockout experiments. It connects the GRN literature to the perturbation-prediction problem that foundation models are also aimed at.

**7. GMFGRN: matrix factorization and graph neural network for GRN inference**  
*Briefings in Bioinformatics*, 2023 · 35 citations · outside the top 100 on citations · [10.1093/bib/bbad529](https://doi.org/10.1093/bib/bbad529)  
`factorization-initialized GNN`

GMFGRN initializes gene representations by matrix factorization of the cell–gene expression matrix and then refines them with a graph neural network before predicting regulatory links. The factorization step gives the GNN a sensible starting point and reduces the sensitivity to random initialization that makes many GRN methods hard to reproduce. It is a small, practical idea, well supported by ablations, and worth knowing for any task where a GNN must be trained on a graph built from noisy data.

**8. Deciphering driver regulators of cell fate decisions with CEFCON**  
*Nature Communications*, 2023 · 25 citations · outside the top 100 on citations · [10.1038/s41467-023-44103-3](https://doi.org/10.1038/s41467-023-44103-3)  
`network control theory on scRNA-seq GRNs`

CEFCON applies network control theory to single-cell-derived regulatory networks: after constructing a context-specific GRN, it identifies the minimal set of driver nodes whose control is sufficient to steer the system between cell fates, using a graph attention encoder to learn the network representation and a controllability objective to rank regulators. Importing control theory gives a principled definition of "driver regulator" in place of the usual centrality heuristics. One of the more mathematically substantial papers in this list.

**9. GRACE: causal mechanistic graph neural networks for GRN inference**  
*IEEE Trans. Neural Netw. Learn. Syst.*, 2024 · 19 citations · outside the top 100 on citations · [10.1109/tnnls.2024.3412753](https://doi.org/10.1109/tnnls.2024.3412753)  
`causal mechanism modelling`

GRACE builds causal mechanism modelling into a graph neural network for GRN inference, aiming to distinguish causal regulatory edges from merely correlated ones by encoding structural assumptions about how regulation propagates. Causal identifiability from observational single-cell data is limited, and the paper is explicit about the assumptions required. It is the right entry point for a discussion of what "inferring a regulatory network" can and cannot mean without interventions.

---

### 7. Patient-level multi-omics

A parallel literature that developed largely independently of the single-cell one, on bulk cohorts such as TCGA. The unit of analysis is the patient or the gene rather than the cell, sample sizes are in the hundreds rather than the hundreds of thousands, and prior biological networks carry correspondingly more of the load. Worth reading precisely because the constraints are different.

#### 7.1 Patient-similarity networks: classification and subtyping

*Patients as nodes, omics as node features, subtype as label. The design question throughout is where fusion across modalities should happen: at the feature level, the graph level, or the label level.*

**1. MOGONET integrates multi-omics data using graph convolutional networks**  
*Nature Communications*, 2021 · 660 citations · citation rank 16/182 · [10.1038/s41467-021-23774-w](https://doi.org/10.1038/s41467-021-23774-w)  
`per-omic GCN + view correlation discovery`

MOGONET is the template for graph-based patient classification and the most cited method in this section. For each omics layer it builds a patient-similarity network — patients are nodes, edges connect similar profiles — and trains a separate graph convolutional network, then fuses the per-omic predictions with a View Correlation Discovery Network that learns from the *cross-omics label correlation tensor* rather than by averaging or concatenating. Late fusion at the label level lets each modality keep its own graph and its own noise characteristics. The paper also extracts the features driving each prediction, which is how most follow-ups justify their biomarker claims.

**2. MoGCN: a multi-omics integration method based on GCN for cancer subtype analysis**  
*Frontiers in Genetics*, 2022 · 166 citations · citation rank 38/182 · [10.3389/fgene.2022.806842](https://doi.org/10.3389/fgene.2022.806842)  
`AE fusion + patient similarity GCN`

MoGCN first compresses each omics layer with an autoencoder, then builds a patient similarity network by similarity network fusion over the compressed representations, and finally classifies subtypes with a GCN on the fused graph. Separating dimensionality reduction from graph construction from classification makes each stage easy to swap and to diagnose, which is why this pipeline shape recurs throughout the sub-section. The paper's ablations on the fusion step are the useful part: naive concatenation of omics layers performs noticeably worse than similarity network fusion.

**3. A multimodal graph neural network framework for cancer molecular subtype classification**  
*BMC Bioinformatics*, 2024 · 77 citations · citation rank 63/182 · [10.1186/s12859-023-05622-4](https://doi.org/10.1186/s12859-023-05622-4)  
`supervised + contrastive multi-omic GNN`

A multimodal GNN for cancer molecular subtype classification that combines supervised classification with a contrastive objective over the patient graph, so representations of patients of the same subtype are pulled together even when their raw omics profiles differ. The contrastive term acts as a regularizer against the very small sample sizes typical of TCGA subtype problems — a few hundred patients against tens of thousands of features. It is a good illustration of self-supervision used to counter overfitting rather than to avoid labels.

**4. MOGAT: multi-omics integration using graph attention networks for cancer subtype prediction**  
*Int. J. Molecular Sciences*, 2024 · 70 citations · citation rank 67/182 · [10.3390/ijms25052788](https://doi.org/10.3390/ijms25052788)  
`per-omic GAT with attention fusion`

MOGAT replaces MOGONET's graph convolutions with graph attention and fuses modalities with an attention mechanism, so both the influence of each neighbouring patient and the contribution of each omics layer are learned rather than fixed. The attention weights over modalities give a per-patient readout of which data type drove the prediction, which is clinically more interesting than a global feature importance. Read it directly after MOGONET as the attention variant of the same design.

**5. Semi-supervised multi-omics integration with transformer self-attention and GCN**  
*BMC Genomics*, 2024 · 63 citations · citation rank 72/182 · [10.1186/s12864-024-09985-7](https://doi.org/10.1186/s12864-024-09985-7)  
`transformer fusion + patient graph`

This method uses transformer self-attention to fuse omics features within each patient and a graph convolutional network to propagate over the patient similarity graph, trained semi-supervised so unlabelled patients contribute to the representation. The combination addresses the two structural facts of multi-omic cohorts: features interact within a sample, and samples are related to one another. Semi-supervised training is the practically relevant part, since labelled cohorts are small while unlabelled molecular profiles are comparatively plentiful.

**6. Graph neural networks with multiple prior knowledge for multi-omics data analysis**  
*IEEE J. Biomed. Health Inform.*, 2023 · 58 citations · citation rank 73/182 · [10.1109/jbhi.2023.3284794](https://doi.org/10.1109/jbhi.2023.3284794)  
`pathway/PPI priors as graph structure`

Rather than deriving the graph from the data, this method injects multiple sources of prior knowledge — pathway membership, protein–protein interactions, regulatory relations — as graph structure, and learns how much to trust each prior. Using priors as structure rather than as post-hoc annotation is the recurring argument of this sub-section, and here it is tested systematically across several priors, which most papers do not do. The finding that priors help most when sample size is smallest is intuitive and well supported.

**7. Multi-omics integration using adaptive graph learning and attention**  
*Computers in Biology and Medicine*, 2023 · 55 citations · citation rank 77/182 · [10.1016/j.compbiomed.2023.107303](https://doi.org/10.1016/j.compbiomed.2023.107303)  
`learns the patient graph jointly`

This model learns the patient graph jointly with the classifier through adaptive graph learning, instead of fixing it beforehand with a k-nearest-neighbour rule on a chosen distance. Since a patient similarity graph built on noisy high-dimensional omics data is at best a rough guess, making it a trainable object is well motivated, and the paper shows the learned graph diverging substantially from the kNN initialization. Compare with the fixed-graph methods above to see how much of their error is attributable to the graph.

**8. Molecular subtyping of cancer based on robust GNN and multi-omics integration**  
*Frontiers in Genetics*, 2022 · 40 citations · citation rank 99/182 · [10.3389/fgene.2022.884028](https://doi.org/10.3389/fgene.2022.884028)  
`noise-robust patient graph`

A subtyping model designed for robustness: it uses a noise-tolerant construction of the patient graph and a training procedure that down-weights unreliable edges, motivated by the observation that a handful of spurious edges between subtypes can dominate message passing and corrupt predictions for whole neighbourhoods. The stress tests with deliberately corrupted graphs are the informative experiments and are rarely reported elsewhere in this literature.

**9. Global and cross-modal feature aggregation for multi-omics classification and drug response**  
*Information Fusion*, 2023 · 39 citations · outside the top 100 on citations · [10.1016/j.inffus.2023.102077](https://doi.org/10.1016/j.inffus.2023.102077)  
`cross-modal attention fusion`

This paper performs global and cross-modal feature aggregation with attention for both multi-omics classification and drug response prediction, allowing features from one modality to attend directly to features in another before any patient-level graph is built. Feature-level cross-modal attention is a genuinely different fusion point from the label-level fusion of MOGONET, and the paper compares fusion depths systematically, which makes it the best single reference for the early-versus-late fusion question in this setting.

**10. Prior knowledge-guided multilevel GNN for tumor risk prediction**  
*Briefings in Bioinformatics*, 2024 · 39 citations · outside the top 100 on citations · [10.1093/bib/bbae184](https://doi.org/10.1093/bib/bbae184)  
`gene-pathway-patient hierarchy`

A multilevel GNN that models a gene–pathway–patient hierarchy explicitly, propagating information up from genes to the pathways containing them and then to patients, so predictions are grounded in pathway-level intermediates that a clinician can inspect. Building the biological hierarchy into the architecture rather than recovering it from attention is the design choice worth discussing; it constrains the model but makes its intermediate representations meaningful by construction.

**11. DeepMoIC: multi-omics integration via deep graph convolutional networks for subtyping**  
*BMC Genomics*, 2024 · 37 citations · outside the top 100 on citations · [10.1186/s12864-024-11112-5](https://doi.org/10.1186/s12864-024-11112-5)  
`deep GCN with residual connections`

DeepMoIC applies a deep graph convolutional network with residual connections to multi-omics subtyping, testing whether the additional depth that residual connections permit actually helps on patient graphs. The answer is a qualified yes — a few extra layers help on large cohorts and hurt on small ones — which is a useful concrete data point in the over-smoothing discussion, since patient graphs are small and dense and behave differently from the citation networks where those methods were developed.

**12. MRGCN: cancer subtyping with multi-reconstruction graph convolutional network**  
*Bioinformatics*, 2023 · 35 citations · outside the top 100 on citations · [10.1093/bioinformatics/btad353](https://doi.org/10.1093/bioinformatics/btad353)  
`handles partial multi-omic data`

MRGCN handles the common and awkward case in which not every patient has every omics layer measured. Instead of dropping incomplete samples or imputing whole modalities, it uses a multi-reconstruction objective in which the model learns to reconstruct missing views from available ones while performing subtyping. Cohorts with complete multi-omic coverage are the exception, so this is one of the more practically important methods in the sub-section.

**13. SUPREME: multiomics data integration using graph convolutional networks**  
*NAR Genomics and Bioinformatics*, 2023 · 34 citations · outside the top 100 on citations · [10.1093/nargab/lqad063](https://doi.org/10.1093/nargab/lqad063)  
`stacked patient-similarity GCNs`

SUPREME builds a separate GCN per omics-derived patient similarity network and then combines their node embeddings by exhaustively evaluating combinations of networks, reporting which subsets of data types actually contribute for a given prediction task. The systematic combination search is the contribution: most papers assume more modalities are better, and SUPREME shows that for many endpoints a subset performs as well or better. It also releases a clean, reusable implementation.

**14. Cancer molecular subtype classification by graph convolutional networks on multi-omics data**  
*ACM-BCB*, 2021 · 33 citations · outside the top 100 on citations · [10.1145/3459930.3469542](https://doi.org/10.1145/3459930.3469542)  
`ACM-BCB precursor of later GCN subtyping`

The ACM-BCB conference paper that introduced graph convolution on multi-omics data for cancer molecular subtype classification, and the direct precursor of several journal methods above. It is short and its performance has been superseded, but it establishes the problem formulation — patients as nodes, omics as features, subtype as node label — that the rest of this sub-section inherits. Cite it for priority and read it for its clarity.

**15. Cancer subtype identification by consensus guided graph autoencoders**  
*Bioinformatics*, 2021 · 32 citations · outside the top 100 on citations · [10.1093/bioinformatics/btab535](https://doi.org/10.1093/bioinformatics/btab535)  
`consensus clustering guides the GAE`

This method guides a graph autoencoder with consensus clustering: base clusterings of the multi-omic data are aggregated into a consensus, which then supervises the graph autoencoder's latent space, so the discovered subtypes are those stable across clustering algorithms rather than artefacts of one. Subtype discovery is unsupervised and notoriously irreproducible across studies, which makes stability-based supervision a sensible target. It is the patient-level twin of the consensus-guided scRNA-seq model in Section 2.1.

**16. Deep learning on graphs for multi-omics classification of COPD**  
*PLOS ONE*, 2023 · 31 citations · outside the top 100 on citations · [10.1371/journal.pone.0284563](https://doi.org/10.1371/journal.pone.0284563)  
`non-cancer multi-omic GNN application`

A multi-omic GNN applied to chronic obstructive pulmonary disease rather than cancer — a useful corrective, since the patient-level graph literature is overwhelmingly oncological and inherits assumptions from TCGA that do not hold elsewhere. The paper discusses what changes when subtypes are less discrete and cohorts less deeply profiled. Worth including for breadth even though its methodological content is modest.

**17. Classifying breast cancer using multi-view graph neural network on multi-omics data**  
*Frontiers in Genetics*, 2024 · 27 citations · outside the top 100 on citations · [10.3389/fgene.2024.1363896](https://doi.org/10.3389/fgene.2024.1363896)  
`view-specific GNN encoders`

A multi-view GNN for breast cancer classification with view-specific encoders and a fusion layer, evaluated on the standard TCGA-BRCA subtype task. It is a representative rather than novel method, and its main use in a reading list is as a comparison point: read it beside MOGONET, MOGAT and MultiGATAE on the same dataset to see how narrow the performance differences among these architectures actually are.

**18. omicsGAT: graph attention network for cancer subtype analyses**  
*Int. J. Molecular Sciences*, 2022 · 26 citations · outside the top 100 on citations · [10.3390/ijms231810220](https://doi.org/10.3390/ijms231810220)  
`attention over sample similarity graph`

omicsGAT applies graph attention over a sample-similarity graph for cancer subtype and clinical outcome analysis, with the attention coefficients interpreted as which other patients a prediction relies on. Patient-level attention is an appealing form of interpretability — the model effectively cites comparable cases — and the paper explores this explicitly rather than only reporting accuracy.

**19. iSOM-GSN: transforming multi-omic data into gene similarity networks via self-organizing maps**  
*Bioinformatics*, 2020 · 24 citations · outside the top 100 on citations · [10.1093/bioinformatics/btaa500](https://doi.org/10.1093/bioinformatics/btaa500)  
`graph construction as the integration step`

iSOM-GSN transforms multi-omic feature vectors into gene similarity networks using self-organizing maps, and then applies a convolutional network to the resulting two-dimensional representation. The point of interest is that graph construction, not the predictor, is the method: the SOM arranges genes so that related genes are spatially adjacent, converting a graph problem into an image problem. An unusual approach and a good discussion piece on representation choice.

**20. Attention-based GCN integrates multi-omics for breast cancer subtypes**  
*Briefings in Functional Genomics*, 2023 · 24 citations · outside the top 100 on citations · [10.1093/bfgp/elad013](https://doi.org/10.1093/bfgp/elad013)  
`patient-specific gene marker attribution`

An attention-based GCN for breast cancer subtyping whose emphasis is patient-specific gene marker attribution: for each individual the model reports which genes drove the assignment, rather than reporting one global importance ranking. Per-patient attribution is what a clinical setting actually requires, and the paper's validation of those attributions against known subtype markers is reasonably careful.

**21. Supervised graph contrastive learning for cancer subtype identification**  
*Health Inf. Sci. Syst.*, 2024 · 22 citations · outside the top 100 on citations · [10.1007/s13755-024-00274-x](https://doi.org/10.1007/s13755-024-00274-x)  
`label-aware contrastive multi-omic GNN`

Supervised graph contrastive learning for cancer subtype identification: labels define positive and negative pairs so that the contrastive objective aligns with the classification target, avoiding the false-negative problem that afflicts unsupervised contrastive learning when classes are few. Given how small these cohorts are, using labels twice — once in the contrastive term, once in the classifier — is a sensible use of scarce supervision.

**22. MODILM: complex disease classification via multi-omics data integration learning**  
*BMC Med Inform Decis Mak*, 2023 · 19 citations · outside the top 100 on citations · [10.1186/s12911-023-02173-9](https://doi.org/10.1186/s12911-023-02173-9)  
`GAT-based multi-omic classifier`

MODILM builds a graph attention-based classifier for complex disease from multi-omics, constructing per-modality patient graphs and integrating them with attention. Published in a medical informatics venue, it is framed around clinical decision support and pays more attention to calibration and to the reporting of uncertainty than the bioinformatics-venue equivalents. Useful for that framing as much as for the method.

**23. MultiGATAE: cancer subtype identification based on multi-omics and attention**  
*Frontiers in Genetics*, 2022 · 18 citations · outside the top 100 on citations · [10.3389/fgene.2022.855629](https://doi.org/10.3389/fgene.2022.855629)  
`graph attention autoencoder subtyping`

MultiGATAE combines graph attention with an autoencoder objective for cancer subtype identification, so the model is trained both to reconstruct the omics profiles and to separate subtypes, which regularizes the representation when labels are scarce. It sits in the same family as MoGCN and MOGAT and is best read as one point in a systematic comparison of autoencoder, attention and convolution variants over the same problem.

#### 7.2 Cancer gene and driver gene discovery on biological networks

*Genes as nodes on a protein interaction or pathway network, with multi-omic node features. Label scarcity and ascertainment bias in known driver sets are the central methodological problems.*

**1. Integration of multiomics data with GCN to identify new cancer genes (EMOGI)**  
*Nature Machine Intelligence*, 2021 · 270 citations · citation rank 25/182 · [10.1038/s42256-021-00325-y](https://doi.org/10.1038/s42256-021-00325-y)  
`GCN over PPI with multi-omic node features`

EMOGI is the reference method for this sub-section. It runs a graph convolutional network over a protein–protein interaction network in which each gene node carries multi-omic features — mutation rate, copy number, DNA methylation, expression across cancer types — and is trained to classify genes as cancer drivers. Because the label set is small and known drivers are heavily studied, the model's value is in what it predicts *outside* the training set, and the authors use layer-wise relevance propagation to explain each prediction in terms of both the omics features and the network neighbourhood. This "multi-omic node features on a prior interaction network" formulation is the one nearly every paper below adopts.

**2. GNN-SubNet: disease subnetwork detection with explainable graph neural networks**  
*Bioinformatics*, 2022 · 69 citations · citation rank 68/182 · [10.1093/bioinformatics/btac478](https://doi.org/10.1093/bioinformatics/btac478)  
`GNNExplainer-based subnetwork discovery`

GNN-SubNet detects disease *subnetworks* rather than individual genes, by training a GNN on patient-specific network representations and then applying a GNNExplainer-style optimization to find the connected subgraph that best explains the classification. Returning a module rather than a ranked gene list matches how the biology is usually interpreted, and the paper is explicit about the stability problems of explanation methods, running the explainer repeatedly to report a consensus subnetwork. Read it directly after GNNExplainer in Part II.

**3. CGMega: explainable graph neural network for cancer gene module dissection**  
*Nature Communications*, 2024 · 68 citations · citation rank 70/182 · [10.1038/s41467-024-50426-6](https://doi.org/10.1038/s41467-024-50426-6)  
`attention over multi-omic gene graph`

CGMega dissects cancer gene modules with an attention-based GNN over a multi-omic gene graph built from Hi-C contacts, protein interactions and epigenomic features, and interprets the attention to recover the module structure around each predicted gene. Including 3D genome contacts as edges is unusual and biologically motivated — regulatory relationships often follow chromatin contacts rather than linear proximity. The paper validates predicted modules with independent functional data, which is more than most in this group attempt.

**4. MODIG: multi-omics and multi-dimensional gene network for driver gene identification**  
*Bioinformatics*, 2022 · 52 citations · citation rank 80/182 · [10.1093/bioinformatics/btac622](https://doi.org/10.1093/bioinformatics/btac622)  
`GAT over multiple gene graphs`

MODIG runs graph attention over several gene graphs simultaneously — protein interaction, gene sequence similarity, pathway co-membership, GO semantic similarity — and learns per-graph attention so the model chooses which relational view supports each prediction. Multi-dimensional graph fusion is the contribution, and the ablations showing which graph matters for which gene class are the interesting result. It is one of the better-engineered driver gene methods.

**5. Explainable multilayer graph neural network for cancer gene prediction**  
*Bioinformatics*, 2023 · 43 citations · citation rank 89/182 · [10.1093/bioinformatics/btad643](https://doi.org/10.1093/bioinformatics/btad643)  
`multilayer biological networks`

An explainable multilayer GNN for cancer gene prediction operating on multiple biological networks treated as layers of a single multiplex graph, with explanations produced at both the layer and the node level. The multiplex formulation is more principled than concatenating graphs, since it preserves which relation each edge represents. The paper's explanation analysis is careful about distinguishing what the model used from what is biologically causal — a distinction frequently blurred in this literature.

**6. Effective integration of multi-omics with prior knowledge via explainable GNNs**  
*npj Systems Biology and Applications*, 2025 · 32 citations · outside the top 100 on citations · [10.1038/s41540-025-00519-9](https://doi.org/10.1038/s41540-025-00519-9)  
`biomarker discovery with explanations`

This work integrates multi-omics with prior knowledge through explainable GNNs for biomarker discovery, with an explicit emphasis on making the discovered biomarkers verifiable: predictions are traced back to specific features and network paths, and the resulting candidates are checked against independent evidence. Published in a systems biology venue, it is more concerned with whether the biology holds up than with benchmark rank, which makes it a good closing paper for this sub-section.

**7. Multi-network graph contrastive learning for cancer driver gene identification**  
*IEEE Trans. Netw. Sci. Eng.*, 2024 · 27 citations · outside the top 100 on citations · [10.1109/tnse.2024.3373652](https://doi.org/10.1109/tnse.2024.3373652)  
`contrastive across biological networks`

A multi-network graph contrastive learning approach to driver gene identification, contrasting a gene's representations across different biological networks so that the learned embedding captures what is consistent across relational views. Using different networks as augmented views of the same underlying biology is an elegant reuse of the contrastive recipe, and it avoids the artificial graph perturbations that generic graph contrastive methods rely on.

**8. DGMP: identifying cancer driver genes by jointing DGCN and MLP**  
*Genomics Proteomics Bioinformatics*, 2022 · 26 citations · outside the top 100 on citations · [10.1016/j.gpb.2022.11.004](https://doi.org/10.1016/j.gpb.2022.11.004)  
`directed graph convolution`

DGMP identifies driver genes by combining a directed graph convolutional network with a multilayer perceptron branch. Directionality matters here — regulatory and signalling networks are directed, and symmetric message passing discards that information — and the paper shows the directed variant outperforming its undirected counterpart on the same networks. Its parallel MLP branch also guards against the graph washing out gene-intrinsic features, the same concern as the MLP+GNN fusion paper in Section 2.1.

**9. Biology-inspired graph neural network encodes reactome and reveals biochemical reactions of disease**  
*Patterns*, 2023 · 19 citations · outside the top 100 on citations · [10.1016/j.patter.2023.100758](https://doi.org/10.1016/j.patter.2023.100758)  
`reaction-level graph structure`

This biology-inspired GNN encodes the Reactome pathway database as its graph structure, with nodes representing biochemical reactions and edges the substrate–product relations between them, so the model's internal representation corresponds to actual biochemistry rather than to a generic interaction network. Predictions can be read as statements about specific reactions being perturbed in disease. It is the strongest example in this list of architecture designed around a biological knowledge base rather than around a machine learning convention.

**10. GOAT: gene-level biomarker discovery from multi-omics using graph attention networks**  
*Bioinformatics*, 2023 · 18 citations · outside the top 100 on citations · [10.1093/bioinformatics/btad582](https://doi.org/10.1093/bioinformatics/btad582)  
`asthma subtype biomarkers`

GOAT performs gene-level biomarker discovery from multi-omics with graph attention networks, applied to asthma subtypes rather than cancer. Moving outside oncology exposes assumptions that the cancer-focused methods leave implicit — that driver labels exist, that mutation is the primary omics signal — and the paper adapts the framework accordingly. Worth including for that contrast.

**11. SMG: self-supervised masked graph learning for cancer gene identification**  
*Briefings in Bioinformatics*, 2023 · 18 citations · outside the top 100 on citations · [10.1093/bib/bbad406](https://doi.org/10.1093/bib/bbad406)  
`masked-graph self-supervision`

SMG applies self-supervised masked graph learning to cancer gene identification: nodes and their features are masked and reconstructed, producing gene representations without relying on the small and biased set of known driver labels. Since label scarcity and ascertainment bias are the central difficulties of driver gene prediction, pretraining that requires no labels is a well-aimed idea, and the paper shows the pretrained representations transferring across cancer types.

**12. Identification of cancer driver genes by integrating multiomics data with graph neural networks**  
*Metabolites*, 2023 · 18 citations · outside the top 100 on citations · [10.3390/metabo13030339](https://doi.org/10.3390/metabo13030339)  
`multi-omic node features on PPI`

A straightforward implementation of the EMOGI recipe — multi-omic node features on a protein interaction network, GNN classification of driver genes — with a different feature set and evaluation. Its role here is as a replication: independent implementations reaching similar conclusions are worth something in a literature where most papers report only their own best configuration.

**13. Identifying cancer driver genes with multi-view heterogeneous GCN and self-attention**  
*BMC Bioinformatics*, 2023 · 18 citations · outside the top 100 on citations · [10.1186/s12859-023-05140-3](https://doi.org/10.1186/s12859-023-05140-3)  
`heterogeneous multi-view graph`

This method combines a multi-view heterogeneous graph convolutional network with self-attention for driver gene identification, treating different node types (genes, pathways, samples) and edge types explicitly rather than collapsing them into a homogeneous graph. The heterogeneous formulation lets the model distinguish a gene–gene interaction from a gene–pathway membership, which a homogeneous GCN cannot. Read it alongside R-GCN and HAN in Part II.

**14. GraphPath: graph attention model for molecular stratification via pathway-pathway network**  
*Bioinformatics*, 2024 · 18 citations · outside the top 100 on citations · [10.1093/bioinformatics/btae165](https://doi.org/10.1093/bioinformatics/btae165)  
`interpretable pathway graph`

GraphPath performs molecular stratification through a pathway–pathway interaction network, making pathways rather than genes the nodes, so the model operates at the level of biological processes and its attention weights are directly interpretable as process-level importance. Raising the level of abstraction reduces the dimensionality problem substantially and yields explanations that are easier to act on. One of the more thoughtfully designed interpretable models in this section.

**15. AMOGEL: associative graph neural networks with prior knowledge for biomarker identification**  
*BMC Bioinformatics*, 2025 · 17 citations · outside the top 100 on citations · [10.1186/s12859-025-06111-6](https://doi.org/10.1186/s12859-025-06111-6)  
`association-rule-derived graphs`

AMOGEL derives graph structure from association rule mining over multi-omic data — edges are added where feature co-occurrence patterns are statistically strong — and combines this data-derived graph with prior knowledge for biomarker identification. Association rules are an unfashionable technique but produce interpretable, explicitly stated edges, which is a reasonable trade against the opacity of learned graphs. A useful methodological outlier in this list.

#### 7.3 Survival and prognosis

*Censored outcomes and calibrated risk, which change what the model must optimize. Do not assume a method tuned for classification transfers.*

**1. Geometric graph neural networks on multi-omics data to predict cancer survival**  
*Computers in Biology and Medicine*, 2023 · 41 citations · citation rank 96/182 · [10.1016/j.compbiomed.2023.107117](https://doi.org/10.1016/j.compbiomed.2023.107117)  
`geometric deep learning for prognosis`

This paper applies geometric deep learning to multi-omics data for cancer survival prediction, treating the patient representation as living on a graph-structured manifold and using geometric convolutions to respect that structure. Survival prediction differs from classification in its censored outcomes and its need for calibrated risk, and the paper handles the censoring with a Cox partial-likelihood loss on the graph model's output. It is the clearest statement of the geometric framing in a prognostic setting.

**2. Graph attention-based fusion of pathology images and gene expression for survival**  
*IEEE Trans. Medical Imaging*, 2024 · 37 citations · outside the top 100 on citations · [10.1109/tmi.2024.3386108](https://doi.org/10.1109/tmi.2024.3386108)  
`WSI graph + expression fusion`

A graph attention model fusing whole-slide pathology images with gene expression for survival prediction: the slide is represented as a graph of image patches, expression provides a second modality, and cross-modal attention lets morphological regions be weighted by their transcriptional context. Published in a medical imaging venue, it is rigorous about evaluation — proper patient-level splits, concordance index with confidence intervals — which is not universal in this literature. It is the strongest multimodal prognostic method in the list.

**3. DeepMOCCA: pan-cancer prognostic model with graph attention and multi-omics**  
*preprint (bioRxiv)*, 2021 · 27 citations · outside the top 100 on citations · [10.1101/2021.03.02.433454](https://doi.org/10.1101/2021.03.02.433454)  
`graph attention over PPI; preprint`

DeepMOCCA is a pan-cancer prognostic model applying graph attention over a protein interaction network with multi-omic node features, trained across cancer types so that shared prognostic signal can be borrowed between rare and common tumours. Pan-cancer training is the interesting design decision, and the paper examines where it helps and where cancer-specific models remain better. Still a preprint, but well-specified and frequently used as a comparison.

**4. FGCNSurv: dually fused graph convolutional network for multi-omics survival prediction**  
*Bioinformatics*, 2023 · 18 citations · outside the top 100 on citations · [10.1093/bioinformatics/btad472](https://doi.org/10.1093/bioinformatics/btad472)  
`dual fusion for survival`

FGCNSurv uses dually fused graph convolutional networks for survival: fusion happens both at the feature level within each omics layer and at the graph level across layers, with a Cox objective on the final representation. The paper's contribution is the systematic treatment of *where* fusion should occur for a survival endpoint specifically, and it reports that the answer differs from what classification-oriented papers conclude — a point worth flagging, since methods are routinely transplanted between the two tasks without checking.

**5. Local augmented graph neural network for multi-omics cancer prognosis**  
*Methods*, 2023 · 18 citations · outside the top 100 on citations · [10.1016/j.ymeth.2023.02.011](https://doi.org/10.1016/j.ymeth.2023.02.011)  
`local augmentation of patient graphs`

This method augments patient graphs locally — generating perturbed versions of each node's neighbourhood during training — to address the small-sample problem in multi-omics prognosis. Local augmentation is a graph-native form of data augmentation and is better suited than global perturbations to graphs where each node is a patient and neighbourhoods are the meaningful unit. The paper reports the largest gains exactly where expected: on the smallest cohorts.

#### 7.4 Drug response prediction

*Two graphs at once: the compound's molecular structure and the biological context of the cell line.*

**1. DeepCDR: a hybrid graph convolutional network for predicting cancer drug response**  
*Bioinformatics*, 2020 · 271 citations · citation rank 24/182 · [10.1093/bioinformatics/btaa822](https://doi.org/10.1093/bioinformatics/btaa822)  
`drug graph + cell-line multi-omics`

DeepCDR is the standard reference for graph-based drug response prediction. It represents each drug as a molecular graph processed by a graph convolutional network, encodes cell-line multi-omics — expression, mutation, methylation — with separate subnetworks, and combines the two to predict IC50. Putting the drug on the graph side rather than using fixed fingerprints lets the model generalize to unseen compounds, which is the practically important form of generalization. Every method below is a variation on this two-branch structure.

**2. GraphCDR: GNN with contrastive learning for cancer drug response prediction**  
*Briefings in Bioinformatics*, 2021 · 128 citations · citation rank 43/182 · [10.1093/bib/bbab457](https://doi.org/10.1093/bib/bbab457)  
`contrastive drug-cell bipartite graph`

GraphCDR builds a bipartite drug–cell-line graph and trains it with a contrastive objective, treating known sensitive and resistant pairs as positive and negative examples so that the representation is shaped by the response relation itself rather than only by reconstruction. Contrastive learning suits this problem because response data are sparse and unevenly distributed across drug–line pairs. The paper reports notably better cold-start performance on unseen cell lines than DeepCDR.

**3. Predicting drug response based on multi-omics fusion and graph convolution**  
*IEEE J. Biomed. Health Inform.*, 2021 · 109 citations · citation rank 51/182 · [10.1109/jbhi.2021.3102186](https://doi.org/10.1109/jbhi.2021.3102186)  
`similarity-network graph convolution`

This method fuses multi-omic profiles into a similarity network over cell lines and applies graph convolution to predict drug response, using the network to share information between similar lines when response data for a given drug are sparse. It is the similarity-network counterpart to DeepCDR's molecular-graph approach, and comparing the two makes the point that "the graph" in drug response prediction can be over drugs, over samples, or over both.

**4. DualGCN: a dual graph convolutional network model to predict cancer drug response**  
*BMC Bioinformatics*, 2022 · 39 citations · outside the top 100 on citations · [10.1186/s12859-022-04664-4](https://doi.org/10.1186/s12859-022-04664-4)  
`drug graph + PPI-constrained omics graph`

DualGCN runs two graph convolutional networks in parallel — one over the drug's molecular structure, one over a protein interaction network carrying the cell line's omics features — so that the model reasons about the compound and the biological context in comparable representations before combining them. Constraining the omics branch with the PPI network rather than treating features as a flat vector is what distinguishes it from DeepCDR, and the ablation isolating that choice is convincing.

**5. CancerOmicsNet: a multi-omics network-based approach to anti-cancer drug profiling**  
*Oncotarget*, 2022 · 26 citations · outside the top 100 on citations · [10.18632/oncotarget.28234](https://doi.org/10.18632/oncotarget.28234)  
`GCN over drug-target-omics network`

CancerOmicsNet builds a heterogeneous network linking drugs, targets, genes and omics measurements, and propagates over it with graph convolution to profile anti-cancer drug activity, so a prediction is supported by an explicit path through the network from compound to target to pathway to phenotype. The traceable path is the appeal: it gives a mechanistic story alongside a score, which pure two-branch regressors cannot. A good closing paper for Part I's applied sections.

---

### 8. Single-cell foundation models

Four models, four tokenization schemes, and an unresolved argument about whether large-scale pretraining pays off on this data. Read them for the tokenization choices — binned values, rank orderings, depth conditioning — which are the substantive contributions, and read the critical benchmarking literature alongside them.

**1. scGPT: toward building a foundation model for single-cell multi-omics**  
*Nature Methods*, 2024 · 1,227 citations · citation rank 9/182 · [10.1038/s41592-024-02201-0](https://doi.org/10.1038/s41592-024-02201-0)  
`generative pretraining, multi-omic fine-tuning`

scGPT is the most widely used single-cell foundation model and the reference point for the entire debate about whether pretraining pays off in this domain. It tokenizes each cell as a set of gene tokens with binned expression values, pretrains a transformer on 33 million cells with a masked-value objective adapted to the non-sequential nature of gene sets, and fine-tunes for annotation, batch integration, multi-omic integration, perturbation prediction and GRN inference. Its attention maps are also used to extract gene networks. Read it together with the critical literature: several benchmarks find that simple baselines match fine-tuned scGPT on some of these tasks, and that comparison is the most instructive part.

**2. Transfer learning enables predictions in network biology (Geneformer)**  
*Nature*, 2023 · 1,176 citations · citation rank 11/182 · [10.1038/s41586-023-06139-9](https://doi.org/10.1038/s41586-023-06139-9)  
`rank-value encoding, network-biology transfer`

Geneformer takes a different tokenization: each cell becomes a *rank-ordered* list of genes by expression normalized to the gene's median across a 30-million-cell corpus, discarding magnitudes and keeping only ordering. This makes the representation robust to depth and platform differences, at the cost of quantitative information. Pretrained with masked-token prediction, it supports *in silico* deletion — removing a gene token and measuring the shift in the cell's embedding — which the authors use to nominate therapeutic targets in cardiomyopathy. The rank-value encoding is the idea most worth studying, independent of one's view of foundation models.

**3. scBERT as a large-scale pretrained deep language model for cell type annotation**  
*Nature Machine Intelligence*, 2022 · 635 citations · citation rank 19/182 · [10.1038/s42256-022-00534-z](https://doi.org/10.1038/s42256-022-00534-z)  
`gene-expression BERT for annotation`

scBERT predates both of the above and is the cleanest demonstration of the basic recipe: gene expression is embedded through a gene2vec-style representation plus a binned expression embedding, a Performer backbone handles the full gene set without truncating to highly variable genes, and the pretrained model is fine-tuned for cell type annotation. Using the whole transcriptome rather than a selected subset is its distinguishing choice, and it argues this is what allows recognition of novel and rare types. As the earliest of the four, it is the right one to read first.

**4. Large-scale foundation model on single-cell transcriptomics (scFoundation)**  
*Nature Methods*, 2024 · 541 citations · citation rank 20/182 · [10.1038/s41592-024-02305-7](https://doi.org/10.1038/s41592-024-02305-7)  
`100M-parameter read-depth-aware model`

scFoundation is a 100-million-parameter model trained on over 50 million cells whose central contribution is read-depth awareness: sequencing depth is supplied as an explicit conditioning signal, and the pretraining task asks the model to reconstruct a high-depth profile from a low-depth one. This directly addresses the confound that most distinguishes single-cell datasets from one another. The asymmetric encoder–decoder design also lets the encoder process only non-zero genes, which is what makes a model of this size trainable on this data. Evaluated on drug response and perturbation prediction as well as the standard tasks.

---

### 9. Reviews, benchmarks and software

Where to start, and how to tell whether any of the above actually works. The independent re-evaluations here disagree substantially with the self-reported rankings in the primary papers, which is the most useful thing in this section.

**1. Benchmarking atlas-level data integration in single-cell genomics (scIB)**  
*Nature Methods*, 2021 · 1,515 citations · citation rank 7/182 · [10.1038/s41592-021-01336-8](https://doi.org/10.1038/s41592-021-01336-8)  
`the standard metric suite for integration`

scIB is the benchmarking paper that defined how single-cell integration is evaluated. It formalizes the trade-off between batch correction and conservation of biological variation, provides a suite of metrics for each side — kBET, graph iLISI, PCR comparison against ARI, NMI, isolated-label scores and trajectory conservation — and combines them into an aggregate score, then applies the suite to sixteen methods across many tasks. Every integration paper published since reports these metrics. Read it not just for the ranking, which is now dated, but for the metric definitions and their known failure modes, which are still what everyone uses.

**2. Deep learning-based approaches for multi-omics data integration and analysis**  
*BioData Mining*, 2024 · 243 citations · citation rank 27/182 · [10.1186/s13040-024-00391-z](https://doi.org/10.1186/s13040-024-00391-z)  
`survey of fusion strategies`

A survey of deep learning for multi-omics integration organized by fusion strategy — early, intermediate and late fusion — rather than by application, which makes it a useful conceptual map for placing any new method. It covers autoencoders, GNNs and attention-based fusion, and discusses the practical problems of missing modalities and unequal feature dimensionality. Broad rather than deep; use it to orient before reading primary methods.

**3. Graph machine learning for integrated multi-omics analysis**  
*British Journal of Cancer*, 2024 · 122 citations · citation rank 46/182 · [10.1038/s41416-024-02706-7](https://doi.org/10.1038/s41416-024-02706-7)  
`review of graph-based omic fusion`

A review of graph machine learning for integrated multi-omics analysis with a cancer-biology focus, covering graph construction choices, the main GNN architectures, and interpretability approaches, together with a discussion of what these methods have and have not delivered clinically. Its treatment of graph construction — what should be a node, what should be an edge, and where the prior network comes from — is the most useful part, because that choice determines more about performance than the architecture does.

**4. OmicVerse: bridging and deepening insights across bulk and single-cell sequencing**  
*Nature Communications*, 2024 · 93 citations · citation rank 56/182 · [10.1038/s41467-024-50194-3](https://doi.org/10.1038/s41467-024-50194-3)  
`unified analysis framework`

OmicVerse is a unified Python framework spanning bulk and single-cell analysis, wrapping a large number of methods behind consistent interfaces and adding its own implementations for several tasks. For a textbook its value is practical: it removes much of the environment and format friction that makes reproducing published single-cell pipelines painful, and it makes side-by-side comparison of methods feasible in a course setting.

**5. A comprehensive overview of GNN-based approaches to clustering for spatial transcriptomics**  
*Comput. Struct. Biotechnol. J.*, 2023 · 41 citations · citation rank 95/182 · [10.1016/j.csbj.2023.11.055](https://doi.org/10.1016/j.csbj.2023.11.055)  
`systematic comparison of spatial GNNs`

A systematic comparison of GNN-based clustering approaches for spatial transcriptomics, running the main methods on common datasets under a common protocol rather than reporting each paper's self-assessed numbers. Independent re-evaluations of this kind are scarce and the results are sobering — the ranking differs substantially from what the original papers imply, and preprocessing choices account for much of the variance. Essential reading before trusting any single spatial-method benchmark.

**6. Graph neural networks for single-cell omics data: a review of approaches and applications**  
*Briefings in Bioinformatics*, 2025 · 40 citations · citation rank 100/182 · [10.1093/bib/bbaf109](https://doi.org/10.1093/bib/bbaf109)  
`task-by-task review of sc GNNs`

A recent, task-organized review of graph neural networks for single-cell omics: clustering, annotation, imputation, integration, trajectory inference, communication and regulatory inference each get a section with the representative methods and the open problems. It is the closest thing to a map of Part I of this reading list and the most current of the surveys here. Good as an assigned overview at the start of a course, with the primary papers following.

**7. Comparative analysis of multi-omics integration using GNNs for cancer classification**  
*IEEE Access*, 2025 · 36 citations · outside the top 100 on citations · [10.1109/access.2025.3540769](https://doi.org/10.1109/access.2025.3540769)  
`head-to-head GNN comparison`

A head-to-head comparative analysis of GNN architectures for multi-omics cancer classification, holding data and evaluation fixed and varying the graph construction and the GNN layer type. The finding that graph construction matters more than the choice among GCN, GAT and GraphSAGE echoes the spatial re-evaluation above and is worth stating explicitly to students, who tend to assume the opposite.

**8. Graph neural network approaches for single-cell data: a recent overview**  
*Neural Computing and Applications*, 2024 · 23 citations · outside the top 100 on citations · [10.1007/s00521-024-09662-6](https://doi.org/10.1007/s00521-024-09662-6)  
`compact task-oriented survey`

A compact, task-oriented overview of graph neural network approaches for single-cell data, shorter and more accessible than the *Briefings in Bioinformatics* review above and with a stronger emphasis on the machine-learning side of the methods. Useful as a first reading for someone coming from computer science rather than biology.

**9. DANCE: a deep learning library and benchmark platform for single-cell analysis**  
*Genome Biology*, 2024 · 11 citations · outside the top 100 on citations · [10.1186/s13059-024-03211-z](https://doi.org/10.1186/s13059-024-03211-z)  
`standardized baselines incl. GNN models`

DANCE is a deep learning library and benchmark platform for single-cell analysis, providing standardized implementations, data loaders and evaluation protocols for a large set of tasks — imputation, clustering, annotation, spatial domain identification, deconvolution and multi-omic integration — including many of the GNN methods in this list. Its citation count badly understates its usefulness: it is the most practical way to run a fair comparison, and it is the natural baseline platform for a course's programming exercises.

---

## Part II — Methods: the classical GNN canon

59 papers, the machine-learning literature the applied methods in Part I are assembled from. Citation range 11,077 – 37. Several entries carry citation counts far below their true influence because OpenAlex splits preprint and published records; GIN, Deep Graph Infomax and Graphormer are the worst affected and are core curriculum regardless of where they rank.

### II.1 Origins and surveys

Two decades of framing, from the first proposal of learning on graph domains to the taxonomies that fixed the field's vocabulary. Read one survey properly rather than four superficially.

**1. A comprehensive survey on graph neural networks**  
*IEEE Trans. Neural Netw. Learn. Syst.*, 2020 · 9,903 citations · citation rank 2/59 · [10.1109/tnnls.2020.2978386](https://doi.org/10.1109/tnnls.2020.2978386)  
`Wu et al.; the standard taxonomy`

Wu et al.'s survey is the standard taxonomy for the field and the one most single-cell papers cite when they need a definition. It partitions graph neural networks into recurrent, convolutional (spectral and spatial), autoencoder and spatio-temporal families, gives a consistent notation for message passing across all of them, and catalogues benchmark datasets and open problems. If you read one survey, read this one first: the vocabulary it fixes — aggregation, readout, spectral versus spatial — is the vocabulary the rest of this list uses.

**2. The graph neural network model**  
*IEEE Trans. Neural Networks*, 2008 · 9,687 citations · citation rank 3/59 · [10.1109/tnn.2008.2005605](https://doi.org/10.1109/tnn.2008.2005605)  
`Scarselli et al.; the original recurrent GNN`

Scarselli et al.'s original graph neural network, a decade before the modern wave. It defines node states through a contraction mapping applied repeatedly until a fixed point is reached, with the Banach fixed-point theorem guaranteeing convergence, and learns the transition function with the Almeida–Pineda algorithm. The recurrent, converge-to-equilibrium formulation is quite different from today's fixed-depth message passing, but it states the core idea — a node's representation is a function of its neighbourhood, applied recursively — with unusual clarity. Read it for the framing and for the historical record.

**3. Graph neural networks: a review of methods and applications**  
*AI Open*, 2020 · 5,832 citations · citation rank 7/59 · [10.1016/j.aiopen.2021.01.001](https://doi.org/10.1016/j.aiopen.2021.01.001)  
`Zhou et al.; design-space framing`

Zhou et al. organize the field as a design space rather than a taxonomy of models: pick a graph type, a propagation module, a sampling module, a pooling module and a training objective, and any particular architecture is one point in that space. This framing is more useful than a list of named models when you are building something new, which is the position most readers of this list are in. It also has the broadest coverage of applications of any of the surveys here.

**4. A comprehensive survey of graph embedding: problems, techniques and applications**  
*IEEE Trans. Knowl. Data Eng.*, 2018 · 2,079 citations · citation rank 21/59 · [10.1109/tkde.2018.2807452](https://doi.org/10.1109/tkde.2018.2807452)  
`pre-GNN embedding landscape`

A comprehensive survey of graph embedding covering the pre-GNN landscape — matrix factorization, random walks, deep learning approaches — organized by problem setting and by the type of graph. It is the best reference for understanding what graph neural networks replaced and why, and for recognizing when a shallow embedding method is still the right tool. Several single-cell methods are, in effect, rediscoveries of techniques catalogued here.

**5. A new model for learning in graph domains**  
*IJCNN*, 2006 · 1,946 citations · citation rank 24/59 · [10.1109/ijcnn.2005.1555942](https://doi.org/10.1109/ijcnn.2005.1555942)  
`Gori et al.; the paper that started it`

Gori, Monfardini and Scarselli's IJCNN paper is where the graph neural network model was first proposed, a year before the fuller journal treatment. It is short and worth reading precisely because it is short: the argument for learning directly on graph domains rather than flattening them into vectors is made in a few pages, and it is the same argument the field still makes.

**6. Graph convolutional networks: a comprehensive review**  
*Computational Social Networks*, 2019 · 1,887 citations · citation rank 25/59 · [10.1186/s40649-019-0069-y](https://doi.org/10.1186/s40649-019-0069-y)  
`spectral vs. spatial framing`

A review focused specifically on graph convolutional networks, with the clearest side-by-side derivation of the spectral and spatial formulations and of how the spectral view collapses into the spatial one under the localization approximations that ChebNet and GCN make. If the relationship between graph Fourier transforms, Chebyshev polynomials and neighbourhood averaging is not yet clear, this is the paper that makes it so.

**7. Graph embedding techniques, applications and performance: a survey**  
*Knowledge-Based Systems*, 2018 · 1,845 citations · citation rank 27/59 · [10.1016/j.knosys.2018.03.022](https://doi.org/10.1016/j.knosys.2018.03.022)  
`benchmark-oriented survey`

Goyal and Ferrara's survey is benchmark-oriented: it re-implements the major graph embedding methods, runs them on common datasets for node classification, link prediction and visualization, and reports where each wins. Independent re-evaluation of this kind is rare and valuable, and the accompanying library made the comparisons reproducible. Read it for the empirical results rather than the taxonomy.

**8. Deep learning on graphs: a survey**  
*IEEE Trans. Knowl. Data Eng.*, 2020 · 1,563 citations · citation rank 30/59 · [10.1109/tkde.2020.2981333](https://doi.org/10.1109/tkde.2020.2981333)  
`complements the Wu and Zhou surveys`

Zhang, Cui and Zhu's survey complements the Wu and Zhou surveys with a stronger emphasis on graph autoencoders, generative models and reinforcement learning on graphs — the parts the other two treat briefly. Since graph autoencoders are the workhorse of the single-cell literature, this is the survey most directly relevant to Part I of this reading list.

**9. Graph neural networks: a review of methods and applications (preprint)**  
*preprint (arXiv)*, 2018 · 1,448 citations · citation rank 35/59 · [10.48550/arxiv.1812.08434](https://doi.org/10.48550/arxiv.1812.08434)  
`the widely cited arXiv record`

The arXiv record of Zhou et al.'s review, which accumulated a large citation count in its own right before the journal version appeared. It is included here because much of the literature cites this version specifically, and because the preprint's treatment of some topics is longer than the published one. See the AI Open entry above for the substance.

**10. A survey on network embedding**  
*IEEE Trans. Knowl. Data Eng.*, 2018 · 1,294 citations · citation rank 39/59 · [10.1109/tkde.2018.2849727](https://doi.org/10.1109/tkde.2018.2849727)  
`unifies matrix-factorisation and walk methods`

Cui et al.'s network embedding survey unifies matrix-factorization and random-walk approaches, showing that DeepWalk, LINE and node2vec are all implicitly factorizing particular matrices — a result that clarifies what these methods actually optimize and why their hyperparameters behave as they do. This equivalence is the single most useful thing to know about shallow embeddings, and this survey states it well.

---

### II.2 Shallow node embeddings

The pre-GNN generation: random walks and matrix factorization. Still competitive on link prediction, and all of them turn out to be factorizing particular matrices.

**1. node2vec: scalable feature learning for networks**  
*KDD*, 2016 · 11,077 citations · citation rank 1/59 · [10.1145/2939672.2939754](https://doi.org/10.1145/2939672.2939754)  
`biased random walks; the pre-GNN baseline`

node2vec generalizes DeepWalk's uniform random walks with two parameters, p and q, that interpolate the walk between breadth-first and depth-first exploration, letting the embedding emphasize either structural equivalence (nodes with similar roles) or homophily (nodes in the same community). Walks are then fed to skip-gram with negative sampling. The biased-walk idea remains the most intuitive way to explain what a node embedding is choosing to encode, and node2vec is still a competitive baseline on link prediction. It is the pre-GNN method most worth understanding before reading anything else here.

**2. DeepWalk: online learning of social representations**  
*KDD*, 2014 · 8,679 citations · citation rank 4/59 · [10.1145/2623330.2623732](https://doi.org/10.1145/2623330.2623732)  
`truncated random walks + skip-gram`

DeepWalk is the paper that connected graphs to language modelling: sample truncated random walks from each node, treat each walk as a sentence and each node as a word, and run skip-gram. That analogy unlocked a decade of work, and it is worth pausing on how strong an assumption it makes — that co-occurrence in a short walk is the right notion of similarity. Everything from node2vec to metapath2vec is a refinement of the walk-sampling step.

**3. LINE: large-scale information network embedding**  
*WWW*, 2015 · 4,755 citations · citation rank 10/59 · [10.1145/2736277.2741093](https://doi.org/10.1145/2736277.2741093)  
`first- and second-order proximity`

LINE embeds nodes by explicitly preserving first-order proximity (observed edges) and second-order proximity (shared neighbourhoods) with two separate objectives, then concatenating the resulting embeddings. Making the two notions of proximity explicit, rather than blending them implicitly through a walk length, is clearer than the walk-based alternatives and easier to reason about. LINE also introduced the edge-sampling trick that makes stochastic training on weighted graphs unbiased.

**4. Structural deep network embedding (SDNE)**  
*KDD*, 2016 · 2,845 citations · citation rank 15/59 · [10.1145/2939672.2939753](https://doi.org/10.1145/2939672.2939753)  
`deep autoencoder on proximity`

SDNE was among the first to use a deep autoencoder for network embedding, with a reconstruction loss on the adjacency vector capturing second-order proximity and a Laplacian eigenmaps-style penalty capturing first-order proximity. It also handles the sparsity of adjacency vectors by weighting reconstruction errors on non-zero entries more heavily — a detail that reappears, unattributed, in many single-cell graph autoencoders.

**5. Asymmetric transitivity preserving graph embedding (HOPE)**  
*KDD*, 2016 · 1,304 citations · citation rank 38/59 · [10.1145/2939672.2939751](https://doi.org/10.1145/2939672.2939751)  
`directed-graph proximity`

HOPE preserves *asymmetric* transitivity, which matters for directed graphs where a path from u to v implies nothing about a path from v to u. It represents each node by a source and a target vector and uses a generalized SVD formulation that admits several high-order proximity measures (Katz, rooted PageRank, common neighbours) under one framework, with a closed-form solution. Directed structure is under-modelled throughout the biological literature, and this is the cleanest reference for it.

---

### II.3 Convolution and message passing

The core of the field. Follow the line from spectral convolution to Chebyshev localization to GCN, then read the simplifications that ask how much of it was necessary.

**1. Semi-supervised classification with graph convolutional networks (GCN)**  
*preprint (arXiv)*, 2016 · 8,058 citations · citation rank 6/59 · [10.48550/arxiv.1609.02907](https://doi.org/10.48550/arxiv.1609.02907)  
`Kipf & Welling; the first-order spectral simplification`

Kipf and Welling's GCN is the single most important paper in this list. It starts from spectral graph convolution, restricts the filter to first-order Chebyshev polynomials, applies a renormalization trick to the adjacency matrix, and arrives at a layer that is nothing more than a symmetric-normalized neighbourhood average followed by a linear map and a non-linearity. The result is cheap, easy to implement, and works, and it is the layer used by the majority of the methods in Part I of this reading list. Read the derivation carefully; almost every simplification and criticism in the papers below is a comment on one of its steps.

**2. Neural message passing for quantum chemistry (MPNN)**  
*preprint (arXiv)*, 2017 · 3,015 citations · citation rank 13/59 · [10.48550/arxiv.1704.01212](https://doi.org/10.48550/arxiv.1704.01212)  
`the message-passing abstraction itself`

Gilmer et al. introduced the message-passing neural network abstraction: a message function, an update function and a readout function, into which GCN, GAT, GraphSAGE and most other architectures can be written as special cases. Having a common formalism is what makes the literature comparable at all, and it is what libraries like PyTorch Geometric implement. The paper's own application — predicting quantum-chemical properties of molecules — is also the origin of the molecular-graph branch used by the drug response methods in Part I.

**3. Spectral networks and locally connected networks on graphs**  
*preprint (arXiv)*, 2013 · 2,726 citations · citation rank 17/59 · [10.48550/arxiv.1312.6203](https://doi.org/10.48550/arxiv.1312.6203)  
`Bruna et al.; the spectral origin`

Bruna et al.'s spectral networks are where graph convolution begins. Convolution on a graph is defined through the eigendecomposition of the graph Laplacian, with filters as functions of the eigenvalues. The formulation is exact and general but requires an O(n³) eigendecomposition and produces filters that are not localized and do not transfer between graphs — the three problems that ChebNet and then GCN were designed to solve. Read it to understand what those later approximations gave up.

**4. Hypergraph neural networks (HGNN)**  
*AAAI*, 2019 · 1,729 citations · citation rank 28/59 · [10.1609/aaai.v33i01.33013558](https://doi.org/10.1609/aaai.v33i01.33013558)  
`higher-order incidence instead of edges`

HGNN extends convolution from graphs to hypergraphs, where an edge may connect any number of nodes, using the hypergraph Laplacian built from the incidence matrix and a truncated Chebyshev approximation analogous to GCN's. Many biological relations are genuinely higher-order — a pathway, a protein complex, a multi-way chromatin contact — and decomposing them into pairwise edges loses information. Pair it with Hyper-SAGNN in Part I.

**5. Convolutional neural networks on graphs with fast localized spectral filtering (ChebNet)**  
*preprint (arXiv)*, 2016 · 1,703 citations · citation rank 29/59 · [10.48550/arxiv.1606.09375](https://doi.org/10.48550/arxiv.1606.09375)  
`Chebyshev polynomial filters`

ChebNet is the essential intermediate step between spectral networks and GCN. It parameterizes the spectral filter as a Chebyshev polynomial of order K in the scaled Laplacian, which makes the filter exactly K-hop localized and computable through sparse matrix multiplication with no eigendecomposition at all. Setting K=1 and applying the renormalization trick gives Kipf and Welling's GCN. Understanding this paper is what makes the phrase "first-order spectral simplification" meaningful rather than incantatory.

**6. Simplifying graph convolutional networks (SGC)**  
*preprint (arXiv)*, 2019 · 1,187 citations · citation rank 44/59 · [10.48550/arxiv.1902.07153](https://doi.org/10.48550/arxiv.1902.07153)  
`removes nonlinearity; the linear baseline`

SGC removes the non-linearities from a K-layer GCN, collapsing the model into a fixed low-pass filter A^K X followed by a single logistic regression — and shows that on standard benchmarks this loses almost nothing while training orders of magnitude faster. The paper is the most important negative result in graph learning: much of what a GCN does is neighbourhood smoothing, not learning. Every claim in Part I that a graph neural network beat a simpler model should be checked against an SGC baseline.

**7. Variational graph auto-encoders (GAE/VGAE)**  
*preprint (arXiv)*, 2016 · 891 citations · citation rank 45/59 · [10.48550/arxiv.1611.07308](https://doi.org/10.48550/arxiv.1611.07308)  
`the encoder behind most sc graph autoencoders`

Kipf and Welling's graph autoencoder and its variational form: a GCN encoder produces node embeddings, and the decoder is simply the inner product of embedding pairs passed through a sigmoid to reconstruct the adjacency matrix. It is a short workshop paper that has become the backbone of an enormous applied literature — nearly every "graph autoencoder" in Part I is this model with a different decoder or an extra loss term. The variational version's prior also supplies the regularization that stAA and similar methods later replace adversarially.

**8. Predict then propagate: GNNs meet personalized PageRank (APPNP)**  
*preprint (arXiv)*, 2018 · 435 citations · outside the top 50 on citations · [10.48550/arxiv.1810.05997](https://doi.org/10.48550/arxiv.1810.05997)  
`decouples propagation from prediction`

APPNP decouples propagation from prediction: a small neural network first predicts from node features alone, and those predictions are then propagated with a personalized PageRank scheme that includes a teleport term back to the original prediction. Because propagation carries no parameters, the receptive field can be made large without deepening the network and without over-smoothing — the teleport term is what preserves locality. It is the most elegant answer in this list to the depth problem.

**9. Geometric deep learning on graphs and manifolds using mixture model CNNs (MoNet)**  
*preprint (arXiv)*, 2016 · 37 citations · outside the top 50 on citations · [10.48550/arxiv.1611.08402](https://doi.org/10.48550/arxiv.1611.08402)  
`generalises spatial convolution`

MoNet gives a general framework for convolution on non-Euclidean domains: define a local pseudo-coordinate system around each node and learn a set of parametric kernels (Gaussians, in the paper) over those coordinates, so aggregation weights are learned functions of relative position. GCN, GAT and several manifold CNNs are recoverable as particular choices of pseudo-coordinates. It is the geometric deep learning view of what message passing is doing.

---

### II.4 Attention and graph transformers

Learned edge weights, and then the removal of the edge constraint altogether in favour of structural encodings inside a transformer.

**1. Graph Attention Networks (GAT)**  
*preprint (arXiv)*, 2017 · 8,402 citations · citation rank 5/59 · [10.48550/arxiv.1710.10903](https://doi.org/10.48550/arxiv.1710.10903)  
`masked self-attention over neighbours`

GAT replaces the fixed normalized-adjacency weights of GCN with masked self-attention: each edge's weight is computed from the two endpoint representations and normalized across each node's neighbourhood, with multiple attention heads for stability. Because the weights depend on features rather than on graph structure alone, GAT handles graphs whose edges have unequal reliability — exactly the situation for kNN cell graphs and spatial proximity graphs, which is why nearly every spatial method in Part I is a GAT variant. It is also inductive, needing no knowledge of the full graph at training time.

**2. Do transformers really perform bad for graph representation? (Graphormer)**  
*preprint (arXiv)*, 2021 · 131 citations · outside the top 50 on citations · [10.48550/arxiv.2106.05234](https://doi.org/10.48550/arxiv.2106.05234)  
`structural encodings inside a transformer`

Graphormer shows that a standard transformer, given the right structural encodings, is a very strong graph model: it adds a centrality encoding to the node inputs, a spatial encoding (shortest-path distance) as an attention bias, and an edge encoding along the connecting path. With these, global attention over all node pairs no longer discards the graph — the graph is expressed through the biases. It won the OGB Large-Scale Challenge and is the reference point for the graph-transformer direction that scMoFormer and DeepMAPS follow. Its OpenAlex count is split across preprint and published records and badly understates its influence.

---

### II.5 Heterogeneous and knowledge graphs

Multiple node and edge types. Biological graphs are almost always of this kind, and this is the machinery for handling them properly rather than by collapsing types.

**1. Modeling relational data with graph convolutional networks (R-GCN)**  
*ESWC*, 2018 · 5,254 citations · citation rank 8/59 · [10.1007/978-3-319-93417-4_38](https://doi.org/10.1007/978-3-319-93417-4_38)  
`per-relation weight matrices`

R-GCN extends graph convolution to graphs with multiple edge types by giving each relation its own weight matrix, and controls the resulting parameter explosion with basis decomposition or block-diagonal decomposition of those matrices. It handles both node classification and link prediction, the latter by pairing the encoder with a DistMult decoder. Biological graphs are almost always multi-relational — regulates, binds, is-part-of — and this is the foundational paper for treating them as such.

**2. Heterogeneous graph attention network (HAN)**  
*WWW*, 2019 · 3,008 citations · citation rank 14/59 · [10.1145/3308558.3313562](https://doi.org/10.1145/3308558.3313562)  
`node- and semantic-level attention over metapaths`

HAN introduces two levels of attention for heterogeneous graphs: node-level attention over the neighbours reachable along a given metapath, and semantic-level attention over the metapaths themselves, so the model learns which types of relation matter for the task. The hierarchical structure is the idea worth carrying: it is exactly the two-level design SpatialGlue uses for spatial multi-omics, one level within modality and one across.

**3. metapath2vec: scalable representation learning for heterogeneous networks**  
*KDD*, 2017 · 2,297 citations · citation rank 19/59 · [10.1145/3097983.3098036](https://doi.org/10.1145/3097983.3098036)  
`metapath-guided walks`

metapath2vec adapts random-walk embedding to heterogeneous graphs by constraining walks to follow a specified metapath schema — author–paper–author, or gene–pathway–gene — so the sampled context is semantically coherent rather than a mix of incomparable node types. It also introduces a heterogeneous negative-sampling scheme. The metapath concept is what most heterogeneous methods, including CAME in Part I, ultimately rely on.

**4. KGAT: knowledge graph attention network**  
*KDD*, 2019 · 2,158 citations · citation rank 20/59 · [10.1145/3292500.3330989](https://doi.org/10.1145/3292500.3330989)  
`attentive propagation over a knowledge graph`

KGAT propagates over a knowledge graph with attention, combining collaborative signal and knowledge-graph structure in one embedding by recursively aggregating from a node's higher-order neighbours with attention-weighted edges. It is a recommendation paper, but the mechanism — attentive propagation over a heterogeneous knowledge graph to produce explainable paths — transfers directly to biomedical knowledge graphs, and CancerOmicsNet in Part I is essentially this idea.

**5. Heterogeneous graph neural network (HetGNN)**  
*KDD*, 2019 · 1,538 citations · citation rank 31/59 · [10.1145/3292500.3330961](https://doi.org/10.1145/3292500.3330961)  
`type-aware sampling and aggregation`

HetGNN performs type-aware sampling followed by type-aware aggregation: a restart-based random walk collects a fixed-size, type-balanced neighbourhood for each node, features within each type are encoded with a per-type encoder, and the types are then combined with attention. The explicit sampling step is what makes it scale to large heterogeneous graphs where full neighbourhood aggregation is infeasible.

**6. Heterogeneous graph transformer (HGT)**  
*WWW*, 2020 · 1,445 citations · citation rank 36/59 · [10.1145/3366423.3380027](https://doi.org/10.1145/3366423.3380027)  
`type-dependent attention; basis of DeepMAPS`

HGT parameterizes attention by node and edge type — separate projection matrices for each type triple — so heterogeneity is handled inside the attention mechanism rather than by pre-specifying metapaths, and adds relative temporal encoding for dynamic graphs. It scales through a heterogeneous mini-batch sampling algorithm. This is the architecture underneath DeepMAPS, and reading it first makes that paper considerably easier.

---

### II.6 Scalability: sampling and batching

How to train on a graph that does not fit in memory. Prerequisite for every atlas-scale application in Part I.

**1. Inductive representation learning on large graphs (GraphSAGE)**  
*preprint (arXiv)*, 2017 · 4,538 citations · citation rank 11/59 · [10.48550/arxiv.1706.02216](https://doi.org/10.48550/arxiv.1706.02216)  
`neighbour sampling + aggregators; inductive`

GraphSAGE made graph learning inductive. Instead of learning a fixed embedding per node, it learns aggregator functions — mean, LSTM or max-pooling — that generate a node's embedding from a *sampled* fixed-size neighbourhood, so unseen nodes and entirely new graphs can be embedded at test time with the trained weights. Sampling also bounds the computation per node, which is what makes minibatch training on large graphs possible. Both contributions — inductive capability and neighbourhood sampling — are prerequisites for essentially all atlas-scale single-cell applications.

**2. Graph convolutional neural networks for web-scale recommender systems (PinSAGE)**  
*KDD*, 2018 · 2,845 citations · citation rank 16/59 · [10.1145/3219819.3219890](https://doi.org/10.1145/3219819.3219890)  
`production-scale sampling and MapReduce inference`

PinSAGE is GraphSAGE taken to production at Pinterest: three billion nodes, eighteen billion edges. Its contributions are systems contributions — importance-based neighbourhood sampling using random-walk visit counts, curriculum training with progressively harder negatives, and a MapReduce pipeline for inference — and they are the ones that matter when a method has to run on real data rather than a benchmark. A useful antidote to the benchmark-scale mindset of most academic papers.

**3. Cluster-GCN: an efficient algorithm for training deep and large GCNs**  
*KDD*, 2019 · 1,267 citations · citation rank 40/59 · [10.1145/3292500.3330925](https://doi.org/10.1145/3292500.3330925)  
`subgraph batching by graph clustering`

Cluster-GCN partitions the graph with a clustering algorithm (METIS) and forms each minibatch from one or a few clusters, so almost all edges within a batch are retained and the neighbourhood explosion of naive minibatching disappears. Memory use becomes independent of depth, which is what allows genuinely deep GCNs to be trained. The stochastic multi-cluster variant is what keeps the batches unbiased.

**4. FastGCN: fast learning with GCNs via importance sampling**  
*preprint (arXiv)*, 2018 · 646 citations · citation rank 50/59 · [10.48550/arxiv.1801.10247](https://doi.org/10.48550/arxiv.1801.10247)  
`layer-wise sampling with variance reduction`

FastGCN reinterprets graph convolution as an integral over node distributions, then samples a fixed number of nodes independently per layer rather than sampling neighbourhoods per node, with importance sampling to reduce variance. This makes cost linear in depth instead of exponential. The probabilistic reformulation is the interesting part and is worth reading even if you use one of the other samplers in practice.

**5. GraphSAINT: graph sampling based inductive learning method**  
*preprint (arXiv)*, 2019 · 337 citations · outside the top 50 on citations · [10.48550/arxiv.1907.04931](https://doi.org/10.48550/arxiv.1907.04931)  
`subgraph sampling with bias correction`

GraphSAINT samples *subgraphs* rather than nodes or edges, builds a full GNN on each sampled subgraph, and corrects the resulting bias with normalization coefficients derived from the sampler's node and edge probabilities. Decoupling the sampling from the architecture means any GNN layer can be dropped in unchanged. It is generally the best-performing sampler in this group and the one most current libraries default to.

---

### II.7 Depth and over-smoothing

Why deep GNNs fail, and five different fixes. Read the diagnosis first.

**1. Deeper insights into graph convolutional networks for semi-supervised learning**  
*AAAI*, 2018 · 2,661 citations · citation rank 18/59 · [10.1609/aaai.v32i1.11604](https://doi.org/10.1609/aaai.v32i1.11604)  
`named over-smoothing; GCN as Laplacian smoothing`

Li, Han and Wu's paper named the central pathology of deep graph networks. It shows that graph convolution is a form of Laplacian smoothing, that repeated application drives node representations within a connected component toward the same value, and that this is why GCNs perform worse as they get deeper — a fact practitioners had observed but not explained. It also proposes co-training and self-training remedies for the label-scarce regime. Everything in this sub-section is a response to this paper.

**2. DeepGCNs: can GCNs go as deep as CNNs?**  
*IEEE*, 2019 · 1,380 citations · citation rank 37/59 · [10.1109/iccv.2019.00936](https://doi.org/10.1109/iccv.2019.00936)  
`residual/dense connections and dilated aggregation`

DeepGCNs imports the tools that made very deep CNNs trainable — residual and dense connections, plus dilated aggregation to widen the receptive field without more layers — and demonstrates a 56-layer GCN that keeps improving with depth on point cloud segmentation. It establishes that depth in graph networks is not impossible, only that the plain architecture is wrong. scRGCL in Part I is the single-cell application of exactly this recipe.

**3. Representation learning on graphs with jumping knowledge networks**  
*preprint (arXiv)*, 2018 · 733 citations · citation rank 48/59 · [10.48550/arxiv.1806.03536](https://doi.org/10.48550/arxiv.1806.03536)  
`per-node adaptive receptive field`

Jumping Knowledge Networks let each node choose its own effective depth: representations from every layer are kept and combined per node by concatenation, max-pooling or an LSTM attention, so nodes in dense regions can use shallow representations while nodes in sparse regions draw on deeper ones. The observation motivating it — that the right receptive field varies enormously across nodes in a graph with heterogeneous degree — applies directly to cell graphs, where rare cell types sit in sparse neighbourhoods.

**4. DropEdge: towards deep graph convolutional networks on node classification**  
*preprint (arXiv)*, 2019 · 632 citations · outside the top 50 on citations · [10.48550/arxiv.1907.10903](https://doi.org/10.48550/arxiv.1907.10903)  
`edge dropout against over-smoothing`

DropEdge randomly removes a fraction of edges at each training epoch, which both acts as data augmentation and provably slows the convergence toward over-smoothing by reducing the graph's connectivity. It is a one-line change that reliably helps, particularly on deeper models, and it costs nothing at inference time. The theoretical analysis relating edge dropping to the rate of information loss is the part worth reading.

**5. PairNorm: tackling oversmoothing in GNNs**  
*preprint (arXiv)*, 2019 · 159 citations · outside the top 50 on citations · [10.48550/arxiv.1909.12223](https://doi.org/10.48550/arxiv.1909.12223)  
`normalisation that preserves pair distances`

PairNorm normalizes representations after each layer so that the *total pairwise squared distance* between node representations is preserved, directly counteracting the contraction that over-smoothing produces. It is a normalization layer, not an architectural change, so it can be added to any GNN. Together with DropEdge it forms the standard pair of cheap fixes to try before restructuring a deep model.

---

### II.8 Pooling and graph-level readout

Producing one vector per graph, for the tasks where each sample is its own graph.

**1. An end-to-end deep learning architecture for graph classification (DGCNN)**  
*AAAI*, 2018 · 1,520 citations · citation rank 32/59 · [10.1609/aaai.v32i1.11782](https://doi.org/10.1609/aaai.v32i1.11782)  
`SortPooling for graph-level readout`

DGCNN addresses graph *classification*, where the output must be one vector per graph regardless of size or node ordering. Its SortPooling layer sorts nodes by their structural roles — using the continuous WL colours the graph convolutions produce — and keeps a fixed number of the top nodes, yielding a consistent ordering that a standard 1-D CNN can then consume. Solving the ordering problem by sorting rather than by summing is what makes the graph-level representation informative rather than merely permutation-invariant.

**2. Hierarchical graph representation learning with differentiable pooling (DiffPool)**  
*preprint (arXiv)*, 2018 · 844 citations · citation rank 47/59 · [10.48550/arxiv.1806.08804](https://doi.org/10.48550/arxiv.1806.08804)  
`learned soft cluster assignments`

DiffPool learns a hierarchical, differentiable clustering of nodes: at each pooling layer, a GNN predicts a soft assignment matrix mapping nodes to a smaller set of clusters, and both features and adjacency are coarsened accordingly. This gives graph classification something analogous to the spatial hierarchy of a CNN, and the learned clusters are often interpretable as functional modules. Auxiliary link-prediction and entropy losses are needed to keep the assignments from degenerating — a practical detail worth noting.

---

### II.9 Expressivity and theory

What message passing provably can and cannot distinguish, expressed through the Weisfeiler-Leman hierarchy.

**1. Weisfeiler and Leman go neural: higher-order graph neural networks (k-GNN)**  
*AAAI*, 2019 · 1,238 citations · citation rank 42/59 · [10.1609/aaai.v33i01.33014602](https://doi.org/10.1609/aaai.v33i01.33014602)  
`WL hierarchy as an expressivity yardstick`

Morris et al. establish the correspondence between message-passing GNNs and the one-dimensional Weisfeiler–Leman graph isomorphism test: no such GNN can distinguish two graphs that 1-WL cannot. They then define k-GNNs operating on k-tuples of nodes, which match the more powerful k-WL hierarchy at correspondingly higher cost. This gives the field its yardstick for expressive power and explains concretely why standard GNNs fail on tasks requiring cycle counting or substructure detection.

**2. How powerful are graph neural networks? (GIN)**  
*preprint (arXiv)*, 2018 · 385 citations · outside the top 50 on citations · [10.48550/arxiv.1810.00826](https://doi.org/10.48550/arxiv.1810.00826)  
`WL-equivalent expressivity; sum aggregation`

GIN answers what the *most* expressive message-passing GNN looks like: the aggregation function must be injective on multisets, which mean and max pooling are not and sum pooling followed by an MLP is. The resulting architecture is exactly as powerful as 1-WL, the theoretical maximum for this class. The paper's analysis of why mean aggregation fails — it cannot distinguish neighbourhoods differing only in multiplicity — is essential, and applies directly to biological graphs where node counts matter. Its OpenAlex count is split across records and drastically understates its standing as core curriculum.

---

### II.10 Self-supervised and contrastive learning

Pretraining without labels. The augmentation design question here is what every contrastive method in Part I is answering, usually without saying so.

**1. Graph contrastive learning with augmentations (GraphCL)**  
*preprint (arXiv)*, 2020 · 866 citations · citation rank 46/59 · [10.48550/arxiv.2010.13902](https://doi.org/10.48550/arxiv.2010.13902)  
`augmentation taxonomy for graphs`

GraphCL provides the first systematic taxonomy of augmentations for graph contrastive learning — node dropping, edge perturbation, attribute masking, subgraph sampling — and studies which combinations help on which types of graph. The central finding is that the right augmentation is domain-dependent, and that augmentations which destroy the property being predicted actively hurt. Every contrastive method in Part I is choosing a point in the design space this paper maps.

**2. Deep Graph Infomax (DGI)**  
*preprint (arXiv)*, 2018 · 83 citations · outside the top 50 on citations · [10.48550/arxiv.1809.10341](https://doi.org/10.48550/arxiv.1809.10341)  
`mutual-information self-supervision`

Deep Graph Infomax trains node embeddings by maximizing mutual information between local patch representations and a global summary of the graph, with negatives generated by corrupting the feature matrix. It needs no augmentation design and no negative sampling over node pairs, which makes it unusually robust, and it was the first convincing demonstration of unsupervised GNN pretraining. CCST in Part I uses it directly. Its citation count here is a preprint-record artefact, not a measure of influence.

---

### II.11 Dynamic and spatio-temporal graphs

Graphs that change over time, and graphs learned from data when none is given.

**1. Connecting the dots: multivariate time series forecasting with GNNs (MTGNN)**  
*KDD*, 2020 · 1,947 citations · citation rank 23/59 · [10.1145/3394486.3403118](https://doi.org/10.1145/3394486.3403118)  
`learns the graph when none is given`

MTGNN forecasts multivariate time series by *learning* the graph among variables rather than requiring one as input, using a graph learning layer that produces a sparse adjacency from learned node embeddings, combined with dilated temporal convolutions. Learning the graph is directly relevant wherever the relational structure is unknown — which describes most biological applications, where prior networks are incomplete and biased.

**2. EvolveGCN: evolving graph convolutional networks for dynamic graphs**  
*AAAI*, 2020 · 1,214 citations · citation rank 43/59 · [10.1609/aaai.v34i04.5984](https://doi.org/10.1609/aaai.v34i04.5984)  
`RNN over the GCN weights`

EvolveGCN handles graphs whose topology changes over time by using an RNN to evolve the GCN's *weight matrices* across time steps, rather than evolving node embeddings. This keeps the model well-defined when nodes appear and disappear between snapshots, which embedding-based dynamic methods struggle with. The relevant biological analogue is a developmental or perturbation time course where the cell population itself changes.

---

### II.12 Explainability

Turning a prediction into a subgraph a domain expert can read — and the stability problems that come with it.

**1. GNNExplainer: generating explanations for graph neural networks**  
*preprint (arXiv)*, 2019 · 701 citations · citation rank 49/59 · [10.48550/arxiv.1903.03894](https://doi.org/10.48550/arxiv.1903.03894)  
`subgraph + feature masks; used by GNN-SubNet`

GNNExplainer explains an individual prediction by optimizing for a compact subgraph and a small subset of node features that together preserve the model's output, formulated as maximizing mutual information between the masked input and the prediction. It is model-agnostic, works for node, edge and graph-level tasks, and returns something biologists can read: a small connected module. Its known weaknesses — instability across runs and sensitivity to the sparsity penalty — are why GNN-SubNet in Part I runs it repeatedly and takes a consensus.

---

### II.13 Domain landmarks

Papers from outside biology whose architecture became a general template. Each one is the direct ancestor of something in Part I.

**1. Spatial temporal GCN for skeleton-based action recognition (ST-GCN)**  
*AAAI*, 2018 · 5,046 citations · citation rank 9/59 · [10.1609/aaai.v32i1.12328](https://doi.org/10.1609/aaai.v32i1.12328)  
`the spatio-temporal graph template`

ST-GCN, for skeleton-based action recognition, is the template for spatio-temporal graph modelling: a spatial graph over body joints, temporal edges linking the same joint across frames, and convolution over the combined structure with a partitioning strategy that gives different neighbours different weights. The architecture generalizes far beyond its application — any problem with a fixed relational structure evolving over time can be posed this way, including cell-neighbourhood dynamics in live imaging.

**2. LightGCN: simplifying and powering GCN for recommendation**  
*SIGIR*, 2020 · 4,382 citations · citation rank 12/59 · [10.1145/3397271.3401063](https://doi.org/10.1145/3397271.3401063)  
`strips nonlinearity and feature transform`

LightGCN strips a graph convolution used for recommendation down to nothing but neighbourhood aggregation — no feature transformation, no non-linearity — and finds that this performs *better* than the full model, not merely comparably. Alongside SGC it is the strongest evidence that the useful content of many graph networks is the propagation, and that the learned transformations are often just capacity to overfit. A short, salutary paper.

**3. Graph convolutional networks for text classification (TextGCN)**  
*AAAI*, 2019 · 2,003 citations · citation rank 22/59 · [10.1609/aaai.v33i01.33017370](https://doi.org/10.1609/aaai.v33i01.33017370)  
`corpus-level word-document graph`

TextGCN builds a single heterogeneous graph over an entire corpus, with word and document nodes joined by TF-IDF and pointwise-mutual-information edges, and classifies documents as node classification on that graph. Turning a set of independent samples into one graph by connecting them through shared features is exactly the construction that cell–gene bipartite methods use in Part I, and this is the clearest statement of it outside biology.

**4. E(3)-equivariant graph neural networks for interatomic potentials (NequIP)**  
*Nature Communications*, 2022 · 1,846 citations · citation rank 26/59 · [10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5)  
`equivariance as an architectural prior`

NequIP builds E(3) equivariance — rotation, translation and reflection symmetry — into the network through tensor-field operations, so that predicted forces transform correctly with the coordinate frame by construction rather than by learning. The result reaches chemical accuracy on interatomic potentials with orders of magnitude less training data than non-equivariant models. It is the strongest available demonstration that encoding a known symmetry as an architectural constraint buys enormous data efficiency, a lesson that spatial-omics models, which also live in Euclidean space, have barely begun to exploit.

**5. Session-based recommendation with graph neural networks (SR-GNN)**  
*AAAI*, 2019 · 1,514 citations · citation rank 33/59 · [10.1609/aaai.v33i01.3301346](https://doi.org/10.1609/aaai.v33i01.3301346)  
`session graphs over item transitions`

SR-GNN models a user session as a small directed graph over the items viewed, learns item representations by gated message passing over it, and combines them with attention into a session representation for next-item prediction. Its relevance here is the pattern: each sample is its own small graph, and the model must produce a graph-level representation — the same structure as scGraph's per-cell gene networks in Part I.

**6. Modeling polypharmacy side effects with graph convolutional networks (Decagon)**  
*Bioinformatics*, 2018 · 1,492 citations · citation rank 34/59 · [10.1093/bioinformatics/bty294](https://doi.org/10.1093/bioinformatics/bty294)  
`the multi-relational biomedical GCN template`

Decagon predicts polypharmacy side effects on a multi-modal graph of drugs, proteins and side-effect-typed drug–drug edges, framing the task as multi-relational link prediction with a per-side-effect decoder. It is the paper that established graph neural networks as a serious tool in biomedicine, and its architecture — heterogeneous biomedical graph, relation-specific decoders, link prediction as the objective — is the direct ancestor of the patient-level and driver-gene methods in Part I, Section 7.

---

### II.14 Libraries and benchmarks

The implementation everything is written in, and the evaluation suite that showed why older benchmarks had stopped being informative.

**1. Fast graph representation learning with PyTorch Geometric**  
*preprint (arXiv)*, 2019 · 1,258 citations · citation rank 41/59 · [10.48550/arxiv.1903.02428](https://doi.org/10.48550/arxiv.1903.02428)  
`the library most sc-GNN code is written in`

PyTorch Geometric is the library most of the code in Part I is written in. Its contribution is a sparse gather–scatter message-passing abstraction that lets a new layer be defined by writing only the message and update functions, plus efficient CUDA kernels, mini-batching by block-diagonal graph concatenation, and implementations of most published architectures. Understanding its `MessagePassing` base class is the fastest route from the mathematics in this section to running code.

**2. Open Graph Benchmark: datasets for machine learning on graphs**  
*preprint (arXiv)*, 2020 · 491 citations · outside the top 50 on citations · [10.48550/arxiv.2005.00687](https://doi.org/10.48550/arxiv.2005.00687)  
`the standard evaluation suite`

The Open Graph Benchmark exists because results on Cora, Citeseer and Pubmed had stopped being informative — those graphs are small, and standard splits had been overfitted for years. OGB supplies large, realistic datasets with meaningful, non-random splits (by time, by scaffold, by species) and a fixed evaluation protocol. Its central finding, that rankings change substantially under realistic splits, should be read as a warning about every benchmark table in Part I of this list.

---

## How this list was built

Candidates for **Part I** came from two OpenAlex title/abstract queries — graph methods (*graph neural network* / *convolutional* / *attention* / *autoencoder*) crossed with single-cell, multi-omics and spatial transcriptomics — plus a hand-curated set of canonical integration methods and foundation models retrieved by DOI. 182 papers survived screening; clearly out-of-scope hits (drug-discovery reviews, non-biological GNN applications) and preprint duplicates of published papers were removed.

**Part II** was built the same way: a citation-sorted sweep for graph learning terms, then DOI lookups for the canon a keyword search misses (GraphSAGE, GIN, ChebNet, DiffPool and the rest). A large traffic-forecasting cluster (STGCN, T-GCN, ASTGCN, Graph WaveNet; 1,300–3,600 citations each) was screened out as domain-specific; the domain landmarks that stayed are the ones whose architecture became a general template.

**Sub-categories in version 2** were assigned by task rather than by method family, on the grounds that methods solving different problems are not comparable however similar their architectures. Where a paper could sit in two places it was assigned to the task its evaluation actually measures. The introductions were written from the papers' own claims and from the benchmarking literature; where an independent re-evaluation contradicts a paper's self-reported performance, the introduction says so.

### Caveats

**Counts favour older work.** A 2019 method has had six years to accumulate what a 2025 method has had months for. Read the ranking as influence-to-date, not as quality. Several recent core methods — SpaMosaic, MultiGATE, Segger, DANCE — matter far more for this project than their rank suggests.

**OpenAlex splits preprint and published records.** Anything marked *preprint (arXiv)* is a lower bound with its conference citations counted on a separate record. GIN (385) and Deep Graph Infomax (83) are the worst affected — both are core curriculum whose true totals run into the thousands.

**OpenAlex totals run lower than Google Scholar.** Expect roughly 2–4x higher figures there.

**Benchmark tables are not to be trusted individually.** The independent re-evaluations in Section 9 — and the Open Graph Benchmark in Part II — consistently find that rankings change under common preprocessing and realistic data splits. Where two papers claim to beat each other, both are usually right on their own protocol.

