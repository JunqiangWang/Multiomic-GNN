# scMultiomic-integration-GNN — Reading List

An annotated bibliography of **241 papers** on single-cell multi-omic integration with graph neural networks, each with a short critical introduction. Version 2, 10 September 2026.

The list is written to be read, not just searched: every section opens with a paragraph explaining what problem the papers in it solve, and every entry says what the method does, what it buys you, and where it fails. Citation counts come from OpenAlex (retrieved 9 September 2026); every DOI was resolved against OpenAlex rather than generated, so the entries are safe to cite.

## Files

| File | What it is |
| --- | --- |
| `multiomic-gnn-reading-list.md` | The reading list. Markdown source, with a linked table of contents. |
| `multiomic-gnn-reading-list.pdf` | Typeset version of the same content, for reading and printing. |

## How it is organised

**Part I — Applied: single-cell and spatial multi-omics** (182 papers, citation range 17,390–16)

| § | Section | Papers |
| --- | --- | --- |
| 1 | Reference points: integration without graphs | 21 |
| 2 | Graph learning on single cells | 34 |
| 3 | Graph-based multi-omic integration | 9 |
| 4 | Spatial omics | 43 |
| 5 | Cell–cell communication | 5 |
| 6 | Gene regulatory network inference | 9 |
| 7 | Patient-level multi-omics | 48 |
| 8 | Single-cell foundation models | 4 |
| 9 | Reviews, benchmarks and software | 9 |

Sections 1, 2, 3, 4 and 7 are further split into task-level sub-categories (23 in total across the list) — spatial domain identification, deconvolution, histology-to-expression, patient-similarity networks, driver-gene discovery, and so on.

**Part II — Methods: the classical GNN canon** (59 papers, citation range 11,077–37)

Fourteen sections covering origins and surveys, shallow node embeddings, convolution and message passing, attention and graph transformers, heterogeneous and knowledge graphs, scalability, depth and over-smoothing, pooling, expressivity, self-supervised learning, dynamic graphs, explainability, domain landmarks, and libraries and benchmarks.

## Suggested paths through it

- **New to the area.** Section 1 first — it defines the tasks, the metrics and the non-graph performance a graph formulation has to beat. Then Part II.1 (one survey, properly) and II.3, then Section 2.
- **Building a method.** Section 3 for the guidance-graph formulations, Section 2.1 for cell-graph representation learning, Part II.5–II.7 for the architectural choices that decide whether it scales.
- **Working on spatial data.** Section 4, with Part II.3 and II.4 alongside; Section 4.1 is the largest and most crowded sub-category in the list.
- **Choosing a baseline.** Section 9 for the independent benchmarks, plus Section 1 — several non-graph methods there remain the better choice in practice.

## Entry format

```
**N. Title**
*Venue*, year · citations · citation rank N/182 · [DOI](link)
`one-line tag: what the method actually is`

Paragraph: the contribution, the mechanism, and the honest limitation.
```
