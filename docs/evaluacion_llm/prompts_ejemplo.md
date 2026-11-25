# Prompts de la evaluación con modelos de lenguaje

Generados con `src/api/prompts_llm.py` para una cita del test. El mensaje de sistema es fijo por enfoque; el mensaje del usuario cambia con cada cita.

## E1 · zero-shot genérico

**Sistema**

```text
Classify the rhetorical function of a citation in a scientific paper.

You receive the citation context, the section of the citing paper where it appears, and the title and abstract of the cited paper.

Score each of the following 9 citation functions between 0.0 and 1.0, in this exact order:
Background, Gap, Basis, Comparison, Application, Modification_Improvement, Evidence, Identification_of_the_Originator, Further_Reading.

Answer only with a list of 9 numbers, for example [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]. Do not add any other text.
```

**Usuario**

```text
Citation context: "…models have demonstrated substantial capability from function-level synthesis (Chen et al. 2021) to repository-level editing (Jimenez et al. 2023; Jiang et al. 2026), extended to structured generation conforming to domain-specific schemas (Poesia et al. 2022) and creative domains balancing…"
Section: Scientific Body (Seminal: A Survey on Large Language Models for Co)
Cited paper title: A Survey on Large Language Models for Code Generation
Cited paper abstract: Large Language Models (LLMs) have garnered remarkable advancements across diverse code-related tasks, known as Code LLMs, particularly in code generation that generates source code with LLM from natural language descriptions. This burgeoning field has captured significant interest from both academic researchers and industry professionals due to its practical significance in software development, e.g., GitHub Copilot. Despite the active exploration of LLMs for a variety of code tasks, either from the perspective of Natural Language Processing (NLP) or Software Engineering (SE) or both, there is a noticeable absence of a comprehensive and up-to-date literature review dedicated to LLM for code generation. In this survey, we aim to bridge this gap by providing a systematic literature review that serves as a valuable reference for researchers investigating the cutting-edge progress in LLMs for code generation. We introduce a taxonomy to categorize and discuss the recent developments in LLMs for code generation, covering aspects such as data curation, latest advances, performance evaluation, ethical implications, environmental impact, and real-world applications. In addition, we present a historical overview of the evolution of LLMs for code generation and provide a quantitative and qualitative comparative analysis of experimental results of code LLMs, sourced from their original papers to ensure a fair comparison on the HumanEval, MBPP, and BigCodeBench benchmarks, across various levels of difficulty and types of programming tasks, to highlight the progressive enhancements in LLM capabilities for code generation. We identify critical challenges and promising opportunities regarding the gap between academia and practical development. Furthermore, we have established a dedicated resource GitHub page (https://github.com/juyongjiang/CodeLLMSurvey) to continuously document and disseminate the most recent advances in the field.
```

## E2 · zero-shot con criterios

**Sistema**

```text
Classify the rhetorical function of a citation in a scientific paper.

You receive the citation context, the section of the citing paper where it appears, and the title and abstract of the cited paper.

Score each of the following 9 citation functions between 0.0 and 1.0, in this exact order:
Background, Gap, Basis, Comparison, Application, Modification_Improvement, Evidence, Identification_of_the_Originator, Further_Reading.

Definitions and criteria of each citation function:
1. Background
   Definition: Citations used to provide context, summarize the general background of a research topic, or trace the history of a field or idea. They lay the foundation for understanding the focal study.
   Criteria: Reviews existing literature; cites review papers or previous studies depicting the current state of the field; highlights important or prevailing findings; traces historical development or key ideas.
2. Gap
   Definition: Citations that help identify research gaps or unexplored areas, justifying the author's choice of research topic.
   Criteria: Highlights what has or has not been done; identifies gaps in knowledge or areas for further study; justifies the relevance of the current research by contrasting it with existing work.
3. Basis
   Definition: Citations that provide a foundation for the current research, at a macro level (broad influence) or a micro level (specific key contributions).
   Criteria: Serves as the intellectual starting point; significantly shapes the research idea or hypothesis; the focal research builds upon, continues, or expands on the cited work.
4. Comparison
   Definition: Citations used to compare the current work with cited studies or to draw comparisons between cited studies.
   Criteria: Highlights similarities or differences; compares methodologies, findings, algorithms, data, or theoretical concepts; may point out advantages of the current study or link different cited works.
5. Application
   Definition: Citations that directly employ a method, technique, or tool from the cited work without modification.
   Criteria: Utilizes existing methods, algorithms, instruments, or data; employs the cited work as a practical tool (e.g., equations, analysis methods) for calculation or experimentation; applies it without modification.
6. Modification_Improvement
   Definition: Citations in which methods or tools from the cited work are adapted, expanded, or modified for the current research.
   Criteria: The cited method is adapted, improved, or extended to fit new experimental conditions or research objectives; uses the cited work as a foundation but modifies it to enhance precision or scope.
7. Evidence
   Definition: Citations used to support claims, hypotheses, or findings in the current research.
   Criteria: Supports arguments, hypotheses, or factual statements; justifies research design, methodologies, or experimental procedures; substantiates or explains findings; provides evidence to mitigate limitations or support further research suggestions.
8. Identification_of_the_Originator
   Definition: Citations used to acknowledge the original source of an idea, concept, method, or theory.
   Criteria: Identifies the original publication where a key idea or method was first introduced; acknowledges pioneers in the field; gives credit to the priority of a cited work.
9. Further_Reading
   Definition: Citations that direct readers to additional or supplementary literature for more detailed information or context.
   Criteria: Alerts readers to new, different, or relevant sources of information; provides more complete details on data, methods, or background not fully covered in the current work.

Answer only with a list of 9 numbers, for example [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]. Do not add any other text.
```

**Usuario**

```text
Citation context: "…models have demonstrated substantial capability from function-level synthesis (Chen et al. 2021) to repository-level editing (Jimenez et al. 2023; Jiang et al. 2026), extended to structured generation conforming to domain-specific schemas (Poesia et al. 2022) and creative domains balancing…"
Section: Scientific Body (Seminal: A Survey on Large Language Models for Co)
Cited paper title: A Survey on Large Language Models for Code Generation
Cited paper abstract: Large Language Models (LLMs) have garnered remarkable advancements across diverse code-related tasks, known as Code LLMs, particularly in code generation that generates source code with LLM from natural language descriptions. This burgeoning field has captured significant interest from both academic researchers and industry professionals due to its practical significance in software development, e.g., GitHub Copilot. Despite the active exploration of LLMs for a variety of code tasks, either from the perspective of Natural Language Processing (NLP) or Software Engineering (SE) or both, there is a noticeable absence of a comprehensive and up-to-date literature review dedicated to LLM for code generation. In this survey, we aim to bridge this gap by providing a systematic literature review that serves as a valuable reference for researchers investigating the cutting-edge progress in LLMs for code generation. We introduce a taxonomy to categorize and discuss the recent developments in LLMs for code generation, covering aspects such as data curation, latest advances, performance evaluation, ethical implications, environmental impact, and real-world applications. In addition, we present a historical overview of the evolution of LLMs for code generation and provide a quantitative and qualitative comparative analysis of experimental results of code LLMs, sourced from their original papers to ensure a fair comparison on the HumanEval, MBPP, and BigCodeBench benchmarks, across various levels of difficulty and types of programming tasks, to highlight the progressive enhancements in LLM capabilities for code generation. We identify critical challenges and promising opportunities regarding the gap between academia and practical development. Furthermore, we have established a dedicated resource GitHub page (https://github.com/juyongjiang/CodeLLMSurvey) to continuously document and disseminate the most recent advances in the field.
```

## E3 · few-shot genérico

**Sistema**

```text
Classify the rhetorical function of a citation in a scientific paper.

You receive the citation context, the section of the citing paper where it appears, and the title and abstract of the cited paper.

Score each of the following 9 citation functions between 0.0 and 1.0, in this exact order:
Background, Gap, Basis, Comparison, Application, Modification_Improvement, Evidence, Identification_of_the_Originator, Further_Reading.

Solved examples:

Example 1
Citation context: "In contrast, models such as Llama 3.1 [11], Gemma 2 [51], and Mistral [3] show lower computation accuracy, particularly on the Avatar dataset."
Section: Scientific Body (Targeted Ingestion)
Cited paper title: Gemma 2: Improving Open Language Models at a Practical Size
Cited paper abstract: In this work, we introduce Gemma 2, a new addition to the Gemma family of lightweight, state-of-the-art open models, ranging in scale from 2 billion to 27 billion parameters. In this new version, we apply several known technical modifications to the Transformer architecture, such as interleaving local-global attentions (Beltagy et al., 2020a) and group-query attention (Ainslie et al., 2023). We also train the 2B and 9B models with knowledge distillation (Hinton et al., 2015) instead of next token prediction. The resulting models deliver the best performance for their size, and even offer competitive alternatives to models that are 2-3 times bigger. We release all our models to the community.
Answer: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Example 2
Citation context: "The results linking neutrophil ROS production with NETosis and subsequent bacterial killing are consistent with several prior reports (18, 35, 36)."
Section: Discussion
Cited paper title: not available
Cited paper abstract: not available
Answer: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

Example 3
Citation context: "Second, a novel role of endocytosis in turning on activation of the ERK/MAPK cascade by cell surface receptors was first reported for the EGF receptor [15]."
Section: III. DISCUSSION
Cited paper title: Principles of MAP kinase signaling specificity in Saccharomyces cerevisiae.
Cited paper abstract: not available
Answer: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

Example 4
Citation context: "Gene function classification 48 We use single-variable slice sampling (Neal, 2003) to sample from the posterior distribution of hyperparameters."
Section: 3.4.1 Implementation
Cited paper title: Slice Sampling
Cited paper abstract: not available
Answer: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

Example 5
Citation context: "Since a detailed survey of the available ML algorithms is out of the scope of this paper, we refer interested readers to [113, 157, 116, 29]."
Section: Scientific Body (Seminal: A Survey of Transformers)
Cited paper title: A Survey of Transformers
Cited paper abstract: Transformers have achieved great success in many artificial intelligence fields, such as natural language processing, computer vision, and audio processing. Therefore, it is natural to attract lots of interest from academic and industry researchers. Up to the present, a great variety of Transformer variants (a.k.a. X-formers) have been proposed, however, a systematic and comprehensive literature review on these Transformer variants is still missing. In this survey, we provide a comprehensive review of various X-formers. We first briefly introduce the vanilla Transformer and then propose a new taxonomy of X-formers. Next, we introduce the various X-formers from three perspectives: architectural modification, pre-training, and applications. Finally, we outline some potential directions for future research.
Answer: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

Example 6
Citation context: "The implemented deep learning model for real-time segmentation of iris and pupil is based on the MedFormer architecture described in 19 and is illustrated in Figure 1."
Section: Scientific Body (transformer architecture language model)
Cited paper title: A Data-scalable Transformer for Medical Image Segmentation: Architecture, Model Efficiency, and Benchmark
Cited paper abstract: Transformers have demonstrated remarkable performance in natural language processing and computer vision. However, existing vision Transformers struggle to learn from limited medical data and are unable to generalize on diverse medical image tasks. To tackle these challenges, we present MedFormer, a data-scalable Transformer designed for generalizable 3D medical image segmentation. Our approach incorporates three key elements: a desirable inductive bias, hierarchical modeling with linear-complexity attention, and multi-scale feature fusion that integrates spatial and semantic information globally. MedFormer can learn across tiny- to large-scale data without pre-training. Comprehensive experiments demonstrate MedFormer's potential as a versatile segmentation backbone, outperforming CNNs and vision Transformers on seven public datasets covering multiple modalities (e.g., CT and MRI) and various medical targets (e.g., healthy organs, diseased tissues, and tumors). We provide public access to our models and evaluation pipeline, offering solid baselines and unbiased comparisons to advance a wide range of downstream clinical applications.
Answer: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Example 7
Citation context: "We use a modified directed message-passing neural network (D-MPNN) 35,36 to predict retention order indices."
Section: Scientific Body (Targeted Ingestion)
Cited paper title: Chemprop: A Machine Learning Package for Chemical Property Prediction
Cited paper abstract: Deep learning has become a powerful and frequently employed tool for the prediction of molecular properties, thus creating a need for open-source and versatile software solutions that can be operated by nonexperts. Among the current approaches, directed message-passing neural networks (D-MPNNs) have proven to perform well on a variety of property prediction tasks. The software package Chemprop implements the D-MPNN architecture and offers simple, easy, and fast access to machine-learned molecular properties. Compared to its initial version, we present a multitude of new Chemprop functionalities such as the support of multimolecule properties, reactions, atom/bond-level properties, and spectra. Further, we incorporate various uncertainty quantification and calibration methods along with related metrics as well as pretraining and transfer learning workflows, improved hyperparameter optimization, and other customization options concerning loss functions or atom/bond features. We benchmark D-MPNN models trained using Chemprop with the new reaction, atom-level, and spectra functionality on a variety of property prediction data sets, including MoleculeNet and SAMPL, and observe state-of-the-art performance on the prediction of water-octanol partition coefficients, reaction barrier heights, atomic partial charges, and absorption spectra. Chemprop enables out-of-the-box training of D-MPNN models for a variety of problem settings in fast, user-friendly, and open-source software.
Answer: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

Example 8
Citation context: "RSs are widely used in areas such as e-commerce, social networking, entertainment, online learning, and native application platforms [2], [8]."
Section: Scientific Body (systematic review survey deep learning)
Cited paper title: Deep learning approaches to address cold start and long tail challenges in recommendation systems: a systematic review
Cited paper abstract: not available
Answer: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Example 9
Citation context: "However, there are still few studies about the application of deep learning approaches in piRNA prediction [14, 12]."
Section: Scientific Body (computational biology genomics machine learning)
Cited paper title: Computational biology: deep learning
Cited paper abstract: Deep learning is the trendiest tool in a computational biologist's toolbox. This exciting class of methods, based on artificial neural networks, quickly became popular due to its competitive performance in prediction problems. In pioneering early work, applying simple network architectures to abundant data already provided gains over traditional counterparts in functional genomics, image analysis, and medical diagnostics. Now, ideas for constructing and training networks and even off-the-shelf models have been adapted from the rapidly developing machine learning subfield to improve performance in a range of computational biology tasks. Here, we review some of these advances in the last 2 years.
Answer: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Answer only with a list of 9 numbers, for example [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]. Do not add any other text.
```

**Usuario**

```text
Citation context: "…models have demonstrated substantial capability from function-level synthesis (Chen et al. 2021) to repository-level editing (Jimenez et al. 2023; Jiang et al. 2026), extended to structured generation conforming to domain-specific schemas (Poesia et al. 2022) and creative domains balancing…"
Section: Scientific Body (Seminal: A Survey on Large Language Models for Co)
Cited paper title: A Survey on Large Language Models for Code Generation
Cited paper abstract: Large Language Models (LLMs) have garnered remarkable advancements across diverse code-related tasks, known as Code LLMs, particularly in code generation that generates source code with LLM from natural language descriptions. This burgeoning field has captured significant interest from both academic researchers and industry professionals due to its practical significance in software development, e.g., GitHub Copilot. Despite the active exploration of LLMs for a variety of code tasks, either from the perspective of Natural Language Processing (NLP) or Software Engineering (SE) or both, there is a noticeable absence of a comprehensive and up-to-date literature review dedicated to LLM for code generation. In this survey, we aim to bridge this gap by providing a systematic literature review that serves as a valuable reference for researchers investigating the cutting-edge progress in LLMs for code generation. We introduce a taxonomy to categorize and discuss the recent developments in LLMs for code generation, covering aspects such as data curation, latest advances, performance evaluation, ethical implications, environmental impact, and real-world applications. In addition, we present a historical overview of the evolution of LLMs for code generation and provide a quantitative and qualitative comparative analysis of experimental results of code LLMs, sourced from their original papers to ensure a fair comparison on the HumanEval, MBPP, and BigCodeBench benchmarks, across various levels of difficulty and types of programming tasks, to highlight the progressive enhancements in LLM capabilities for code generation. We identify critical challenges and promising opportunities regarding the gap between academia and practical development. Furthermore, we have established a dedicated resource GitHub page (https://github.com/juyongjiang/CodeLLMSurvey) to continuously document and disseminate the most recent advances in the field.
```

## E4 · few-shot con criterios

**Sistema**

```text
Classify the rhetorical function of a citation in a scientific paper.

You receive the citation context, the section of the citing paper where it appears, and the title and abstract of the cited paper.

Score each of the following 9 citation functions between 0.0 and 1.0, in this exact order:
Background, Gap, Basis, Comparison, Application, Modification_Improvement, Evidence, Identification_of_the_Originator, Further_Reading.

Definitions and criteria of each citation function:
1. Background
   Definition: Citations used to provide context, summarize the general background of a research topic, or trace the history of a field or idea. They lay the foundation for understanding the focal study.
   Criteria: Reviews existing literature; cites review papers or previous studies depicting the current state of the field; highlights important or prevailing findings; traces historical development or key ideas.
2. Gap
   Definition: Citations that help identify research gaps or unexplored areas, justifying the author's choice of research topic.
   Criteria: Highlights what has or has not been done; identifies gaps in knowledge or areas for further study; justifies the relevance of the current research by contrasting it with existing work.
3. Basis
   Definition: Citations that provide a foundation for the current research, at a macro level (broad influence) or a micro level (specific key contributions).
   Criteria: Serves as the intellectual starting point; significantly shapes the research idea or hypothesis; the focal research builds upon, continues, or expands on the cited work.
4. Comparison
   Definition: Citations used to compare the current work with cited studies or to draw comparisons between cited studies.
   Criteria: Highlights similarities or differences; compares methodologies, findings, algorithms, data, or theoretical concepts; may point out advantages of the current study or link different cited works.
5. Application
   Definition: Citations that directly employ a method, technique, or tool from the cited work without modification.
   Criteria: Utilizes existing methods, algorithms, instruments, or data; employs the cited work as a practical tool (e.g., equations, analysis methods) for calculation or experimentation; applies it without modification.
6. Modification_Improvement
   Definition: Citations in which methods or tools from the cited work are adapted, expanded, or modified for the current research.
   Criteria: The cited method is adapted, improved, or extended to fit new experimental conditions or research objectives; uses the cited work as a foundation but modifies it to enhance precision or scope.
7. Evidence
   Definition: Citations used to support claims, hypotheses, or findings in the current research.
   Criteria: Supports arguments, hypotheses, or factual statements; justifies research design, methodologies, or experimental procedures; substantiates or explains findings; provides evidence to mitigate limitations or support further research suggestions.
8. Identification_of_the_Originator
   Definition: Citations used to acknowledge the original source of an idea, concept, method, or theory.
   Criteria: Identifies the original publication where a key idea or method was first introduced; acknowledges pioneers in the field; gives credit to the priority of a cited work.
9. Further_Reading
   Definition: Citations that direct readers to additional or supplementary literature for more detailed information or context.
   Criteria: Alerts readers to new, different, or relevant sources of information; provides more complete details on data, methods, or background not fully covered in the current work.

Solved examples:

Example 1
Citation context: "In contrast, models such as Llama 3.1 [11], Gemma 2 [51], and Mistral [3] show lower computation accuracy, particularly on the Avatar dataset."
Section: Scientific Body (Targeted Ingestion)
Cited paper title: Gemma 2: Improving Open Language Models at a Practical Size
Cited paper abstract: In this work, we introduce Gemma 2, a new addition to the Gemma family of lightweight, state-of-the-art open models, ranging in scale from 2 billion to 27 billion parameters. In this new version, we apply several known technical modifications to the Transformer architecture, such as interleaving local-global attentions (Beltagy et al., 2020a) and group-query attention (Ainslie et al., 2023). We also train the 2B and 9B models with knowledge distillation (Hinton et al., 2015) instead of next token prediction. The resulting models deliver the best performance for their size, and even offer competitive alternatives to models that are 2-3 times bigger. We release all our models to the community.
Answer: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Example 2
Citation context: "The results linking neutrophil ROS production with NETosis and subsequent bacterial killing are consistent with several prior reports (18, 35, 36)."
Section: Discussion
Cited paper title: not available
Cited paper abstract: not available
Answer: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

Example 3
Citation context: "Second, a novel role of endocytosis in turning on activation of the ERK/MAPK cascade by cell surface receptors was first reported for the EGF receptor [15]."
Section: III. DISCUSSION
Cited paper title: Principles of MAP kinase signaling specificity in Saccharomyces cerevisiae.
Cited paper abstract: not available
Answer: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

Example 4
Citation context: "Gene function classification 48 We use single-variable slice sampling (Neal, 2003) to sample from the posterior distribution of hyperparameters."
Section: 3.4.1 Implementation
Cited paper title: Slice Sampling
Cited paper abstract: not available
Answer: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

Example 5
Citation context: "Since a detailed survey of the available ML algorithms is out of the scope of this paper, we refer interested readers to [113, 157, 116, 29]."
Section: Scientific Body (Seminal: A Survey of Transformers)
Cited paper title: A Survey of Transformers
Cited paper abstract: Transformers have achieved great success in many artificial intelligence fields, such as natural language processing, computer vision, and audio processing. Therefore, it is natural to attract lots of interest from academic and industry researchers. Up to the present, a great variety of Transformer variants (a.k.a. X-formers) have been proposed, however, a systematic and comprehensive literature review on these Transformer variants is still missing. In this survey, we provide a comprehensive review of various X-formers. We first briefly introduce the vanilla Transformer and then propose a new taxonomy of X-formers. Next, we introduce the various X-formers from three perspectives: architectural modification, pre-training, and applications. Finally, we outline some potential directions for future research.
Answer: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

Example 6
Citation context: "The implemented deep learning model for real-time segmentation of iris and pupil is based on the MedFormer architecture described in 19 and is illustrated in Figure 1."
Section: Scientific Body (transformer architecture language model)
Cited paper title: A Data-scalable Transformer for Medical Image Segmentation: Architecture, Model Efficiency, and Benchmark
Cited paper abstract: Transformers have demonstrated remarkable performance in natural language processing and computer vision. However, existing vision Transformers struggle to learn from limited medical data and are unable to generalize on diverse medical image tasks. To tackle these challenges, we present MedFormer, a data-scalable Transformer designed for generalizable 3D medical image segmentation. Our approach incorporates three key elements: a desirable inductive bias, hierarchical modeling with linear-complexity attention, and multi-scale feature fusion that integrates spatial and semantic information globally. MedFormer can learn across tiny- to large-scale data without pre-training. Comprehensive experiments demonstrate MedFormer's potential as a versatile segmentation backbone, outperforming CNNs and vision Transformers on seven public datasets covering multiple modalities (e.g., CT and MRI) and various medical targets (e.g., healthy organs, diseased tissues, and tumors). We provide public access to our models and evaluation pipeline, offering solid baselines and unbiased comparisons to advance a wide range of downstream clinical applications.
Answer: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Example 7
Citation context: "We use a modified directed message-passing neural network (D-MPNN) 35,36 to predict retention order indices."
Section: Scientific Body (Targeted Ingestion)
Cited paper title: Chemprop: A Machine Learning Package for Chemical Property Prediction
Cited paper abstract: Deep learning has become a powerful and frequently employed tool for the prediction of molecular properties, thus creating a need for open-source and versatile software solutions that can be operated by nonexperts. Among the current approaches, directed message-passing neural networks (D-MPNNs) have proven to perform well on a variety of property prediction tasks. The software package Chemprop implements the D-MPNN architecture and offers simple, easy, and fast access to machine-learned molecular properties. Compared to its initial version, we present a multitude of new Chemprop functionalities such as the support of multimolecule properties, reactions, atom/bond-level properties, and spectra. Further, we incorporate various uncertainty quantification and calibration methods along with related metrics as well as pretraining and transfer learning workflows, improved hyperparameter optimization, and other customization options concerning loss functions or atom/bond features. We benchmark D-MPNN models trained using Chemprop with the new reaction, atom-level, and spectra functionality on a variety of property prediction data sets, including MoleculeNet and SAMPL, and observe state-of-the-art performance on the prediction of water-octanol partition coefficients, reaction barrier heights, atomic partial charges, and absorption spectra. Chemprop enables out-of-the-box training of D-MPNN models for a variety of problem settings in fast, user-friendly, and open-source software.
Answer: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

Example 8
Citation context: "RSs are widely used in areas such as e-commerce, social networking, entertainment, online learning, and native application platforms [2], [8]."
Section: Scientific Body (systematic review survey deep learning)
Cited paper title: Deep learning approaches to address cold start and long tail challenges in recommendation systems: a systematic review
Cited paper abstract: not available
Answer: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Example 9
Citation context: "However, there are still few studies about the application of deep learning approaches in piRNA prediction [14, 12]."
Section: Scientific Body (computational biology genomics machine learning)
Cited paper title: Computational biology: deep learning
Cited paper abstract: Deep learning is the trendiest tool in a computational biologist's toolbox. This exciting class of methods, based on artificial neural networks, quickly became popular due to its competitive performance in prediction problems. In pioneering early work, applying simple network architectures to abundant data already provided gains over traditional counterparts in functional genomics, image analysis, and medical diagnostics. Now, ideas for constructing and training networks and even off-the-shelf models have been adapted from the rapidly developing machine learning subfield to improve performance in a range of computational biology tasks. Here, we review some of these advances in the last 2 years.
Answer: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Answer only with a list of 9 numbers, for example [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]. Do not add any other text.
```

**Usuario**

```text
Citation context: "…models have demonstrated substantial capability from function-level synthesis (Chen et al. 2021) to repository-level editing (Jimenez et al. 2023; Jiang et al. 2026), extended to structured generation conforming to domain-specific schemas (Poesia et al. 2022) and creative domains balancing…"
Section: Scientific Body (Seminal: A Survey on Large Language Models for Co)
Cited paper title: A Survey on Large Language Models for Code Generation
Cited paper abstract: Large Language Models (LLMs) have garnered remarkable advancements across diverse code-related tasks, known as Code LLMs, particularly in code generation that generates source code with LLM from natural language descriptions. This burgeoning field has captured significant interest from both academic researchers and industry professionals due to its practical significance in software development, e.g., GitHub Copilot. Despite the active exploration of LLMs for a variety of code tasks, either from the perspective of Natural Language Processing (NLP) or Software Engineering (SE) or both, there is a noticeable absence of a comprehensive and up-to-date literature review dedicated to LLM for code generation. In this survey, we aim to bridge this gap by providing a systematic literature review that serves as a valuable reference for researchers investigating the cutting-edge progress in LLMs for code generation. We introduce a taxonomy to categorize and discuss the recent developments in LLMs for code generation, covering aspects such as data curation, latest advances, performance evaluation, ethical implications, environmental impact, and real-world applications. In addition, we present a historical overview of the evolution of LLMs for code generation and provide a quantitative and qualitative comparative analysis of experimental results of code LLMs, sourced from their original papers to ensure a fair comparison on the HumanEval, MBPP, and BigCodeBench benchmarks, across various levels of difficulty and types of programming tasks, to highlight the progressive enhancements in LLM capabilities for code generation. We identify critical challenges and promising opportunities regarding the gap between academia and practical development. Furthermore, we have established a dedicated resource GitHub page (https://github.com/juyongjiang/CodeLLMSurvey) to continuously document and disseminate the most recent advances in the field.
```
