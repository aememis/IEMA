# IEMA: Exploring Complex Data Spaces with Body-Motion Interaction
> **Interactive Evolutionary Motion-controlled Algorithm 🎶🤖**

Welcome to the open-source repository for the **Interactive Evolutionary Motion-controlled Algorithm (IEMA)**. This repository provides the codebase for running the IEMA, conducting evaluation tests, and serves as a reference for the research and development of the system. It facilitates the exploration of complex data spaces using evolutionary algorithms with user interaction, specifically within the domain of interactive music systems.

## Table of Contents 📚

1. [About](#about)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Directory Structure](#directory-structure)
5. [Evaluation Tests](#evaluation-tests)
6. [License](#license)
7. [Contact](#contact)

## About 📖

IEMA explores the application of evolutionary algorithms (EA) in an interactive music system. It adapts its search process based on user feedback, using body-motion interaction to guide the exploration of soundscapes. The system leverages concepts like **novel selection**, **dimensionality reduction** (e.g., PCA and t-SNE), and **augmented crossover** to dynamically evolve the sound samples and improve user-driven music composition.

This repository contains the code for:

- Running the IEMA algorithm
- Evaluating different configurations of the system
- Preprocessing and analyzing datasets
- Visualizing results

## Installation ⚙️

To get started with IEMA, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/aememis/IEMA.git
    ```

2. **Install dependencies**:
   Make sure you have `Python 3.10+` installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Usage**:
    ...

4. **Directory Structure** 📂
  <!-- ```
  .
  ├── code/                  # Main codebase for running and evaluating IEMA
  │  ├── config.py           # Configuration settings for the algorithm
  │  ├── corpus_gauss.py     # Handles Gaussian sampling for the corpus
  │  ├── dataset.py          # Dataset loading and preprocessing
  │  ├── evaluation.py       # Evaluation scripts for testing IEMA performance
  │  ├── individual.py       # Defines the individual representation in IEMA
  │  ├── main.py             # Entry point for running IEMA
  │  ├── operations.py       # Evolutionary operations and utilities
  │  ├── path.py             # Path handling for data analysis
  │  └── plot_results.ipynb  # Results visualization
  ├── analyze_datasets/      # Scripts and data for dataset analysis
  │  └── fsd50k/             # FSD50K dataset folder (example)
  ├── output/                # Output of IEMA runs and results
  └── precomputed/           # Precomputed features and samples
  ``` -->

```bash
.
│   README.md
│   requirements.txt
│   
└───code
    │   config.py                  # Configuration settings for the algorithm
    │   corpus_gauss.py            # Creates Gaussian-randomized corpus
    │   corpus_reader.py           # Reads and preprocesses the dataset
    │   dataset.py                 # Dataset ontology-related tasks
    │   evaluation.py              # Evaluation scripts
    │   evaluation_notebook.ipynb  # Playground notebook for evaluation
    │   main.py
    │   operations.py              # Evolutionary operations and utilities
    │   path.py                    # Path creating and handling
    │   plot_results.ipynb         # Notebook for visualizing results
    │   run_configs.json           # Configuration for running IEMA
    │   
    ├───analyze_datasets
    │   └───fsd50k
    │           dataset_ontology.gpickle  # Dataset ontology (networkx)
    │           fsd50k.ipynb              # Analysis and preprocessing
    │           layers.json               # Dataset ontology layers
    │           ontology.json             # Dataset ontology in JSON format
    │
    ├───output
    │   └───00000000_000000
    │           placeholder  # Placeholder for output directory
    │
    ├───plot_output
    │       (plots)  # Visualization of the results
    │
    └───precomputed  # Precomputed features and samples
            features.pkl
            features_raw.pkl
            filenames.pkl
            gaussian_features_norm.pkl
            gaussian_features_raw.pkl
            gaussian_samples.pkl
            paths.pkl
            samples.pkl
```
<!-- > [Detailed Explanation for the repo directory](https://github.com/aememis/thesis/blob/main/directory_tree.txt) -->
