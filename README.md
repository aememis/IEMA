# IEMA: Exploring Complex Data Spaces with Body-Motion Interaction
<!-- > **Interactive Evolutionary Motion-controlled Algorithm 🎶🤖** -->

Welcome to the open-source home of the **Interactive Evolutionary Motion-controlled Algorithm (IEMA)**! This repository is your gateway to a novel approach for navigating and discovering structure in complex data spaces, powered by evolutionary algorithms and body interaction. Here, you’ll find everything you need to run IEMA, experiment with its evaluation framework, and dive into the research that inspired its creation.

IEMA is designed for researchers, developers, and artists interested in interactive music systems and intelligent audio retrieval. By blending evolutionary search with real-time body-motion control, IEMA transforms the process of exploring high-dimensional audio datasets into an intuitive, creative, and engaging experience. This repository provides the tools and documentation to get you started.

Please do not hesitate to reach out if you have any questions or feedback!

## Table of Contents 📚

1. [About the Research](#about)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contact](#contact)


## About the Research 📖

Interactive search in complex data spaces is a crucial and challenging task in audio data retrieval field, where efficient methods are needed to recommend users relevant content from vast audio databases. This task can be addressed through various approaches, often combining multiple techniques together. This thesis proposes the Immersive Evolutionary Motion-controlled Algorithm (IEMA), an interactive system that combines Interactive Evolutionary Algorithms and body motion interaction to guide the exploration of discrete audio data. To aid in the design of such systems, it also presents an evaluation framework that informs how to develop and assess similar interactive evolutionary systems. Through stage-wise prototype design and experimentation, the thesis demonstrates the applicability of IEMA for high-dimensional discrete audio search. The thesis also shares the software implementation of IEMA as open-source. Furthermore, it highlights potential areas for future research in the field of interactive evolutionary music systems[^1].

[^1]: [Memis, A. E. (2025). IEMA: Exploring Complex Data Spaces with Body-Motion Interaction. Master's thesis, University of Oslo.](https://www.duo.uio.no/)

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

## Usage 🪡

*more coming soon...*

### **Directory Structure** 📂

```
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

## Contact 📫
**Ahmet Emin Memis 🤷‍♂️**
- **[Email 📨](mailto:ahmeteminmemis@gmail.com)**
- **[GitHub 🐱](https://github.com/aememis)**
- **[LinkedIn 💼](https://www.linkedin.com/in/aememis/)**

🎶🤖