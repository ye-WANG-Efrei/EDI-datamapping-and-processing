# EDI Datamapping and Processing

## Overview

This repository is designed to handle Electronic Data Interchange (EDI) mapping and processing tasks, with a specific focus on integrating AI-powered tools for data manipulation. The project leverages multiple components for different functionalities, including language graph processing and advanced data handling via libraries like PandasAI.

## Structure

### `data`

This folder contains all the data files used by the system. The data is crucial for mapping and processing tasks.

### `demo`

The `demo` folder contains the main code, which is split into two sections:
1. **Langraph**: This section utilizes a local Large Language Model (LLM) for specific processing tasks.
2. **PandasAI_3 (now replaced by `pandas-ai`)**: Originally empowered by PandasAI_3, this section has transitioned to using the updated `pandas-ai` library (v3 beta) to perform advanced data manipulation tasks. Note that `Pandas_ai` is a replacement for `demo/pandasai_3`.

### `pandas-ai-v3`

This folder contains the beta version of the `pandas-ai` library. The open-source library enables seamless integration with pandas DataFrames and allows users to apply AI techniques for data processing.

### `pandas_ai`

This folder is a replacement for the deprecated `demo/pandasai_3` directory. It contains the updated version of the codebase using `pandas-ai`.

## Requirements

To run the project, you'll need the following dependencies:
- Python 3.10 or less than that
- Required packages listed in `requirements.txt`

## Installation

To install the dependencies, simply run the following command:
```bash
pip install -r requirements.txt
