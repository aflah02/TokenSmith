# TokenSmith Tutorials

This directory contains step-by-step tutorials for using the TokenSmith library for dataset management, sampling, and manipulation.

## Tutorial Overview

The tutorials are designed to be followed in order, building from basic concepts to advanced workflows:

### 📚 Basic Tutorials
1. **[01_basic_setup.ipynb](01_basic_setup.ipynb)** - Setup and initialization
   - Installing dependencies
   - Setting up the DatasetManager
   - Loading tokenizers
   - Basic configuration

2. **[02_inspect_samples.ipynb](02_inspect_samples.ipynb)** - Dataset inspection
   - Inspecting individual samples
   - Batch inspection
   - Understanding document details
   - Viewing tokenized vs detokenized content

3. **[03_sampling_methods.ipynb](03_sampling_methods.ipynb)** - Basic sampling
   - Sampling by indices
   - Batch sampling by IDs
   - Understanding return formats
   - Working with document metadata

### 🎯 Advanced Tutorials
4. **[04_policy_sampling.ipynb](04_policy_sampling.ipynb)** - Policy-based sampling
   - Creating custom sampling policies
   - Random, sequential, and sparse sampling
   - Batch policy sampling
   - Lambda functions as policies

5. **[05_search_functionality.ipynb](05_search_functionality.ipynb)** - Search and filtering
   - Setting up search indices
   - Token-based search
   - Counting occurrences
   - Advanced search patterns

6. **[06_editing_injection.ipynb](06_editing_injection.ipynb)** - Data editing and injection
   - Injecting custom text
   - Different injection types
   - Preview vs actual injection
   - Working with random number generators

7. **[07_export_workflows.ipynb](07_export_workflows.ipynb)** - Export functionality
   - Exporting samples and batches
   - Different export formats
   - Batch export workflows
   - Integration with training pipelines

8. **[08_advanced_workflows.ipynb](08_advanced_workflows.ipynb)** - Complex use cases
   - Combining multiple operations
   - Research workflows
   - Performance optimization
   - Custom integrations

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter notebook or JupyterLab
- TokenSmith library installed
- Required dependencies (transformers, numpy, etc.)

### Quick Start
1. Start with `01_basic_setup.ipynb` to set up your environment
2. Follow the tutorials in order
3. Each tutorial builds on concepts from previous ones
4. Run all cells in sequence for best results

### Data Requirements
Most tutorials use sample data located in the `data/` directory. Some tutorials may require you to provide your own tokenized datasets.

## Troubleshooting

### Common Issues
- **Import errors**: Make sure TokenSmith is installed and in your Python path
- **Missing tokenizer**: Download the required tokenizer models
- **File not found**: Check that data files exist in the expected locations
- **Memory issues**: Reduce batch sizes for large datasets

### Getting Help
- Check the main [README.md](../README.md) for installation instructions
- Review the API documentation in the source code
- Look at the examples in each tutorial for common patterns

## Contributing
If you find issues with tutorials or have suggestions for improvements, please create an issue or submit a pull request.
