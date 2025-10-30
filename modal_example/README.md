# TokenSmith Modal Example

This folder demonstrates how to use TokenSmith with Modal, a serverless platform. The example can be easily run on the generous free tier provided by Modal.

## Prerequisites

- A Modal account (Sign up at [modal.com](https://modal.com))

## Environment Setup

1. **Install and Configure Modal**
   - Follow the Modal setup instructions in their [documentation](https://modal.com/docs/guide)
   - Make sure you have authenticated your Modal client

2. **Deploy the GPU-enabled Container**
   - The `modal_image_generator.py` script creates a custom container with:
     - CUDA 12.8.1 support
     - PyTorch with GPU acceleration
     - All required TokenSmith dependencies
   - Deploy the container by running:
     ```bash
     modal deploy modal_image_generator.py
     ```

3. **Running Examples**
   - Open the included example notebook: `tokensmith-demo-notebook.ipynb`
   - Alternatively, view the hosted version [here](https://modal.com/notebooks/aflah02/_/nb-8kQByHPxWfM71jUuPvsCWv)
   - The notebook demonstrates TokenSmith's capabilities on a shard of DCLM data and can be scaled up/down to custom datasets