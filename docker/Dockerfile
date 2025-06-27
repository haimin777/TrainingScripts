# Use NVIDIA's TensorFlow 2.15 base image with GPU support
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && apt-get clean

# (Optional) Upgrade pip
RUN pip install --upgrade pip

# Copy your training scripts and configuration files
COPY . /workspace

# Install Python dependencies
# If you have a requirements.txt, uncomment the next line
# RUN pip install -r requirements.txt

# Otherwise, install essential libraries
RUN pip install pandas gcsfs matplotlib

# (Optional) Install TensorBoard if not included
RUN pip install tensorboard

# (Optional) If using GCS FUSE or additional cloud tools, install them here

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Default command to run your training script
# Example: python train.py --config configs/your_config.yaml
#CMD ["python", "train.py"]
