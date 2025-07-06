#!/bin/bash

# EC2 VLLM Deployment Script
# This script sets up an EC2 instance for fast video processing with VLLM

echo "🚀 Setting up EC2 instance for VLLM video processing..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install CUDA dependencies
echo "📦 Installing CUDA dependencies..."
sudo apt-get install -y build-essential
sudo apt-get install -y nvidia-cuda-toolkit

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y ffmpeg

# Create virtual environment
echo "🔧 Setting up Python virtual environment..."
python3 -m venv vllm_env
source vllm_env/bin/activate

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install VLLM and other requirements
echo "⚡ Installing VLLM for fast inference..."
pip install vllm[vision]
pip install flash-attn --no-build-isolation

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements_vllm_ec2.txt

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p /tmp/vllm_smolvlm_shots
mkdir -p processed_videos_output

# Test GPU availability
echo "🔍 Testing GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Test VLLM installation
echo "🧪 Testing VLLM installation..."
python3 -c "from vllm import LLM; print('VLLM imported successfully')"

echo "✅ EC2 setup complete! Ready for VLLM video processing."
echo ""
echo "📋 Next steps:"
echo "1. Upload your video files to the instance"
echo "2. Run: python3 src/scene_segmentation/smolvlm_shot_annotator6.py"
echo "3. Monitor GPU usage with: nvidia-smi"
echo ""
echo "💡 Performance tips:"
echo "- Use g5.xlarge or larger for better performance"
echo "- Monitor memory usage with: watch -n 1 nvidia-smi"
echo "- Adjust gpu_memory_utilization in the code if needed" 