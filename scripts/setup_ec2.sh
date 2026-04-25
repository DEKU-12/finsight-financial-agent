#!/bin/bash
# =============================================================================
# scripts/setup_ec2.sh — FinSight EC2 Setup Script
#
# Run this ONCE on a fresh AWS EC2 Ubuntu 22.04 instance to:
#   1. Install Docker and Docker Compose
#   2. Clone the FinSight repo from GitHub
#   3. Create your .env file with API keys
#   4. Start both containers (MLflow + Streamlit)
#
# Usage:
#   ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
#   curl -fsSL https://raw.githubusercontent.com/DEKU-12/finsight-financial-agent/Ayush/scripts/setup_ec2.sh | bash
#
# Or upload and run manually:
#   chmod +x scripts/setup_ec2.sh && ./scripts/setup_ec2.sh
# =============================================================================

set -euo pipefail   # Exit on error, undefined vars, pipe failures

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         FinSight EC2 Setup — Starting...                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. System update ──────────────────────────────────────────────────────────
echo "▶ [1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# ── 2. Install Docker ─────────────────────────────────────────────────────────
echo "▶ [2/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    # Official Docker install script (works on Ubuntu 22.04)
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    rm /tmp/get-docker.sh

    # Allow current user to run docker without sudo
    sudo usermod -aG docker "$USER"
    echo "   Docker installed. NOTE: You may need to log out and back in"
    echo "   for group changes to take effect. Using 'sudo docker' for now."
else
    echo "   Docker already installed: $(docker --version)"
fi

# Install Docker Compose plugin (v2 — 'docker compose' not 'docker-compose')
if ! docker compose version &> /dev/null 2>&1; then
    echo "   Installing Docker Compose plugin..."
    sudo apt-get install -y -qq docker-compose-plugin
fi
echo "   Docker Compose: $(docker compose version)"

# ── 3. Install git and clone repo ─────────────────────────────────────────────
echo "▶ [3/6] Cloning FinSight repository..."
sudo apt-get install -y -qq git

REPO_URL="https://github.com/DEKU-12/finsight-financial-agent.git"
REPO_DIR="$HOME/finsight-financial-agent"
BRANCH="Ayush"

if [ -d "$REPO_DIR" ]; then
    echo "   Repo already exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo "   Repo ready at: $REPO_DIR"

# ── 4. Create .env file ───────────────────────────────────────────────────────
echo "▶ [4/6] Setting up environment variables..."
cd "$REPO_DIR"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "   ┌─────────────────────────────────────────────────────────┐"
    echo "   │  ACTION REQUIRED: Fill in your API keys in .env         │"
    echo "   │                                                          │"
    echo "   │  Run:  nano .env                                         │"
    echo "   │                                                          │"
    echo "   │  Set:  GROQ_API_KEY=your_key_here                       │"
    echo "   │        ALPHA_VANTAGE_API_KEY=your_key_here              │"
    echo "   │        NEWS_API_KEY=your_key_here                       │"
    echo "   └─────────────────────────────────────────────────────────┘"
    echo ""
    echo "   Opening .env for editing now..."
    nano .env
else
    echo "   .env already exists — skipping creation."
fi

# ── 5. Open firewall ports ────────────────────────────────────────────────────
echo "▶ [5/6] Configuring firewall (ufw)..."
sudo ufw allow 22/tcp   > /dev/null 2>&1 || true   # SSH
sudo ufw allow 8501/tcp > /dev/null 2>&1 || true   # Streamlit
sudo ufw allow 5001/tcp > /dev/null 2>&1 || true   # MLflow
sudo ufw --force enable > /dev/null 2>&1 || true
echo "   Ports 22, 8501, 5001 are open."
echo ""
echo "   IMPORTANT: Also open these ports in your EC2 Security Group:"
echo "   - Custom TCP  Port 8501  Source: 0.0.0.0/0  (Streamlit)"
echo "   - Custom TCP  Port 5001  Source: 0.0.0.0/0  (MLflow UI)"

# ── 6. Build and start containers ─────────────────────────────────────────────
echo ""
echo "▶ [6/6] Building and starting Docker containers..."
cd "$REPO_DIR"

# Build the app image and start all services in detached mode
sudo docker compose up --build -d

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         FinSight Deployment Complete! 🎉                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Get the public IP of this EC2 instance
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo "<YOUR-EC2-IP>")

echo "  Your app is running at:"
echo ""
echo "  📈 Streamlit dashboard  →  http://$PUBLIC_IP:8501"
echo "  🔬 MLflow UI            →  http://$PUBLIC_IP:5001"
echo ""
echo "  Useful commands:"
echo "  • View logs:          sudo docker compose logs -f"
echo "  • Stop containers:    sudo docker compose down"
echo "  • Restart:            sudo docker compose restart"
echo "  • Update & restart:   git pull && sudo docker compose up --build -d"
echo ""
