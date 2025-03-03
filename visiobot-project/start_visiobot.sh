#cd /workspaces/codespaces-models/visiobot-project/visiobot-backend
#cd /workspaces/codespaces-models/visiobot-project/visiobot-frontend
#chmod +x start_visiobot.sh --> to make the file executable
#sudo apt update && sudo apt install tmux -y --> to install tmux
#./start_visiobot.sh --> to run the script

#!/bin/bash

echo "🔹 Checking if tmux is installed..."
if ! command -v tmux &> /dev/null; then
    echo "🔹 Installing tmux..."
    sudo apt update && sudo apt install tmux -y
fi

# ✅ Preserve Flask Session Files Before Restarting
SESSION_DIR="/workspaces/codespaces-models/visiobot-project/visiobot-backend/flask_session_data"
mkdir -p $SESSION_DIR
echo "✅ Flask session directory ensured: $SESSION_DIR"

# ✅ Check if backend session exists, then stop it
tmux has-session -t visiobot-backend 2>/dev/null
if [ $? -eq 0 ]; then
    echo "🛑 Stopping existing backend session..."
    tmux kill-session -t visiobot-backend
fi

# ✅ Check if frontend session exists, then stop it
tmux has-session -t visiobot-frontend 2>/dev/null
if [ $? -eq 0 ]; then
    echo "🛑 Stopping existing frontend session..."
    tmux kill-session -t visiobot-frontend
fi

# ✅ Start Flask Backend with Persistent Sessions
echo "🚀 Starting backend..."
tmux new-session -d -s visiobot-backend "cd /workspaces/codespaces-models/visiobot-project/visiobot-backend && python3 app.py"

# ✅ Wait for Backend to Start
sleep 5  

# ✅ Start Frontend HTTP Server
echo "🚀 Starting frontend..."
tmux new-session -d -s visiobot-frontend "cd /workspaces/codespaces-models/visiobot-project/visiobot-frontend && python3 -m http.server 8000"

# ✅ Ensure Ports Are Public in Codespaces
if [ -n "$CODESPACES" ]; then
    echo "🔹 Setting ports 5000 & 8000 to public..."
    gh codespace ports visibility 5000:public 8000:public
    echo "✅ Ports 5000 (backend) & 8000 (frontend) set to public." k

else
    echo "⚠️ Not running in GitHub Codespaces. Ports were not modified."
fi

# ✅ Display Status
echo "✅ VisioBot is running! Backend & Frontend started."
echo "🔎 Run 'tmux attach -t visiobot-backend' to see backend logs."
echo "🔎 Run 'tmux attach -t visiobot-frontend' to see frontend logs."
