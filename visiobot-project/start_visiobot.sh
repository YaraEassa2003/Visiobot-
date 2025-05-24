#To Run:
#cd /workspaces/codespaces-models/visiobot-project
#./start_visiobot.sh --> to run the script     
# sudo kill -9 $(lsof -t -i:5000) --> to kill the process running on port 5000                                   
# sudo kill -9 $(lsof -t -i:8000) --> to kill the process running on port 8000          

echo "Checking if tmux is installed..."
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    sudo apt update && sudo apt install tmux -y
fi
tmux has-session -t visiobot-backend 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Stopping existing backend session..."
    tmux kill-session -t visiobot-backend
fi
tmux has-session -t visiobot-frontend 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Stopping existing frontend session..."
    tmux kill-session -t visiobot-frontend
fi
echo "Starting backend..."
tmux new-session -d -s visiobot-backend "cd /workspaces/codespaces-models/visiobot-project/visiobot-backend && python3 app.py"
sleep 5  
echo "Starting frontend..."
tmux new-session -d -s visiobot-frontend "cd /workspaces/codespaces-models/visiobot-project/visiobot-frontend && python3 -m http.server 8000"
if [ -n "$CODESPACES" ]; then
    echo "🔹 Setting ports 5000 & 8000 to public..."
    gh codespace ports visibility 5000:public 8000:public
    echo "Ports 5000 (backend) & 8000 (frontend) set to public."
else
    echo "Not running in GitHub Codespaces. Ports were not modified."
fi

echo "VisioBot is running! Backend & Frontend started."
echo "Run 'tmux attach -t visiobot-backend' to see backend logs."
echo "Run 'tmux attach -t visiobot-frontend' to see frontend logs."
