#!/bin/bash

remote_server="./examples/remote_dqn_server.py"
remote_client1="./examples/remote_dqn_client1.py"
remote_client2="./examples/remote_dqn_client2.py"
remote_run="./examples/remote_dqn_run.py"

# Open a new terminal window with three tabs, running each Python script
gnome-terminal \
  --tab --title="remote_server" --command="bash -c 'python3 \"$remote_server\"; exec bash'" \
  --tab --title="remote_client1" --command="bash -c 'python3 \"$remote_client1\"; exec bash'" \
  --tab --title="remote_client2" --command="bash -c 'python3 \"$remote_client2\"; exec bash'"


