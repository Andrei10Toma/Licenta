#!/bin/bash
set -e

echo "Starting server"
python3 federate_server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for ((i=0; i<7; i++)); do
    # Your code here
    python3 federate_client.py --client "$i"&
done
# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
