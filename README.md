# Trossen Stationary Dataset Dashboard

Dashboard for analyzing LeRobot/Trossen stationary datasets with four camera streams.

## Features

- Total recordings
- Total recording time
- Typical episode duration
- Longest and shortest episodes
- Average setup time
- Total setup time
- Recording utilization
- Camera completeness
- Missing camera video detection
- CSV export

## Installation

Installation only needs to be done once per computer.

```bash
python3 -m venv ~/.venvs/trossen-stationary-dashboard

source ~/.venvs/trossen-stationary-dashboard/bin/activate

git clone https://github.com/sofiainfantec/trossen-stationary-dataset-dashboard.git

cd trossen-stationary-dataset-dashboard

pip install -r requirements.txt
