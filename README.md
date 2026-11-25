Markdown
# EV Mobility Multi-Planner & AI Negotiator

## Overview
This project is an advanced electric vehicle (EV) mobility planner developed as a final-year engineering project (PFE). It features an intelligent route planner that integrates a dynamic pricing system for charging stations and an automated AI negotiation agent. The system is designed to optimize the EV charging experience by balancing travel time, battery consumption, and charging costs.

## Core Features
* **Smart Routing:** Calculates the fastest, shortest, and eco-friendly routes using OpenStreetMap data.
* **Dynamic Pricing:** Simulates realistic charging station availability and pricing based on grid demand and time of day.
* **AI Delegation:** An automated AI agent that negotiates charging prices with multiple stations simultaneously within a defined radius to secure the best rate based on a user-defined budget.
* **Manual Negotiation:** An interactive UI allowing users to make manual counter-offers with specific charging stations.
* **Analytics Dashboard:** Generates real-time visualizations comparing negotiation performance, bounded time simulations, and cost analysis.

## Getting Started

To maintain a lightweight and optimized repository, mapping data (OSM graphs) and cache files (`.pkl`) are intentionally excluded from version control. The system is designed to auto-generate this data during the initial setup.

### 1. Download the repository

Download the project

bash

cd PFE2

### 2. Create and activate a virtual environment
Using a virtual environment (Python 3.9+) is highly recommended.

Bash

python -m venv venv

### On Windows:
venv\Scripts\activate

### On MacOS / Linux:
source venv/bin/activate

### 3. Install dependencies
Bash

pip install -r requirements.txt

### 4. Generate the data ecosystem
The project requires the creation of the Île-de-France network topology. Run the preparation script, which will automatically build the data/ directory structure, download the graphs, compute elevations, and compile the .pkl caches:

Bash

python data_prep/generer_graphe.py

⚠️ Important Note: This step is computationally heavy. Depending on your internet connection and CPU, it may take 10 to 20 minutes to complete. The terminal will not show a progress bar; please do not interrupt the process.

### 5. Launch the Web Application
Once the data is generated, start the Flask backend server:

Bash

python app/app.py

The user interface and analytics dashboard will be immediately accessible in your browser at: http://127.0.0.1:5000

### Technologies Used
Backend: Python, Flask

Routing & Mapping: OSMnx, NetworkX, Leaflet.js

Data Processing & Visualization: Pandas, Rasterio, Matplotlib, Seaborn

