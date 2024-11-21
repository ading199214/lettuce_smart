# Lettuce Smart

## Installation

### Prerequisites

Ensure the following are installed on your system:

- Python 3.7 or higher  
- Git  
- A virtual environment tool (recommended)  

## Setting Up the Environment

It’s recommended to use a virtual environment to manage dependencies.

### Using `venv`

1. Create a virtual environment:

   ```bash
   python3 -m venv venv

2. Activate the virtual environment:
	•	On macOS/Linux:
    source venv/bin/activate

3. Install Required Packages

Install the required packages using pip:

pip install -r requirements.txt

4. ## Starting the Application
uvicorn app:app --reload

5. Accessing the Application

Open your web browser and navigate to:
http://localhost:8000/

6. Deployment

The application is currently mannualy deployed using Render in the Singapore region.

7. Production URL

Access the production version of the application at:

https://lettuce-smart-server.onrender.com/