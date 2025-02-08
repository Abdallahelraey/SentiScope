import subprocess
import time
import sys

# Get the path to the current Python interpreter (inside the virtual env)
python_executable = sys.executable

# Run FastAPI server in a new terminal
try:
    subprocess.Popen(
        [python_executable, "./src/SentiScope/components/routes/inferance.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE  # Opens in a new terminal (Windows)
    )
    print("FastAPI server started in a new terminal.")
except Exception as e:
    print(f"Error running FastAPI: {e}")

# Wait for a few seconds to ensure the server starts properly
time.sleep(5)

# Run Streamlit in another new terminal
try:
    subprocess.Popen(
        ["streamlit", "run", "./src/SentiScope/components/viewers/straemlit_view.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE  # Opens in a new terminal (Windows)
    )
    print("Streamlit app started in a new terminal.")
except Exception as e:
    print(f"Error running Streamlit: {e}")
