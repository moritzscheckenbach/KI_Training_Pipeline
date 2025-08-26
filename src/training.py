import os
import subprocess
import sys


def start_user_interface():
    """
    Starts the User Interface with Streamlit
    """
    try:
        # Check if User_Interface.py exists
        ui_file = "User_Interface.py"
        if not os.path.exists(ui_file):
            print(f"Error: {ui_file} not found!")
            return False

        # Start Streamlit with User_Interface.py
        print("Starting User Interface...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_file], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
        return False
    except FileNotFoundError:
        print("Error: Streamlit is not installed. Install it with: pip install streamlit")
        return False

    return True


def main():
    """
    Main function of the Training Pipeline
    """
    print("Training Pipeline started...")

    # Start the User Interface
    start_user_interface()


if __name__ == "__main__":
    main()
