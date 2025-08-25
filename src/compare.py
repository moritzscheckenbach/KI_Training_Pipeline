import os
import subprocess
import sys


def start_user_interface():
    """
    Startet das User Interface mit Streamlit
    """
    try:
        # Überprüfe ob User_Interface.py existiert
        ui_file = "evaluation.py"
        if not os.path.exists(ui_file):
            print(f"Fehler: {ui_file} wurde nicht gefunden!")
            return False

        # Starte Streamlit mit User_Interface.py
        print("Starte User Interface...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_file], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Starten von Streamlit: {e}")
        return False
    except FileNotFoundError:
        print("Fehler: Streamlit ist nicht installiert. Installiere es mit: pip install streamlit")
        return False

    return True


def main():
    """
    Hauptfunktion der Training Pipeline
    """
    print("Training Pipeline gestartet...")

    # Starte das User Interface
    start_user_interface()


if __name__ == "__main__":
    main()
