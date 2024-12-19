# setup/environment.py

import os
import ssl
import subprocess
import sys

def fix_ssl_certificate():
    """Reinstall CA certificates to fix SSL issues."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"])
    subprocess.check_call(["sudo", "apt-get", "install", "--reinstall", "ca-certificates", "-y"])
    subprocess.check_call(["sudo", "update-ca-certificates", "--fresh"])

def bypass_ssl_verification():
    """Optional workaround to bypass SSL verification. Use with caution."""
    ssl._create_default_https_context = ssl._create_unverified_context

def mount_google_drive():
    """Mount Google Drive if running in Google Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    except ImportError:
        print("Not running in Google Colab. Skipping Google Drive mount.")
        return False

def install_packages():
    """Install required libraries."""
    packages = [
        "git+https://github.com/qubvel/segmentation_models.pytorch.git",
        "timm==0.9.7",
        "torchmetrics",
        "optuna"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

def setup_environment():
    """Perform all environment setup steps."""
    fix_ssl_certificate()
    bypass_ssl_verification()
    mount_google_drive()
    install_packages()

if __name__ == "__main__":
    setup_environment()
