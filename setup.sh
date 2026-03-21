#!/bin/bash

# Update and install required tools
apt-get update && apt-get install -y wget unzip curl gnupg

# Install Google Chrome (Headless version)
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
dpkg -i google-chrome-stable_current_amd64.deb
apt --fix-broken install -y  # Fix any dependency issues

# Download and install ChromeDriver
wget https://storage.googleapis.com/chrome-for-testing-public/136.0.7103.92/linux64/chromedriver-linux64.zip -P /tmp
unzip /tmp/chromedriver-linux64.zip -d /tmp/chromedriver

# Make it executable
chmod +x /tmp/chromedriver/chromedriver-linux64/chromedriver

# Move it to a location where the script can access
mv /tmp/chromedriver/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver
