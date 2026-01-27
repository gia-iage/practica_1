#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

# Install software
apt-get clean all
echo "==> Updating repositories..."
if ! apt-get update > /tmp/apt-update.log 2>&1; then
    echo "Error when updating repositories, log:"
    cat /tmp/apt-update.log
    exit 1
fi
echo "==> done"
SOFTWARE="nano sshpass unzip python-apt-common dnsutils dos2unix whois python3 python3-venv python3-pip git"
echo "==> Installing software packages..."
if ! apt-get install -y -qq $SOFTWARE > /tmp/apt-install.log 2>&1; then
    echo "Error when installing software, log:"
    cat /tmp/apt-install.log
    exit 1
fi
echo "==> done"
