#!/bin/sh

set -e
set -x

# apt-get update && apt-get install -y mlocate && updatedb

python3 app.py
