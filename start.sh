#!/bin/bash

source ./venv/bin/activate
nohup uvicorn main:app --reload --host=0.0.0.0 --port=29888 &