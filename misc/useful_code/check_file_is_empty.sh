#!/bin/bash

file_path="empty.txt"

if [ -s "$file_path" ]; then
    echo "The file is not empty."
else
    echo "The file is empty."
fi
