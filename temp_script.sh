#!/bin/bash

git status

# Ask for lesson
echo "Current Lesson?: "
read commit_message

# Add all files
git add .

# Commit with your message
git commit -m "Added lesson $commit_message material."

# Push to the remote repository
git push origin main



