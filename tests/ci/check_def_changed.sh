#!/bin/bash

if [[ $(git diff --name-only origin/master -- 'resources/retinal-rl.def') ]]; then
    echo "changed=true"
else
    echo "changed=false"
fi