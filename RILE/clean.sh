#!/bin/bash
find . -type f -regex ".*ckpt.*" -exec rm {} +
find . -maxdepth 1 -type f -regex ".*json$" -exec rm {} +


