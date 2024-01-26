#!/bin/bash
find . -type f -regex ".*ckpt.*" -exec rm {} +
find . -type f -regex ".*json$" -exec rm {} +


