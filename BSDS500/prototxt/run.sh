#!/usr/bin/env sh

TOOLS=../../build/tools

$TOOLS/caffe train \
        --solver=bsds500_solver.prototxt