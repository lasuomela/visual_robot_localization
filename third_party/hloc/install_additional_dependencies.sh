#!/bin/bash

SCRIPT=$(readlink -f "$0")
CWD=$(dirname "$SCRIPT")
TP_LIB=$CWD/Hierarchical-Localization/third_party

touch "${TP_LIB}/__init__.py"

touch "${TP_LIB}/d2net/__init__.py"
touch "${TP_LIB}/d2net/lib/__init__.py"

touch "${TP_LIB}/deep-image-retrieval/__init__.py"
touch "${TP_LIB}/deep-image-retrieval/dirtorch/__init__.py"
touch "${TP_LIB}/deep-image-retrieval/dirtorch/utils/__init__.py"
touch "${TP_LIB}/deep-image-retrieval/dirtorch/nets/layers/__init__.py"
mkdir "${TP_LIB}/deep-image-retrieval/dirtorch/data"

touch "${TP_LIB}/SuperGluePretrainedNetwork/__init__.py"
touch "${TP_LIB}/SuperGluePretrainedNetwork/models/weights/__init__.py"

touch "${TP_LIB}/r2d2/__init__.py"
touch "${TP_LIB}/r2d2/tools/__init__.py"
touch "${TP_LIB}/r2d2/nets/__init__.py"
touch "${TP_LIB}/r2d2/models/__init__.py"
