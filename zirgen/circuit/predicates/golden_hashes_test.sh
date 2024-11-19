#!/bin/sh

if diff -u $1 $2
then
    exit 0
fi

echo "ZKR hashes do not match hashes in golden_hashes.txt.  If these changes are intentional, please update golden_hashes.txt."
exit 1
