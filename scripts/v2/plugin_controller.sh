#!/usr/bin/env bash
# Run the V-JEPA 2-AC plug-in fine-tunes (B, C) and the DINO-WM timing job first, then release the rest of the queue.
cd "$(dirname "$0")/../.."
FIRST="6900 6901 6937 6938 6939 6940"
release_all() { for j in $(squeue -u $USER -t PD -h -o "%i %r" | awk '$2=="JobHeldUser"{print $1}'); do [ "$j" != "6758" ] && scontrol release $j; done; }
started() { ! squeue -h -j "$1" -t PD 2>/dev/null | grep -q .; }
for j in $(squeue -u $USER -t PD -h -o %i); do case " $FIRST " in *" $j "*) scontrol release $j 2>/dev/null;; *) scontrol hold $j 2>/dev/null;; esac; done
echo "$(date -Is) waiting for: $FIRST"
all_started() { for j in $FIRST; do started "$j" || return 1; done; return 0; }
until all_started; do sleep 60; done
release_all; echo "$(date -Is) priority jobs running; queue released"
