#!/bin/bash
set -e
set -x

usage() {
  echo "Usage: $0 ./matches/SgDE9D-Ux.gior 1"
  echo "$0 filename playerid [gif]"
}

if [ -z $1 ]; then
  usage
  exit 1
fi

if [ -z $2 ]; then
  usage
  exit 1
fi

REPLAY_NAME=$(basename $1)
REPLAY_ID=$(echo "$REPLAY_NAME" | cut -d. -f1)
PLAYER_ID=$2

generals-replay game \
  --input ./matches/$REPLAY_NAME \
  --output ./matches_json/${REPLAY_ID}_${PLAYER_ID}.json \
  --player $PLAYER_ID

if [ -n "$3" ]; then
  gifmaker ./matches/$REPLAY_NAME $PLAYER_ID
  google-chrome-stable ./matches/${REPLAY_ID}.gif
fi
