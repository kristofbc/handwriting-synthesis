#!/bin/bash

# Example:
#
#   download_dataset.sh datafile.txt ./tmp
#
# will download all of the files listed in the file, datafiles.txt, into
# a directory, "./tmp".
#
# Each line of the datafiles.txt file should contain the path from the
# bucket root to a file.

ARGC="$#"
LISTING_FILE=datafile.txt
if [ "${ARGC}" -ge 1 ]; then
  LISTING_FILE=$1
fi
OUTPUT_DIR="./"
if [ "${ARGC}" -ge 2 ]; then
  OUTPUT_DIR=$2
fi

echo "OUTPUT_DIR=$OUTPUT_DIR"

mkdir "${OUTPUT_DIR}"

function download_file {
  FILE=$1
  BUCKET="OUR_FTP"
  URL="${BUCKET}/${FILE}"
  OUTPUT_FILE="${OUTPUT_DIR}/${FILE}"
  DIRECTORY=`dirname ${OUTPUT_FILE}`
  echo DIRECTORY=$DIRECTORY
  mkdir -p "${DIRECTORY}"
  curl --output ${OUTPUT_FILE} ${URL}
}

while read filename; do
  download_file $filename
done <${LISTING_FILE}
