#!/bin/sh

# Downloads the Camvid dataset.

# By using this script, you agree to all terms
# and conditions specified by the Camvid
# dataset creators.

CAMVID_DST_DIR=$1

# ----------- Environment Variables + Directory Setup -----------------------

echo "Camvid will be downloaded to "$CAMVID_DST_DIR
mkdir -p $CAMVID_DST_DIR

# ----------- Downloading ---------------------------------------------------

CAMVID_IMGS_URL="https://github.com/johnwlambert/camvid-dataset-mirror/releases/download/v1.0.0/701_StillsRaw_full.zip"


echo "Downloading Camvid dataset..."
# Images are 561 MB, Labels are 8 MB
wget -c -O $CAMVID_DST_DIR/701_StillsRaw_full.zip $CAMVID_IMGS_URL
echo "Camvid dataset downloaded."

# # ----------- Extraction ---------------------------------------------------
echo "Extracting Camvid dataset..."
cd $CAMVID_DST_DIR

unzip 701_StillsRaw_full.zip

#rm 701_StillsRaw_full.zip
echo "Camvid dataset extracted."
