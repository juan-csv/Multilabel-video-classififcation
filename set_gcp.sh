#git clone https://github.com/juan-csv/Multilabel-video-classififcation.git

cd Multilabel-video-classififcation
MY_BUCKET=yt8m-juan
DEST_FOLDER=gcs1
mkdir -p $DEST_FOLDER # Create a folder that will be used as a mount point
gcsfuse --implicit-dirs \
--rename-dir-limit=100 \
--disable-http2 \
--max-conns-per-host=100 \
$MY_BUCKET $DEST_FOLDER


MY_BUCKET=yt8m-ourglass
DEST_FOLDER=gcs2
mkdir -p $DEST_FOLDER # Create a folder that will be used as a mount point
gcsfuse --foreground \
--debug_gcs \
--debug_http \
--debug_fuse \
--debug_invariants \
$MY_BUCKET $DEST_FOLDER

pip install tensorflow
pip install wandb