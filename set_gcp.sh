#git clone https://github.com/juan-csv/Multilabel-video-classififcation.git
#cd Multilabel-video-classififcation

pip install tensorflow

MY_BUCKET=yt8m-juan
cd ~/ # This should take you to /home/jupyter/
mkdir -p gcs # Create a folder that will be used as a mount point
gcsfuse --implicit-dirs \
--rename-dir-limit=100 \
--disable-http2 \
--max-conns-per-host=100 \
$MY_BUCKET "/home/jupyter/Multilabel-video-classififcation/gcs"