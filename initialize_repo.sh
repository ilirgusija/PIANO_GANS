mkdir data
mkdir data/audio/
mkdir data/audio/audio_chunks_10s
mkdir data/audio/audio_chunks_20s
mkdir data/clipped_stft/
mkdir data/resized_stft/
mkdir data/spectrograms/
mkdir data/stft_arrays/
mkdir data/images
mkdir data/audio/maestro-v3.0.0
mkdir output
mkdir logs
mkdir params
mkdir params/resnet_gen
pip install -r ./requirements.txt
cd data/audio/
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
unzip maestro-v3.0.0.zip