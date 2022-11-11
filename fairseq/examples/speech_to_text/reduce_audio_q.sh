COVOST_ROOT=../../covost_data/

cd ${COVOST_ROOT}/en/clips
i=0
for clip in *.mp3; do
    # working ffmpeg command example taken from https://stackoverflow.com/questions/13358287/how-to-convert-any-mp3-file-to-wav-16khz-mono-16bit
    ffmpeg -i "$clip" -acodec pcm_s16le -ac 1 -ar 16000 "${f%.mp3}.wav" &
    i=$i + 1
    if [$i/10 eq 0 ]            # change this to fit your CPU
    then
        wait
    fi
done

