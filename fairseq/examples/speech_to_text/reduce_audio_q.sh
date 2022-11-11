COVOST_ROOT=../../covost_data/

cd ${COVOST_ROOT}/en/clips
i=0
for clip in *.mp3; do
    ffmpeg -i "$clip" -acodec pcm_s16le -ac 1 -ar 16000 "${f%.mp3}.wav" &
    i=$i + 1
    if [$i/10 eq 0 ]
    then
        wait
    fi
done

