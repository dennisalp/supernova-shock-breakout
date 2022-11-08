#!/bin/bash

ids=(
"rp700145a00"
"rp700792n00"
"rp701549n00")
dat_dir=/home/dalp/data/rosat/survey_sample
wd=/home/$USER/Dropbox/astrophysics/survey_rosat

cd $wd

function get_data {
    ra=grep 'RIGHT_ASCENSION' $dat_dir/${ids[i]}/${ids[i]}.public_contents | awk '{print $4 $5 $6}'
    dec=grep 'DECLINATION' $dat_dir/${ids[i]}/${ids[i]}.public_contents | awk '{print $4 $5 $6}'
    start=grep 'UT_START_TIME' $dat_dir/${ids[i]}/${ids[i]}.public_contents | awk '{print $3 " " $4}'
    instr=grep 'INSTRUMENT_NAME' $dat_dir/${ids[i]}/${ids[i]}.public_contents | awk '{print $4}'
    expo=grep 'TOTAL_ACCEPTED_SECONDS' $dat_dir/${ids[i]}/${ids[i]}.public_contents | awk '{print $3}'
}

n=${#ids[*]}
mkdir -p aux/
touch aux/meta_data.txt
for i in $(seq 0 $(($n-1)))
do
    get_data
    append2file
done
