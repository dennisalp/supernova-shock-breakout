#!/bin/bash -x
#/home/$USER/Dropbox/bin/xtractor_pspcb.sh >> output.log 2>&1 &
#find . -name "*.Z" | while read filename; do uncompress "`dirname "$filename"`" "$filename"; done;
#find . -name "*.gz" | while read filename; do uncompress "`dirname "$filename"`" "$filename"; done;
#pspcb_gain1_256.rsp
#        - valid for PSPCB data taken BEFORE 1991 Oct 14
#          (formerly known as pspcb_92mar11.rmf)
#pspcb_gain2_256.rsp
#        - valid for PSPCB data taken AFTER 1991 Oct 14
#          (formerly known as pspcb_93jan12.rmf)

WD=/home/$USER/Dropbox/astrophysics/rosmm/survey_rosat_rsp
DAT_DIR=/home/$USER/data/rosat/survey_sample
BIN_DIR=/home/$USER/Dropbox/bin
META_DATA=meta_data.dat
LC_BINSIZE=400

cd $WD
coords=($(ps -u $USER | awk '{print $1}' aux/${META_DATA}))
ids=($(ps -u $USER | awk '{print $2}' aux/${META_DATA}))
rsp=($(ps -u $USER | awk '{print $7}' aux/${META_DATA}))
src=($(ps -u $USER | awk '{print $8}' aux/${META_DATA}))
bkg=($(ps -u $USER | awk '{print $9}' aux/${META_DATA}))

function make_src {
    printf "# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
${src[i]}" > ${ids[i]}_src.reg
}

function make_bkg {
    printf "# Region file format: DS9 version 4.1
global color=red dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
${bkg[i]}" > ${ids[i]}_bkg.reg
}

function make_input {
    printf "${ids[i]}
read events
$DAT_DIR/${ids[i]}
${ids[i]}_bas.fits
yes

extract image
save image ${ids[i]}_img.fits

filter region ${ids[i]}_src.reg
set binsize $LC_BINSIZE
extract curve
save curve ${ids[i]}_slc.fits
extract spectrum
save spectrum ${ids[i]}_src_unb.fits
no

clear region all
filter region ${ids[i]}_bkg.reg
extract curve
save curve ${ids[i]}_blc.fits
extract spectrum
save spectrum ${ids[i]}_bkg_unb.fits
no

exit
no
" > xselect_input.txt
}

function print_img {
    ds9 ${ids[i]}_img.fits -scale log -zoom to fit -cmap Heat -regions load "*.reg" -print destination file -print filename ${ids[i]}_img.ps -print -exit
    print_help ${ids[i]}_img
}

function print_lc {
    printf "hardcopy ${ids[i]}_lc.ps/ps \nexit">${ids[i]}_help.pco
    lcmath infile=${ids[i]}_slc.fits bgfile=${ids[i]}_blc.fits outfile=${ids[i]}_lc.fits multi=9 multb=1 addsubr=no
    lcurve nser=1 cfile1=${ids[i]}_lc.fits window=- dtnb=400 nbint=INDEF outfile=temp.fits plot=yes plotdev=/xw plotdnum=3 plotfile=${ids[i]}_help.pco
    rm temp.fits
    print_help ${ids[i]}_lc
}

function print_help {
    ps2pdf $1.ps
    pdfcrop $1.pdf
    rm $1.ps $1.pdf
    mv $1-crop.pdf $1.pdf
}

function do_group {
cp $BIN_DIR/spec_grp.py .
cp ../../aux/${rsp[i]} .
python spec_grp.py ${rsp[i]} ${ids[i]}_src_unb.fits binning.txt
rbnrmf infile=${rsp[i]} binfile=binning.txt outfile=${ids[i]}_rsp.fits
rbnpha infile=${ids[i]}_src_unb.fits binfile=binning.txt outfile=${ids[i]}_src.fits
rbnpha infile=${ids[i]}_bkg_unb.fits binfile=binning.txt outfile=${ids[i]}_bkg.fits
grppha << FLAG
${ids[i]}_src.fits
${ids[i]}_grp.fits
chkey BACKF ${ids[i]}_bkg.fits
chkey RESPF ${ids[i]}_rsp.fits
exit
FLAG
}

#Convert, merge, delete, and move some stuff
function cleanup {
    pdftk *.pdf cat output output.pdf
    rm *_*.pdf
    mv output.pdf ../outputs/${ids[i]}_output.pdf
}

n=${#ids[*]}
mkdir -p reduction/outputs
for i in $(seq 0 $(($n-1)))
do
    mkdir -p reduction/${ids[i]}
    cd reduction/${ids[i]}
    make_src
    make_bkg
    make_input
    xselect < xselect_input.txt > ${ids[i]}_xselect.log
    print_img
    print_lc
    do_group
    cleanup
    cd ../../
done

echo "EOF"
