#!/bin/bash -x
#/home/$USER/Dropbox/bin/xray_pspc_xtractor.sh >> output.log 2>&1 &
#find . -name "*.Z" | while read filename; do uncompress "`dirname "$filename"`" "$filename"; done;
#find . -name "*.gz" | while read filename; do uncompress "`dirname "$filename"`" "$filename"; done;
#pspcb_gain1_256.rmf
#        - valid for PSPCB data taken BEFORE 1991 Oct 14
#          (formerly known as pspcb_92mar11.rmf)
#pspcb_gain2_256.rmf
#        - valid for PSPCB data taken AFTER 1991 Oct 14
#          (formerly known as pspcb_93jan12.rmf)

WD=/home/$USER/Dropbox/science/rosmm/survey_rosat
DAT_DIR=/home/$USER/data/rosat/zhou06_60min
BIN_DIR=/home/$USER/Dropbox/bin
META_DATA=meta_data_interactive.dat
LC_BINSIZE=400
MIN_CTS=-99999999999

mkdir -p $WD
cd $WD
coords=($(ps -u $USER | awk '{print $1}' aux/${META_DATA}))
ids=($(ps -u $USER | awk '{print $2}' aux/${META_DATA}))
rmf=($(ps -u $USER | awk '{print $7}' aux/${META_DATA}))
crf=($(ps -u $USER | awk '{print $8}' aux/${META_DATA}))
src=($(ps -u $USER | awk '{print $9}' aux/${META_DATA}))
bkg=($(ps -u $USER | awk '{print $10}' aux/${META_DATA}))

function make_src {
    printf "# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
${src[i]}" > ${coords[i]}_src.reg
}

function make_bkg {
    printf "# Region file format: DS9 version 4.1
global color=red dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
${bkg[i]}" > ${coords[i]}_bkg.reg
}

function make_input {
    printf "${coords[i]}
read events
$DAT_DIR/${ids[i]}
${ids[i]}_bas.fits
yes

extract image
save image ${coords[i]}_img.fits

filter region ${coords[i]}_src.reg
set binsize $LC_BINSIZE
extract curve
save curve ${coords[i]}_slc.fits
extract spectrum
save spectrum ${coords[i]}_src_unb.fits
no

clear region all
filter region ${coords[i]}_bkg.reg
extract curve
save curve ${coords[i]}_blc.fits
extract spectrum
save spectrum ${coords[i]}_bkg_unb.fits
no

exit
no
" > xselect_input.txt
}

function significant {
    src_cts=$(hexdump -e '80/1 "%_p" "\n"' ${coords[i]}_src_unb.fits | grep -m 1 TOTCTS | awk '{print $3}')
    bkg_cts=$(hexdump -e '80/1 "%_p" "\n"' ${coords[i]}_bkg_unb.fits | grep -m 1 TOTCTS | awk '{print $3}')

    src_scal=$(hexdump -e '80/1 "%_p" "\n"' ${coords[i]}_src_unb.fits | grep -m 1 BACKSCAL | awk '{print $2}')
    bkg_scal=$(hexdump -e '80/1 "%_p" "\n"' ${coords[i]}_bkg_unb.fits | grep -m 1 BACKSCAL | awk '{print $2}')
    echo $(python -c "print $src_cts-$src_scal/$bkg_scal*$bkg_cts")
    return $(python -c "print 0 if $src_cts-$src_scal/$bkg_scal*$bkg_cts > $MIN_CTS else 1")
}

function print_img {
    ds9 ${coords[i]}_img.fits -scale log -zoom to fit -cmap Heat -regions load "*.reg" -print destination file -print filename ${coords[i]}_img.ps -print -exit
    print_help ${coords[i]}_img
}

function print_lc {
    src_scal=$(hexdump -e '80/1 "%_p" "\n"' ${coords[i]}_src_unb.fits | grep -m 1 BACKSCAL | awk '{print $2}')
    bkg_scal=$(hexdump -e '80/1 "%_p" "\n"' ${coords[i]}_bkg_unb.fits | grep -m 1 BACKSCAL | awk '{print $2}')
    src_scal=$(python -c "print 1/$src_scal")
    bkg_scal=$(python -c "print 1/$bkg_scal")

    printf "hardcopy ${coords[i]}_lc.ps/ps \nexit">${coords[i]}_help.pco
    lcmath infile=${coords[i]}_slc.fits bgfile=${coords[i]}_blc.fits outfile=${coords[i]}_lc.fits multi=$src_scal multb=$bkg_scal addsubr=no
    lcurve nser=1 cfile1=${coords[i]}_lc.fits window=- dtnb=400 nbint=INDEF outfile=temp.fits plot=yes plotdev=/xw plotdnum=3 plotfile=${coords[i]}_help.pco
    rm temp.fits
    print_help ${coords[i]}_lc
}

function print_help {
    ps2pdf $1.ps
    pdfcrop $1.pdf
    rm $1.ps $1.pdf
    mv $1-crop.pdf $1.pdf
}

function make_rsp {
    cp ../../aux/${rmf[i]} .
    cp ../../aux/${crf[i]} .
    pcarf phafil=${coords[i]}_src_unb.fits rmffil=${rmf[i]} outfil=${coords[i]}_arf_unb.fits crffil=${crf[i]}
    marfrmf rmfil=${rmf[i]} arfil=${coords[i]}_arf_unb.fits outfil=${coords[i]}_rsp_unb.fits
}

function do_group {
    cp $BIN_DIR/xray_spec_grp.py .
    python xray_spec_grp.py ${coords[i]}_rsp_unb.fits ${coords[i]}_src_unb.fits binning.txt
    rbnrmf infile=${coords[i]}_rsp_unb.fits binfile=binning.txt outfile=${coords[i]}_rsp.fits
    rbnpha infile=${coords[i]}_src_unb.fits binfile=binning.txt outfile=${coords[i]}_src.fits
    rbnpha infile=${coords[i]}_bkg_unb.fits binfile=binning.txt outfile=${coords[i]}_bkg.fits
    grppha << FLAG
${coords[i]}_src.fits
${coords[i]}_grp.fits
chkey BACKF ${coords[i]}_bkg.fits
chkey RESPF ${coords[i]}_rsp.fits
exit
FLAG
}

#Convert, merge, delete, and move some stuff
function cleanup {
    pdftk *.pdf cat output output.pdf
    rm *_*.pdf
    mv output.pdf ../outputs/${coords[i]}_output.pdf
}

n=${#coords[*]}
mkdir -p reduction/outputs
for i in $(seq 0 $(($n-1)))
do
    mkdir -p reduction/${coords[i]}
    cd reduction/${coords[i]}
    make_src
    make_bkg
    make_input
    xselect < xselect_input.txt > ${coords[i]}_xselect.log
    if significant
    then
	print_img
	print_lc
	make_rsp
	do_group
	cleanup
	cd ../../
    else
	cd ../
	rm -rf ${coords[i]}
	cd ../
    fi
done

echo "EOF"
