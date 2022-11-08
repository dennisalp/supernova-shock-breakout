#!/bin/bash -x



oi=("0149780101" "0203560201" "0300240501" "0300930301" "0502020101" "0555780101" "0604740101" "0675010401" "0743650701" "0760380201" "0765041301" "0770380401" "0781890401")
ra=("29.28818" "169.53625" "196.83165" "311.43725" "24.27520" "52.99533" "321.79670" "37.89595" "43.65356" "167.07789" "204.19991" "173.53096" "263.23663")
de=("37.62732" "7.70266" "-40.46064" "-67.64702" "-12.95280" "-27.65780" "-12.03906" "-60.62934" "41.07430" "-5.07517" "-41.33736" "0.87357" "43.51250")



for (( ii=0; ii<${#oi[@]}; ii++ ));
do
    echo ${oi[$ii]}
    /Applications/SAOImageDS9.app/Contents/MacOS/ds9 -bin buffersize 4096 \
        -catalog cds "VII/283" \
        -catalog radius 30 arcmin \
        -catalog skyformat degrees \
        -catalog ${ra[$ii]} ${de[$ii]} icrs "$(ls /Users/silver/dat/xmm/sbo/${oi[$ii]}_repro/${oi[$ii]}_ep_?????_img.fits)" \
        -catalog retrieve \
        -catalog symbol color cyan \
        -cmap heat \
        -scale log \
        -pan to ${ra[$ii]} ${de[$ii]} wcs icrs \
        -regions command "icrs;circle(${ra[$ii]}d,${de[$ii]}d,2\") # color=white width=1" \
        -regions load "/Users/silver/dat/xmm/sbo/${oi[ii]}_repro/${oi[ii]}_emllist.reg" \
        -print destination file \
        -print filename "/Users/silver/box/phd/pro/sne/sbo/art/fig/alignment/${oi[ii]}_xmm.ps" \
        -print resolution 600 \
        -print
    ps2pdf "/Users/silver/box/phd/pro/sne/sbo/art/fig/alignment/${oi[ii]}_xmm.ps" "/Users/silver/box/phd/pro/sne/sbo/art/fig/alignment/${oi[ii]}_xmm.pdf"
    rm "/Users/silver/box/phd/pro/sne/sbo/art/fig/alignment/${oi[ii]}_xmm.ps"

    # /Applications/SAOImageDS9.app/Contents/MacOS/ds9 -bin buffersize 4096 \
    #     -catalog cds gsc \
    #     -catalog radius 90 arcsec \
    #     -catalog skyformat degrees \
    #     -catalog ${ra[$ii]} ${de[$ii]} icrs /Users/silver/box/phd/pro/sne/sbo/art/fig/${oi[ii]}_img.fits \
    #     -catalog retrieve \
    #     -catalog cds nomad \
    #     -catalog radius 90 arcsec \
    #     -catalog skyformat degrees \
    #     -catalog ${ra[$ii]} ${de[$ii]} icrs \
    #     -catalog retrieve \
    #     -catalog symbol color cyan \
    #     -catalog symbol shape box \
    #     -catalog symbol size 10 \
    #     -catalog symbol size2 10 \
    #     -cmap heat \
    #     -pan to ${ra[$ii]} ${de[$ii]} wcs icrs \
    #     -zoom to fit \
    #     -regions command "icrs;circle(${ra[$ii]}d,${de[$ii]}d,2\") # color=white width=1" \
    #     -print destination file \
    #     -print filename "/Users/silver/box/phd/pro/sne/sbo/art/fig/alignment/${oi[ii]}_sky.ps" \
    #     -print resolution 600 \
    #     -print \
    #     -exit
    # ps2pdf "/Users/silver/box/phd/pro/sne/sbo/art/fig/alignment/${oi[ii]}_sky.ps" "/Users/silver/box/phd/pro/sne/sbo/art/fig/alignment/${oi[ii]}_sky.pdf"
    # rm "/Users/silver/box/phd/pro/sne/sbo/art/fig/alignment/${oi[ii]}_sky.ps"
done
