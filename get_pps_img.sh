curl -sS -O -J "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=0021140801&level=PPS&name=IMAGE"
curl -sS -O -J "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=0021140901&level=PPS&name=IMAGE"
ds9 /Users/silver/Downloads/0021140901/pps/P0021140901EPX000OIMAGE8000.FTZ -pan to 17:32:56.791 +43:30:45.00 -scale mode 99.5
ds9 /Users/silver/Downloads/0021140801/pps/P0021140801PNS003IMAGE_8000.FTZ -pan to 17:32:56.791 +43:30:45.00 -scale mode 99.5









curl -sS -O -J "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=0502020201&level=PPS&name=IMAGE"
ds9  0502020201/pps/P0502020201EPX000OIMAGE8000.FTZ -pan to 01:37:06.048 -12:57:10.08 -scale mode 99.5



curl -sS -O -J "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=0694170101&level=PPS&name=IMAGE"
ds9  /Users/silver/Downloads/0694170101/pps/P0694170101EPX000OIMAGE8000.FTZ -pan to 13:07:19.596 -40:27:38.30 -scale mode 99.5


curl -sS -O -J "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=0203560401&level=PPS&name=IMAGE"
curl -sS -O -J "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=0082340201&level=PPS&name=IMAGE"
curl -sS -O -J "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=0082340101&level=PPS&name=IMAGE"
ds9 /Users/silver/Downloads/0203560401/pps/P0203560401EPX000OIMAGE8000.FTZ  -pan to 11:18:8.700 +07:42:9.58 -scale mode 99.5
