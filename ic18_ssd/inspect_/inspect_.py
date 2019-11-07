from ic18_ssd.config import dlib_front_rear_config as config
from pyimagesearch.utils.tfannotation import TFAnnotation
from bs4 import BeautifulSoup
from PIL import Image
import tensorflow as tf
import os
import bs4

#
# with open(config.CLASSES_FILE, 'w') as f:
#     for k, v in config.CLASSES.items():
#         item = ("item {\n" +
#                 "\tid: " + str(v) + "\n" +
#                 "\tname: '" + k + "'\n" +
#                 "}\n")
#         f.write(item)


# contents = open(input_path).read()
contents = """
<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>imglab dataset</name>
<comment>Created by imglab tool.</comment>
<images>
  <image file='la_hill_st/la_hill_st_000001.jpg'>
    <box top='1406' left='668' width='203' height='134'>
      <label>front</label>
    </box>
    <box top='1389' left='812' width='174' height='127'>
      <label>front</label>
    </box>
    <box top='1389' left='968' width='210' height='145'>
      <label>front</label>
    </box>
    <box top='1375' left='1063' width='160' height='141'>
      <label>front</label>
    </box>
    <box top='1383' left='1134' width='133' height='133' ignore='1'>
      <label>front</label>
    </box>
    <box top='1376' left='912' width='126' height='145' ignore='1'>
      <label>front</label>
    </box>
    <box top='1309' left='1231' width='350' height='243'>
      <label>rear</label>
    </box>
    <box top='1378' left='2364' width='1017' height='357' ignore='1'>
      <label>rear</label>
    </box>
    <box top='1476' left='3020' width='1003' height='753'>
      <label>rear</label>
    </box>
    <box top='1370' left='12' width='882' height='315' ignore='1'>
      <label>front</label>
    </box>
  </image>
    <image file='la_hill_st/la_hill_st_000002.jpg'>
    <box top='1460' left='3074' width='931' height='733'>
      <label>rear</label>
    </box>
    <box top='1248' left='2836' width='1103' height='385' ignore='1'>
      <label>rear</label>
    </box>
    <box top='1415' left='270' width='284' height='142'>
      <label>front</label>
    </box>
    <box top='1359' left='501' width='771' height='287' ignore='1'>
      <label>front</label>
    </box>
    <box top='1290' left='1308' width='415' height='399' ignore='1'>
      <label>front</label>
    </box>
  </image>
  <image file='la_hill_st/la_hill_st_000003.jpg'>
    <box top='1413' left='273' width='283' height='147'>
      <label>front</label>
    </box>
    <box top='1406' left='656' width='211' height='133'>
      <label>front</label>
    </box>
    <box top='1391' left='815' width='162' height='140'>
      <label>front</label>
    </box>
    <box top='1252' left='2736' width='1195' height='395' ignore='1'>
      <label>front</label>
    </box>
    <box top='1325' left='1453' width='205' height='195'>
      <label>rear</label>
    </box>
    <box top='1385' left='918' width='402' height='199'>
      <label>rear</label>
    </box>
    <box top='1454' left='3104' width='827' height='673'>
      <label>rear</label>
    </box>
  </image>
</images>

    """
soup = BeautifulSoup(contents)
images = soup.find_all('image')
for image in images:
    for box in image.find_all('box'):
        print(box)
