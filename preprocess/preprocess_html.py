#!/usr/bin/env python

#Original: Dato starter solution
#Modified: Alexey Svyatkovskiy

from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel

from bs4 import BeautifulSoup as bs
import os, sys, logging, string, glob
import cssutils as cu
import json

ferr = open("errors_in_scraping.log","w")

import re
def clean_text(text_as_list):
    #map(lambda x: re.sub(' +',' ',x),text_as_list)
    text_as_string = " ".join(text_as_list)
    text_as_string = text_as_string.encode("utf8").translate(None,'=@&$/%?<>,[]{}()*.0123456789:;-\n\'"_').lower()
    text_as_string = re.sub(' +',' ',text_as_string)

    return text_as_string


def parse_page_rdd(input_page_as_tuple):
    page = input_page_as_tuple[1]
    soup = bs(page)

    filenameDetails = input_page_as_tuple[0].split("/")
    urlId = filenameDetails[-1].split('_')[0]
    #bucket = filenameDetails[-2]
    #return (urlId,[parse_text(soup),parse_title(soup),parse_links(soup),parse_images(soup)])

    doc = {
            "id":urlId,
            "text":parse_text(soup),
            "title":parse_title(soup ),
            "links":parse_links(soup),
            "images":parse_images(soup),
           }

    return doc


def parse_text(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - textdata: a list of parsed text output by looping over html paragraph tags
        note:
            - could soup.get_text() instead but the output is more noisy """
    textdata = ['']

    for tag in soup.find_all("div", {"class":"text"}):
        try:
           textdata.append(tag.text.encode('ascii','ignore').strip())
        except Exception:
           continue

    for text in soup.find_all('p'):
        try:
            textdata.append(text.text.encode('ascii','ignore').strip())
        except Exception:
            continue

    #FIXME if you need to clean it
    textdata = filter(None,textdata)
    textdata = clean_text(textdata)
    return textdata

def parse_title(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - title: parsed title """

    title = ['']

    try:
        title.append(soup.title.string.encode('ascii','ignore').strip())
    except Exception:
        return title

    return filter(None,title)

def parse_links(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - linkdata: a list of parsed links by looping over html link tags
        note:
            - some bad links in here, this could use more processing """

    linkdata = ['']

    for link in soup.find_all('a'):
        try:
            linkdata.append(str(link.get('href').encode('ascii','ignore')))
        except Exception:
            continue

    #return len(filter(None,linkdata))
    return filter(None,linkdata)


def parse_images(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - imagesdata: a list of parsed image names by looping over html img tags """
    imagesdata = ['']

    for image in soup.findAll("img"):
        try:
            imagesdata.append("%(src)s"%image)
        except Exception:
            continue

    #return len(filter(None,imagesdata))
    return filter(None,imagesdata)


def parse_input(x_rdd):
    sid,label = x_rdd.split(",")
    id = sid.split("_")[0]
    doc = {'id':id, 'label':int(label)}
    return doc

def main(argv):
    sc = SparkContext(appName="KaggleDato")

    #parse labels as JSON
    PATH_TO_TRAIN_LABELS = "file:///scratch/network/alexeys/KaggleDato/train_v2.csv"
    train_label_rdd = sc.textFile(PATH_TO_TRAIN_LABELS).filter(lambda x: 'file' not in x).map(lambda x: parse_input(x)).map(lambda x: json.dumps(x)).repartition(1).saveAsTextFile('/user/alexeys/KaggleDato/train_csv_json')

    nbuckets =  1
    for bucket in range(nbuckets):
        for section in range(1,2):
            print "Processing bucket ",bucket," section ", section
            fIn_rdd = sc.wholeTextFiles("file:///scratch/network/alexeys/KaggleDato/"+str(bucket)+"/"+str(section)+"*_raw_html.txt",10).map(parse_page_rdd).map(lambda x: json.dumps(x))
            fIn_rdd.repartition(1).saveAsTextFile('/user/alexeys/KaggleDato/'+str(bucket)+'_'+str(section)+'/')

if __name__ == "__main__":
   main(sys.argv)
