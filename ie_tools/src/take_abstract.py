"""
@author: ZacVND

Retrieve abstracts and their respective titles
The logic is from Antonio Trenta's code but now written in Python 3
"""

from bs4 import BeautifulSoup
import urllib3
import certifi
import re
import os

import ie_tools.src.util as util

http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())


def take(pmid):
    '''
    retrieves the abstract with id pmid from PUBMED and saves it in path
    in a standard format
    '''
    global http

    path = util.get_new_data_dir()

    url = 'https://www.ncbi.nlm.nih.gov/pubmed/' + str(pmid) + \
          '?report=xml&format=text'

    response = http.request('GET', url)

    xmlpage = response.data.decode('utf-8')
    xmlpage = xmlpage.replace('&lt;', '<')
    xmlpage = xmlpage.replace('&gt;', '>')

    soup = BeautifulSoup(xmlpage, "html5lib")

    if soup.abstract:
        abstract = soup.abstract
    else:
        print('*'),
        soup = BeautifulSoup(xmlpage, "xml")
        title = soup.Articletitle.encode('utf-8')
        abstract = soup.Abstract.encode('utf-8')
        abstract = re.sub(r'\<Abstract', '<abstract', abstract)
        abstract = re.sub(r'\<\/Abstract', '</abstract', abstract)
        abstract = re.sub(r'bstractText', 'bstracttext', abstract)
        abstract = re.sub(r'Label\=', 'label=', abstract)
        abstract = re.sub(r'NlmCategory\=', 'nlmcategory=', abstract)

    abstract = re.sub(r'\n\s+', '\n', str(abstract))
    title_tag = soup.new_tag('title')
    title_tag.string = soup.articletitle.text

    output = str(soup.pmid) + '\n' + str(title_tag) + '\n' + abstract
    out_file = open(os.path.join(path, pmid + '.xml'), 'w')
    out_file.write(output)
    out_file.close()


def take_title(pmid):
    '''
    Use to populate the old dataset with their corresponding titles
    '''

    global http

    url = 'https://www.ncbi.nlm.nih.gov/pubmed/' + str(pmid) + \
          '?report=xml&format=text'

    response = http.request('GET', url)

    xmlpage = response.data.decode('utf-8')
    xmlpage = xmlpage.replace('&lt;', '<')
    xmlpage = xmlpage.replace('&gt;', '>')

    soup = BeautifulSoup(xmlpage, "html5lib")
    if soup.articletitle:
        return soup.articletitle.text
    else:
        return None


def main():
    # change new_pmids to values you want to collect the abstracts
    # new_pmids = [28777224, 28400374, 28178150, 25393036, 24844538, 28887006,
    #              27269279, 21921953, 27567675, 26907933, 24247999, 25907999,
    #              25893514, 25270273, 25908001, 25847610, 30022618, 29133641]
    #
    # for pmid in new_pmids:
    #     print("processing #{}".format(pmid))
    #     take(str(pmid))

    take_title(12345)

    # populate data with title tags
    # paper_paths = util.get_paper_paths()
    # for file_path in sorted(paper_paths):
    #     with open(file_path, 'r+') as f:
    #         raw = f.read()
    #         soup = BeautifulSoup(raw, "html5lib")
    #         pmid = soup.pmid.text
    #         print("processing paper #", pmid)
    #         title = take_title(pmid)
    #         title_tag = soup.new_tag('title')
    #         title_tag.string = title
    #         f.write("\n" + str(title_tag))


if __name__ == "__main__":
    main()
