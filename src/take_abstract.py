import urllib3, re, os
from bs4 import BeautifulSoup
import certifi

http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where())


def take(pmid):
    '''
    retrieves the abstract with id pmid from PUBMED and saves it in path
    in a standard format
    '''
    global http

    path = os.path.join(os.getcwd(), 'new_data')

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
    output = str(soup.pmid) + '\n' + abstract
    out_file = open(os.path.join(path, pmid + '.xml'), 'w')
    out_file.write(output)
    out_file.close()


def take_title(pmid):
    '''
    returns the title of the document registered with pmid
    :param pmid:str
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
