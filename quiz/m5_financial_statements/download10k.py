import requests
from bs4 import BeautifulSoup as Soup


def get_links(cik, priorto, count):
    link = "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="+ \
        str(cik)+"&type=10-K&dateb="+str(priorto)+"&owner=exclude&output=xml&count="+str(count)
    
    # parse the website and extract links
    data = requests.get(link).text
    # print("see tentative links for all documents:")
    # print(link)
    
    soup = Soup(data, "lxml")
    # store the link in the list
    links = []

    # If the link is .htm convert it to .html
    for link in soup.find_all('filinghref'):

        # convert http://*-index.htm to http://*.txt
        url = link.string
        if link.string.split(".")[len(link.string.split("."))-1] == "htm":
            url += "l"
        required_url = url.replace('-index.html', '')
        txtdoc = required_url + ".txt"
        #docname = txtdoc.split("/")[-1]
        links.append(txtdoc)
    return links

# clean up the soup we construct from the links
def clean_soup(link):
    data = requests.get(link).text
    soup = Soup(data, "lxml")
    blacklist = ["script", "style"]
    attrlist = ["class", "id", "name", "style", 'cellpadding', 'cellspacing']
    skiptags = ['font', 'a', 'b', 'i', 'u']
    
    for tag in soup.findAll():
        if tag.name.lower() in blacklist:
            # blacklisted tags are removed in their entirety
            tag.extract()

        if tag.name.lower() in skiptags:
            tag.replaceWithChildren()
            
        for attribute in attrlist:
            del tag[attribute]
            
                    
    return soup


import unicodedata

# normalize the text
# remove some escape characters
def normtxt(txt):
    return unicodedata.normalize("NFKD",txt)

# get section from 10K
# looks for the term "item 1a" and collects text until "item 1b" is found
# returns None if there is no appropriate section found
# raise error when it cannot find the end of the section

def extract_section(soup, section='1a', section_end='1b'):
    
    search_next = ["p", "div", "table"]
    
    # loop over all tables
    items = soup.find_all(("table", "div"))

    myitem = None
    
    search_txt = ['item '+ section ]
    
    for i, item in enumerate(items):
        txt = normtxt(item.get_text())
        
        # this is to avoid long sentences or tables that contain the item
        if len(txt.split()) > 5:
            continue
        if any([w in txt.lower() for w in search_txt]):
            myitem = item
            
    if myitem is None:
        # print("section not found, returned None")
        return None
        
    lines = ""
    des = myitem.find_next(search_next)
    
    search_txt = [ 'item '+section_end ]

    while not des is None:
        des = des.find_next(search_next)
        
        if des is None:
            raise ValueError("end section not properly found")
            
        if any([w in normtxt(des.get_text()).lower() for w in search_txt]):
            break
            
        elif des is not None:
            if len(des.get_text().split()) > 2 and '|' not in normtxt(des.get_text()):
                # need to get rid of escape characters
                lines += normtxt(" "+des.get_text())
            #elif len(des.get_text().split()) > 2:
                #print("removing text: ",des.get_text())
            
        else:
            continue
    
    return lines[1:]
    
    
import os.path


def get_files(cik, company, n=5, max_n=20):
    mylinks = get_links(cik, '20170601', str(max_n))
    dates = range(2017, 1000, -1)
    print("downloading 10-Ks item 1A for CIK =",cik, "...")
    result_txt = []
    i=0
    for link in mylinks:
        filename = company+"_10k_"+str(dates[i])+".txt"

        if os.path.isfile(filename):
            print("skipping "+filename)
            i+=1
            
            if i >= n:
                break

            continue

        soup = clean_soup(link)
        section = extract_section(soup)
        
        if section is None:
            continue
        
        print("writing "+filename+"...")
        
        with open(filename,"w") as f:
            f.write(section)
            
        i+=1

        if i >= n:
            break
    
    