from multiprocessing import Pool
from bs4 import BeautifulSoup
from tqdm import tqdm

import requests
import shutil
import os

def download(data):
    href            = data['href']
    url             = data['url']
    destination_dir = data['destination_dir']

    req = requests.get(f'{url}{href}', stream=True)
    with open(os.path.join(destination_dir, href), 'wb') as f:
        shutil.copyfileobj(req.raw, f)

def scrap(destination_dir):
    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)

    url   = 'https://www.vgmusic.com/music/other/miscellaneous/piano/'
    html  = requests.get(url)
    soup  = BeautifulSoup(html.text, 'html.parser')
    hrefs = [a['href'] for a in soup.find_all('a', href=True) if '.mid' in a['href']]
    datas = [
        {
            'href'           : href,
            'url'            : url,
            'destination_dir': destination_dir
        }
        for href in hrefs
    ]

    pbar = tqdm(datas, total=len(datas), desc=f'Downloading dataset from {url}')
    with Pool(6) as pool:
        list(pool.imap(download, pbar))

    url   = 'https://utapriforever1.webs.com/'
    html  = requests.get(url)
    soup  = BeautifulSoup(html.text, 'html.parser')
    hrefs = [a['href'].split('/')[-1] for a in soup.find_all('a', href=True) if '.mid' in a['href']]
    datas = [
        {
            'href'           : str(href),
            'url'            : url,
            'destination_dir': destination_dir
        }
        for href in hrefs
    ]

    pbar = tqdm(datas, total=len(datas), desc=f'Downloading dataset from {url}')
    with Pool(6) as pool:
        list(pool.imap(download, pbar))
