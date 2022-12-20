# python3
import requests
import time
import os


def checkDir():
    path = "scrapedImages"
    isExist = os.path.exists(path)
    print(f"Directory exists? {isExist}") if isExist == True else os.mkdir(
        path)


def download(fileName):
    currentDirectory = os.getcwd()
    dirpath = "scrapedImages/"
    fullpath = os.path.join(currentDirectory, dirpath)

    filelocation = os.path.join(fullpath, fileName)
    f = open(filelocation, 'wb')
    f.write(requests.get('https://thispersondoesnotexist.com/image',
                         headers={'User-Agent': 'My User Agent 1.0'}).content)
    f.close()


checkDir()
couter = 5  # change the counter to increase count of scraped images
print("Scraping images...")
for i in range(couter):
    time.sleep(1)
    download(str(i)+'.jpg')
print("Done!")
