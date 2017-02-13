from bs4 import BeautifulSoup
import urllib2

def scrap(artist,title):
    artist = artist.lower().replace(" ","")
    title = title.lower().replace(" ","")
    url = 'http://azlyrics.com/lyrics/'+artist+'/'+title +'.html'
    # url = "http://www.azlyrics.com/lyrics/jaysean/rideit.html"

    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    html = response.read()

    soup = BeautifulSoup(html,"html.parser")
    lyrics = soup.find_all("div", attrs={"class": None, "id": None})
    lyrics = [x.getText() for x in lyrics]
    return lyrics

f = open("listRapSongs.txt")
songsList = f.read()
songs = songsList.split("\n")
artists = []
songNames = []
for s in songs:
    a,sn = s.split("-")
    artists.append(a)
    songNames.append(sn)

g = open("rapLyrics.txt","a")

for a,sn in zip(artists,songNames):
    lyrics = scrap(a,sn)
    print a,sn
    try:
        g.write(lyrics[0])
    except:
        print a,sn," failed"
        continue

f.close()
g.close()