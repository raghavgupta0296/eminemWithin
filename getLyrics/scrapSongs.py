from urllib.request import urlopen
from bs4 import BeautifulSoup

artist = ["eminem","drdre"]
s = []

def songs(artist):
    html = urlopen("http://www.azlyrics.com/"+artist[0]+"/"+artist+".html")

    soup = BeautifulSoup(html,"html.parser")
    songList = soup.find_all("div",attrs={"class":None,"id":"listAlbum"})
    songList = songList[0].find_all("a")
    for i in range(1,len(songList)):
        a = str(songList[i].get("href"))
        if "intro" not in a:
            if a[0:2] == "..":
                a = a[2:]
                a = "http://www.azlyrics.com"+a
                s.append(a)
            elif a[0:2]=="ht":
                s.append(a)

for a in artist:
    songs(a)
# print(s)

# g = open("songList.txt",'w')
# for e in s:
#     g.write(e+"\n")
# g.close()
# g = open("songList.txt",'r')
# s = g.read()
# s = s.split("\n")
# g.close()

def get_lyrics(song):
    html = urlopen(song)
    soup = BeautifulSoup(html,"html.parser")
    lyrics = soup.find_all("div", attrs={"class": None, "id": None})
    lyrics = [x.getText() for x in lyrics]
    return lyrics

bad = 0
f = open("lyrics2.txt",'a')
for e in s:
    try:
        l = get_lyrics(e)
        f.write(l[0] +"\n")
    except:
        print(e)
        bad +=1
        continue
print (bad)
f.close()
