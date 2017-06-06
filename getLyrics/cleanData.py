# Clean rap generation data

f = open("rapLyrics.txt")
g = open("cleanedRapLyrics.txt","w")
data = f.read().lower()

d = ""
x=1
for i in data:
    if i == "[":
        x = 0
    elif i == "]":
        x = 1
    if x == 0:
        continue
    else:
        d = d + str(i)

data = d

d = ""
x=1
for i in data:
    if i == "(":
        x = 0
    elif i == ")":
        x = 1
    if x == 0:
        continue
    else:
        d = d + str(i)

data = d

d = ""
x=1
for i in data:
    if i == "{":
        x = 0
    elif i == "}":
        x = 1
    if x == 0:
        continue
    else:
        d = d + str(i)

data = d
# Remove [] words... yet to do : remove words that come b/w " ' ", " [] ", " () "
# for i in data.split():
#     if i[0]=="[" and i[-1]=="]":
#         data = data.replace(i,"")
#
# for i in data.split():
#     if i[0]=="(" and i[-1]==")":
#         data = data.replace(i,"")
#
# for i in data.split("\n"):
#     try:
#         if i[0][0]=="[" and i[-1][-1]=="]":
#             data = data.replace(i,"")
#     except:
#         continue


# Remove ',' spaces, '!' spaces
data = data.replace('in\'','ing')
data = data.replace(',','')
data = data.replace('!','')
data = data.replace('?','')
data = data.replace('#','')
data = data.replace('$','')
data = data.replace('©','')
data = data.replace('à','a')
data = data.replace('ä','a')
data = data.replace('å','a')
data = data.replace('—','-')
data = data.replace('–','-')
data = data.replace('ç','c')
data = data.replace('é','e')
data = data.replace('ü','u')
data = data.replace('"','')
data = data.replace('‘','')
data = data.replace('’','')
data = data.replace('“','')
data = data.replace('”','')
data = data.replace('…','')
data = data.replace('.','')
data = data.replace('. . .','')
data = data.replace('-',' - ')
data = data.replace(']','')
data = data.replace(')','')
data = data.replace('}','')
data = data.replace('\n\n','\n')
data = data.replace('\n\n\n','\n')

print(len(data))

data = data.split("\n")
lines_seen = set()
for line in data:
    if line not in lines_seen:
        g.write(line + "\n")
        lines_seen.add(line)

f.close()
g.close()

print("press enter")
input()
g = open("cleanedRapLyrics.txt","r")
data = g.read().lower()
print(data)
print(len(data))