# Clean rap generation data

f = open("raplyrics.txt")
g = open("cleanedRapLyrics.txt","w")
data = f.read()

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
data = data.replace(',',' ,')
data = data.replace('!','')
data = data.replace('?','')
data = data.replace('"','')
data = data.replace('.','')
data = data.replace('. . .','')
data = data.replace('-',' - ')
data = data.replace(']','')
data = data.replace(')','')



g.write(data)
f.close()
g.close()