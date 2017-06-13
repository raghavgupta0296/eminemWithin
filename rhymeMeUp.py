import pronouncing as pr
import numpy as np

# word = "me"
# print(pr.phones_for_word(word))
# a = (pr.phones_for_word(word))
# print(a[0])
# print(pr.search(a[0]))
# print(pr.rhymes(word))
#
# aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
# d = np.random.choice(aa_milne_arr)
# print(d)

# w1 = "a"
# w1 = pr.phones_for_word(w1)
# w2 = "pea"
# w2 = pr.phones_for_word(w2)
# print(w1)
# print(w2)
# l = pr.rhyming_part(w1[0])+ " " +pr.rhyming_part(w2[0])
# print (pr.search(l))

a = pr.phones_for_word("empty")
print(a)
l = pr.search(a[0])
find = l[0]
for i in l:
    if pr.phones_for_word(i)==a:
        find = i
        print(i)
print(find)