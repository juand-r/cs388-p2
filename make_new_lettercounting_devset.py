with open("data/text8-100k.txt","r") as fd:
    x = fd.read()


#x300 = [x[i:i+300] for i in range(0,len(x)-350,350)]
x100 = [x[i:i+100] for i in range(0,len(x),100)]








#NOTE wait a minute if I use this dataset I will leak training text into my dev set..

#with open("data/lettercounting-dev20.txt","r") as fd:
#    x= fd.readlines()

#x = [i.replace("\n","") for i in x]




