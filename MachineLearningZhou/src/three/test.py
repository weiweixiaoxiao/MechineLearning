def count():
    fs = []
    for i in range(1, 4):
        def f(j):
            #def g():
            return j*j
            #return g
        print(f(3))    
        r = f(i)
        fs.append(r)
    return fs

f1, f2, f3 = count()
print (f1(), f2(), f3())