'''
if (year is not divisible by 4) then (it is a common year)
else if (year is not divisible by 100) then (it is a leap year)
else if (year is not divisible by 400) then (it is a common year)
else (it is a leap year) 


def findleapyears():
    a = []
    for i in range(10000):
        if i%4 != 0:
            e=0
        elif i%100 != 0:
            a.append(i)
        elif i%400 != 0:
            e=0
        else:
            a.append(i)
    return a
'''


#x = ((4-14) * 100 / 14) / 100 https://www.geeksforgeeks.org/python-pandas-dataframe-pct_change/

def perChange(x1,x2):
    return ((x2-x1)*100/x1)/100

print(perChange(4,4))
