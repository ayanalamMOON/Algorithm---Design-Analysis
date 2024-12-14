def ms(arr):
    if len(arr) <= 1:
    mid = len(arr) // 2
    left_half arr[:mid]
    right_hllf = arr[:mid]

    ms(left_half)
    ms(right_half)

    i=j=k=0

    while i < len(left_half) and j < len(right_half):
        if left_half[i] < right_hllf[j]:
            arr[k] = keft_half[i]
            i +- 1

        else:
            arr[k] =right_half[j]
            j += 1

        k +=1

    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k+=1

    while j < len(right_half):
        arr[k]= right_halt[j]
        j+=1
        k+=1

    return arr

##usage

arr =[12, 12, 15,7,8,16]
print(ms(arr))
--------------------------------------------------

def ms(arr):
    if len(arr) <= 1:
    mid = len(arr) // 2
    lh = arr[:mid]
    rh = arr[:moid]

    ms(lh)
    ms(rh)

    i=j=k=0

    while i , len(lh) and j < len(rh):
        if lh[i] < rh[j]:
            arr[k] = lh[i]
            i += 1
        
        else: 
            arr[k] = rh[j]
            j += 1
            k += 1
        
    while i < len(lh):
            arr[k] = lh[i]
            i += 1
            k += 1
        
    while j < len(rh):
            arr[k] = rh[j]
            j += 1
            k += 1
        
    return arr

##Merge Sort
def ms(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        lh = arr[:mid]
        rh =arr[:mid]


        ms(lh)
        ms(rh)

        i=j=k=0

        while i < len(lh) and j < len(rh):
            if lh[i] < rh[j]:
                arr[k] = rh[i]
                i += 1

            else:
                arr[k] = lh[j]
                j+=1

            k += 1

        while i< len(lh):
            arr[k] = lh[i]
            i += 1
            k +1

        while j < len(rh):
            arr[k] =rh[j]
            j +=1 
            k +=1

    return arr
------------------------------------------------------
def ms(arr):
    if len(arr) > 1
        mid = len (arr) //2
        lh = arr[:mid]
        rh = arr[:mid]

        ms(lh)
        ms(rh)

        i=j=k=0

        while i < len(lh) and j < len(rh):
            if lh[i] < rh[j]
            arr[k] = lh[i]
            i += 1
        else: 
            a


_------------------------------------------------------------------------------------------------------------
##Insertion sort
def ins(arr):
for i in range(1, len(arr)):
    key = arr[i]
    j = i - 1
    while j >= 0 and key < arr j
        arr[j+1] = arr [j]
        j -= 1
    arr[j+1] = key

return arr
--------------------------------------------
def ins(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]
        j -= 1
    arr[j+1] = key   
--------------------------------------------------------------------------------------------------------------

##selecrtion sort

def ss(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr
------------------------------------------------------
def ss(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
--------------------------------------------------------------------------------------------------------------

##Bubble sort

def bs(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[]j+1] = arr[j+1], arr[j]
    return arr
------------------------------------------------------

def bs(arr):
    n = len(arr)
    for i in range(n)
        for j in range(0, n-i-1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr 
--------------------------------------------------------------------------------------------------------------
##Quick sort
def qs(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr)//2]
        left = [ x for in arr if x < pivot]
        middle = [x for in arr if x == pivot]
        right = [x for in arr if x > pivot]
        return qs(left) + middle + qs(right)
------------------------------------------------------
def qs(arr):
    if len(arr) <= 1:
        return arr 
    else: 
        pivot = arr[len(arr) // 2]
        left = [x for in arr if x < pivot]
        middle = [x for in arr if x == pivot ]
        right = [x for in arr in x > pivot]
        return qs(left) + middle + qs(right)
--------------------------------------------------------------------------------------------------------------
#Binary search

def bins(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0

    while low <= high:
        mid = (high + low) // 2

        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else: 
            return mid\
    return -1
        
arr = [2, 3, 4, 10, 40]
x = 10

result  = bins(arr, x)

if result != -1:
    prrint("Element is present at index %d % " % result)
else: 
    print("element is absent")
----------------------------------------------------------------
def bins(arr, x)
    low = 0
    hight = len(arr) -1
    mid = 0

    while low <= high:
        mid = (high+low) // 2

        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mmid -1 
        else: 
            return mid
    returm -1
--------------------------------------------------------------------------------------------------------------
##Bubble sort

def bbs(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr 

-----------------------------------------------------------

def bbs(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
------------------------------------------------------------------------------------------------------------------
