def quicksort(eip):
    if len(eip) <= 1:
        return eip
    mlblr = eip[len(eip) // 2]
    eip_in = [my_num_eip for my_num_eip in eip if my_num_eip < mlblr]
    eip_out = [my_num_eip for my_num_eip in eip if my_num_eip == mlblr]
    eip_list = [my_num_eip for my_num_eip in eip if my_num_eip > mlblr]
    return quicksort(eip_in) + eip_out + quicksort(eip_list)

print(quicksort([3,6,8,10,1,2,1]))


eip = 3
print(type(eip)) # Prints "<class 'int'>"
print(eip)       # Prints "3"
print(eip + 1)   # Addition; prints "4"
print(eip - 1)   # Subtraction; prints "2"
print(eip * 2)   # Multiplication; prints "6"
print(eip ** 2)  # Exponentiation; prints "9"
eip += 1
print(eip)  # Prints "4"
eip *= 2
print(eip)  # Prints "8"
mlblr = 2.5
print(type(mlblr)) # Prints "<class 'float'>"
print(mlblr, mlblr + 1, mlblr * 2, mlblr ** 2) # Prints "2.5 3.5 5.0 6.25"

eip = True
mlblr_in = False
print(type(eip)) # Prints "<class 'bool'>"
print(eip and mlblr_in) # Logical AND; prints "False"
print(eip or mlblr_in)  # Logical OR; prints "True"
print(not eip)   # Logical NOT; prints "False"
print(eip != mlblr_in)  # Logical XOR; prints "True"

mlblr_out = 'hello'    # String literals can use single quotes
elblr_in = "world"    # or double quotes; it does not matter.
print(mlblr_out)       # Prints "hello"
print(len(mlblr_out))  # String length; prints "5"
my_name = mlblr_out + ' ' + elblr_in  # String concatenation
print(my_name)  # prints "hello world"
my_name_12 = '%s %s %d' % (mlblr_out, elblr_in, 12)  # sprintf style string formatting
print(my_name_12)  # prints "hello world 12"

my_string = "hello"
print(my_string.capitalize())  # Capitalize a string; prints "Hello"
print(my_string.upper())       # Convert a string to uppercase; prints "HELLO"
print(my_string.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(my_string.center(7))     # Center a string, padding with spaces; prints " hello "
print(my_string.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"

elblr_out = [3, 1, 2]    # Create a list
print(elblr_out, elblr_out[2])  # Prints "[3, 1, 2] 2"
print(elblr_out[-1])     # Negative indices count from the end of the list; prints "2"
elblr_out[2] = 'foo'     # Lists can contain elements of different types
print(elblr_out)         # Prints "[3, 1, 'foo']"
elblr_out.append('bar')  # Add a new element to the end of the list
print(elblr_out)         # Prints "[3, 1, 'foo', 'bar']"
elblr_group = elblr_out.pop()      # Remove and return the last element of the list
print(elblr_group, elblr_out)      # Prints "bar [3, 1, 'foo']"


eip_list = list(range(5))     # range is a built-in function that creates a list of integers
print(eip_list)               # Prints "[0, 1, 2, 3, 4]"
print(eip_list[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(eip_list[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(eip_list[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(eip_list[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(eip_list[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
eip_list[2:4] = [8, 9]        # Assign a new sublist to a slice
print(eip_list)               # Prints "[0, 1, 8, 9, 4]"

mlblr_animals_list = ['cat', 'dog', 'monkey']
for eip_animal in mlblr_animals_list:
    print(eip_animal)
# Prints "cat", "dog", "monkey", each on its own line.

mlblr_animals = ['cat', 'dog', 'monkey']
for my_index_eip, eip_animal in enumerate(mlblr_animals):
    print('#%d: %s' % (my_index_eip + 1, eip_animal))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line

mlblr_nums_list = [0, 1, 2, 3, 4]
list_of_squares = []
for my_num_eip in mlblr_nums_list:
    list_of_squares.append(my_num_eip ** 2)
print(list_of_squares)   # Prints [0, 1, 4, 9, 16]

eip_nums_list = [0, 1, 2, 3, 4]
squares_eip_nums = [my_num_eip ** 2 for my_num_eip in eip_nums_list]
print(squares_eip_nums)   # Prints [0, 1, 4, 9, 16]

eip_eip_list = [0, 1, 2, 3, 4]
even_squares_eip = [my_num_eip ** 2 for my_num_eip in eip_eip_list if my_num_eip % 2 == 0]
print(even_squares_eip)  # Prints "[0, 4, 16]"

my_eip_dict = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(my_eip_dict['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in my_eip_dict)     # Check if a dictionary has a given key; prints "True"
my_eip_dict['fish'] = 'wet'     # Set an entry in a dictionary
print(my_eip_dict['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(my_eip_dict.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(my_eip_dict.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del my_eip_dict['fish']         # Remove an element from a dictionary
print(my_eip_dict.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"


my_dict_eip = {'person': 2, 'cat': 4, 'spider': 8}
for eip_animal in my_dict_eip:
    eip_legs = my_dict_eip[eip_animal]
    print('A %s has %d legs' % (eip_animal, eip_legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"

eip_dict = {'person': 2, 'cat': 4, 'spider': 8}
for eip_animal, eip_legs in eip_dict.items():
    print('A %s has %d legs' % (eip_animal, eip_legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"

mlblr_num_list = [0, 1, 2, 3, 4]
mlblr_num_squares_even = {my_num_eip: my_num_eip ** 2 for my_num_eip in mlblr_num_list if my_num_eip % 2 == 0}
print(mlblr_num_squares_even)  # Prints "{0: 0, 2: 4, 4: 16}"

eip_animals_dict = {'cat', 'dog'}
print('cat' in eip_animals_dict)   # Check if an element is in a set; prints "True"
print('fish' in eip_animals_dict)  # prints "False"
eip_animals_dict.add('fish')       # Add an element to a set
print('fish' in eip_animals_dict)  # Prints "True"
print(len(eip_animals_dict))       # Number of elements in a set; prints "3"
eip_animals_dict.add('cat')        # Adding an element that is already in the set does nothing
print(len(eip_animals_dict))       # Prints "3"
eip_animals_dict.remove('cat')     # Remove an element from a set
print(len(eip_animals_dict))       # Prints "2"

eip_animals_dict = {'cat', 'dog', 'fish'}
for my_index_eip, eip_animal in enumerate(eip_animals_dict):
    print('#%d: %s' % (my_index_eip + 1, eip_animal))
# Prints "#1: fish", "#2: dog", "#3: cat"

from math import sqrt
mlblr_set = {int(sqrt(eip_num)) for eip_num in range(30)}
print(mlblr_set)  # Prints "{0, 1, 2, 3, 4, 5}"

eip_tuple_dict = {(eip_num, eip_num + 1): eip_num for eip_num in range(10)}  # Create a dictionary with tuple keys
eip_tuple = (5, 6)        # Create a tuple
print(type(eip_tuple))    # Prints "<class 'tuple'>"
print(eip_tuple_dict[eip_tuple])       # Prints "5"
print(eip_tuple_dict[(1, 2)])  # Prints "1"

def sign(eip_num):
    if eip_num > 0:
        return 'positive'
    elif eip_num < 0:
        return 'negative'
    else:
        return 'zero'

for my_num_eip in [-1, 0, 1]:
    print(sign(my_num_eip))
# Prints "negative", "zero", "positive"

def hello(eip_name, eip_loud=False):
    if eip_loud:
        print('HELLO, %s!' % eip_name.upper())
    else:
        print('Hello, %s' % eip_name)

hello('Bob') # Prints "Hello, Bob"
hello('Fred', eip_loud=True)  # Prints "HELLO, FRED!"

class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"


import numpy as np

eip_array = np.array([1, 2, 3])   # Create a rank 1 array
print(type(eip_array))            # Prints "<class 'numpy.ndarray'>"
print(eip_array.shape)            # Prints "(3,)"
print(eip_array[0], eip_array[1], eip_array[2])   # Prints "1 2 3"
eip_array[0] = 5                  # Change an element of the array
print(eip_array)                  # Prints "[5, 2, 3]"

mlblr_array = np.array([[1, 2, 3], [4, 5, 6]])    # Create a rank 2 array
print(mlblr_array.shape)                     # Prints "(2, 3)"
print(mlblr_array[0, 0], mlblr_array[0, 1], mlblr_array[1, 0])   # Prints "1 2 4"


import numpy as np

eip_array = np.zeros((2, 2))   # Create an array of all zeros
print(eip_array)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

mlblr_array = np.ones((1, 2))    # Create an array of all ones
print(mlblr_array)              # Prints "[[ 1.  1.]]"

my_array = np.full((2, 2), 7)  # Create a constant array
print(my_array)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d_eip_array = np.eye(2)         # Create a 2x2 identity matrix
print(d_eip_array)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e_eip_array = np.random.random((2, 2))  # Create an array filled with random values
print(e_eip_array)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"


# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
eip_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
mlblr_array = eip_array[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(eip_array[0, 1])   # Prints "2"
mlblr_array[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(eip_array[0, 1])   # Prints "77"

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
eip_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_eip_array = eip_array[1, :]    # Rank 1 view of the second row of a
row_mlblr_array = eip_array[1:2, :]  # Rank 2 view of the second row of a
print(row_eip_array, row_eip_array.shape)  # Prints "[5 6 7 8] (4,)"
print(row_mlblr_array, row_mlblr_array.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_eip_array = eip_array[:, 1]
col_mlblr_array = eip_array[:, 1:2]
print(col_eip_array, col_eip_array.shape)  # Prints "[ 2  6 10] (3,)"
print(col_mlblr_array, col_mlblr_array.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"



eip_array = np.array([[1, 2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(eip_array[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([eip_array[0, 0], eip_array[1, 1], eip_array[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(eip_array[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([eip_array[0, 1], eip_array[0, 1]]))  # Prints "[2 2]"



eip_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

print(eip_array)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
mlblr_array = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(eip_array[np.arange(4), mlblr_array])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
eip_array[np.arange(4), mlblr_array] += 10

print(eip_array)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])



eip_array = np.array([[1, 2], [3, 4], [5, 6]])

bool_index_eip = (eip_array > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_index_eip)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(eip_array[bool_index_eip])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(eip_array[eip_array > 2])     # Prints "[3 4 5 6]"



eip_numpy_array = np.array([1, 2])   # Let numpy choose the datatype
print(eip_numpy_array.dtype)         # Prints "int64"

eip_numpy_array = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(eip_numpy_array.dtype)             # Prints "float64"

eip_numpy_array = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(eip_numpy_array.dtype)                         # Prints "int64"

eip_numpy_array = np.array([[1, 2], [3, 4]], dtype=np.float64)
mlblr_numpy_array = np.array([[5, 6], [7, 8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(eip_numpy_array + mlblr_numpy_array)
print(np.add(eip_numpy_array, mlblr_numpy_array))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(eip_numpy_array - mlblr_numpy_array)
print(np.subtract(eip_numpy_array, mlblr_numpy_array))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(eip_numpy_array * mlblr_numpy_array)
print(np.multiply(eip_numpy_array, mlblr_numpy_array))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(eip_numpy_array / mlblr_numpy_array)
print(np.divide(eip_numpy_array, mlblr_numpy_array))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(eip_numpy_array))

eip_numpy_array = np.array([[1, 2], [3, 4]])
mlblr_numpy_array = np.array([[5, 6], [7, 8]])

v_eip_numpy = np.array([9, 10])
w_eip_numpy = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v_eip_numpy.dot(w_eip_numpy))
print(np.dot(v_eip_numpy, w_eip_numpy))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(eip_numpy_array.dot(v_eip_numpy))
print(np.dot(eip_numpy_array, v_eip_numpy))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(eip_numpy_array.dot(mlblr_numpy_array))
print(np.dot(eip_numpy_array, mlblr_numpy_array))

eip_numpy_array = np.array([[1, 2], [3, 4]])

print(np.sum(eip_numpy_array))  # Compute sum of all elements; prints "10"
print(np.sum(eip_numpy_array, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(eip_numpy_array, axis=1))  # Compute sum of each row; prints "[3 7]"

eip_numpy_array = np.array([[1, 2], [3, 4]])
print(eip_numpy_array)    # Prints "[[1 2]
            #          [3 4]]"
print(eip_numpy_array.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v_eip_numpy = np.array([1, 2, 3])
print(v_eip_numpy)    # Prints "[1 2 3]"
print(v_eip_numpy.T)  # Prints "[1 2 3]"


# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip_numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v_eip_numpy = np.array([1, 0, 1])
mlblr_numpy_array = np.empty_like(eip_numpy_array)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    mlblr_numpy_array[i, :] = eip_numpy_array[i, :] + v_eip_numpy

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(mlblr_numpy_array)

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip_numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v_eip_numpy = np.array([1, 0, 1])
vv_mlblr_array = np.tile(v_eip_numpy, (4, 1))   # Stack 4 copies of v on top of each other
print(vv_mlblr_array)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
mlblr_numpy_array = eip_numpy_array + vv_mlblr_array  # Add x and vv elementwise
print(mlblr_numpy_array)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip_numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v_eip_numpy = np.array([1, 0, 1])
mlblr_numpy_array = eip_numpy_array + v_eip_numpy  # Add v to each row of x using broadcasting
print(mlblr_numpy_array)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"


# Compute outer product of vectors
v_eip_numpy = np.array([1, 2, 3])  # v has shape (3,)
w_eip_numpy = np.array([4, 5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v_eip_numpy, (3, 1)) * w_eip_numpy)

# Add a vector to each row of a matrix
eip_numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(eip_numpy_array + v_eip_numpy)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((eip_numpy_array.T + w_eip_numpy).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(eip_numpy_array + np.reshape(w_eip_numpy, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(eip_numpy_array * 2)


from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
eip_image = imread('cat.jpg')
print(eip_image.dtype, eip_image.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
tinted_eip_image = eip_image * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
tinted_eip_image = imresize(tinted_eip_image, (300, 300))

# Write the tinted image back to disk
imsave('cat_tinted.jpg', tinted_eip_image)

import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
eip_points = np.array([[0, 1], [1, 0], [2, 0]])
print(eip_points)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
eip_eud_dist = squareform(pdist(eip_points, 'euclidean'))
print(eip_eud_dist)


import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
eip_sin_x = np.arange(0, 3 * np.pi, 0.1)
eip_sin_y = np.sin(eip_sin_x)

# Plot the points using matplotlib
plt.plot(eip_sin_x, eip_sin_y)
plt.show()  # You must call plt.show() to make graphics appear.

import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
eip_x_coordinate = np.arange(0, 3 * np.pi, 0.1)
eip_y_sin = np.sin(eip_x_coordinate)
mlblr_y_cos = np.cos(eip_x_coordinate)

# Plot the points using matplotlib
plt.plot(eip_x_coordinate, eip_y_sin)
plt.plot(eip_x_coordinate, mlblr_y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

# Compute the x and y coordinates for points on sine and cosine curves
eip_x_coordinate = np.arange(0, 3 * np.pi, 0.1)
eip_y_sin = np.sin(eip_x_coordinate)
mlblr_y_cos = np.cos(eip_x_coordinate)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(eip_x_coordinate, eip_y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(eip_x_coordinate, mlblr_y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

mlblr_img = imread('cat.jpg')
img_tinted_mlblr = mlblr_img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(mlblr_img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted_mlblr))
plt.show()