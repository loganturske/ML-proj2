import numpy
import pandas
from copy import deepcopy

glass_data_set = None
iris_data_set = None
spam_data_set = None
def subtract_points(a,b):
	return a - b

def square_it(a):
	return a * a
def divide_a_by_b(a,b):
	if b == 0:
		return 0
	return a/b

def get_max_of(a, b):
	if a > b:
		return a
	return b
def take_square_root(a):
	return numpy.sqrt(int(a))

def euclidean_dist(arr1, arr2):
	total = 0

	for x,y in zip(arr1,arr2):
		sub = subtract_points(x,y)
		sqr = square_it(sub)
		total = total + sqr
	return take_square_root(total)

def get_absolute_value(x):
	return numpy.absolute(x)

def move_id_from_glass_dataframe():
	global glass_data_set
	# glass_data_set = glass_data_set.drop(['Id'], axis=1)
	cols = list(glass_data_set.columns.values)
	cols.pop(cols.index('Id'))
	glass_data_set = glass_data_set[cols+['Id']]
	pass

def read_glass_csv():
	global glass_data_set
	glass_data_set = pandas.read_csv('glass.csv')
	pass

def read_iris_csv():
	global iris_data_set
	iris_data_set = pandas.read_csv('iris.csv')
	pass

def get_num_columns(df):
	return len(df.columns)

def get_num_rows(df):
	return df.shape[0]

def get_max_value_of_data_set(df):
	d = df.max(numeric_only=True)
	return d.max()

def read_spam_csv():
	global spam_data_set
	spam_data_set = pandas.read_csv('spam.csv')
	pass

def give_initial_random_k_means(num_of_k, num_of_dim, max_num):
	return numpy.random.randint(max_num, size=(num_of_k, num_of_dim))

# def change_in_centroids(old, new):
# 	change = False
# 	if 'Class' in old.columns:
# 		old= old.drop(['Class'], axis=1)
# 	if 'Class' in new.columns:
# 		new = new.drop(['Class'], axis=1)
# 	if 'cluster' in old.columns:
# 		old = old.drop(['cluster'], axis=1)
# 	if 'cluster' in new.columns:
# 		new = new.drop(['cluster'], axis=1)

# 	for row,row2 in zip(old.iterrows(),new.iterrows()):
# 		print row.values
# 		for col in row[0].columns:
# 			diff = subtract_points(row[col], row2[col])
# 			diff = get_absolute_value(diff)
# 			if diff > .5:
# 				change = True
# 	return change

# def find_closest_centroid(point, centroids):

# 	argmin = None
# 	chosen_centroid = None
# 	i = 0

# 	for index,cent in centroids.iterrows():
# 		if argmin is None:
# 			argmin = euclidean_dist(point.values, cent.values)
# 			chosen_centroid = i
# 		if euclidean_dist(point.values, cent) < argmin:
# 			argmin = euclidean_dist(point.values, cent.values)
# 			chosen_centroid = i
# 		i += 1

# 	# print "CHOSE: " + str(chosen_centroid)
# 	return chosen_centroid

# def recalculate_new_centroids(df, centroids):

# 	if 'Class' in centroids.columns:
# 		centroids = centroids.drop(['Class'], axis=1)
# 	new_centroids = pandas.DataFrame()	
# 	i = 0
# 	for index,cent in centroids.iterrows():
# 		clusters = df.loc[df['cluster'] == i]
# 		if clusters.empty:
# 			new_centroids = new_centroids.append(cent, ignore_index=True)
# 		else:
# 			centroid = clusters.mean(skipna=True)
# 			centroid['cluster'] = i
# 			new_centroids = new_centroids.append(centroid, ignore_index=True)
# 		i += 1

# 	if 'Class' in new_centroids.columns:
# 			new_centroids = new_centroids.drop(['Class'], axis=1)
# 	return new_centroids

def remove_ele(arr, ele):
	# Remove the element from the arr
	index = numpy.argwhere(arr==ele)
	arr = numpy.delete(arr, index, None)
	return arr
def get_num_of_k(data):
	# Get the data and use as a matrix
	data = data.as_matrix()
	# Get all of the rows with just the last column
	arr = [data[j][-1] for j in range(len(data))]
	# Get all of the unique things that were in the last rows
	# these should be the number of classes
	k = numpy.unique(arr)
	# Return the number of k
	return k.size


def dist1(a, b):
	# Get distance for points compared to centroids and 
	# return the one that is the closest

	# Set a minimim to a super high number
	mini = 100000
	i = 0
	r = 0
	# For each centroid which should be 'b'
	for cent in b:
		# Set a total to 0
		total = 0
		# Get calculation for under the square root 
		for x,y in zip(a,cent):
			sub = subtract_points(x,y)
			sqr = square_it(sub)
			total = total + sqr
		t = take_square_root(total)
		# If you are less than than the minimum, set it
		# keep track of what centroid is closest
		if t < mini:
			r = i
			mini = t
		i += 1
	# return centroid number that is closest
	return r
def dist(a, b, ax=1):
	# This will get the distance between two vectors
	return numpy.linalg.norm(a - b, axis=ax)

def ai(point, points, C, clusters, k):

	# Get all of the points that are in your clusters
	other_points = [points[j][:-1] for j in range(len(points)) if clusters[j] == C]
	# Set a total to 0
	total = 0
	# For each point in you cluster
	for r in range(len(other_points)):
		# Add to the total the distance to the other points
		total += euclidean_dist(point, other_points[r])
	# Divide the total by the number of points to get average distance
	return divide_a_by_b(total, len(other_points))


def bi(point, points, C, clusters, k):
	# Set a super high minimum
	mini = 100000000
	# For each k, which is the number of centroids
	for i in range(k):
		# If the centroid is the cluster you are assigned, skip it
		if i == C:
			continue
		# Get all other points that are assigned different clusters
		other_points = [points[j][:-1] for j in range(len(points)) if clusters[j] == i]
		# If there are no other points, skip
		if not other_points:
			continue
		# Set a total to 0
		total = 0
		# For each of the other points
		for r in range(len(other_points)):
			# Add to the total the distance to the other points
			total += euclidean_dist(point, other_points[r])
			# print "Total: " + str(total)

		# Divide the total by the number of points
		final = divide_a_by_b(total, len(other_points))
		# If the final equation was less than the minimum, set it
		if final < mini:
			mini = final
			# print "SETTING"
	# Return the minimum
	return mini

def sil_coe(point, points, C, clusters, k):
	# Sillouette coeficient for the point passed in
	# Get the b sub i
	b_i = bi(point, points, C, clusters, k)
	# print "B: " + str(b_i)

	# Get the a sub i
	a_i = ai(point, points, C, clusters, k)
	# print "A: " + str(a_i)

	# The top of the equation is b_sub_i minus a_sub_i
	top = b_i - a_i
	# The bottom of the equation is the maximum of either a_sub_i or b_sub_i
	bottom = get_max_of(a_i, b_i)
	# Return the top divided by the bottom
	return divide_a_by_b(top, bottom)

def overall_sil(points, clusters, k):
	# This is the sillouette coefficient for the points given to it
	# d is the number of points
	d = len(points)
	# Set a total to 0
	total = 0
	# for each 
	for i in range(d):
		# Set what cluster the point to evaluate is associated with
		C = clusters[i]
		# Add to the total
		total += sil_coe(points[i][:-1], points, C, clusters, k)
	# Return total divided by the number of points
	return divide_a_by_b(total, d)


def stepwise_backward_feature_selection(data, features):
	
	features= numpy.asarray(features.tolist()[:-1])
	k = get_num_of_k(data)
	f_sub_0 = features
	basePref = -100000

	while features.size != 1:
		bestPerf = -100000
		bestF = None
		for candidate in features:
			f_sub_0 = remove_ele(f_sub_0, candidate)
			num_of_features = f_sub_0.size
			currPerf = k_means(data,k,num_of_features)
			if currPerf > bestPerf:
				bestPerf = currPerf
				bestF = candidate
			f_sub_0 = numpy.append(f_sub_0, candidate)
		if bestPerf > basePref:
			basePref = bestPerf
			features = remove_ele(features, bestF)
			f_sub_0 = remove_ele(f_sub_0, bestF)
		else:
			break
	print basePref
	print f_sub_0

def stepwise_forward_feature_selection(data, features):
	# Get all of the features from data and remove the last column
	features= numpy.asarray(features.tolist()[:-1])
	# The number of classifiers there are
	k = get_num_of_k(data)
	# An empty array to put features
	f_sub_0 = numpy.asarray([])
	# The base performace that you will run against
	basePref = -100000

	# While you still have features to test out
	while features.size != 0:
		# Best preformance so far
		bestPerf = -100000
		# Best feature that you found so far
		bestF = None
		# For each feature in the features list
		for candidate in features:
			# Add the feature to the array to test the algo
			f_sub_0 = numpy.append(f_sub_0, candidate)
			# Number of features we are testing
			num_of_features = f_sub_0.size
			# Run the algo and get performance score back
			currPerf = k_means(data,k,num_of_features)
			# If the performace of the last run is better than the best so far
			# set it. And make the best feature the one you just used
			if currPerf > bestPerf:
				bestPerf = currPerf
				bestF = candidate
			# Remove the element from the features to try
			f_sub_0 = remove_ele(f_sub_0, candidate)
		# If the best performance so far is better than the base
		# set it then remove the element from the features to test
		# then put the best feature in the set to try it
		if bestPerf > basePref:
			basePref = bestPerf
			features = remove_ele(features, bestF)
			f_sub_0 = numpy.append(f_sub_0, bestF)
		else:
			break
	print "\t Silhouette Coefficient:" + str(basePref)
	print "\t Features: "+ str(f_sub_0)

def k_means(data, k, num_of_features):
	# Make a matrix out of the data
	X = data.as_matrix()
	# Get k random points from the data
	C =  X[numpy.random.choice(X.shape[0], k, replace=False), :]
	# Remove the last col
	C = [C[j][:-1] for j in range(len(C))]
	# Turn it into a numpy array
	C = numpy.asarray(C)
	# To store the value of centroids when it updates
	C_old = numpy.zeros(C.shape)
	# Make an array that will assign clusters to each point
	clusters = numpy.zeros(len(X))
	# Error func. - Distance between new centroids and old centroids
	error = dist(C, C_old, None)
	# Loop will run till the error becomes zero of 5 tries
	tries = 0
	while error != 0 and tries < 1:
		# Assigning each value to its closest cluster
		for i in range(len(X)):
			# Get closest cluster in terms of distance
			clusters[i] = dist1(X[i][:-1], C)
		# Storing the old centroid values
		C_old = deepcopy(C)
		# Finding the new centroids by taking the average value
		for i in range(k):
			# Get all of the points that match the cluster you are on
			points = [X[j][:-1] for j in range(len(X)) if clusters[j] == i]
			# If there were no points assigned to cluster, put at origin
			if not points:
				C[i][:] = numpy.zeros(C[i].shape)
			else:
				# Get the average of all the points and put that centroid there
				C[i] = numpy.mean(points, axis=0)
		# Erro is the distance between where the centroids use to be and where they are now
		error = dist(C, C_old, None)
		# Increase tries
		tries += 1
	return overall_sil(X,clusters,k)

read_glass_csv()
read_iris_csv()
move_id_from_glass_dataframe()
read_spam_csv()
print "Iris SFS: "
stepwise_forward_feature_selection(iris_data_set, iris_data_set.columns)
# print
# print "Iris SBS: "
# stepwise_backward_feature_selection(iris_data_set, iris_data_set.columns)

# print 

print "Glass SFS: "
stepwise_forward_feature_selection(glass_data_set, glass_data_set.columns)
# print
# print "Glass SBS: "
# stepwise_backward_feature_selection(glass_data_set, glass_data_set.columns)

# print 

# print "Spam SFS: "
# stepwise_forward_feature_selection(spam_data_set, spam_data_set.columns)
# print
# print "Spam SBS: "
# stepwise_backward_feature_selection(spam_data_set, spam_data_set.columns)
# k_means(iris_data_set,3)
# stepwise_forward_feature_selection(iris_data_set)
# stepwise_forward_feature_selection(iris_data_set)