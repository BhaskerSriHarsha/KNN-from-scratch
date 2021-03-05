import numpy as np
import statistics

def manhattanDistance(image1, image2):
	return np.linalg.norm((image1 - image2), ord=1)

class KNN:
	
	def __init__(self,k,loss):
		self.k = k
		self.loss = loss
		print("\nInitialized KNN classifier with k=",str(k)," loss=",loss,"\n")
		
	def train(self, train_data,train_labels):
		print("\nBegan training")
		self.train_data = train_data
		self.train_labels = train_labels
		print("Completed training\n")
		
	def predict_sample(self, test_sample):
		distances = []		
		for index in range(self.train_data.shape[0]):
			train_sample = self.train_data[index]
			distance = manhattanDistance(test_sample,train_sample)
			distances.append(distance)
			
		# find the k closest sample indices
		k_nearest_indices = sorted(range(len(distances)), key = lambda sub: distances[sub])[:self.k]
		k_nearest_labels = self.train_labels[k_nearest_indices] 
		prediction = statistics.mode(k_nearest_labels)
#		print("\nNearest neighbours indices: ",k_nearest_indices)
#		print("Nearest neighbours labels: ",k_nearest_labels)
#		print("prediction: ",prediction)
		
		return prediction
		
	def predict(self, test_data):
		
		predictions = []
		print("Total samples:",test_data.shape[0])
		for index in range(test_data.shape[0]):
			print(index)
			# Extract individual sample from the test data
			test_sample = test_data[index]
			# Get predictions for that sample and append to list
			predictions.append(self.predict_sample(test_sample))
			# return the predictions
		return np.array(predictions)
		
	def evaluate(self, test_data, test_labels):
		
		predictions = self.predict(test_data)
		correct_count=0
		for i in range(predictions.shape[0]):
			if predictions[i] == test_labels[i]:
				correct_count += 1
		
		return (correct_count/predictions.shape[0])*100
			
		
		
		
		
			

	
	
